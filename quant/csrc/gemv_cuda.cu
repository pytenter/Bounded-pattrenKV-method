// Inspired by https://github.com/ankan-ban/llama_cu_awq 
// and the official implementation of AWQ
/*

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

*/

#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include <algorithm>
#include <ATen/ATen.h>
#include "gemv_cuda.h"
#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MAX_CENTROIDS 64  // 够用：一般 M∈{8,16,32}


// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  for(int i = 4; i >= 0; i--){
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  /*
  // Equivalent to the following tree reduction implementation:
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  */
  return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor){
  return (c + divisor - 1) / divisor;
}


/*
Computes GEMV (group_size = 64).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g64(
  const float4* _inputs, const uint32_t* weight, const half* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 64;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    // This is essentially zeros_w.
    const int num_groups_packed = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
    // consistent with input shape
    const int sf_w = make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2 * PACK_FACTOR;
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w, sf_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed / 2; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g64: two threads -> 64 numbers -> 1 group; 1 warp = 16 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);
      float current_zeros =  __half2float(zeros[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * current_single_weight_fp + current_zeros;
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (group_size = 128).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g128(
  const float4* _inputs, const uint32_t* weight, const half* zeros, const half* scaling_factors, half* _outputs, 
  const int IC, const int OC){
    const int group_size = 128;
    float psum = 0;
    const int batch_idx = blockIdx.z;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
    half* outputs = _outputs + batch_idx * OC;
    const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
    const int weight_w = IC / PACK_FACTOR;
    // TODO (Haotian): zeros_w is incorrect, after fixing we got misaligned address
    const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
    // consistent with input shape
    const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;
    //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) printf("%d %d %d %d\n", IC, group_size, PACK_FACTOR, zeros_w);
    // tile size: 4 OC x 1024 IC per iter
    for(int packed_group_idx = 0; packed_group_idx < num_groups_packed; packed_group_idx++){
      // 1024 numbers in one iteration across warp. Need 1024 / group_size zeros.
      uint32_t packed_weights[4];
      // use float4 to load weights, each thread load 32 int4 numbers (1 x float4)
      *((float4*)(packed_weights)) = *((float4*)(weight + oc_idx * weight_w + packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));
      // load scaling factors
      // g128: four threads -> 128 numbers -> 1 group; 1 warp = 8 groups.
      float scaling_factor = __half2float(scaling_factors[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
      float current_zeros = __half2float(zeros[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);
      int inputs_ptr_delta = packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4; 
      const float4* inputs_ptr = inputs + inputs_ptr_delta;
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        // iterate over different uint32_t packed_weights in this loop
        uint32_t current_packed_weight = packed_weights[ic_0];
        half packed_inputs[PACK_FACTOR];
        // each thread load 8 inputs, starting index is packed_group_idx * 128 * 8 (because each iter loads 128*8)
        if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
          *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
          #pragma unroll
          for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++){
            // iterate over 8 numbers packed within each uint32_t number
            float current_single_weight_fp = (float)(current_packed_weight & 0xF);
            float dequantized_weight = scaling_factor * current_single_weight_fp + current_zeros;
            //if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && ic_0 == 0 && ic_1 == 0 && packed_group_idx == 0) printf("%f %f %f %f %X %X\n", dequantized_weight, current_single_weight_fp, scaling_factor, current_zeros, current_packed_weight, packed_zeros);
            psum += dequantized_weight * __half2float(packed_inputs[ic_1]);
            current_packed_weight = current_packed_weight >> 4;
          }
        }
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
     outputs[oc_idx] = __float2half(psum); 
    }
}


/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC, IC // 8];
  _zeros: int tensor of shape [OC, IC // G // 8];
  _scaling_factors: tensor of shape [OC, IC // G];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;

Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size)
{
    int num_in_feats = _in_feats.size(0);
    int num_in_channels = _in_feats.size(1);
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<float4*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [OC, IC]
    at::Tensor _out_feats = torch::empty({num_in_feats, _kernel.size(0)}, options);
    int num_out_feats = _out_feats.size(-2);
    int num_out_channels = _out_feats.size(-1);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    int blockDim_z = num_out_feats;
    dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
    dim3 num_threads(32, 4);
    if (group_size == 64)
    {
      gemv_kernel_g64<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels
      );
    }
    else if (group_size == 128)
    {
      gemv_kernel_g128<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels
      );
    }
    return _out_feats;
;}




/*
Computes Batched 4-bit GEMV (group_size = 64).

Args:
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  zeros: matrix of shape [BS, OC // group_size, IC];
  scaling_factors: matrix of shape [BS, OC // group_size, IC];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv4_kernel_outer_dim(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int nh, const int nh_kv){
    const int bit = 4;
    const int pack_factor = 8;
    const int batch_idx = blockIdx.x;
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; 
    const half* inputs = _inputs + batch_idx * IC;
    half* outputs = _outputs + batch_idx * OC;
    const int ratio = nh / nh_kv;
    int _batch_idx = batch_idx / ratio;
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    // 1float4 == 8 half number
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // each thread load 32 int4 number
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      // each thread load 32 int4 number
      // int weight_offset = packed_oc_idx * IC + k * TILE_DIM + threadIdx.x*4;
      // if (weight_offset < OC * IC / pack_factor)
      //   *((float4*)(qw)) = *((float4*)(weight + packed_oc_idx * IC + k * TILE_DIM + threadIdx.x*4));
      // int scale_mn_offset = group_idx * IC + k * TILE_DIM + threadIdx.x*4;
      // if (scale_mn_offset < OC * IC / group_size){
      //   *((float2*)(cscale)) = *((float2*)(scaling_factors + scale_mn_offset));
      //   *((float2*)(czero)) = *((float2*)(zeros + scale_mn_offset));
      // }
      // int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      // if (inputs_ptr_delta < IC){
      //   const half* inputs_ptr = inputs + inputs_ptr_delta;
      //   *((float2*)(inp)) = *((float2*)(inputs_ptr));
      // }
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            cur_packed_weight = cur_packed_weight >> bit;
            psum[ic_1] += dequantized_weight * cur_inp;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        psum[i] = warp_reduce_sum(psum[i]);
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}


__global__ void bgemv2_kernel_outer_dim(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int nh, const int nh_kv){
    // const int group_size = 64;
    const int bit = 2;
    const int pack_factor = 16;
    const int batch_idx = blockIdx.x;
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; 
    const int ICR = IC;
    const half* inputs = _inputs + batch_idx * ICR;
    half* outputs = _outputs + batch_idx * OC;
    const int ratio = nh / nh_kv;
    int _batch_idx = batch_idx / ratio;
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    // 1float4 == 8 half number
    float psum[pack_factor]{};
    for (int k=0; k < (ICR + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // each thread load 32 int4 number
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      // if (weight_offset < OC * ICR / pack_factor)
      //   *((float4*)(qw)) = *((float4*)(weight + packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4));
      // int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      // if (scale_mn_offset < OC * ICR / group_size){
      //   *((float2*)(cscale)) = *((float2*)(scaling_factors + scale_mn_offset));
      //   *((float2*)(czero)) = *((float2*)(zeros + scale_mn_offset));
      // }
      // int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      // if (inputs_ptr_delta < ICR){
      //   const half* inputs_ptr = inputs + inputs_ptr_delta;
      //   *((float2*)(inp)) = *((float2*)(inputs_ptr));
      // }
      // multiply 32 weights with 32 inputs
      #pragma unroll
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            cur_packed_weight = cur_packed_weight >> bit;
            psum[ic_1] += dequantized_weight * cur_inp;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        psum[i] = warp_reduce_sum(psum[i]);
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

// __global__ void bgemv2_kernel_g64_outer_dim(
//   const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
//   const int IC, const int OC){
//     const int group_size = 64;
//     const int bit = 2;
//     const int pack_factor = 16;
//     const int batch_idx = blockIdx.x;
//     const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
//     const int oc_start_idx = packed_oc_idx * pack_factor;
//     const int group_idx = oc_start_idx / group_size; 
//     const int ICR = IC;
//     const half* inputs = _inputs + batch_idx * ICR;
//     half* outputs = _outputs + batch_idx * OC;
//     const uint32_t*  weight = _weight + batch_idx * OC * IC / pack_factor;
//     const half* scaling_factors = _scale + batch_idx * OC * IC / group_size;
//     const half* zeros = _zeros + batch_idx * OC * IC / group_size;
//     const int TILE_DIM = 128;
//     const int num = 0xFF >> (8-bit);
//     // 1float4 == 8 half number
//     float psum[pack_factor]{};
//     for (int k=0; k < (ICR + TILE_DIM - 1) / TILE_DIM; k++){
//       uint32_t qw[4]{};
//       half cscale[4]{};
//       half czero[4]{};
//       half inp[4]{};
//       // each thread load 32 int4 number
//       int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
//       if (weight_offset < OC * ICR / pack_factor)
//         *((float4*)(qw)) = *((float4*)(weight + packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4));
//       int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
//       if (scale_mn_offset < OC * ICR / group_size){
//         *((float2*)(cscale)) = *((float2*)(scaling_factors + scale_mn_offset));
//         *((float2*)(czero)) = *((float2*)(zeros + scale_mn_offset));
//       }
//       int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
//       if (inputs_ptr_delta < ICR){
//         const half* inputs_ptr = inputs + inputs_ptr_delta;
//         *((float2*)(inp)) = *((float2*)(inputs_ptr));
//       }
//       // multiply 32 weights with 32 inputs
//       #pragma unroll
//       for (int ic_0 = 0; ic_0 < 4; ic_0++){
//         uint32_t cur_packed_weight =  qw[ic_0];
//         float cur_inp = __half2float(inp[ic_0]);
//         float cur_scale = __half2float(cscale[ic_0]);
//         float cur_zero = __half2float(czero[ic_0]);
//         for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
//           int oc_idx = oc_start_idx + ic_1;
//           if (oc_idx < OC){
//             float cur_single_weight_fp = (float)(cur_packed_weight & num);
//             float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
//             // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
//             cur_packed_weight = cur_packed_weight >> bit;
//             psum[ic_1] += dequantized_weight * cur_inp;
//           }
//         }
//       }
//     }
//     for (int i=0; i < pack_factor; i++){
//       int oc_idx = oc_start_idx + i;
//       if (oc_idx < OC){
//         psum[i] = warp_reduce_sum(psum[i]);
//         if (threadIdx.x == 0) 
//           outputs[oc_idx] = __float2half(psum[i]); 
//       }
//     }
// }


/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC // PACK_Factor, IC];
  _zeros: int tensor of shape [OC // G, IC];
  _scaling_factors: tensor of shape [OC // G, IC];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;
Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_forward_cuda_outer_dim(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv)
{
    int BS = _in_feats.size(0);
    int num_in_feats = _in_feats.size(1);
    int num_in_channels = _in_feats.size(2);
    int num_out_channels = _zeros.size(1) * group_size;
    // int kernel_volume = _out_in_map.size(1);
    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [OC, IC]
    at::Tensor _out_feats = torch::empty({BS, num_in_feats, num_out_channels}, options);
    int num_out_feats = _out_feats.size(-2);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    int pack_factor = 32 / bit;
    dim3 num_blocks(BS, (num_out_channels / pack_factor + 3) / 4, num_out_feats);
    dim3 num_threads(32, 4);
    if (bit == 4){
      bgemv4_kernel_outer_dim<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels, group_size, nh, nh_kv
      );}
    else{
      // note: in this case, pack factor == 16
      bgemv2_kernel_outer_dim<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, zeros, scaling_factors, out_feats,
        // constants
        num_in_channels, num_out_channels, group_size, nh, nh_kv
      );     
      }
    return _out_feats;
;}




// __device__ __forceinline__ float warp_reduce_sum_local(float v) {
//   #pragma unroll
//   for (int i = 16; i > 0; i >>= 1) {
//     v += __shfl_down_sync(0xffffffff, v, i);
//   }
//   return v;
// }
// 
// // ========================== 4-bit + base 融合核 ==========================
// __global__ void bgemv4_kernel_outer_dim_with_base(
//   const half* __restrict__ _inputs,      // [B*nh, IC]
//   const uint32_t* __restrict__ _weight,  // [B*nh_kv, OC/8, IC]
//   const half* __restrict__ _zeros,       // [B*nh_kv, OC/group, IC]
//   const half* __restrict__ _scale,       // [B*nh_kv, OC/group, IC]
//   const half* __restrict__ _centroids,   // [nh_kv, M, IC]
//   const void* __restrict__ _assign,      // [B, nh_kv, OC] (u8/u16/i32)
//   half* __restrict__ _outputs,           // [B*nh, OC]
//   const int IC, const int OC,
//   const int group_size, const int nh, const int nh_kv,
//   const int M_centroids, const int assign_bytes)
// {
//   const int bit = 4;
//   const int pack_factor = 8;
//   const int TILE_DIM = 128;

//   const int batch_idx     = blockIdx.x;          // 0..B*nh-1
//   const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
//   const int oc_start      = packed_oc_idx * pack_factor;

//   if (oc_start >= OC) return;

//   const int ratio = nh / nh_kv;                  // GQA 映射
//   const int b  = batch_idx / nh;
//   const int hq = batch_idx % nh;
//   const int kv = hq / ratio;

//   const half* inputs = _inputs + (size_t)batch_idx * IC;
//   half* outputs      = _outputs + (size_t)batch_idx * OC;

//   const int batch_kv_flat = b * nh_kv + kv;
//   const uint32_t* weight  = _weight + (size_t)batch_kv_flat * (OC * IC / pack_factor);
//   const half* scale       = _scale  + (size_t)batch_kv_flat * (OC * IC / group_size);
//   const half* zeros       = _zeros  + (size_t)batch_kv_flat * (OC * IC / group_size);

//   // centroids: [nh_kv, M, IC]
//   const half* cbase = _centroids + (size_t)kv * (M_centroids * IC);

//   // assignments: [B, nh_kv, OC], 第三维与 OC 对齐
//   const char* arow = reinterpret_cast<const char*>(_assign)
//                    + ((size_t)b * nh_kv + kv) * (size_t)OC * assign_bytes;

//   // -------- Pass 0: 并行计算 q·centroid（写 SMEM） --------
//   __shared__ float s_qC[MAX_CENTROIDS];
//   // 多 warp 并行：每个 warp 处理若干 m
//   int lane = threadIdx.x & (WARP_SIZE - 1);
//   int warp = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
//   int warps_per_CTA = (blockDim.x * blockDim.y) / WARP_SIZE;

//   for (int m = warp; m < M_centroids; m += warps_per_CTA) {
//     float acc = 0.f;
//     // 跨 K 维分条带累加
//     for (int k = lane; k < IC; k += WARP_SIZE) {
//       float qv = __half2float(inputs[k]);
//       float cv = __half2float(cbase[(size_t)m * IC + k]);
//       acc += qv * cv;
//     }
//     // warp 归约
//     acc = warp_reduce_sum_local(acc);
//     if (lane == 0) s_qC[m] = acc;
//   }
//   __syncthreads();

//   // -------- Pass 1: 量化残差 QK + 基向量补偿 --------
//   float psum[pack_factor] = {0};

//   // group 索引只与 oc 相关
//   const int group_idx = oc_start / group_size;

//   for (int k = 0; k < (IC + TILE_DIM - 1) / TILE_DIM; ++k) {
//     // 载入 4×(8half) inputs、4×uint32 weights、scale/zero
//     uint32_t qw[4] = {0,0,0,0};
//     half sc4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
//     half ze4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
//     half in4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};

//     const int w_off  = packed_oc_idx * IC + k * TILE_DIM + threadIdx.x * 4;
//     const int sz_off = group_idx      * IC + k * TILE_DIM + threadIdx.x * 4;
//     const int in_off = k * TILE_DIM + threadIdx.x * 4;

//     #if defined(KIVI_VEC_LOAD)
//       if (w_off + 3 < OC * IC / pack_factor) {
//         *((uint4*)qw) = *((const uint4*)(weight + w_off));
//       } else {
//         #pragma unroll
//         for (int i=0;i<4;++i) if (w_off + i < OC * IC / pack_factor) qw[i] = *(weight + w_off + i);
//       }
//       if (sz_off + 3 < OC * IC / group_size) {
//         *((half4*)sc4) = *((const half4*)(scale + sz_off));
//         *((half4*)ze4) = *((const half4*)(zeros + sz_off));
//       } else {
//         #pragma unroll
//         for (int i=0;i<4;++i) if (sz_off + i < OC * IC / group_size) {
//           sc4[i] = *(scale + sz_off + i);
//           ze4[i] = *(zeros + sz_off + i);
//         }
//       }
//       if (in_off + 3 < IC) {
//         *((half4*)in4) = *((const half4*)(inputs + in_off));
//       } else {
//         #pragma unroll
//         for (int i=0;i<4;++i) if (in_off + i < IC) in4[i] = *(inputs + in_off + i);
//       }
//     #else
//       #pragma unroll
//       for (int i=0;i<4;++i) {
//         if (w_off + i < OC * IC / pack_factor) qw[i] = *(weight + w_off + i);
//         if (sz_off + i < OC * IC / group_size) {
//           sc4[i] = *(scale + sz_off + i);
//           ze4[i] = *(zeros + sz_off + i);
//         }
//         if (in_off + i < IC) in4[i] = *(inputs + in_off + i);
//       }
//     #endif

//     // 4 个“8权重”块
//     #pragma unroll
//     for (int t=0; t<4; ++t) {
//       float cur_inp   = __half2float(in4[t]);
//       float cur_scale = __half2float(sc4[t]);
//       float cur_zero  = __half2float(ze4[t]);
//       float alpha = cur_scale * cur_inp;   // 提常数：与 nibble 无关
//       float beta  = cur_zero  * cur_inp;

//       uint32_t cur = qw[t];

//       #pragma unroll
//       for (int lane8=0; lane8<pack_factor; ++lane8) {
//         int oc = oc_start + lane8;
//         if (oc < OC) {
//           // nibble 展开（可用 __bfe(cur, lane8*4, 4) 替代）
//           float wq = float(cur & 0xF);
//           cur >>= bit;
//           psum[lane8] += wq * alpha + beta;
//         }
//       }
//     }
//   }

//   // warp 归约 + 基向量补偿 + 写回
//   #pragma unroll
//   for (int i=0;i<pack_factor;++i) {
//     int oc = oc_start + i;
//     if (oc < OC) {
//       float v = psum[i];
//       v = warp_reduce_sum_local(v);
//       if (threadIdx.x == 0) {
//         // 读 assignment
//         int aidx;
//         if (assign_bytes == 1)      aidx = *((const uint8_t *)(arow + oc));
//         else if (assign_bytes == 2) aidx = *((const uint16_t*)(arow + oc * 2));
//         else                        aidx = *((const int32_t *)(arow + oc * 4));
//         float add = (aidx >= 0 && aidx < M_centroids) ? s_qC[aidx] : 0.f;
//         outputs[oc] = __float2half(v + add);
//       }
//     }
//   }
// }

// // ========================== 2-bit + base 融合核 ==========================
// __global__ void bgemv2_kernel_outer_dim_with_base(
//   const half* __restrict__ _inputs,
//   const uint32_t* __restrict__ _weight,
//   const half* __restrict__ _zeros,
//   const half* __restrict__ _scale,
//   const half* __restrict__ _centroids,
//   const void* __restrict__ _assign,
//   half* __restrict__ _outputs,
//   const int IC, const int OC,
//   const int group_size, const int nh, const int nh_kv,
//   const int M_centroids, const int assign_bytes)
// {
//   const int bit = 2;
//   const int pack_factor = 16;
//   const int TILE_DIM = 256;

//   const int batch_idx     = blockIdx.x;          // 0..B*nh-1
//   const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
//   const int oc_start      = packed_oc_idx * pack_factor;

//   if (oc_start >= OC) return;

//   const int ratio = nh / nh_kv;                  // GQA 映射
//   const int b  = batch_idx / nh;
//   const int hq = batch_idx % nh;
//   const int kv = hq / ratio;

//   const half* inputs = _inputs + (size_t)batch_idx * IC;
//   half* outputs      = _outputs + (size_t)batch_idx * OC;

//   const int batch_kv_flat = b * nh_kv + kv;
//   const uint32_t* weight  = _weight + (size_t)batch_kv_flat * (OC * IC / pack_factor);
//   const half* scale       = _scale  + (size_t)batch_kv_flat * (OC * IC / group_size);
//   const half* zeros       = _zeros  + (size_t)batch_kv_flat * (OC * IC / group_size);

//   const half* cbase = _centroids + (size_t)kv * (M_centroids * IC);
//   const char* arow  = reinterpret_cast<const char*>(_assign)
//                     + ((size_t)b * nh_kv + kv) * (size_t)OC * assign_bytes;

//   // ---- 并行 q·centroid
//   __shared__ float s_qC[MAX_CENTROIDS];
//   int lane = threadIdx.x & (WARP_SIZE - 1);
//   int warp = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
//   int warps_per_CTA = (blockDim.x * blockDim.y) / WARP_SIZE;

//   for (int m = warp; m < M_centroids; m += warps_per_CTA) {
//     float acc = 0.f;
//     for (int k = lane; k < IC; k += WARP_SIZE) {
//       float qv = __half2float(inputs[k]);
//       float cv = __half2float(cbase[(size_t)m * IC + k]);
//       acc += qv * cv;
//     }
//     acc = warp_reduce_sum_local(acc);
//     if (lane == 0) s_qC[m] = acc;
//   }
//   __syncthreads();

//   // ---- 量化残差 QK + 基向量补偿
//   float psum[pack_factor] = {0};
//   const int group_idx = oc_start / group_size;

//   for (int k = 0; k < (IC + TILE_DIM - 1) / TILE_DIM; ++k) {
//     uint32_t qw[4] = {0,0,0,0};
//     half sc4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
//     half ze4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
//     half in4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};

//     const int w_off  = packed_oc_idx * IC + k * TILE_DIM + threadIdx.x * 4;
//     const int sz_off = group_idx      * IC + k * TILE_DIM + threadIdx.x * 4;
//     const int in_off = k * TILE_DIM + threadIdx.x * 4;

//     #if defined(KIVI_VEC_LOAD)
//       if (w_off + 3 < OC * IC / pack_factor) {
//         *((uint4*)qw) = *((const uint4*)(weight + w_off));
//       } else {
//         #pragma unroll
//         for (int i=0;i<4;++i) if (w_off + i < OC * IC / pack_factor) qw[i] = *(weight + w_off + i);
//       }
//       if (sz_off + 3 < OC * IC / group_size) {
//         *((half4*)sc4) = *((const half4*)(scale + sz_off));
//         *((half4*)ze4) = *((const half4*)(zeros + sz_off));
//       } else {
//         #pragma unroll
//         for (int i=0;i<4;++i) if (sz_off + i < OC * IC / group_size) {
//           sc4[i] = *(scale + sz_off + i);
//           ze4[i] = *(zeros + sz_off + i);
//         }
//       }
//       if (in_off + 3 < IC) {
//         *((half4*)in4) = *((const half4*)(inputs + in_off));
//       } else {
//         #pragma unroll
//         for (int i=0;i<4;++i) if (in_off + i < IC) in4[i] = *(inputs + in_off + i);
//       }
//     #else
//       #pragma unroll
//       for (int i=0;i<4;++i) {
//         if (w_off + i < OC * IC / pack_factor) qw[i] = *(weight + w_off + i);
//         if (sz_off + i < OC * IC / group_size) {
//           sc4[i] = *(scale + sz_off + i);
//           ze4[i] = *(zeros + sz_off + i);
//         }
//         if (in_off + i < IC) in4[i] = *(inputs + in_off + i);
//       }
//     #endif

//     #pragma unroll
//     for (int t=0; t<4; ++t) {
//       float cur_inp   = __half2float(in4[t]);
//       float cur_scale = __half2float(sc4[t]);
//       float cur_zero  = __half2float(ze4[t]);
//       float alpha = cur_scale * cur_inp;
//       float beta  = cur_zero  * cur_inp;

//       uint32_t cur = qw[t];

//       #pragma unroll
//       for (int lane16=0; lane16<pack_factor; ++lane16) {
//         int oc = oc_start + lane16;
//         if (oc < OC) {
//           float wq = float(cur & 0x3);
//           cur >>= bit;
//           psum[lane16] += wq * alpha + beta;
//         }
//       }
//     }
//   }

//   #pragma unroll
//   for (int i=0;i<pack_factor;++i) {
//     int oc = oc_start + i;
//     if (oc < OC) {
//       float v = psum[i];
//       v = warp_reduce_sum_local(v);
//       if (threadIdx.x == 0) {
//         int aidx;
//         if (assign_bytes == 1)      aidx = *((const uint8_t *)(arow + oc));
//         else if (assign_bytes == 2) aidx = *((const uint16_t*)(arow + oc * 2));
//         else                        aidx = *((const int32_t *)(arow + oc * 4));
//         float add = (aidx >= 0 && aidx < M_centroids) ? s_qC[aidx] : 0.f;
//         outputs[oc] = __float2half(v + add);
//       }
//     }
//   }
// }

// // ========================== C++ 导出函数 ==========================
// torch::Tensor gemv_forward_cuda_outer_dim_with_base(
//     torch::Tensor _in_feats,          // [B*nh, 1, K]
//     torch::Tensor _kernel,            // [B*nh_kv, N/pack, K]
//     torch::Tensor _scaling_factors,   // [B*nh_kv, N/group, K]
//     torch::Tensor _zeros,             // [B*nh_kv, N/group, K]
//     const int bit,
//     const int group_size,
//     const int nh,
//     const int nh_kv,
//     torch::Tensor _centroids,         // [nh_kv, M, K]
//     torch::Tensor _assignments        // [B, nh_kv, N] (u8/u16/i32)
// ){
//   TORCH_CHECK(_in_feats.dim()==3 && _in_feats.size(1)==1, "in_feats must be [B*nh,1,K]");
//   TORCH_CHECK(bit==2 || bit==4, "only 2/4 bit supported");

//   const int BS_nh = _in_feats.size(0);
//   const int IC    = _in_feats.size(2);
//   const int OC    = _zeros.size(1) * group_size;

//   auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
//   at::Tensor _out = torch::empty({BS_nh, 1, OC}, options);
//   half* out_ptr = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

//   const half* in_ptr   = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
//   const uint32_t* wptr = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
//   const half* zptr     = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
//   const half* sptr     = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());

//   TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv && _centroids.size(2)==IC,
//               "centroids must be [nh_kv, M, IC]");
//   const int M_cent = _centroids.size(1);
//   TORCH_CHECK(M_cent <= MAX_CENTROIDS, "M exceeds MAX_CENTROIDS");
//   const half* cptr  = reinterpret_cast<half*>(_centroids.data_ptr<at::Half>());

//   TORCH_CHECK(_assignments.dim()==3 && _assignments.size(2)==OC,
//               "assignments must be [B, nh_kv, OC]");
//   const int assign_bytes =
//     (_assignments.dtype()==torch::kUInt8) ? 1 :
//     (_assignments.dtype()==torch::kInt16) ? 2 : 4;
//   const void* aptr = static_cast<const void*>(_assignments.data_ptr());

//   // grid 配置沿用你现有的模式
//   const int pack_factor = 32 / bit;
//   dim3 num_blocks(BS_nh, (OC / pack_factor + 3) / 4, 1);
//   dim3 num_threads(32, 4);

//   if (bit == 4) {
//     bgemv4_kernel_outer_dim_with_base<<<num_blocks, num_threads>>>(
//       in_ptr, wptr, zptr, sptr, cptr, aptr, out_ptr,
//       IC, OC, group_size, nh, nh_kv, M_cent, assign_bytes
//     );
//   } else {
//     bgemv2_kernel_outer_dim_with_base<<<num_blocks, num_threads>>>(
//       in_ptr, wptr, zptr, sptr, cptr, aptr, out_ptr,
//       IC, OC, group_size, nh, nh_kv, M_cent, assign_bytes
//     );
//   }
//   return _out;
// }


// ---------------- warp reduce (sum) ----------------
__inline__ __device__ float warp_reduce_sum_local(float v) {
  unsigned mask = 0xffffffffu;
  #pragma unroll
  for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}

// ---------------- ceil_div helper ----------------
__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
  return (a + b - 1) / b;
}

// ---------------- 核心模板（Bit = 2 或 4） ----------------
template<int Bit>
__global__ void bgemv_kernel_outer_dim_with_base_tiled(
  const half* __restrict__ _inputs,      // [B*nh, IC]
  const uint32_t* __restrict__ _weight,  // [B*nh_kv, OC/pack, IC]，每个uint32包含 pack 个量化权值的若干子位
  const half* __restrict__ _zeros,       // [B*nh_kv, OC/group, IC]
  const half* __restrict__ _scale,       // [B*nh_kv, OC/group, IC]
  const half* __restrict__ _centroids,   // [nh_kv, M, IC]
  const void* __restrict__ _assign,      // [B, nh_kv, OC] (u8/u16/i32)
  half* __restrict__ _outputs,           // [B*nh, OC]
  const int IC, const int OC,
  const int group_size, const int nh, const int nh_kv,
  const int M_centroids, const int assign_bytes
){
  constexpr int pack_factor = 32 / Bit;   // 4-bit: 8  /  2-bit: 16
  constexpr int TILE_DIM    = 128;        // **统一设为 128（修复 2-bit 版本）**
  const uint32_t mask = (1u << Bit) - 1u;

  const int batch_idx = blockIdx.x;   // 0..B*nh-1

  // ---- GQA 映射 ----
  const int ratio = nh / nh_kv;
  const int b  = batch_idx / nh;
  const int hq = batch_idx % nh;
  const int kv = hq / ratio;

  const half* inputs = _inputs  + (size_t)batch_idx * IC;
  half* outputs      = _outputs + (size_t)batch_idx * OC;

  const int batch_kv_flat = b * nh_kv + kv;

  const uint32_t* weight  = _weight + (size_t)batch_kv_flat * (OC * IC / pack_factor);
  const half*     scale   = _scale  + (size_t)batch_kv_flat * (OC * IC / group_size);
  const half*     zeros   = _zeros  + (size_t)batch_kv_flat * (OC * IC / group_size);

  // centroids: [nh_kv, M, IC]
  const half* cbase = _centroids + (size_t)kv * (M_centroids * IC);
  // assignments: [B, nh_kv, OC] （与 OC 对齐）
  const char* arow  = reinterpret_cast<const char*>(_assign)
                    + ((size_t)b * nh_kv + kv) * (size_t)OC * assign_bytes;

  // ---------------- 动态共享内存布局 ----------------
  extern __shared__ unsigned char smem_raw[];
  float* s_qC = reinterpret_cast<float*>(smem_raw);                         // M_centroids floats
  half*  s_in = reinterpret_cast<half*>(s_qC + M_centroids);                // TILE_DIM halfs (inputs tile)

  // ---------------- Pass 0: 并行计算 q · centroid → s_qC ----------------
  {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warps_per_CTA = (blockDim.x * blockDim.y) / WARP_SIZE;

    for (int m = warp; m < M_centroids; m += warps_per_CTA) {
      float acc = 0.f;
      // K 维跨 lane 累加
      for (int k = lane; k < IC; k += WARP_SIZE) {
        float qv = __half2float(inputs[k]);
        float cv = __half2float(cbase[(size_t)m * IC + k]);
        acc += qv * cv;
      }
      acc = warp_reduce_sum_local(acc);
      if (lane == 0) s_qC[m] = acc;
    }
    __syncthreads();
  }

  // ---------------- Pass 1: 残差 QK + 基向量补偿（持久化 oc 循环 + inputs tile 复用） ----------------
  const int nPacked = ceil_div(OC, pack_factor);
  const int start_packed = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_packed = gridDim.y * blockDim.y;

  for (int packed = start_packed; packed < nPacked; packed += stride_packed) {

    const int oc_start = packed * pack_factor;
    if (oc_start >= OC) continue;

    const int group_idx = oc_start / group_size;

    // 本 warp 为该 oc-pack 的累加器
    float psum[pack_factor];
    #pragma unroll
    for (int i = 0; i < pack_factor; ++i) psum[i] = 0.f;

    float beta_acc = 0.f;  // << beta-hoist：只对 beta 做一次累计，写回前统一相加

    const int nKTiles = ceil_div(IC, TILE_DIM);

    for (int kt = 0; kt < nKTiles; ++kt) {

      // ---- CTA 共享：把本 tile 的 inputs 拉到 SMEM（由 y==0 的 warp 负责向量化加载） ----
      // tile 起始全局 K 索引
      const int k_base = kt * TILE_DIM;

      if (threadIdx.y == 0) {
        // 32 个线程（threadIdx.x） × 4 half = 128 half
        const int in_off = k_base + threadIdx.x * 4;
        half v0 = __float2half(0.f), v1 = __float2half(0.f),
             v2 = __float2half(0.f), v3 = __float2half(0.f);
        if (in_off + 0 < IC) v0 = inputs[in_off + 0];
        if (in_off + 1 < IC) v1 = inputs[in_off + 1];
        if (in_off + 2 < IC) v2 = inputs[in_off + 2];
        if (in_off + 3 < IC) v3 = inputs[in_off + 3];

        // 写入 s_in（tile 内局部下标）
        const int loc = threadIdx.x * 4;
        s_in[loc + 0] = v0;
        s_in[loc + 1] = v1;
        s_in[loc + 2] = v2;
        s_in[loc + 3] = v3;
      }
      __syncthreads(); // s_in 就绪

      // ---- 每个 warp 继续完成自己 oc-pack 的 K×tile 累加 ----
      // 载入与该 oc-pack、该 tile 对应的权重&量化参数
      const int w_off  = packed   * IC + k_base + threadIdx.x * 4;
      const int sz_off = group_idx* IC + k_base + threadIdx.x * 4;

      // 4 份 32-bit 打包权重（每份包含 pack_factor 个子权重的一部分）
      uint32_t qw[4] = {0,0,0,0};
      if (w_off + 0 < (OC * IC / pack_factor)) qw[0] = weight[w_off + 0];
      if (w_off + 1 < (OC * IC / pack_factor)) qw[1] = weight[w_off + 1];
      if (w_off + 2 < (OC * IC / pack_factor)) qw[2] = weight[w_off + 2];
      if (w_off + 3 < (OC * IC / pack_factor)) qw[3] = weight[w_off + 3];

      // 对应的 scale/zero（与 lane 无关；同一 pack 的 oc 共享一组）
      half sc4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
      half ze4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
      if (sz_off + 0 < (OC * IC / group_size)) { sc4[0] = scale[sz_off + 0]; ze4[0] = zeros[sz_off + 0]; }
      if (sz_off + 1 < (OC * IC / group_size)) { sc4[1] = scale[sz_off + 1]; ze4[1] = zeros[sz_off + 1]; }
      if (sz_off + 2 < (OC * IC / group_size)) { sc4[2] = scale[sz_off + 2]; ze4[2] = zeros[sz_off + 2]; }
      if (sz_off + 3 < (OC * IC / group_size)) { sc4[3] = scale[sz_off + 3]; ze4[3] = zeros[sz_off + 3]; }

      // 该线程负责的 tile 内 4 个 K 位置（从 s_in 取）
      const int loc = threadIdx.x * 4;

      #pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int k_local = loc + t;
        if (k_base + k_local >= IC) break;  // 边界

        const float cur_inp   = __half2float(s_in[k_local]);
        const float cur_scale = __half2float(sc4[t]);
        const float cur_zero  = __half2float(ze4[t]);

        const float alpha = cur_scale * cur_inp;  // 与 oc lane 无关
        beta_acc += cur_zero * cur_inp;           // << beta-hoist 累加一次

        const uint32_t q = qw[t];                 // 不变，避免移位链

        // 展开 pack_factor 个子权重（无依赖位提取）
        #pragma unroll
        for (int i = 0; i < pack_factor; ++i) {
          const int wi = (q >> (i * Bit)) & mask;    // or: __bfe(q, i*Bit, Bit)
          psum[i] = fmaf(static_cast<float>(wi), alpha, psum[i]);
        }
      }

      __syncthreads(); // 下一个 tile 前同步（确保 s_in 不被覆盖）
    } // k tiles

    // ---- 归约 + 写回（基向量补偿） ----
    beta_acc = warp_reduce_sum_local(beta_acc);

    #pragma unroll
    for (int i = 0; i < pack_factor; ++i) {
      const int oc = oc_start + i;
      if (oc < OC) {
        float v = psum[i];
        v = warp_reduce_sum_local(v);
        if (threadIdx.x == 0) {
          int aidx;
          if (assign_bytes == 1)      aidx = *((const uint8_t *)(arow + oc));
          else if (assign_bytes == 2) aidx = *((const uint16_t*)(arow + oc * 2));
          else                        aidx = *((const int32_t *)(arow + oc * 4));
          const float add = (aidx >= 0 && aidx < M_centroids) ? s_qC[aidx] : 0.f;
          outputs[oc] = __float2half(v + beta_acc + add);
        }
      }
    }

  } // persistent oc loop
}

// ---------------- C++ 导出函数 ----------------
torch::Tensor gemv_forward_cuda_outer_dim_with_base(
    torch::Tensor _in_feats,          // [B*nh, 1, K]  => inputs: [B*nh, K]
    torch::Tensor _kernel,            // [B*nh_kv, N/pack, K] （打包后的uint32）
    torch::Tensor _scaling_factors,   // [B*nh_kv, N/group, K]
    torch::Tensor _zeros,             // [B*nh_kv, N/group, K]
    const int bit,                    // 2 或 4
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,         // [nh_kv, M, K]
    torch::Tensor _assignments        // [B, nh_kv, N] (u8/u16/i32)
){
  TORCH_CHECK(_in_feats.dim()==3 && _in_feats.size(1)==1, "in_feats must be [B*nh,1,K]");
  TORCH_CHECK(bit==2 || bit==4, "only 2/4 bit supported");

  const int BS_nh = _in_feats.size(0);
  const int IC    = _in_feats.size(2);
  const int OC    = _zeros.size(1) * group_size;

  auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out = torch::empty({BS_nh, 1, OC}, options);

  half*       out_ptr = reinterpret_cast<half*>      (_out.data_ptr<at::Half>());
  const half* in_ptr  = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
  const uint32_t* wptr= reinterpret_cast<const uint32_t*>(_kernel.data_ptr<int>());
  const half* zptr    = reinterpret_cast<const half*>(_zeros.data_ptr<at::Half>());
  const half* sptr    = reinterpret_cast<const half*>(_scaling_factors.data_ptr<at::Half>());

  TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv && _centroids.size(2)==IC,
              "centroids must be [nh_kv, M, IC]");
  const int M_cent = _centroids.size(1);
  const half* cptr = reinterpret_cast<const half*>(_centroids.data_ptr<at::Half>());

  TORCH_CHECK(_assignments.dim()==3 && _assignments.size(2)==OC,
              "assignments must be [B, nh_kv, OC]");
  const int assign_bytes =
    (_assignments.dtype()==torch::kUInt8) ? 1 :
    (_assignments.dtype()==torch::kInt16) ? 2 : 4;
  const void* aptr = static_cast<const void*>(_assignments.data_ptr());

  // ---- Launch 配置 ----
  constexpr int TILE_DIM = 128;                       // 与核中一致
  const int pack_factor  = 32 / bit;
  const int nPacked      = (OC + pack_factor - 1) / pack_factor;

  // 一个 CTA = 32x4 = 128 线程（4 个 warp）。y 维可按需调大/调小。
  dim3 num_threads(32, 8, 1);
  // dim3 num_blocks(BS_nh, std::max(1, (nPacked + num_threads.y - 1) / num_threads.y), 1);
  const int blocks_y_i = std::max(
    1,
    (nPacked + static_cast<int>(num_threads.y) - 1) / static_cast<int>(num_threads.y)
  );
  dim3 num_blocks(BS_nh, static_cast<unsigned int>(blocks_y_i), 1);

  // 动态共享内存：M 个 float + TILE_DIM 个 half
  size_t smem_bytes = (size_t)M_cent * sizeof(float) + (size_t)TILE_DIM * sizeof(half);

  // ---- 启动 ----
  if (bit == 4) {
    bgemv_kernel_outer_dim_with_base_tiled<4>
      <<<num_blocks, num_threads, smem_bytes>>>(
        in_ptr, wptr, zptr, sptr, cptr, aptr, out_ptr,
        IC, OC, group_size, nh, nh_kv, M_cent, assign_bytes
      );
  } else { // bit == 2
    bgemv_kernel_outer_dim_with_base_tiled<2>
      <<<num_blocks, num_threads, smem_bytes>>>(
        in_ptr, wptr, zptr, sptr, cptr, aptr, out_ptr,
        IC, OC, group_size, nh, nh_kv, M_cent, assign_bytes
      );
  }

  // 注意：实际工程里可在此处加 cudaGetLastError() / cudaDeviceSynchronize() 做调试
  return _out;
}

// 需要的 warp 规约工具
__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
  #pragma unroll
  for (int i = 4; i >= 0; --i) {
    v += __shfl_down_sync(0xffffffff, v, 1 << i);
  }
  return v;
}

enum AttnVAblationMode {
  ABLATION_FULL = 0,
  ABLATION_RESIDUAL_ONLY = 1,
  ABLATION_NO_CENTROID_HISTOGRAM = 2,
  ABLATION_CENTROID_ONLY = 3,
  ABLATION_WARP_AGG_FULL = 4,
  ABLATION_PER_WARP_HIST_FULL = 5,
  ABLATION_NO_TABLE_CONTRIBUTION = 6,
  ABLATION_LANE0_TABLE_FULL = 7,
};

// ===================== 最终修正版 V 融合核 =====================
template<int BIT, int MODE>
__global__ void battn_v_kernel_with_base(
  const half*      __restrict__ _alpha_q,   // [B*nh, K]
  const uint32_t*  __restrict__ _vq_lin,    // [B*nh_kv, (OC/pack)*K] 线性化
  const half*      __restrict__ _vscale_lin,// [B*nh_kv, (OC/group)*K] 线性化
  const half*      __restrict__ _vzero_lin, // [B*nh_kv, (OC/group)*K] 线性化
  const half*      __restrict__ _centroids, // [nh_kv, Mcent, OC]
  const uint8_t*   __restrict__ _mask_q,    // [B, nh_kv, K]
  const void*      __restrict__ _idx_q,     // [B, nh_kv, K] (u8/u16/i32)
  const half*      __restrict__ _alpha_f,   // [B*nh, Lf] (可空)
  const half*      __restrict__ _v_full,    // [B, nh_kv, Lf, OC] (可空)
  half*            __restrict__ _out,       // [B*nh, OC]
  const int K, const int OC, const int Lf,
  const int group_size, const int nh, const int nh_kv,
  const int Mcent, const int idx_bytes)
{
  static_assert(BIT==2 || BIT==4, "BIT must be 2 or 4");
  static_assert(MODE==ABLATION_FULL || MODE==ABLATION_RESIDUAL_ONLY ||
                MODE==ABLATION_NO_CENTROID_HISTOGRAM || MODE==ABLATION_CENTROID_ONLY ||
                MODE==ABLATION_WARP_AGG_FULL || MODE==ABLATION_PER_WARP_HIST_FULL ||
                MODE==ABLATION_NO_TABLE_CONTRIBUTION || MODE==ABLATION_LANE0_TABLE_FULL,
                "invalid V attention ablation mode");
  constexpr bool DO_RESIDUAL = MODE != ABLATION_CENTROID_ONLY;
  constexpr bool DO_HISTOGRAM = MODE == ABLATION_FULL || MODE == ABLATION_CENTROID_ONLY ||
                                MODE == ABLATION_WARP_AGG_FULL || MODE == ABLATION_PER_WARP_HIST_FULL ||
                                MODE == ABLATION_NO_TABLE_CONTRIBUTION || MODE == ABLATION_LANE0_TABLE_FULL;
  constexpr bool DO_CENTROID_TABLE = MODE == ABLATION_FULL || MODE == ABLATION_CENTROID_ONLY ||
                                     MODE == ABLATION_WARP_AGG_FULL || MODE == ABLATION_PER_WARP_HIST_FULL ||
                                     MODE == ABLATION_LANE0_TABLE_FULL;
  constexpr bool DO_FULL_RECENT = MODE == ABLATION_FULL || MODE == ABLATION_WARP_AGG_FULL ||
                                  MODE == ABLATION_PER_WARP_HIST_FULL || MODE == ABLATION_NO_TABLE_CONTRIBUTION ||
                                  MODE == ABLATION_LANE0_TABLE_FULL;
  constexpr bool DO_WARP_AGG_HISTOGRAM = MODE == ABLATION_WARP_AGG_FULL;
  constexpr bool DO_PER_WARP_HISTOGRAM = MODE == ABLATION_PER_WARP_HIST_FULL ||
                                         MODE == ABLATION_NO_TABLE_CONTRIBUTION ||
                                         MODE == ABLATION_LANE0_TABLE_FULL;
  constexpr bool DO_LANE0_TABLE = MODE == ABLATION_LANE0_TABLE_FULL;
  constexpr int PACK = 32 / BIT;        // 2bit=16, 4bit=8
  const uint32_t CODE_MASK = (1u << BIT) - 1u;
  const int TILE = 128;

  // --- 线程块映射 ---
  const int bnh = blockIdx.x;                                   // over [B*nh]
  const int wy  = threadIdx.y;                                  // 0..(blockDim.y-1)
  const int lane= threadIdx.x;                                  // 0..31

  const int packed_oc_idx = blockIdx.y * blockDim.y + wy;       // 以 PACK 聚类的 oc 块
  const int oc_start = packed_oc_idx * PACK;
  if (oc_start >= OC) return;

  // --- GQA 头映射 ---
  const int ratio = nh / nh_kv;
  const int b  = bnh / nh;
  const int hq = bnh % nh;
  const int hk = hq / ratio;

  // --- 指针基址（含 batch/kv 偏移）---
  const half* alpha_q = _alpha_q + (size_t)bnh * K;             // [K]
  half* out_row = _out + (size_t)bnh * OC;                      // [OC]

  const size_t bkv = (size_t)b * nh_kv + hk;
  // vq: [OC/pack, K] 逐行存放，线性化成一维
  const uint32_t* vq_base = _vq_lin + bkv * (size_t)(OC / PACK) * K;
  // scale/zero: [OC/group, K] 逐行存放，线性化成一维
  const half*     vsc_base= _vscale_lin + bkv * (size_t)(OC / group_size) * K;
  const half*     vzr_base= _vzero_lin  + bkv * (size_t)(OC / group_size) * K;

  const half* C = _centroids + (size_t)hk * (size_t)Mcent * OC; // [Mcent, OC]
  const uint8_t* mask_row = _mask_q + bkv * (size_t)K;          // [K]
  const char*    idx_row  = reinterpret_cast<const char*>(_idx_q)
                           + bkv * (size_t)K * idx_bytes;

  // --- 共享内存直方图 Sacc[c] ---
  extern __shared__ float s_Sacc[];  // 大小=Mcent
  const int sacc_rows = DO_PER_WARP_HISTOGRAM ? blockDim.y : 1;
  const int sacc_elems = Mcent * sacc_rows;
  for (int c = wy * blockDim.x + lane; c < sacc_elems; c += blockDim.x * blockDim.y)
    s_Sacc[c] = 0.f;
  __syncthreads();

  // --- 量化残差 GEMV 累加器 ---
  float psum[PACK];
  #pragma unroll
  for (int p=0; p<PACK; ++p) psum[p] = 0.f;

  // 预取本 oc 块所在的组（沿 OC 聚组）
  const int oc_group = oc_start / group_size;

  // --- 沿 K 维分块 ---
  for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
    const int t_base = kt * TILE + lane * 4;

    // α_q 装载
    half a4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
    #pragma unroll
    for (int i=0;i<4;++i) {
      const int t = t_base + i;
      if (t < K) a4[i] = __ldg(alpha_q + t);
    }

    // 直方图：Sacc[idx[t]] += α_q[t] * mask[t]
    if constexpr (DO_HISTOGRAM) {
    if constexpr (DO_PER_WARP_HISTOGRAM) {
      const int t = kt * TILE + wy * blockDim.x + lane;
      if (t < K) {
        const uint8_t m = __ldg(mask_row + t);
        if (m) {
          int idx;
          if (idx_bytes==1)      idx = *((const uint8_t *)(idx_row + t));
          else if (idx_bytes==2) idx = *((const uint16_t*)(idx_row + t*2));
          else                   idx = *((const int32_t *)(idx_row + t*4));
          if (0 <= idx && idx < Mcent) {
            atomicAdd(&s_Sacc[wy * Mcent + idx], __half2float(__ldg(alpha_q + t)));
          }
        }
      }
    } else {
    if (wy == 0) {                           // ★ 仅由 wy==0 的 warp 累加 s_Sacc
      #pragma unroll
      for (int i=0;i<4;++i) {
        const int t = t_base + i;
        bool valid = false;
        int idx = 0;
        float mass = 0.f;
        if (t < K) {
          const uint8_t m = __ldg(mask_row + t);
          if (m) {
            if (idx_bytes==1)      idx = *((const uint8_t *)(idx_row + t));
            else if (idx_bytes==2) idx = *((const uint16_t*)(idx_row + t*2));
            else                   idx = *((const int32_t *)(idx_row + t*4));
            if (0 <= idx && idx < Mcent) {
              valid = true;
              mass = __half2float(a4[i]);
            }
          }
        }
        if constexpr (DO_WARP_AGG_HISTOGRAM) {
          const unsigned active = __ballot_sync(0xffffffff, valid);
          if (valid) {
            const unsigned peers = __match_any_sync(active, idx);
            float sum = 0.f;
            #pragma unroll
            for (int src = 0; src < 32; ++src) {
              if (peers & (1u << src)) {
                sum += __shfl_sync(peers, mass, src);
              }
            }
            if (lane == (__ffs(peers) - 1)) {
              atomicAdd(&s_Sacc[idx], sum);
            }
          }
        } else {
          if (valid) {
            atomicAdd(&s_Sacc[idx], mass);
          }
        }
      }
    }
    }
    }

    // 量化 V：对 [oc_start .. oc_start+PACK-1] 载入 packed 行、scale/zero 行
    // vq 行基址：第 packed_oc_idx 行
    const uint32_t* vq_row = vq_base + (size_t)packed_oc_idx * K;
    // scale/zero 行基址：第 oc_group 行
    const half* vsc_row = vsc_base + (size_t)oc_group * K;
    const half* vzr_row = vzr_base + (size_t)oc_group * K;

    uint32_t qw[4] = {0,0,0,0};
    half sc4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
    half zr4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};

    if constexpr (DO_RESIDUAL) {
      #pragma unroll
      for (int i=0;i<4;++i) {
        const int t = t_base + i;
        if (t < K) {
          qw[i]  = __ldg(vq_row + t);
          sc4[i] = __ldg(vsc_row + t);
          zr4[i] = __ldg(vzr_row + t);
        }
      }

      // FMA：psum[p] += (s*code + z) * a
      #pragma unroll
      for (int j=0;j<4;++j) {
        const float a = __half2float(a4[j]);
        uint32_t cur = qw[j];
        const float s = __half2float(sc4[j]);
        const float z = __half2float(zr4[j]);

        #pragma unroll
        for (int p=0;p<PACK;++p) {
          const int oc = oc_start + p;
          if (oc < OC) {
            const float code = float(cur & CODE_MASK);
            psum[p] += (s * code + z) * a;
          }
          cur >>= BIT;
        }
      }
    }
  } // end kt
  __syncthreads();

  // --- 基向量补偿：add_base[p] = Σ_c Sacc[c] * C[c, oc_start+p] ---
  float add_base[PACK];
  #pragma unroll
  for (int p=0;p<PACK;++p) add_base[p] = 0.f;

  if constexpr (DO_CENTROID_TABLE) {
    if (!DO_LANE0_TABLE || lane == 0) {
    for (int c=0; c<Mcent; ++c) {
      float s = 0.f;
      if constexpr (DO_PER_WARP_HISTOGRAM) {
        #pragma unroll
        for (int w=0; w<4; ++w) {
          if (w < blockDim.y) s += s_Sacc[w * Mcent + c];
        }
      } else {
        s = s_Sacc[c];
      }
      if (s != 0.f) {
        const half* crow = C + (size_t)c * OC + oc_start;
        #pragma unroll
        for (int p=0;p<PACK;++p) {
          const int oc = oc_start + p;
          if (oc < OC) add_base[p] += s * __half2float(__ldg(crow + p));
        }
      }
    }
    }
  }

  // --- 最近窗口全精分量：add_full[p] = Σ_t α_f[t]·V_full[t, oc] ---
  float add_full[PACK];
  #pragma unroll
  for (int p=0;p<PACK;++p) add_full[p] = 0.f;

  if constexpr (DO_FULL_RECENT) {
  if (Lf > 0 && _alpha_f && _v_full) {
    const half* aF = _alpha_f + (size_t)bnh * Lf;                          // [Lf]
    const half* vF = _v_full  + ((size_t)b * nh_kv + hk) * (size_t)Lf * OC; // [Lf, OC]
    for (int t = lane; t < Lf; t += blockDim.x) {
      const float a = __half2float(__ldg(aF + t));
      const half* row = vF + (size_t)t * OC + oc_start;
      #pragma unroll
      for (int p=0;p<PACK;++p) {
        const int oc = oc_start + p;
        if (oc < OC) add_full[p] += a * __half2float(__ldg(row + p));
      }
    }
    // 一个 warp 对应一个 oc tile → 只需 warp 内规约
    #pragma unroll
    for (int p=0;p<PACK;++p) {
      float v = add_full[p];
      v = warp_reduce_sum_f32(v);
      if (lane == 0) add_full[p] = v;
    }
  }
  }

  // --- 写回 ---
  #pragma unroll
  for (int p=0;p<PACK;++p) {
    const int oc = oc_start + p;
    if (oc < OC) {
      float vqsum = DO_RESIDUAL ? warp_reduce_sum_f32(psum[p]) : 0.f;
      if (lane == 0) {
        const float val = vqsum + add_base[p] + add_full[p];
        out_row[oc] = __float2half(val);
      }
    }
  }
}



// 导出：选择 2bit/4bit（与 battn_v_kernel_with_base<BIT> 对齐）
torch::Tensor attn_v_forward_cuda_outer_dim_with_base(
    torch::Tensor _alpha_q,    // [B*nh, 1, K]
    torch::Tensor _vq,         // [B*nh_kv, OC/pack, K]
    torch::Tensor _vscale,     // [B*nh_kv, OC/group, K]
    torch::Tensor _vzero,      // [B*nh_kv, OC/group, K]
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,  // [nh_kv, Mcent, OC]
    torch::Tensor _mask_q,     // [B, nh_kv, K]  (uint8)
    torch::Tensor _idx_q,      // [B, nh_kv, K]  (u8/u16/i32)
    torch::Tensor _alpha_f,    // [B*nh, Lf]     (可空 size=0)
    torch::Tensor _v_full      // [B, nh_kv, Lf, OC] (可空 size=0)
){
  // ---- 基本检查 ----
  TORCH_CHECK(_alpha_q.dim()==3 && _alpha_q.size(1)==1, "alpha_q must be [B*nh,1,K]");
  const int BSnh = _alpha_q.size(0);
  const int K    = _alpha_q.size(2);

  TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv, "centroids must be [nh_kv,M,OC]");
  const int OC   = _centroids.size(2);

  TORCH_CHECK(_vq.dim()==3 && _vq.size(2)==K, "vq must be [B*nh_kv, OC/pack, K]");
  const int PACK = 32 / bit;
  TORCH_CHECK(_vq.size(1) * PACK == OC, "vq.pack mismatch: (OC/pack)*pack must equal OC");

  TORCH_CHECK(_vscale.dim()==3 && _vzero.dim()==3, "scale/zero must be 3D [B*nh_kv, OC/group, K]");
  const int Mcent = _centroids.size(1);

  TORCH_CHECK(_mask_q.dim()==3 && _mask_q.size(2)==K && _mask_q.size(1)==nh_kv, "mask_q must be [B,nh_kv,K]");
  TORCH_CHECK(_idx_q .dim()==3 && _idx_q .size(2)==K && _idx_q .size(1)==nh_kv, "idx_q must be [B,nh_kv,K]");

  const int Lf = (_alpha_f.numel()==0 || _v_full.numel()==0) ? 0 : _alpha_f.size(1);
  if (Lf > 0) {
    TORCH_CHECK(_v_full.dim()==4 && _v_full.size(2)==Lf && _v_full.size(3)==OC, "v_full must be [B,nh_kv,Lf,OC]");
  }

  // ---- 输出张量 ----
  auto options = torch::TensorOptions().dtype(_alpha_q.dtype()).device(_alpha_q.device());
  at::Tensor _out = torch::empty({BSnh, 1, OC}, options);

  // ---- 原始指针（名称与 kernel 一一对应）----
  const half*      alpha_q = reinterpret_cast<const half*     >(_alpha_q.data_ptr<at::Half>());
  const uint32_t*  vq      = reinterpret_cast<const uint32_t* >(_vq      .data_ptr<int>());
  const half*      vsc     = reinterpret_cast<const half*     >(_vscale  .data_ptr<at::Half>());
  const half*      vzr     = reinterpret_cast<const half*     >(_vzero   .data_ptr<at::Half>());
  const half*      cent    = reinterpret_cast<const half*     >(_centroids.data_ptr<at::Half>());
  const uint8_t*   mask    = reinterpret_cast<const uint8_t*  >(_mask_q  .data_ptr<uint8_t>());
  const void*      idx     = static_cast<const void*>(_idx_q.data_ptr());
  const half*      alpha_f = (_alpha_f.numel()==0) ? nullptr : reinterpret_cast<const half*>(_alpha_f.data_ptr<at::Half>());
  const half*      v_full  = (_v_full .numel()==0) ? nullptr : reinterpret_cast<const half*>(_v_full .data_ptr<at::Half>());
  half*            outp    = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  // ---- launch 形状 & 共享内存 ----
  dim3 threads(32, 4, 1);  // 每个 warp 负责一个 oc tile
  dim3 blocks(BSnh, (OC / PACK + threads.y - 1) / threads.y, 1);
  size_t shmem = (size_t)Mcent * 4 * sizeof(float);  // per-warp Sacc[4][Mcent]

  // ---- 索引宽度 ----
  const int idx_bytes =
      (_idx_q.dtype() == torch::kUInt8) ? 1 :
      (_idx_q.dtype() == torch::kInt16) ? 2 : 4;

  // ---- 调度 ----
  if (bit == 4) {
    battn_v_kernel_with_base<4, ABLATION_LANE0_TABLE_FULL><<<blocks, threads, shmem>>>(
      alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
      K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
    );
  } else if (bit == 2) {
    battn_v_kernel_with_base<2, ABLATION_LANE0_TABLE_FULL><<<blocks, threads, shmem>>>(
      alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
      K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
    );
  } else {
    TORCH_CHECK(false, "Only 2-bit or 4-bit are supported.");
  }
  return _out;
}

torch::Tensor attn_v_forward_cuda_outer_dim_with_base_debug(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full,
    const int debug_mode
){
  TORCH_CHECK(debug_mode == ABLATION_FULL ||
              debug_mode == ABLATION_RESIDUAL_ONLY ||
              debug_mode == ABLATION_NO_CENTROID_HISTOGRAM ||
              debug_mode == ABLATION_CENTROID_ONLY ||
              debug_mode == ABLATION_WARP_AGG_FULL ||
              debug_mode == ABLATION_PER_WARP_HIST_FULL ||
              debug_mode == ABLATION_NO_TABLE_CONTRIBUTION ||
              debug_mode == ABLATION_LANE0_TABLE_FULL,
              "Invalid debug_mode. Expected 0=FULL, 1=RESIDUAL_ONLY, 2=NO_CENTROID_HISTOGRAM, 3=CENTROID_ONLY, 4=WARP_AGG_FULL, 5=PER_WARP_HIST_FULL, 6=NO_TABLE_CONTRIBUTION, 7=LANE0_TABLE_FULL.");
  TORCH_CHECK(_alpha_q.dim()==3 && _alpha_q.size(1)==1, "alpha_q must be [B*nh,1,K]");
  const int BSnh = _alpha_q.size(0);
  const int K    = _alpha_q.size(2);

  TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv, "centroids must be [nh_kv,M,OC]");
  const int OC   = _centroids.size(2);

  TORCH_CHECK(_vq.dim()==3 && _vq.size(2)==K, "vq must be [B*nh_kv, OC/pack, K]");
  const int PACK = 32 / bit;
  TORCH_CHECK(_vq.size(1) * PACK == OC, "vq.pack mismatch: (OC/pack)*pack must equal OC");

  TORCH_CHECK(_vscale.dim()==3 && _vzero.dim()==3, "scale/zero must be 3D [B*nh_kv, OC/group, K]");
  const int Mcent = _centroids.size(1);

  TORCH_CHECK(_mask_q.dim()==3 && _mask_q.size(2)==K && _mask_q.size(1)==nh_kv, "mask_q must be [B,nh_kv,K]");
  TORCH_CHECK(_idx_q .dim()==3 && _idx_q .size(2)==K && _idx_q .size(1)==nh_kv, "idx_q must be [B,nh_kv,K]");

  const int Lf = (_alpha_f.numel()==0 || _v_full.numel()==0) ? 0 : _alpha_f.size(1);
  if (Lf > 0) {
    TORCH_CHECK(_v_full.dim()==4 && _v_full.size(2)==Lf && _v_full.size(3)==OC, "v_full must be [B,nh_kv,Lf,OC]");
  }

  auto options = torch::TensorOptions().dtype(_alpha_q.dtype()).device(_alpha_q.device());
  at::Tensor _out = torch::empty({BSnh, 1, OC}, options);

  const half*      alpha_q = reinterpret_cast<const half*     >(_alpha_q.data_ptr<at::Half>());
  const uint32_t*  vq      = reinterpret_cast<const uint32_t* >(_vq      .data_ptr<int>());
  const half*      vsc     = reinterpret_cast<const half*     >(_vscale  .data_ptr<at::Half>());
  const half*      vzr     = reinterpret_cast<const half*     >(_vzero   .data_ptr<at::Half>());
  const half*      cent    = reinterpret_cast<const half*     >(_centroids.data_ptr<at::Half>());
  const uint8_t*   mask    = reinterpret_cast<const uint8_t*  >(_mask_q  .data_ptr<uint8_t>());
  const void*      idx     = static_cast<const void*>(_idx_q.data_ptr());
  const half*      alpha_f = (_alpha_f.numel()==0) ? nullptr : reinterpret_cast<const half*>(_alpha_f.data_ptr<at::Half>());
  const half*      v_full  = (_v_full .numel()==0) ? nullptr : reinterpret_cast<const half*>(_v_full .data_ptr<at::Half>());
  half*            outp    = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  dim3 threads(32, 4, 1);
  dim3 blocks(BSnh, (OC / PACK + threads.y - 1) / threads.y, 1);
  const int sacc_rows = (debug_mode == ABLATION_PER_WARP_HIST_FULL ||
                         debug_mode == ABLATION_NO_TABLE_CONTRIBUTION ||
                         debug_mode == ABLATION_LANE0_TABLE_FULL) ? 4 : 1;
  size_t shmem = (size_t)Mcent * (size_t)sacc_rows * sizeof(float);

  const int idx_bytes =
      (_idx_q.dtype() == torch::kUInt8) ? 1 :
      (_idx_q.dtype() == torch::kInt16) ? 2 : 4;

#define DISPATCH_V_ABLATION(BIT_VALUE, MODE_VALUE) \
  battn_v_kernel_with_base<BIT_VALUE, MODE_VALUE><<<blocks, threads, shmem>>>( \
    alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp, \
    K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes)

  if (bit == 4) {
    if (debug_mode == ABLATION_FULL) DISPATCH_V_ABLATION(4, ABLATION_FULL);
    else if (debug_mode == ABLATION_RESIDUAL_ONLY) DISPATCH_V_ABLATION(4, ABLATION_RESIDUAL_ONLY);
    else if (debug_mode == ABLATION_NO_CENTROID_HISTOGRAM) DISPATCH_V_ABLATION(4, ABLATION_NO_CENTROID_HISTOGRAM);
    else if (debug_mode == ABLATION_CENTROID_ONLY) DISPATCH_V_ABLATION(4, ABLATION_CENTROID_ONLY);
    else if (debug_mode == ABLATION_WARP_AGG_FULL) DISPATCH_V_ABLATION(4, ABLATION_WARP_AGG_FULL);
    else if (debug_mode == ABLATION_PER_WARP_HIST_FULL) DISPATCH_V_ABLATION(4, ABLATION_PER_WARP_HIST_FULL);
    else if (debug_mode == ABLATION_NO_TABLE_CONTRIBUTION) DISPATCH_V_ABLATION(4, ABLATION_NO_TABLE_CONTRIBUTION);
    else DISPATCH_V_ABLATION(4, ABLATION_LANE0_TABLE_FULL);
  } else if (bit == 2) {
    if (debug_mode == ABLATION_FULL) DISPATCH_V_ABLATION(2, ABLATION_FULL);
    else if (debug_mode == ABLATION_RESIDUAL_ONLY) DISPATCH_V_ABLATION(2, ABLATION_RESIDUAL_ONLY);
    else if (debug_mode == ABLATION_NO_CENTROID_HISTOGRAM) DISPATCH_V_ABLATION(2, ABLATION_NO_CENTROID_HISTOGRAM);
    else if (debug_mode == ABLATION_CENTROID_ONLY) DISPATCH_V_ABLATION(2, ABLATION_CENTROID_ONLY);
    else if (debug_mode == ABLATION_WARP_AGG_FULL) DISPATCH_V_ABLATION(2, ABLATION_WARP_AGG_FULL);
    else if (debug_mode == ABLATION_PER_WARP_HIST_FULL) DISPATCH_V_ABLATION(2, ABLATION_PER_WARP_HIST_FULL);
    else if (debug_mode == ABLATION_NO_TABLE_CONTRIBUTION) DISPATCH_V_ABLATION(2, ABLATION_NO_TABLE_CONTRIBUTION);
    else DISPATCH_V_ABLATION(2, ABLATION_LANE0_TABLE_FULL);
  } else {
    TORCH_CHECK(false, "Only 2-bit or 4-bit are supported.");
  }

#undef DISPATCH_V_ABLATION
  return _out;
}

// // ---------------------- warp reduce --------------------------
// __device__ __forceinline__ float warp_reduce_sum_f32(float v) {
//   #pragma unroll
//   for (int i = 4; i >= 0; --i) {
//     v += __shfl_down_sync(0xffffffff, v, 1 << i);
//   }
//   return v;
// }

// // ---------------------- idx loader ---------------------------
// template<int IDX_BYTES>
// __device__ __forceinline__ int load_idx(const char* base, int t);

// template<>
// __device__ __forceinline__ int load_idx<1>(const char* base, int t) {
//   return static_cast<int>(reinterpret_cast<const uint8_t*>(base)[t]);
// }
// template<>
// __device__ __forceinline__ int load_idx<2>(const char* base, int t) {
//   return static_cast<int>(reinterpret_cast<const uint16_t*>(base)[t]);
// }
// template<>
// __device__ __forceinline__ int load_idx<4>(const char* base, int t) {
//   return reinterpret_cast<const int32_t*>(base)[t];
// }

// // ---------------------- 核函数主体 ---------------------------
// template<int BIT, int IDX_BYTES>
// __global__ void battn_v_kernel_with_base_opt(
//   const half*      __restrict__ _alpha_q,   // [B*nh, K]
//   const uint32_t*  __restrict__ _vq_lin,    // [B*nh_kv, (OC/pack)*K]
//   const half*      __restrict__ _vscale_lin,// [B*nh_kv, (OC/group)*K]
//   const half*      __restrict__ _vzero_lin, // [B*nh_kv, (OC/group)*K]
//   const half*      __restrict__ _centroids, // [nh_kv, Mcent, OC]
//   const uint8_t*   __restrict__ _mask_q,    // [B, nh_kv, K]
//   const void*      __restrict__ _idx_q,     // [B, nh_kv, K] (u8/u16/i32)
//   const half*      __restrict__ _alpha_f,   // [B*nh, Lf] (可空)
//   const half*      __restrict__ _v_full,    // [B, nh_kv, Lf, OC] (可空)
//   half*            __restrict__ _out,       // [B*nh, OC]
//   const int K, const int OC, const int Lf,
//   const int group_size, const int nh, const int nh_kv,
//   const int Mcent, const int idx_bytes_runtime // 仅用于断言/无分支
// ){
//   static_assert(BIT==2 || BIT==4, "BIT must be 2 or 4");
//   constexpr int PACK = 32 / BIT;        // 2bit=16, 4bit=8
//   const uint32_t CODE_MASK = (1u << BIT) - 1u;
//   const int TILE = 128;

//   // --- 线程块映射 ---
//   const int bnh = blockIdx.x;               // over [B*nh]
//   const int wy  = threadIdx.y;              // 0..(blockDim.y-1)
//   const int lane= threadIdx.x;              // 0..31

//   const int packed_oc_idx = blockIdx.y * blockDim.y + wy;   // 以 PACK 聚类的 oc 块
//   const int oc_start = packed_oc_idx * PACK;
//   if (oc_start >= OC) return;

//   // --- GQA 头映射 ---
//   const int ratio = nh / nh_kv;
//   const int b  = bnh / nh;
//   const int hq = bnh % nh;
//   const int hk = hq / ratio;

//   // --- 指针基址（含 batch/kv 偏移）---
//   const half* alpha_q = _alpha_q + (size_t)bnh * K;         // [K]
//   half* out_row = _out + (size_t)bnh * OC;                  // [OC]

//   const size_t bkv = (size_t)b * nh_kv + hk;
//   const uint32_t* vq_base = _vq_lin + bkv * (size_t)(OC / PACK) * K;
//   const half*     vsc_base= _vscale_lin + bkv * (size_t)(OC / group_size) * K;
//   const half*     vzr_base= _vzero_lin  + bkv * (size_t)(OC / group_size) * K;

//   const half* C = _centroids + (size_t)hk * (size_t)Mcent * OC; // [Mcent, OC]
//   const uint8_t* mask_row = _mask_q + bkv * (size_t)K;          // [K]
//   const char*    idx_row  = reinterpret_cast<const char*>(_idx_q)
//                            + bkv * (size_t)K * IDX_BYTES;

//   // --- 共享内存: Sacc[Mcent_pad] + alpha_tile[TILE] ---
//   extern __shared__ char __smem[];
//   int Mcent_pad = (Mcent & 31) ? Mcent : (Mcent + 1); // 避免 32 倍数引发银行对齐冲突
//   float* s_Sacc = reinterpret_cast<float*>(__smem);                 // size=Mcent_pad
//   half*  s_alpha= reinterpret_cast<half*>(s_Sacc + Mcent_pad);      // size=TILE

//   // 清零 Sacc（所有线程参与，跨步）
//   for (int c = wy * blockDim.x + lane; c < Mcent_pad; c += blockDim.x * blockDim.y)
//     s_Sacc[c] = 0.f;
//   __syncthreads();

//   // 量化残差 GEMV 累加器
//   float psum[PACK];
//   #pragma unroll
//   for (int p=0; p<PACK; ++p) psum[p] = 0.f;

//   // 预取本 oc 块所在的组（沿 OC 聚组）
//   const int oc_group = oc_start / group_size;

//   // 每 warp 对应自身的 vq/scale/zero 行
//   const uint32_t* vq_row = vq_base + (size_t)packed_oc_idx * K;
//   const half*     vsc_row= vsc_base + (size_t)oc_group * K;
//   const half*     vzr_row= vzr_base + (size_t)oc_group * K;

//   const int nTiles = (K + TILE - 1) / TILE;

//   // --- K 维分块 ---
//   for (int kt = 0; kt < nTiles; ++kt) {
//     const int t0 = kt * TILE;
//     const int tile_rem = min(TILE, K - t0);

//     // alpha_q -> smem（由 wy==0 的 warp 填充，块内复用）
//     if (wy == 0) {
//       // 简洁稳妥：按 32*4 的细粒度装载，保持边界判断简单
//       #pragma unroll
//       for (int i=0;i<4;++i) {
//         int t = t0 + lane*4 + i;
//         if (t < t0 + tile_rem)
//           s_alpha[lane*4 + i] = __ldg(alpha_q + t);
//       }
//       // 如需更极致，可改为 uint2/uint4 矢量化或 cp.async 双缓冲
//     }
//     __syncthreads();

//     // 直方图：Sacc[idx[t]] += α_q[t] * mask[t]  （仅 wy==0 的 warp 参与）
//     if (wy == 0) {
//       #pragma unroll
//       for (int i=0;i<4;++i) {
//         int t = t0 + lane*4 + i;
//         if (t < t0 + tile_rem) {
//           const uint8_t m = __ldg(mask_row + t);
//           if (m) {
//             const float a = __half2float(s_alpha[lane*4 + i]);
//             int idx = load_idx<IDX_BYTES>(idx_row, t);
//             if (0 <= idx && idx < Mcent) {
//               // s_Sacc 经过 padding，无需改 idx
//               atomicAdd(&s_Sacc[idx], a);
//             }
//           }
//         }
//       }
//     }

//     // 量化 V：对 [oc_start .. oc_start+PACK-1] 载入 packed 行、scale/zero 行
//     // vq 用 128-bit 矢量化，一次读 4 个 32b code
//     uint32_t qw[4] = {0,0,0,0};
//     half sc4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
//     half zr4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
//     half a4[4]  = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};

//     // 基地址：t_base = t0 + lane*4
//     // const int t_base = t0 + lane*4;

//     // // vq：若对齐且不越界，用 uint4 一次性取 16B
//     // if (t_base + 3 < t0 + tile_rem) {
//     //   const uint4 pack = reinterpret_cast<const uint4*>(vq_row + t_base)[0];
//     //   qw[0] = pack.x; qw[1] = pack.y; qw[2] = pack.z; qw[3] = pack.w;
//     // } else {
//     //   #pragma unroll
//     //   for (int i=0;i<4;++i) {
//     //     int t = t_base + i;
//     //     if (t < t0 + tile_rem) qw[i] = __ldg(vq_row + t);
//     //   }
//     // }
//     const int t_base = t0 + lane*4;
//     const uint32_t* vq_ptr = vq_row + t_base;
//     if (t_base + 3 < t0 + tile_rem && (((uintptr_t)vq_ptr & 0xF) == 0)) {
//       const uint4 pack = reinterpret_cast<const uint4*>(vq_ptr)[0];
//       qw[0]=pack.x; qw[1]=pack.y; qw[2]=pack.z; qw[3]=pack.w;
//     } else {
//       #pragma unroll
//       for (int i=0;i<4;++i) {
//         const int t = t_base + i;
//         if (t < t0 + tile_rem) qw[i] = __ldg(vq_row + t);
//       }
//     }

//     // scale/zero：用标量读，逻辑简单稳妥（可按需再矢量化）
//     #pragma unroll
//     for (int i=0;i<4;++i) {
//       int t = t_base + i;
//       if (t < t0 + tile_rem) {
//         sc4[i] = __ldg(vsc_row + t);
//         zr4[i] = __ldg(vzr_row + t);
//         a4[i]  = s_alpha[lane*4 + i]; // 由 smem 读 alpha
//       }
//     }

//     // FMA：psum[p] += (s*code + z) * a  →  先算 sa/za，内环仅一次 FMA + 加法
//     #pragma unroll
//     for (int j=0;j<4;++j) {
//       const float a = __half2float(a4[j]);
//       uint32_t cur  = qw[j];
//       const float s = __half2float(sc4[j]);
//       const float z = __half2float(zr4[j]);
//       const float sa = s * a;
//       const float za = z * a;

//       #pragma unroll
//       for (int p=0;p<PACK;++p) {
//         const int oc = oc_start + p;
//         if (oc < OC) {
//           const float code = float(cur & CODE_MASK);
//           psum[p] = fmaf(code, sa, psum[p]); // += code*sa
//         }
//         cur >>= BIT;
//       }
//       #pragma unroll
//       for (int p=0;p<PACK;++p) {
//         const int oc = oc_start + p;
//         if (oc < OC) psum[p] += za;         // 统一加偏置
//       }
//     }
//     __syncthreads(); // 保护 s_alpha 在下个 tile 被重写（wy==0 与其他 warp 同步）
//   } // end kt

//   __syncthreads();

//   // --- 基向量补偿：warp 协作 + 规约（避免 32/128 次重复） ---
//   float add_base_local[PACK];
//   #pragma unroll
//   for (int p=0;p<PACK;++p) add_base_local[p] = 0.f;

//   for (int c = lane; c < Mcent; c += 32) {
//     const float s = s_Sacc[c]; // padding 位置未被写，不影响
//     if (s != 0.f) {
//       const half* crow = C + (size_t)c * OC + oc_start; // [PACK] 连续
//       #pragma unroll
//       for (int p=0;p<PACK;++p) {
//         const int oc = oc_start + p;
//         if (oc < OC) add_base_local[p] = fmaf(s, __half2float(__ldg(crow + p)), add_base_local[p]);
//       }
//     }
//   }
//   float add_base[PACK];
//   #pragma unroll
//   for (int p=0;p<PACK;++p) {
//     float v = add_base_local[p];
//     v = warp_reduce_sum_f32(v);
//     if (lane == 0) add_base[p] = v;
//   }

//   // --- 最近窗口全精分量：add_full[p] = Σ_t α_f[t]·V_full[t, oc] ---
//   float add_full[PACK];
//   #pragma unroll
//   for (int p=0;p<PACK;++p) add_full[p] = 0.f;

//   if (Lf > 0 && _alpha_f && _v_full) {
//     const half* aF = _alpha_f + (size_t)bnh * Lf;                          // [Lf]
//     const half* vF = _v_full  + ((size_t)b * nh_kv + hk) * (size_t)Lf * OC; // [Lf, OC]
//     for (int t = lane; t < Lf; t += blockDim.x) {
//       const float a = __half2float(__ldg(aF + t));
//       const half* row = vF + (size_t)t * OC + oc_start;
//       #pragma unroll
//       for (int p=0;p<PACK;++p) {
//         const int oc = oc_start + p;
//         if (oc < OC) add_full[p] += a * __half2float(__ldg(row + p));
//       }
//     }
//     // 一个 warp 对应一个 oc tile → 只需 warp 内规约
//     #pragma unroll
//     for (int p=0;p<PACK;++p) {
//       float v = add_full[p];
//       v = warp_reduce_sum_f32(v);
//       if (lane == 0) add_full[p] = v;
//     }
//   }

//   // --- 写回 ---
//   #pragma unroll
//   for (int p=0;p<PACK;++p) {
//     const int oc = oc_start + p;
//     if (oc < OC) {
//       float vqsum = warp_reduce_sum_f32(psum[p]);
//       if (lane == 0) {
//         const float val = vqsum + add_base[p] + add_full[p];
//         out_row[oc] = __float2half(val);
//       }
//     }
//   }
// }

// // ---------------------- 外层导出 -----------------------------
// torch::Tensor attn_v_forward_cuda_outer_dim_with_base(
//     torch::Tensor _alpha_q,    // [B*nh, 1, K]
//     torch::Tensor _vq,         // [B*nh_kv, OC/pack, K]
//     torch::Tensor _vscale,     // [B*nh_kv, OC/group, K]
//     torch::Tensor _vzero,      // [B*nh_kv, OC/group, K]
//     const int bit,
//     const int group_size,
//     const int nh,
//     const int nh_kv,
//     torch::Tensor _centroids,  // [nh_kv, Mcent, OC]
//     torch::Tensor _mask_q,     // [B, nh_kv, K]  (uint8)
//     torch::Tensor _idx_q,      // [B, nh_kv, K]  (u8/u16/i32)
//     torch::Tensor _alpha_f,    // [B*nh, Lf]     (可空 size=0)
//     torch::Tensor _v_full      // [B, nh_kv, Lf, OC] (可空 size=0)
// ){
//   // ---- 基本检查 ----
//   TORCH_CHECK(_alpha_q.dim()==3 && _alpha_q.size(1)==1, "alpha_q must be [B*nh,1,K]");
//   const int BSnh = _alpha_q.size(0);
//   const int K    = _alpha_q.size(2);

//   TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv, "centroids must be [nh_kv,M,OC]");
//   const int OC   = _centroids.size(2);

//   TORCH_CHECK(_vq.dim()==3 && _vq.size(2)==K, "vq must be [B*nh_kv, OC/pack, K]");
//   TORCH_CHECK(bit==2 || bit==4, "Only 2-bit or 4-bit are supported.");
//   const int PACK = 32 / bit;
//   TORCH_CHECK(_vq.size(1) * PACK == OC, "vq.pack mismatch: (OC/pack)*pack must equal OC");

//   TORCH_CHECK(_vscale.dim()==3 && _vzero.dim()==3, "scale/zero must be 3D [B*nh_kv, OC/group, K]");
//   const int Mcent = _centroids.size(1);

//   TORCH_CHECK(_mask_q.dim()==3 && _mask_q.size(2)==K && _mask_q.size(1)==nh_kv, "mask_q must be [B,nh_kv,K]");
//   TORCH_CHECK(_idx_q .dim()==3 && _idx_q .size(2)==K && _idx_q .size(1)==nh_kv, "idx_q must be [B,nh_kv,K]");

//   const int Lf = (_alpha_f.numel()==0 || _v_full.numel()==0) ? 0 : _alpha_f.size(1);
//   if (Lf > 0) {
//     TORCH_CHECK(_v_full.dim()==4 && _v_full.size(2)==Lf && _v_full.size(3)==OC, "v_full must be [B,nh_kv,Lf,OC]");
//   }

//   // ---- 输出张量 ----
//   auto options = torch::TensorOptions().dtype(_alpha_q.dtype()).device(_alpha_q.device());
//   at::Tensor _out = torch::empty({BSnh, 1, OC}, options);

//   // ---- 原始指针（名称与 kernel 一一对应）----
//   const half*      alpha_q = reinterpret_cast<const half*     >(_alpha_q.data_ptr<at::Half>());
//   const uint32_t*  vq      = reinterpret_cast<const uint32_t* >(_vq      .data_ptr<int>());
//   const half*      vsc     = reinterpret_cast<const half*     >(_vscale  .data_ptr<at::Half>());
//   const half*      vzr     = reinterpret_cast<const half*     >(_vzero   .data_ptr<at::Half>());
//   const half*      cent    = reinterpret_cast<const half*     >(_centroids.data_ptr<at::Half>());
//   const uint8_t*   mask    = reinterpret_cast<const uint8_t*  >(_mask_q  .data_ptr<uint8_t>());
//   const void*      idx     = static_cast<const void*>(_idx_q.data_ptr());
//   const half*      alpha_f = (_alpha_f.numel()==0) ? nullptr : reinterpret_cast<const half*>(_alpha_f.data_ptr<at::Half>());
//   const half*      v_full  = (_v_full .numel()==0) ? nullptr : reinterpret_cast<const half*>(_v_full .data_ptr<at::Half>());
//   half*            outp    = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

//   // ---- launch 形状 ----
//   dim3 threads(32, 4, 1);  // 每个 warp 负责一个 oc tile
//   // dim3 blocks(BSnh, (OC / PACK + threads.y - 1) / threads.y, 1);
//   // 修正（向上取整）：
//   const int packed_tiles = (OC + PACK - 1) / PACK;   // ceil(OC/PACK)
//   dim3 blocks(BSnh, (packed_tiles + threads.y - 1) / threads.y, 1);

//   // ---- 动态共享内存：Sacc[Mcent_pad] + alpha_tile[TILE] ----
//   const int TILE = 128;
//   const int Mcent_pad = (Mcent & 31) ? Mcent : (Mcent + 1);
//   size_t shmem = (size_t)Mcent_pad * sizeof(float) + (size_t)TILE * sizeof(half);

//   // ---- 索引宽度（模板路径选择）----
//   const int idx_bytes =
//       (_idx_q.dtype() == torch::kUInt8) ? 1 :
//       (_idx_q.dtype() == torch::kInt16) ? 2 : 4;

//   // ---- 调度 ----
//   if (bit == 4) {
//     if (idx_bytes == 1) {
//       battn_v_kernel_with_base_opt<4,1><<<blocks, threads, shmem>>>(
//         alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
//         K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
//       );
//     } else if (idx_bytes == 2) {
//       battn_v_kernel_with_base_opt<4,2><<<blocks, threads, shmem>>>(
//         alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
//         K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
//       );
//     } else {
//       battn_v_kernel_with_base_opt<4,4><<<blocks, threads, shmem>>>(
//         alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
//         K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
//       );
//     }
//   } else { // bit == 2
//     if (idx_bytes == 1) {
//       battn_v_kernel_with_base_opt<2,1><<<blocks, threads, shmem>>>(
//         alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
//         K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
//       );
//     } else if (idx_bytes == 2) {
//       battn_v_kernel_with_base_opt<2,2><<<blocks, threads, shmem>>>(
//         alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
//         K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
//       );
//     } else {
//       battn_v_kernel_with_base_opt<2,4><<<blocks, threads, shmem>>>(
//         alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
//         K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
//       );
//     }
//   }

//   // 可选：调试期同步检查
//   // CUDA_CHECK(cudaGetLastError());

//   return _out;
// }
