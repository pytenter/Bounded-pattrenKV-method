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
  const half* __restrict__ _centroids,   // [nh_kv, M, IC] or [B, nh_kv, M, IC]
  const void* __restrict__ _assign,      // [B, nh_kv, OC] (u8/u16/i32)
  half* __restrict__ _outputs,           // [B*nh, OC]
  const int IC, const int OC,
  const int group_size, const int nh, const int nh_kv,
  const int M_centroids, const int B_centroids, const int assign_bytes
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

  // centroids: [nh_kv, M, IC] or [B, nh_kv, M, IC]
  const size_t c_offset = (B_centroids == 1)
                        ? ((size_t)kv * M_centroids * IC)
                        : (((size_t)b * nh_kv + kv) * M_centroids * IC);
  const half* cbase = _centroids + c_offset;
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
    torch::Tensor _centroids,         // [nh_kv, M, K] or [B, nh_kv, M, K]
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

  const int B = BS_nh / nh;
  TORCH_CHECK(
      (_centroids.dim()==3 && _centroids.size(0)==nh_kv && _centroids.size(2)==IC) ||
      (_centroids.dim()==4 && _centroids.size(0)==B && _centroids.size(1)==nh_kv && _centroids.size(3)==IC),
      "centroids must be [nh_kv, M, IC] or [B, nh_kv, M, IC]");
  const int B_cent = (_centroids.dim()==4) ? B : 1;
  const int M_cent = (_centroids.dim()==4) ? _centroids.size(2) : _centroids.size(1);
  auto centroids = _centroids.to(torch::kFloat16).contiguous();
  const half* cptr = reinterpret_cast<const half*>(centroids.data_ptr<at::Half>());

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
        IC, OC, group_size, nh, nh_kv, M_cent, B_cent, assign_bytes
      );
  } else { // bit == 2
    bgemv_kernel_outer_dim_with_base_tiled<2>
      <<<num_blocks, num_threads, smem_bytes>>>(
        in_ptr, wptr, zptr, sptr, cptr, aptr, out_ptr,
        IC, OC, group_size, nh, nh_kv, M_cent, B_cent, assign_bytes
      );
  }

  // 注意：实际工程里可在此处加 cudaGetLastError() / cudaDeviceSynchronize() 做调试
  return _out;
}

template<int Bit>
__global__ void bgemv_kernel_outer_dim_with_base_strided_k(
  const half* __restrict__ _inputs,      // [B*nh, IC]
  const uint32_t* __restrict__ _weight,  // [B, nh_kv, IC, ceil(OC/pack)]
  const half* __restrict__ _zeros,       // [B, nh_kv, IC, ceil(OC/group)]
  const half* __restrict__ _scale,       // [B, nh_kv, IC, ceil(OC/group)]
  const half* __restrict__ _centroids,   // [nh_kv, M, IC]
  const void* __restrict__ _assign,      // [B, nh_kv, OC]
  half* __restrict__ _outputs,           // [B*nh, OC]
  const int IC, const int OC,
  const int group_size, const int nh, const int nh_kv,
  const int M_centroids, const int assign_bytes,
  const int64_t w_s0, const int64_t w_s1, const int64_t w_s2, const int64_t w_s3,
  const int64_t sc_s0, const int64_t sc_s1, const int64_t sc_s2, const int64_t sc_s3,
  const int64_t z_s0, const int64_t z_s1, const int64_t z_s2, const int64_t z_s3,
  const int64_t a_s0, const int64_t a_s1, const int64_t a_s2
){
  constexpr int pack_factor = 32 / Bit;
  constexpr int TILE_DIM    = 128;
  const uint32_t mask = (1u << Bit) - 1u;

  const int batch_idx = blockIdx.x;
  const int ratio = nh / nh_kv;
  const int b  = batch_idx / nh;
  const int hq = batch_idx % nh;
  const int kv = hq / ratio;

  const half* inputs = _inputs  + (size_t)batch_idx * IC;
  half* outputs      = _outputs + (size_t)batch_idx * OC;
  const half* cbase  = _centroids + (size_t)kv * (M_centroids * IC);

  extern __shared__ unsigned char smem_raw[];
  float* s_qC = reinterpret_cast<float*>(smem_raw);
  half*  s_in = reinterpret_cast<half*>(s_qC + M_centroids);

  {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = (threadIdx.y * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warps_per_CTA = (blockDim.x * blockDim.y) / WARP_SIZE;
    for (int m = warp; m < M_centroids; m += warps_per_CTA) {
      float acc = 0.f;
      for (int k = lane; k < IC; k += WARP_SIZE) {
        acc += __half2float(inputs[k]) * __half2float(cbase[(size_t)m * IC + k]);
      }
      acc = warp_reduce_sum_local(acc);
      if (lane == 0) s_qC[m] = acc;
    }
    __syncthreads();
  }

  const int nPacked = ceil_div(OC, pack_factor);
  const int start_packed = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_packed = gridDim.y * blockDim.y;

  for (int packed = start_packed; packed < nPacked; packed += stride_packed) {
    const int oc_start = packed * pack_factor;
    if (oc_start >= OC) continue;
    const int group_idx = oc_start / group_size;

    float psum[pack_factor];
    #pragma unroll
    for (int i = 0; i < pack_factor; ++i) psum[i] = 0.f;

    float beta_acc = 0.f;
    const int nKTiles = ceil_div(IC, TILE_DIM);

    for (int kt = 0; kt < nKTiles; ++kt) {
      const int k_base = kt * TILE_DIM;
      if (threadIdx.y == 0) {
        const int in_off = k_base + threadIdx.x * 4;
        half v0 = __float2half(0.f), v1 = __float2half(0.f),
             v2 = __float2half(0.f), v3 = __float2half(0.f);
        if (in_off + 0 < IC) v0 = inputs[in_off + 0];
        if (in_off + 1 < IC) v1 = inputs[in_off + 1];
        if (in_off + 2 < IC) v2 = inputs[in_off + 2];
        if (in_off + 3 < IC) v3 = inputs[in_off + 3];
        const int loc = threadIdx.x * 4;
        s_in[loc + 0] = v0;
        s_in[loc + 1] = v1;
        s_in[loc + 2] = v2;
        s_in[loc + 3] = v3;
      }
      __syncthreads();

      uint32_t qw[4] = {0, 0, 0, 0};
      half sc4[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
      half ze4[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
      const int k0 = k_base + threadIdx.x * 4;
      #pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int kk = k0 + t;
        if (kk < IC) {
          qw[t] = _weight[(int64_t)b * w_s0 + (int64_t)kv * w_s1 + (int64_t)kk * w_s2 + (int64_t)packed * w_s3];
          sc4[t] = _scale[(int64_t)b * sc_s0 + (int64_t)kv * sc_s1 + (int64_t)kk * sc_s2 + (int64_t)group_idx * sc_s3];
          ze4[t] = _zeros[(int64_t)b * z_s0 + (int64_t)kv * z_s1 + (int64_t)kk * z_s2 + (int64_t)group_idx * z_s3];
        }
      }

      const int loc = threadIdx.x * 4;
      #pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int k_local = loc + t;
        if (k_base + k_local >= IC) break;
        const float cur_inp   = __half2float(s_in[k_local]);
        const float cur_scale = __half2float(sc4[t]);
        const float cur_zero  = __half2float(ze4[t]);
        const float alpha = cur_scale * cur_inp;
        beta_acc += cur_zero * cur_inp;
        const uint32_t q = qw[t];
        #pragma unroll
        for (int i = 0; i < pack_factor; ++i) {
          const int wi = (q >> (i * Bit)) & mask;
          psum[i] = fmaf(static_cast<float>(wi), alpha, psum[i]);
        }
      }
      __syncthreads();
    }

    beta_acc = warp_reduce_sum_local(beta_acc);
    #pragma unroll
    for (int i = 0; i < pack_factor; ++i) {
      const int oc = oc_start + i;
      if (oc < OC) {
        float v = warp_reduce_sum_local(psum[i]);
        if (threadIdx.x == 0) {
          const char* abase = reinterpret_cast<const char*>(_assign);
          const int64_t elem_off = (int64_t)b * a_s0 + (int64_t)kv * a_s1 + (int64_t)oc * a_s2;
          int aidx;
          if (assign_bytes == 1)      aidx = *((const uint8_t *)(abase + elem_off));
          else if (assign_bytes == 2) aidx = *((const int16_t *)(abase + elem_off * 2));
          else if (assign_bytes == 4) aidx = *((const int32_t *)(abase + elem_off * 4));
          else                        aidx = static_cast<int>(*((const int64_t *)(abase + elem_off * 8)));
          const float add = (aidx >= 0 && aidx < M_centroids) ? s_qC[aidx] : 0.f;
          outputs[oc] = __float2half(v + beta_acc + add);
        }
      }
    }
  }
}

torch::Tensor gemv_forward_cuda_outer_dim_with_base_strided_k(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _assignments
){
  TORCH_CHECK(_in_feats.dim()==3 && _in_feats.size(1)==1, "in_feats must be [B*nh,1,K]");
  TORCH_CHECK(_kernel.dim()==4, "kernel must be [B,nh_kv,K,ceil(N/pack)]");
  TORCH_CHECK(_scaling_factors.dim()==4 && _zeros.dim()==4, "scale/zero must be [B,nh_kv,K,ceil(N/group)]");
  TORCH_CHECK(_assignments.dim()==3, "assignments must be [B,nh_kv,N]");
  TORCH_CHECK(bit==2 || bit==4, "only 2/4 bit supported");
  TORCH_CHECK(_in_feats.scalar_type()==torch::kFloat16, "in_feats must be float16");
  TORCH_CHECK(_kernel.scalar_type()==torch::kInt32, "kernel must be int32");
  TORCH_CHECK(_scaling_factors.scalar_type()==torch::kFloat16 && _zeros.scalar_type()==torch::kFloat16,
              "scale/zero must be float16");
  TORCH_CHECK(_centroids.scalar_type()==torch::kFloat16, "centroids must be float16");
  TORCH_CHECK(_assignments.scalar_type()==torch::kUInt8 || _assignments.scalar_type()==torch::kInt16 ||
              _assignments.scalar_type()==torch::kInt32 || _assignments.scalar_type()==torch::kInt64,
              "assignments must be uint8, int16, int32, or int64");

  const int B = _kernel.size(0);
  const int IC = _in_feats.size(2);
  const int BS_nh = _in_feats.size(0);
  const int OC = _assignments.size(2);
  const int pack_factor = 32 / bit;
  const int nPacked = (OC + pack_factor - 1) / pack_factor;
  const int nGroups = (OC + group_size - 1) / group_size;
  TORCH_CHECK(nh % nh_kv == 0, "nh must be divisible by nh_kv");
  TORCH_CHECK(BS_nh == B * nh, "in_feats batch/head mismatch");
  TORCH_CHECK(_kernel.size(1)==nh_kv && _kernel.size(2)==IC && _kernel.size(3) >= nPacked,
              "kernel expected [B,nh_kv,K,ceil(N/pack)]");
  TORCH_CHECK(_scaling_factors.size(0)==B && _scaling_factors.size(1)==nh_kv &&
              _scaling_factors.size(2)==IC && _scaling_factors.size(3) >= nGroups,
              "scale expected [B,nh_kv,K,ceil(N/group)]");
  TORCH_CHECK(_zeros.size(0)==B && _zeros.size(1)==nh_kv &&
              _zeros.size(2)==IC && _zeros.size(3) >= nGroups,
              "zero expected [B,nh_kv,K,ceil(N/group)]");
  TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv && _centroids.size(2)==IC,
              "centroids must be [nh_kv,M,IC]");
  TORCH_CHECK(_assignments.size(0)==B && _assignments.size(1)==nh_kv,
              "assignments expected [B,nh_kv,N]");

  auto options = torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out = torch::empty({BS_nh, 1, OC}, options);

  const half* in_ptr = reinterpret_cast<const half*>(_in_feats.data_ptr<at::Half>());
  const uint32_t* wptr = reinterpret_cast<const uint32_t*>(_kernel.data_ptr<int>());
  const half* zptr = reinterpret_cast<const half*>(_zeros.data_ptr<at::Half>());
  const half* sptr = reinterpret_cast<const half*>(_scaling_factors.data_ptr<at::Half>());
  const half* cptr = reinterpret_cast<const half*>(_centroids.data_ptr<at::Half>());
  const void* aptr = static_cast<const void*>(_assignments.data_ptr());
  half* out_ptr = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  const int M_cent = _centroids.size(1);
  const int assign_bytes =
    (_assignments.dtype()==torch::kUInt8) ? 1 :
    (_assignments.dtype()==torch::kInt16) ? 2 :
    (_assignments.dtype()==torch::kInt32) ? 4 : 8;

  constexpr int TILE_DIM = 128;
  dim3 num_threads(32, 8, 1);
  const int blocks_y_i = std::max(
    1,
    (nPacked + static_cast<int>(num_threads.y) - 1) / static_cast<int>(num_threads.y)
  );
  dim3 num_blocks(BS_nh, static_cast<unsigned int>(blocks_y_i), 1);
  size_t smem_bytes = (size_t)M_cent * sizeof(float) + (size_t)TILE_DIM * sizeof(half);

  if (bit == 4) {
    bgemv_kernel_outer_dim_with_base_strided_k<4>
      <<<num_blocks, num_threads, smem_bytes>>>(
        in_ptr, wptr, zptr, sptr, cptr, aptr, out_ptr,
        IC, OC, group_size, nh, nh_kv, M_cent, assign_bytes,
        _kernel.stride(0), _kernel.stride(1), _kernel.stride(2), _kernel.stride(3),
        _scaling_factors.stride(0), _scaling_factors.stride(1), _scaling_factors.stride(2), _scaling_factors.stride(3),
        _zeros.stride(0), _zeros.stride(1), _zeros.stride(2), _zeros.stride(3),
        _assignments.stride(0), _assignments.stride(1), _assignments.stride(2)
      );
  } else {
    bgemv_kernel_outer_dim_with_base_strided_k<2>
      <<<num_blocks, num_threads, smem_bytes>>>(
        in_ptr, wptr, zptr, sptr, cptr, aptr, out_ptr,
        IC, OC, group_size, nh, nh_kv, M_cent, assign_bytes,
        _kernel.stride(0), _kernel.stride(1), _kernel.stride(2), _kernel.stride(3),
        _scaling_factors.stride(0), _scaling_factors.stride(1), _scaling_factors.stride(2), _scaling_factors.stride(3),
        _zeros.stride(0), _zeros.stride(1), _zeros.stride(2), _zeros.stride(3),
        _assignments.stride(0), _assignments.stride(1), _assignments.stride(2)
      );
  }
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

__device__ __forceinline__ const uint32_t* page_ptr_u32(const int64_t* ptrs, int page_id) {
  return reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(ptrs[page_id]));
}

__device__ __forceinline__ const uint8_t* strided_idx_ptr(
    const void* base,
    const long long b_stride,
    const long long h_stride,
    const long long token_stride,
    const int b,
    const int hk,
    const int t,
    const int idx_bytes) {
  const long long elem_offset =
      (long long)b * b_stride + (long long)hk * h_stride + (long long)t * token_stride;
  return reinterpret_cast<const uint8_t*>(base) + elem_offset * idx_bytes;
}

// S5A-1 experimental V2-only reader. This is intentionally a close copy of the
// production ABLATION_LANE0_TABLE_FULL math path with only historical-cache
// addressing changed from tight K-derived strides to explicit tensor strides.
template<int BIT>
__global__ void battn_v_kernel_with_base_strided(
  const half*      __restrict__ _alpha_q,   // [B*nh, K]
  const uint32_t*  __restrict__ _vq,        // [B, nh_kv, K, OC/16] strided
  const half*      __restrict__ _vscale,    // [B, nh_kv, K, OC/group] strided
  const half*      __restrict__ _vzero,     // [B, nh_kv, K, OC/group] strided
  const half*      __restrict__ _centroids, // [nh_kv, Mcent, OC]
  const uint8_t*   __restrict__ _mask_q,    // [B, nh_kv, K] strided
  const void*      __restrict__ _idx_q,     // [B, nh_kv, K] strided
  const half*      __restrict__ _alpha_f,   // [B*nh, Lf] (可空)
  const half*      __restrict__ _v_full,    // [B, nh_kv, Lf, OC] (可空)
  half*            __restrict__ _out,       // [B*nh, OC]
  const int K, const int OC, const int Lf,
  const int group_size, const int nh, const int nh_kv,
  const int Mcent, const int idx_bytes,
  const long long vq_stride_b,
  const long long vq_stride_h,
  const long long vq_stride_t,
  const long long vq_stride_pack,
  const long long vscale_stride_b,
  const long long vscale_stride_h,
  const long long vscale_stride_t,
  const long long vscale_stride_group,
  const long long vzero_stride_b,
  const long long vzero_stride_h,
  const long long vzero_stride_t,
  const long long vzero_stride_group,
  const long long mask_stride_b,
  const long long mask_stride_h,
  const long long mask_stride_t,
  const long long idx_stride_b,
  const long long idx_stride_h,
  const long long idx_stride_t)
{
  static_assert(BIT==2 || BIT==4, "BIT must be 2 or 4");
  constexpr int PACK = 32 / BIT;
  const uint32_t CODE_MASK = (1u << BIT) - 1u;
  const int TILE = 128;

  const int bnh = blockIdx.x;
  const int wy  = threadIdx.y;
  const int lane= threadIdx.x;

  const int packed_oc_idx = blockIdx.y * blockDim.y + wy;
  const int oc_start = packed_oc_idx * PACK;
  if (oc_start >= OC) return;

  const int ratio = nh / nh_kv;
  const int b  = bnh / nh;
  const int hq = bnh % nh;
  const int hk = hq / ratio;

  const half* alpha_q = _alpha_q + (size_t)bnh * K;
  half* out_row = _out + (size_t)bnh * OC;

  const uint32_t* vq_bh = _vq + (long long)b * vq_stride_b + (long long)hk * vq_stride_h;
  const half* vsc_bh = _vscale + (long long)b * vscale_stride_b + (long long)hk * vscale_stride_h;
  const half* vzr_bh = _vzero + (long long)b * vzero_stride_b + (long long)hk * vzero_stride_h;
  const uint8_t* mask_bh = _mask_q + (long long)b * mask_stride_b + (long long)hk * mask_stride_h;
  const half* C = _centroids + (size_t)hk * (size_t)Mcent * OC;

  extern __shared__ float s_Sacc[];
  const int sacc_rows = blockDim.y;
  const int sacc_elems = Mcent * sacc_rows;
  for (int c = wy * blockDim.x + lane; c < sacc_elems; c += blockDim.x * blockDim.y)
    s_Sacc[c] = 0.f;
  __syncthreads();

  float psum[PACK];
  #pragma unroll
  for (int p=0; p<PACK; ++p) psum[p] = 0.f;

  const int oc_group = oc_start / group_size;
  const uint32_t* vq_pack_row = vq_bh + (long long)packed_oc_idx * vq_stride_pack;
  const half* vsc_group_row = vsc_bh + (long long)oc_group * vscale_stride_group;
  const half* vzr_group_row = vzr_bh + (long long)oc_group * vzero_stride_group;

  for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
    const int t_base = kt * TILE + lane * 4;

    half a4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
    #pragma unroll
    for (int i=0;i<4;++i) {
      const int t = t_base + i;
      if (t < K) a4[i] = __ldg(alpha_q + t);
    }

    const int t_hist = kt * TILE + wy * blockDim.x + lane;
    if (t_hist < K) {
      const uint8_t m = __ldg(mask_bh + (long long)t_hist * mask_stride_t);
      if (m) {
        int idx;
        const uint8_t* ip = strided_idx_ptr(
            _idx_q, idx_stride_b, idx_stride_h, idx_stride_t, b, hk, t_hist, idx_bytes);
        if (idx_bytes==1)      idx = *((const uint8_t *)(ip));
        else if (idx_bytes==2) idx = *((const uint16_t*)(ip));
        else                   idx = *((const int32_t *)(ip));
        if (0 <= idx && idx < Mcent) {
          atomicAdd(&s_Sacc[wy * Mcent + idx], __half2float(__ldg(alpha_q + t_hist)));
        }
      }
    }

    uint32_t qw[4] = {0,0,0,0};
    half sc4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
    half zr4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};

    #pragma unroll
    for (int i=0;i<4;++i) {
      const int t = t_base + i;
      if (t < K) {
        qw[i]  = __ldg(vq_pack_row + (long long)t * vq_stride_t);
        sc4[i] = __ldg(vsc_group_row + (long long)t * vscale_stride_t);
        zr4[i] = __ldg(vzr_group_row + (long long)t * vzero_stride_t);
      }
    }

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
  __syncthreads();

  float add_base[PACK];
  #pragma unroll
  for (int p=0;p<PACK;++p) add_base[p] = 0.f;

  if (lane == 0) {
    for (int c=0; c<Mcent; ++c) {
      float s = 0.f;
      #pragma unroll
      for (int w=0; w<4; ++w) {
        if (w < blockDim.y) s += s_Sacc[w * Mcent + c];
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

  float add_full[PACK];
  #pragma unroll
  for (int p=0;p<PACK;++p) add_full[p] = 0.f;

  if (Lf > 0 && _alpha_f && _v_full) {
    const half* aF = _alpha_f + (size_t)bnh * Lf;
    const half* vF = _v_full  + ((size_t)b * nh_kv + hk) * (size_t)Lf * OC;
    for (int t = lane; t < Lf; t += blockDim.x) {
      const float a = __half2float(__ldg(aF + t));
      const half* row = vF + (size_t)t * OC + oc_start;
      #pragma unroll
      for (int p=0;p<PACK;++p) {
        const int oc = oc_start + p;
        if (oc < OC) add_full[p] += a * __half2float(__ldg(row + p));
      }
    }
    #pragma unroll
    for (int p=0;p<PACK;++p) {
      float v = add_full[p];
      v = warp_reduce_sum_f32(v);
      if (lane == 0) add_full[p] = v;
    }
  }

  #pragma unroll
  for (int p=0;p<PACK;++p) {
    const int oc = oc_start + p;
    if (oc < OC) {
      float vqsum = warp_reduce_sum_f32(psum[p]);
      if (lane == 0) {
        const float val = vqsum + add_base[p] + add_full[p];
        out_row[oc] = __float2half(val);
      }
    }
  }
}

__device__ __forceinline__ const half* page_ptr_half(const int64_t* ptrs, int page_id) {
  return reinterpret_cast<const half*>(static_cast<uintptr_t>(ptrs[page_id]));
}

__device__ __forceinline__ const uint8_t* page_ptr_u8(const int64_t* ptrs, int page_id) {
  return reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ptrs[page_id]));
}

__device__ __forceinline__ const char* page_ptr_char(const int64_t* ptrs, int page_id) {
  return reinterpret_cast<const char*>(static_cast<uintptr_t>(ptrs[page_id]));
}

__device__ __forceinline__ int load_idx_paged_runtime(const int64_t* page_ptrs, int page_id, size_t offset, int idx_bytes) {
  const char* base = page_ptr_char(page_ptrs, page_id);
  if (idx_bytes == 1) return int(*((const uint8_t*)(base + offset)));
  if (idx_bytes == 2) return int(*((const uint16_t*)(base + offset * 2)));
  return int(*((const int32_t*)(base + offset * 4)));
}

__global__ void battn_v_kernel_with_base_paged_v2(
  const half*      __restrict__ _alpha_q,   // [B*nh, K]
  const int64_t*   __restrict__ _vq_pages,  // num_pages, each [B*nh_kv, OC/16, page_size]
  const int64_t*   __restrict__ _vsc_pages, // num_pages, each [B*nh_kv, OC/group, page_size]
  const int64_t*   __restrict__ _vzr_pages, // num_pages, each [B*nh_kv, OC/group, page_size]
  const half*      __restrict__ _centroids, // [nh_kv, Mcent, OC]
  const int64_t*   __restrict__ _mask_pages,// num_pages, each [B, nh_kv, page_size]
  const int64_t*   __restrict__ _idx_pages, // num_pages, each [B, nh_kv, page_size]
  const half*      __restrict__ _alpha_f,   // [B*nh, Lf] (optional)
  const half*      __restrict__ _v_full,    // [B, nh_kv, Lf, OC] (optional)
  half*            __restrict__ _out,       // [B*nh, OC]
  const int K, const int OC, const int Lf,
  const int group_size, const int nh, const int nh_kv,
  const int Mcent, const int page_size, const int idx_bytes)
{
  constexpr int BIT = 2;
  constexpr int PACK = 16;
  const uint32_t CODE_MASK = 3u;
  const int TILE = 128;

  const int bnh = blockIdx.x;
  const int wy = threadIdx.y;
  const int lane = threadIdx.x;
  const int packed_oc_idx = blockIdx.y * blockDim.y + wy;
  const int oc_start = packed_oc_idx * PACK;
  if (oc_start >= OC) return;

  const int ratio = nh / nh_kv;
  const int b = bnh / nh;
  const int hq = bnh % nh;
  const int hk = hq / ratio;
  const size_t bkv = (size_t)b * nh_kv + hk;

  const half* alpha_q = _alpha_q + (size_t)bnh * K;
  half* out_row = _out + (size_t)bnh * OC;
  const half* C = _centroids + (size_t)hk * (size_t)Mcent * OC;
  const int oc_group = oc_start / group_size;

  extern __shared__ float s_Sacc[];
  const int sacc_elems = Mcent * blockDim.y;
  for (int c = wy * blockDim.x + lane; c < sacc_elems; c += blockDim.x * blockDim.y) {
    s_Sacc[c] = 0.f;
  }
  __syncthreads();

  float psum[PACK];
  #pragma unroll
  for (int p = 0; p < PACK; ++p) psum[p] = 0.f;

  for (int kt = 0; kt < (K + TILE - 1) / TILE; ++kt) {
    const int t_base = kt * TILE + lane * 4;

    half a4[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = t_base + i;
      if (t < K) a4[i] = __ldg(alpha_q + t);
    }

    const int hist_t = kt * TILE + wy * blockDim.x + lane;
    if (hist_t < K) {
      const int page_id = hist_t / page_size;
      const int page_off = hist_t - page_id * page_size;
      const size_t off = bkv * (size_t)page_size + page_off;
      const uint8_t* mask_page = page_ptr_u8(_mask_pages, page_id);
      const uint8_t m = __ldg(mask_page + off);
      if (m) {
        const int idx = load_idx_paged_runtime(_idx_pages, page_id, off, idx_bytes);
        if (0 <= idx && idx < Mcent) {
          atomicAdd(&s_Sacc[wy * Mcent + idx], __half2float(__ldg(alpha_q + hist_t)));
        }
      }
    }

    uint32_t qw[4] = {0, 0, 0, 0};
    half sc4[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
    half zr4[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int t = t_base + i;
      if (t < K) {
        const int page_id = t / page_size;
        const int page_off = t - page_id * page_size;
        const uint32_t* vq_page = page_ptr_u32(_vq_pages, page_id);
        const half* vsc_page = page_ptr_half(_vsc_pages, page_id);
        const half* vzr_page = page_ptr_half(_vzr_pages, page_id);
        qw[i] = __ldg(vq_page + bkv * (size_t)(OC / PACK) * page_size + (size_t)packed_oc_idx * page_size + page_off);
        sc4[i] = __ldg(vsc_page + bkv * (size_t)(OC / group_size) * page_size + (size_t)oc_group * page_size + page_off);
        zr4[i] = __ldg(vzr_page + bkv * (size_t)(OC / group_size) * page_size + (size_t)oc_group * page_size + page_off);
      }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float a = __half2float(a4[j]);
      uint32_t cur = qw[j];
      const float s = __half2float(sc4[j]);
      const float z = __half2float(zr4[j]);
      #pragma unroll
      for (int p = 0; p < PACK; ++p) {
        const int oc = oc_start + p;
        if (oc < OC) {
          const float code = float(cur & CODE_MASK);
          psum[p] += (s * code + z) * a;
        }
        cur >>= BIT;
      }
    }
  }
  __syncthreads();

  float add_base[PACK];
  #pragma unroll
  for (int p = 0; p < PACK; ++p) add_base[p] = 0.f;

  if (lane == 0) {
    for (int c = 0; c < Mcent; ++c) {
      float s = 0.f;
      #pragma unroll
      for (int w = 0; w < 4; ++w) {
        if (w < blockDim.y) s += s_Sacc[w * Mcent + c];
      }
      if (s != 0.f) {
        const half* crow = C + (size_t)c * OC + oc_start;
        #pragma unroll
        for (int p = 0; p < PACK; ++p) {
          const int oc = oc_start + p;
          if (oc < OC) add_base[p] += s * __half2float(__ldg(crow + p));
        }
      }
    }
  }

  float add_full[PACK];
  #pragma unroll
  for (int p = 0; p < PACK; ++p) add_full[p] = 0.f;

  if (Lf > 0 && _alpha_f && _v_full) {
    const half* aF = _alpha_f + (size_t)bnh * Lf;
    const half* vF = _v_full + ((size_t)b * nh_kv + hk) * (size_t)Lf * OC;
    for (int t = lane; t < Lf; t += blockDim.x) {
      const float a = __half2float(__ldg(aF + t));
      const half* row = vF + (size_t)t * OC + oc_start;
      #pragma unroll
      for (int p = 0; p < PACK; ++p) {
        const int oc = oc_start + p;
        if (oc < OC) add_full[p] += a * __half2float(__ldg(row + p));
      }
    }
    #pragma unroll
    for (int p = 0; p < PACK; ++p) {
      float v = add_full[p];
      v = warp_reduce_sum_f32(v);
      if (lane == 0) add_full[p] = v;
    }
  }

  #pragma unroll
  for (int p = 0; p < PACK; ++p) {
    const int oc = oc_start + p;
    if (oc < OC) {
      const float vqsum = warp_reduce_sum_f32(psum[p]);
      if (lane == 0) {
        out_row[oc] = __float2half(vqsum + add_base[p] + add_full[p]);
      }
    }
  }
}

__device__ __forceinline__ int load_idx_runtime(const char* base, int t, int idx_bytes) {
  if (idx_bytes == 1) return int(*((const uint8_t*)(base + t)));
  if (idx_bytes == 2) return int(*((const uint16_t*)(base + (size_t)t * 2)));
  return int(*((const int32_t*)(base + (size_t)t * 4)));
}

template<int TILE>
__global__ void battn_v_kernel_gqa4_v2_with_base(
  const half*      __restrict__ _alpha_q,   // [B*nh, K]
  const uint32_t*  __restrict__ _vq_lin,    // [B*nh_kv, (OC/16)*K]
  const half*      __restrict__ _vscale_lin,// [B*nh_kv, (OC/group)*K]
  const half*      __restrict__ _vzero_lin, // [B*nh_kv, (OC/group)*K]
  const half*      __restrict__ _centroids, // [nh_kv, Mcent, OC]
  const uint8_t*   __restrict__ _mask_q,    // [B, nh_kv, K]
  const void*      __restrict__ _idx_q,     // [B, nh_kv, K]
  const half*      __restrict__ _alpha_f,   // [B*nh, Lf]
  const half*      __restrict__ _v_full,    // [B, nh_kv, Lf, OC]
  half*            __restrict__ _out,       // [B*nh, OC]
  const int K, const int OC, const int Lf,
  const int group_size, const int nh, const int nh_kv,
  const int Mcent, const int idx_bytes)
{
  constexpr int PACK = 16;
  constexpr int RATIO = 4;
  const uint32_t CODE_MASK = 3u;

  const int lane = threadIdx.x;
  const int wy = threadIdx.y;
  const int qslot = threadIdx.z;
  const int tile_oc = blockDim.y * PACK;
  const int packed_oc_idx = blockIdx.y * blockDim.y + wy;
  const int oc_start = packed_oc_idx * PACK;
  if (oc_start >= OC) return;

  const int bkv = blockIdx.x;
  const int b = bkv / nh_kv;
  const int hk = bkv % nh_kv;
  const int hq = hk * RATIO + qslot;
  const int bnh = b * nh + hq;

  const half* alpha_q = _alpha_q + (size_t)bnh * K;
  half* out_row = _out + (size_t)bnh * OC;

  const uint32_t* vq_base = _vq_lin + (size_t)bkv * (size_t)(OC / PACK) * K;
  const half* vsc_base = _vscale_lin + (size_t)bkv * (size_t)(OC / group_size) * K;
  const half* vzr_base = _vzero_lin + (size_t)bkv * (size_t)(OC / group_size) * K;
  const half* C = _centroids + (size_t)hk * (size_t)Mcent * OC;
  const uint8_t* mask_row = _mask_q + (size_t)bkv * K;
  const char* idx_row = reinterpret_cast<const char*>(_idx_q) + (size_t)bkv * (size_t)K * idx_bytes;

  extern __shared__ __align__(16) unsigned char smem[];
  size_t off = 0;
  float* s_Sacc = reinterpret_cast<float*>(smem + off);
  off += (size_t)RATIO * blockDim.y * Mcent * sizeof(float);
  uint32_t* s_vq = reinterpret_cast<uint32_t*>(smem + off);
  off += (size_t)blockDim.y * TILE * sizeof(uint32_t);
  half* s_scale = reinterpret_cast<half*>(smem + off);
  off += (size_t)blockDim.y * TILE * sizeof(half);
  half* s_zero = reinterpret_cast<half*>(smem + off);
  off += (size_t)blockDim.y * TILE * sizeof(half);
  int* s_mask = reinterpret_cast<int*>(smem + off);
  off += (size_t)TILE * sizeof(int);
  int* s_idx = reinterpret_cast<int*>(smem + off);
  off += (size_t)TILE * sizeof(int);
  half* s_cent = reinterpret_cast<half*>(smem + off);

  const int hist_elems = RATIO * blockDim.y * Mcent;
  for (int i = ((qslot * blockDim.y + wy) * blockDim.x + lane);
       i < hist_elems;
       i += blockDim.x * blockDim.y * blockDim.z) {
    s_Sacc[i] = 0.f;
  }
  __syncthreads();

  float psum[PACK];
  #pragma unroll
  for (int p = 0; p < PACK; ++p) psum[p] = 0.f;

  const int oc_group = oc_start / group_size;
  const int kt_count = (K + TILE - 1) / TILE;
  for (int kt = 0; kt < kt_count; ++kt) {
    const int base_t = kt * TILE;
    if (qslot == 0) {
      for (int e = lane; e < TILE; e += blockDim.x) {
        const int t = base_t + e;
        uint32_t qw = 0;
        half sc = __float2half(0.f);
        half zr = __float2half(0.f);
        if (t < K) {
          qw = __ldg(vq_base + (size_t)packed_oc_idx * K + t);
          sc = __ldg(vsc_base + (size_t)oc_group * K + t);
          zr = __ldg(vzr_base + (size_t)oc_group * K + t);
        }
        s_vq[wy * TILE + e] = qw;
        s_scale[wy * TILE + e] = sc;
        s_zero[wy * TILE + e] = zr;
      }
      if (wy == 0) {
        for (int e = lane; e < TILE; e += blockDim.x) {
          const int t = base_t + e;
          int m = 0;
          int idx = 0;
          if (t < K) {
            m = int(__ldg(mask_row + t));
            idx = load_idx_runtime(idx_row, t, idx_bytes);
          }
          s_mask[e] = m;
          s_idx[e] = idx;
        }
      }
    }
    __syncthreads();

    for (int e = lane; e < TILE; e += blockDim.x) {
      const int t = base_t + e;
      if (t < K) {
        const float a = __half2float(__ldg(alpha_q + t));
        uint32_t cur = s_vq[wy * TILE + e];
        const float sc = __half2float(s_scale[wy * TILE + e]);
        const float zr = __half2float(s_zero[wy * TILE + e]);
        #pragma unroll
        for (int p = 0; p < PACK; ++p) {
          const int oc = oc_start + p;
          if (oc < OC) {
            const float code = float(cur & CODE_MASK);
            psum[p] += (sc * code + zr) * a;
          }
          cur >>= 2;
        }
      }
    }

    const int hist_e = wy * blockDim.x + lane;
    if (hist_e < TILE) {
      const int t = base_t + hist_e;
      if (t < K && s_mask[hist_e]) {
        const int idx = s_idx[hist_e];
        if (0 <= idx && idx < Mcent) {
          const float mass = __half2float(__ldg(alpha_q + t));
          atomicAdd(&s_Sacc[(qslot * blockDim.y + wy) * Mcent + idx], mass);
        }
      }
    }
    __syncthreads();
  }

  if (qslot == 0) {
    const int cent_elems = Mcent * tile_oc;
    for (int i = wy * blockDim.x + lane; i < cent_elems; i += blockDim.x * blockDim.y) {
      const int c = i / tile_oc;
      const int oc_delta = i - c * tile_oc;
      const int oc = blockIdx.y * tile_oc + oc_delta;
      s_cent[i] = (oc < OC) ? __ldg(C + (size_t)c * OC + oc) : __float2half(0.f);
    }
  }
  __syncthreads();

  float add_base[PACK];
  #pragma unroll
  for (int p = 0; p < PACK; ++p) add_base[p] = 0.f;

  if (lane == 0) {
    for (int c = 0; c < Mcent; ++c) {
      float s = 0.f;
      #pragma unroll
      for (int w = 0; w < 4; ++w) {
        s += s_Sacc[(qslot * blockDim.y + w) * Mcent + c];
      }
      if (s != 0.f) {
        const half* crow = s_cent + c * tile_oc + wy * PACK;
        #pragma unroll
        for (int p = 0; p < PACK; ++p) {
          const int oc = oc_start + p;
          if (oc < OC) add_base[p] += s * __half2float(crow[p]);
        }
      }
    }
  }

  float add_full[PACK];
  #pragma unroll
  for (int p = 0; p < PACK; ++p) add_full[p] = 0.f;

  if (Lf > 0 && _alpha_f && _v_full) {
    const half* aF = _alpha_f + (size_t)bnh * Lf;
    const half* vF = _v_full + (size_t)bkv * (size_t)Lf * OC;
    for (int t = lane; t < Lf; t += blockDim.x) {
      const float a = __half2float(__ldg(aF + t));
      const half* row = vF + (size_t)t * OC + oc_start;
      #pragma unroll
      for (int p = 0; p < PACK; ++p) {
        const int oc = oc_start + p;
        if (oc < OC) add_full[p] += a * __half2float(__ldg(row + p));
      }
    }
    #pragma unroll
    for (int p = 0; p < PACK; ++p) {
      float v = add_full[p];
      v = warp_reduce_sum_f32(v);
      if (lane == 0) add_full[p] = v;
    }
  }

  #pragma unroll
  for (int p = 0; p < PACK; ++p) {
    const int oc = oc_start + p;
    if (oc < OC) {
      float vqsum = warp_reduce_sum_f32(psum[p]);
      if (lane == 0) {
        out_row[oc] = __float2half(vqsum + add_base[p] + add_full[p]);
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

static torch::Tensor attn_v_forward_cuda_outer_dim_with_base_strided_bit(
    torch::Tensor _alpha_q,    // [B*nh, 1, K]
    torch::Tensor _vq,         // [B, nh_kv, K, OC/pack], may be strided on K
    torch::Tensor _vscale,     // [B, nh_kv, K, OC/group], may be strided on K
    torch::Tensor _vzero,      // [B, nh_kv, K, OC/group], may be strided on K
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,  // [nh_kv, Mcent, OC]
    torch::Tensor _mask_q,     // [B, nh_kv, K] uint8, may be strided on K
    torch::Tensor _idx_q,      // [B, nh_kv, K] u8/i16/i32, may be strided on K
    torch::Tensor _alpha_f,    // [B*nh, Lf] size=0 allowed
    torch::Tensor _v_full      // [B, nh_kv, Lf, OC] size=0 allowed
){
  TORCH_CHECK(bit==2 || bit==4, "strided Value reader only supports bit=2 or bit=4");
  TORCH_CHECK(_alpha_q.dim()==3 && _alpha_q.size(1)==1, "alpha_q must be [B*nh,1,K]");
  TORCH_CHECK(_vq.dim()==4, "vq must be [B,nh_kv,K,OC/pack]");
  TORCH_CHECK(_vscale.dim()==4 && _vzero.dim()==4, "scale/zero must be [B,nh_kv,K,OC/group]");
  TORCH_CHECK(_mask_q.dim()==3 && _idx_q.dim()==3, "mask/idx must be [B,nh_kv,K]");
  TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv, "centroids must be [nh_kv,M,OC]");
  TORCH_CHECK(_vq.scalar_type()==torch::kInt32, "strided vq must be int32");
  TORCH_CHECK(_vscale.scalar_type()==torch::kFloat16 && _vzero.scalar_type()==torch::kFloat16,
              "strided scale/zero must be float16");
  TORCH_CHECK(_mask_q.scalar_type()==torch::kUInt8, "strided mask must be uint8");
  TORCH_CHECK(_idx_q.scalar_type()==torch::kUInt8 || _idx_q.scalar_type()==torch::kInt16 ||
              _idx_q.scalar_type()==torch::kInt32, "strided idx must be uint8, int16, or int32");

  const int B = _vq.size(0);
  const int K = _alpha_q.size(2);
  const int BSnh = _alpha_q.size(0);
  const int OC = _centroids.size(2);
  const int PACK = 32 / bit;
  const int Mcent = _centroids.size(1);
  TORCH_CHECK(nh % nh_kv == 0, "nh must be divisible by nh_kv");
  TORCH_CHECK(BSnh == B * nh, "alpha_q batch/head mismatch");
  TORCH_CHECK(_vq.size(1)==nh_kv && _vq.size(2)==K && _vq.size(3)*PACK==OC,
              "vq expected [B,nh_kv,K,OC/pack]");
  TORCH_CHECK(_vscale.size(0)==B && _vscale.size(1)==nh_kv && _vscale.size(2)==K &&
              _vscale.size(3)==OC/group_size, "vscale expected [B,nh_kv,K,OC/group]");
  TORCH_CHECK(_vzero.size(0)==B && _vzero.size(1)==nh_kv && _vzero.size(2)==K &&
              _vzero.size(3)==OC/group_size, "vzero expected [B,nh_kv,K,OC/group]");
  TORCH_CHECK(_mask_q.size(0)==B && _mask_q.size(1)==nh_kv && _mask_q.size(2)==K,
              "mask expected [B,nh_kv,K]");
  TORCH_CHECK(_idx_q.size(0)==B && _idx_q.size(1)==nh_kv && _idx_q.size(2)==K,
              "idx expected [B,nh_kv,K]");

  const int Lf = (_alpha_f.numel()==0 || _v_full.numel()==0) ? 0 : _alpha_f.size(1);
  if (Lf > 0) {
    TORCH_CHECK(_v_full.dim()==4 && _v_full.size(0)==B && _v_full.size(1)==nh_kv &&
                _v_full.size(2)==Lf && _v_full.size(3)==OC, "v_full must be [B,nh_kv,Lf,OC]");
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
  size_t shmem = (size_t)Mcent * 4 * sizeof(float);

  const int idx_bytes =
      (_idx_q.dtype() == torch::kUInt8) ? 1 :
      (_idx_q.dtype() == torch::kInt16) ? 2 : 4;

  if (bit == 2) {
    battn_v_kernel_with_base_strided<2><<<blocks, threads, shmem>>>(
      alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
      K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes,
      _vq.stride(0), _vq.stride(1), _vq.stride(2), _vq.stride(3),
      _vscale.stride(0), _vscale.stride(1), _vscale.stride(2), _vscale.stride(3),
      _vzero.stride(0), _vzero.stride(1), _vzero.stride(2), _vzero.stride(3),
      _mask_q.stride(0), _mask_q.stride(1), _mask_q.stride(2),
      _idx_q.stride(0), _idx_q.stride(1), _idx_q.stride(2)
    );
  } else {
    battn_v_kernel_with_base_strided<4><<<blocks, threads, shmem>>>(
      alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
      K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes,
      _vq.stride(0), _vq.stride(1), _vq.stride(2), _vq.stride(3),
      _vscale.stride(0), _vscale.stride(1), _vscale.stride(2), _vscale.stride(3),
      _vzero.stride(0), _vzero.stride(1), _vzero.stride(2), _vzero.stride(3),
      _mask_q.stride(0), _mask_q.stride(1), _mask_q.stride(2),
      _idx_q.stride(0), _idx_q.stride(1), _idx_q.stride(2)
    );
  }
  return _out;
}

torch::Tensor attn_v_forward_cuda_outer_dim_with_base_strided_v2(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full
){
  return attn_v_forward_cuda_outer_dim_with_base_strided_bit(
      _alpha_q, _vq, _vscale, _vzero, 2, group_size, nh, nh_kv,
      _centroids, _mask_q, _idx_q, _alpha_f, _v_full);
}

torch::Tensor attn_v_forward_cuda_outer_dim_with_base_strided_v4(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full
){
  return attn_v_forward_cuda_outer_dim_with_base_strided_bit(
      _alpha_q, _vq, _vscale, _vzero, 4, group_size, nh, nh_kv,
      _centroids, _mask_q, _idx_q, _alpha_f, _v_full);
}

torch::Tensor attn_v_forward_cuda_outer_dim_with_base_paged_v2(
    torch::Tensor _alpha_q,
    torch::Tensor _vq_page_ptrs,
    torch::Tensor _vscale_page_ptrs,
    torch::Tensor _vzero_page_ptrs,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_page_ptrs,
    torch::Tensor _idx_page_ptrs,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full,
    const int K,
    const int page_size,
    const int idx_bytes
){
  TORCH_CHECK(_alpha_q.dim()==3 && _alpha_q.size(1)==1, "alpha_q must be [B*nh,1,K]");
  TORCH_CHECK(_alpha_q.size(2)==K, "alpha_q K mismatch");
  TORCH_CHECK(_alpha_q.is_cuda(), "alpha_q must be CUDA");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(idx_bytes == 1 || idx_bytes == 2 || idx_bytes == 4, "idx_bytes must be 1, 2, or 4");
  TORCH_CHECK(nh > 0 && nh_kv > 0 && nh % nh_kv == 0, "nh must be divisible by nh_kv");
  const int BSnh = _alpha_q.size(0);
  TORCH_CHECK(BSnh % nh == 0, "B*nh must be divisible by nh");
  const int B = BSnh / nh;

  TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv, "centroids must be [nh_kv,M,OC]");
  const int Mcent = _centroids.size(1);
  const int OC = _centroids.size(2);
  constexpr int PACK = 16;
  TORCH_CHECK(OC % PACK == 0, "V2 page reader requires OC divisible by 16");
  TORCH_CHECK(OC % group_size == 0, "OC must be divisible by group_size");
  TORCH_CHECK(Mcent <= MAX_CENTROIDS, "too many centroids");

  const int num_pages = (K + page_size - 1) / page_size;
  TORCH_CHECK(_vq_page_ptrs.dim()==1 && _vq_page_ptrs.size(0)>=num_pages, "vq_page_ptrs must cover K pages");
  TORCH_CHECK(_vscale_page_ptrs.dim()==1 && _vscale_page_ptrs.size(0)>=num_pages, "vscale_page_ptrs must cover K pages");
  TORCH_CHECK(_vzero_page_ptrs.dim()==1 && _vzero_page_ptrs.size(0)>=num_pages, "vzero_page_ptrs must cover K pages");
  TORCH_CHECK(_mask_page_ptrs.dim()==1 && _mask_page_ptrs.size(0)>=num_pages, "mask_page_ptrs must cover K pages");
  TORCH_CHECK(_idx_page_ptrs.dim()==1 && _idx_page_ptrs.size(0)>=num_pages, "idx_page_ptrs must cover K pages");
  TORCH_CHECK(_vq_page_ptrs.dtype()==torch::kInt64 && _vscale_page_ptrs.dtype()==torch::kInt64 &&
              _vzero_page_ptrs.dtype()==torch::kInt64 && _mask_page_ptrs.dtype()==torch::kInt64 &&
              _idx_page_ptrs.dtype()==torch::kInt64, "page pointer tables must be int64");
  TORCH_CHECK(_vq_page_ptrs.is_cuda() && _vscale_page_ptrs.is_cuda() && _vzero_page_ptrs.is_cuda() &&
              _mask_page_ptrs.is_cuda() && _idx_page_ptrs.is_cuda(), "page pointer tables must be CUDA tensors");

  const int Lf = (_alpha_f.numel()==0 || _v_full.numel()==0) ? 0 : _alpha_f.size(1);
  if (Lf > 0) {
    TORCH_CHECK(_v_full.dim()==4 && _v_full.size(0)==B && _v_full.size(1)==nh_kv &&
                _v_full.size(2)==Lf && _v_full.size(3)==OC, "v_full must be [B,nh_kv,Lf,OC]");
  }

  auto options = torch::TensorOptions().dtype(_alpha_q.dtype()).device(_alpha_q.device());
  at::Tensor _out = torch::empty({BSnh, 1, OC}, options);

  const half* alpha_q = reinterpret_cast<const half*>(_alpha_q.data_ptr<at::Half>());
  const int64_t* vq_pages = _vq_page_ptrs.data_ptr<int64_t>();
  const int64_t* vsc_pages = _vscale_page_ptrs.data_ptr<int64_t>();
  const int64_t* vzr_pages = _vzero_page_ptrs.data_ptr<int64_t>();
  const half* cent = reinterpret_cast<const half*>(_centroids.data_ptr<at::Half>());
  const int64_t* mask_pages = _mask_page_ptrs.data_ptr<int64_t>();
  const int64_t* idx_pages = _idx_page_ptrs.data_ptr<int64_t>();
  const half* alpha_f = (_alpha_f.numel()==0) ? nullptr : reinterpret_cast<const half*>(_alpha_f.data_ptr<at::Half>());
  const half* v_full = (_v_full.numel()==0) ? nullptr : reinterpret_cast<const half*>(_v_full.data_ptr<at::Half>());
  half* outp = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  dim3 threads(32, 4, 1);
  dim3 blocks(BSnh, (OC / PACK + threads.y - 1) / threads.y, 1);
  size_t shmem = (size_t)Mcent * (size_t)threads.y * sizeof(float);
  battn_v_kernel_with_base_paged_v2<<<blocks, threads, shmem>>>(
    alpha_q, vq_pages, vsc_pages, vzr_pages, cent, mask_pages, idx_pages,
    alpha_f, v_full, outp, K, OC, Lf, group_size, nh, nh_kv, Mcent,
    page_size, idx_bytes
  );
  return _out;
}

template<int BIT>
__device__ __forceinline__ float page_pool_block_reduce_sum_f32(float v) {
  __shared__ float shared[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_reduce_sum_f32(v);
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
  if (warp == 0) {
    v = warp_reduce_sum_f32(v);
  }
  return v;
}

template<int BIT>
__global__ void page_mixed_pool_value_kernel(
  const half* __restrict__ alpha,          // [B, nh, 1, T]
  const uint32_t* __restrict__ v2_payload, // [nh_kv, N2, OC/16]
  const uint32_t* __restrict__ v4_payload, // [nh_kv, N4, OC/8]
  const half* __restrict__ v2_scale,       // [nh_kv, N2, OC/group]
  const half* __restrict__ v2_zero,
  const half* __restrict__ v4_scale,
  const half* __restrict__ v4_zero,
  const uint8_t* __restrict__ v2_pattern,  // [nh_kv, N2]
  const uint8_t* __restrict__ v4_pattern,
  const int32_t* __restrict__ v2_assignment,
  const int32_t* __restrict__ v4_assignment,
  const half* __restrict__ centroids,      // [nh_kv, M, OC]
  const int32_t* __restrict__ v2_page_offsets,
  const int32_t* __restrict__ v4_page_offsets,
  const int32_t* __restrict__ v2_page_table,
  const int32_t* __restrict__ v4_page_table,
  const int32_t* __restrict__ metadata_page_table,
  const int16_t* __restrict__ v4_prefix_counts,
  const int32_t* __restrict__ seq_lens,
  half* __restrict__ out,                  // [B, nh, 1, OC]
  const int B,
  const int T,
  const int OC,
  const int Mcent,
  const int Bcent,
  const int N2,
  const int N4,
  const int pages_per_request,
  const int group_size,
  const int nh,
  const int nh_kv,
  const int page_size)
{
  const int bnh = blockIdx.x;
  const int oc = blockIdx.y;
  if (oc >= OC) return;
  const int b = bnh / nh;
  const int hq = bnh - b * nh;
  const int ratio = nh / nh_kv;
  const int hk = hq / ratio;
  const int tid = threadIdx.x;
  const int oc_group = oc / group_size;
  const int T_valid = min(max((int)__ldg(seq_lens + b), 0), T);

  float sum = 0.0f;
  for (int t = tid; t < T_valid; t += blockDim.x) {
    const int logical_page = t / page_size;
    const int page_off = t - logical_page * page_size;
    const int table_off = b * pages_per_request + logical_page;
    const int metadata_page = __ldg(metadata_page_table + table_off);
    const int16_t* prefix = v4_prefix_counts + (size_t)metadata_page * (page_size + 1);
    const int v4_before = (int)__ldg(prefix + page_off);
    const int v4_after = (int)__ldg(prefix + page_off + 1);
    const bool is_v4 = v4_after > v4_before;
    const float a = __half2float(__ldg(alpha + ((size_t)bnh * T + t)));
    if (is_v4) {
      const int page_id = __ldg(v4_page_table + table_off);
      if (page_id >= 0) {
        const int phys = __ldg(v4_page_offsets + page_id) + v4_before;
        const size_t payload_off = ((size_t)hk * N4 + phys) * (OC / 8) + (oc / 8);
        uint32_t word = __ldg(v4_payload + payload_off);
        const float code = (float)((word >> ((oc & 7) * 4)) & 0xFu);
        const size_t affine_off = ((size_t)hk * N4 + phys) * (OC / group_size) + oc_group;
        float value = __half2float(__ldg(v4_scale + affine_off)) * code + __half2float(__ldg(v4_zero + affine_off));
        if (__ldg(v4_pattern + (size_t)hk * N4 + phys)) {
          const int idx = __ldg(v4_assignment + (size_t)hk * N4 + phys);
          if (0 <= idx && idx < Mcent) {
            const size_t cent_base = (Bcent == 1) ? ((size_t)hk * Mcent + idx) : (((size_t)b * nh_kv + hk) * Mcent + idx);
            value += __half2float(__ldg(centroids + cent_base * OC + oc));
          }
        }
        sum += a * value;
      }
    } else {
      const int page_id = __ldg(v2_page_table + table_off);
      if (page_id >= 0) {
        const int phys = __ldg(v2_page_offsets + page_id) + (page_off - v4_before);
        const size_t payload_off = ((size_t)hk * N2 + phys) * (OC / 16) + (oc / 16);
        uint32_t word = __ldg(v2_payload + payload_off);
        const float code = (float)((word >> ((oc & 15) * 2)) & 0x3u);
        const size_t affine_off = ((size_t)hk * N2 + phys) * (OC / group_size) + oc_group;
        float value = __half2float(__ldg(v2_scale + affine_off)) * code + __half2float(__ldg(v2_zero + affine_off));
        if (__ldg(v2_pattern + (size_t)hk * N2 + phys)) {
          const int idx = __ldg(v2_assignment + (size_t)hk * N2 + phys);
          if (0 <= idx && idx < Mcent) {
            const size_t cent_base = (Bcent == 1) ? ((size_t)hk * Mcent + idx) : (((size_t)b * nh_kv + hk) * Mcent + idx);
            value += __half2float(__ldg(centroids + cent_base * OC + oc));
          }
        }
        sum += a * value;
      }
    }
  }

  sum = page_pool_block_reduce_sum_f32<BIT>(sum);
  if (tid == 0) {
    out[((size_t)bnh * OC) + oc] = __float2half(sum);
  }
}

torch::Tensor attn_v_forward_cuda_page_mixed_pool(
    torch::Tensor _alpha_q,
    torch::Tensor _v2_payload,
    torch::Tensor _v4_payload,
    torch::Tensor _v2_scale,
    torch::Tensor _v2_zero,
    torch::Tensor _v4_scale,
    torch::Tensor _v4_zero,
    torch::Tensor _v2_pattern,
    torch::Tensor _v4_pattern,
    torch::Tensor _v2_assignment,
    torch::Tensor _v4_assignment,
    torch::Tensor _centroids,
    torch::Tensor _v2_page_offsets,
    torch::Tensor _v4_page_offsets,
    torch::Tensor _v2_page_table,
    torch::Tensor _v4_page_table,
    torch::Tensor _metadata_page_table,
    torch::Tensor _v4_prefix_counts,
    torch::Tensor _seq_lens,
    const int group_size,
    const int nh,
    const int nh_kv,
    const int page_size)
{
  TORCH_CHECK(_alpha_q.dim()==4 && _alpha_q.size(2)==1, "alpha_q must be [B,nh,1,T]");
  const int B = _alpha_q.size(0);
  const int T = _alpha_q.size(3);
  TORCH_CHECK(_alpha_q.size(1)==nh, "alpha_q nh mismatch");
  TORCH_CHECK((_centroids.dim()==3 && _centroids.size(0)==nh_kv) || (_centroids.dim()==4 && _centroids.size(0)==B && _centroids.size(1)==nh_kv), "centroids must be [nh_kv,M,OC] or [B,nh_kv,M,OC]");
  const int Bcent = (_centroids.dim()==4) ? B : 1;
  const int Mcent = (_centroids.dim()==4) ? _centroids.size(2) : _centroids.size(1);
  const int OC = (_centroids.dim()==4) ? _centroids.size(3) : _centroids.size(2);
  TORCH_CHECK(OC % group_size == 0, "OC must be divisible by group_size");
  TORCH_CHECK(_v2_payload.dim()==3 && _v2_payload.size(0)==nh_kv, "v2 payload must be [nh_kv,N2,OC/16]");
  TORCH_CHECK(_v4_payload.dim()==3 && _v4_payload.size(0)==nh_kv, "v4 payload must be [nh_kv,N4,OC/8]");
  const int N2 = _v2_payload.size(1);
  const int N4 = _v4_payload.size(1);
  TORCH_CHECK(_v2_payload.size(2) * 16 == OC, "v2 payload pack mismatch");
  TORCH_CHECK(_v4_payload.size(2) * 8 == OC, "v4 payload pack mismatch");
  TORCH_CHECK(_v2_scale.sizes()==torch::IntArrayRef({_v2_payload.size(0), _v2_payload.size(1), OC / group_size}), "v2 scale shape mismatch");
  TORCH_CHECK(_v4_scale.sizes()==torch::IntArrayRef({_v4_payload.size(0), _v4_payload.size(1), OC / group_size}), "v4 scale shape mismatch");
  TORCH_CHECK(_v2_page_table.dim()==2 && _v2_page_table.size(0)==B, "v2 page table must be [B,pages]");
  const int pages_per_request = _v2_page_table.size(1);
  TORCH_CHECK(_v4_page_table.sizes()==_v2_page_table.sizes(), "v4 page table mismatch");
  TORCH_CHECK(_metadata_page_table.sizes()==_v2_page_table.sizes(), "metadata page table mismatch");
  TORCH_CHECK(_v4_prefix_counts.dim()==2 && _v4_prefix_counts.size(1)==page_size+1, "v4 prefix must be [pages,page_size+1]");
  TORCH_CHECK(_seq_lens.dim()==1 && _seq_lens.size(0)==B, "seq_lens must be [B]");

  auto alpha = _alpha_q.to(torch::kFloat16).contiguous();
  auto centroids = _centroids.to(torch::kFloat16).contiguous();
  auto v2_payload = _v2_payload.contiguous();
  auto v4_payload = _v4_payload.contiguous();
  auto v2_scale = _v2_scale.to(torch::kFloat16).contiguous();
  auto v2_zero = _v2_zero.to(torch::kFloat16).contiguous();
  auto v4_scale = _v4_scale.to(torch::kFloat16).contiguous();
  auto v4_zero = _v4_zero.to(torch::kFloat16).contiguous();
  auto v2_pattern = _v2_pattern.to(torch::kUInt8).contiguous();
  auto v4_pattern = _v4_pattern.to(torch::kUInt8).contiguous();
  auto v2_assignment = _v2_assignment.to(torch::kInt32).contiguous();
  auto v4_assignment = _v4_assignment.to(torch::kInt32).contiguous();
  auto v2_page_offsets = _v2_page_offsets.to(torch::kInt32).contiguous();
  auto v4_page_offsets = _v4_page_offsets.to(torch::kInt32).contiguous();
  auto v2_page_table = _v2_page_table.to(torch::kInt32).contiguous();
  auto v4_page_table = _v4_page_table.to(torch::kInt32).contiguous();
  auto metadata_page_table = _metadata_page_table.to(torch::kInt32).contiguous();
  auto v4_prefix_counts = _v4_prefix_counts.to(torch::kInt16).contiguous();
  auto seq_lens = _seq_lens.to(torch::kInt32).contiguous();

  auto out = torch::empty({B, nh, 1, OC}, alpha.options());
  dim3 blocks(B * nh, OC, 1);
  dim3 threads(256, 1, 1);
  page_mixed_pool_value_kernel<2><<<blocks, threads>>>(
    reinterpret_cast<const half*>(alpha.data_ptr<at::Half>()),
    reinterpret_cast<const uint32_t*>(v2_payload.data_ptr<int>()),
    reinterpret_cast<const uint32_t*>(v4_payload.data_ptr<int>()),
    reinterpret_cast<const half*>(v2_scale.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(v2_zero.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(v4_scale.data_ptr<at::Half>()),
    reinterpret_cast<const half*>(v4_zero.data_ptr<at::Half>()),
    reinterpret_cast<const uint8_t*>(v2_pattern.data_ptr<uint8_t>()),
    reinterpret_cast<const uint8_t*>(v4_pattern.data_ptr<uint8_t>()),
    reinterpret_cast<const int32_t*>(v2_assignment.data_ptr<int>()),
    reinterpret_cast<const int32_t*>(v4_assignment.data_ptr<int>()),
    reinterpret_cast<const half*>(centroids.data_ptr<at::Half>()),
    reinterpret_cast<const int32_t*>(v2_page_offsets.data_ptr<int>()),
    reinterpret_cast<const int32_t*>(v4_page_offsets.data_ptr<int>()),
    reinterpret_cast<const int32_t*>(v2_page_table.data_ptr<int>()),
    reinterpret_cast<const int32_t*>(v4_page_table.data_ptr<int>()),
    reinterpret_cast<const int32_t*>(metadata_page_table.data_ptr<int>()),
    reinterpret_cast<const int16_t*>(v4_prefix_counts.data_ptr<int16_t>()),
    reinterpret_cast<const int32_t*>(seq_lens.data_ptr<int>()),
    reinterpret_cast<half*>(out.data_ptr<at::Half>()),
    B, T, OC, Mcent, Bcent, N2, N4, pages_per_request, group_size, nh, nh_kv, page_size
  );
  return out;
}

__device__ __forceinline__ float block_reduce_sum_128(float value) {
  __shared__ float shared[128];
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ float block_reduce_max_128(float value) {
  __shared__ float shared[128];
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = 64; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ int logical_to_physical_idx(
    const int logical,
    const int sink_valid,
    const int packed_valid,
    const int pending_valid,
    const int recent_valid,
    const int sink_physical,
    const int packed_physical,
    const int pending_physical) {
  if (logical < sink_valid) {
    return logical;
  }
  int rem = logical - sink_valid;
  if (rem < packed_valid) {
    return sink_physical + rem;
  }
  rem -= packed_valid;
  if (rem < pending_valid) {
    return sink_physical + packed_physical + rem;
  }
  rem -= pending_valid;
  if (rem < recent_valid) {
    return sink_physical + packed_physical + pending_physical + rem;
  }
  return -1;
}

__global__ void request_invariant_fixed_split_softmax_kernel(
    const half* scores,
    half* out,
    const int32_t* total_lens,
    const int32_t* sink_lens,
    const int32_t* packed_lens,
    const int32_t* pending_lens,
    const int32_t* recent_lens,
    const int B,
    const int H,
    const int T,
    const int sink_physical,
    const int packed_physical,
    const int pending_physical,
    const int recent_physical,
    const int split_size) {
  const int bh = blockIdx.x;
  const int b = bh / H;
  const int h = bh - b * H;
  const int tid = threadIdx.x;
  const int total = total_lens[b];
  const int sink_valid = sink_lens[b];
  const int packed_valid = packed_lens[b];
  const int pending_valid = pending_lens[b];
  const int recent_valid = recent_lens[b];
  const int splits = (total + split_size - 1) / split_size;
  const size_t base = ((size_t)b * H + h) * (size_t)T;

  __shared__ float merged_max_shared;
  __shared__ float merged_sum_shared;

  if (tid == 0) {
    merged_max_shared = -INFINITY;
    merged_sum_shared = 0.0f;
  }
  __syncthreads();

  for (int split = 0; split < splits; ++split) {
    const int start = split * split_size;
    const int end = min(start + split_size, total);
    float local_max = -INFINITY;
    for (int logical = start + tid; logical < end; logical += blockDim.x) {
      const int physical = logical_to_physical_idx(
          logical, sink_valid, packed_valid, pending_valid, recent_valid,
          sink_physical, packed_physical, pending_physical);
      if (physical >= 0 && physical < T) {
        local_max = fmaxf(local_max, __half2float(scores[base + physical]));
      }
    }
    const float split_max = block_reduce_max_128(local_max);
    float local_sum = 0.0f;
    for (int logical = start + tid; logical < end; logical += blockDim.x) {
      const int physical = logical_to_physical_idx(
          logical, sink_valid, packed_valid, pending_valid, recent_valid,
          sink_physical, packed_physical, pending_physical);
      if (physical >= 0 && physical < T) {
        local_sum += expf(__half2float(scores[base + physical]) - split_max);
      }
    }
    const float split_sum = block_reduce_sum_128(local_sum);
    if (tid == 0) {
      if (split == 0) {
        merged_max_shared = split_max;
        merged_sum_shared = split_sum;
      } else if (split_sum > 0.0f) {
        const float old_max = merged_max_shared;
        const float old_sum = merged_sum_shared;
        const float merged_max = fmaxf(old_max, split_max);
        merged_sum_shared = old_sum * expf(old_max - merged_max) + split_sum * expf(split_max - merged_max);
        merged_max_shared = merged_max;
      }
    }
    __syncthreads();
  }

  const float merged_max = merged_max_shared;
  const float merged_sum = merged_sum_shared;
  for (int physical = tid; physical < T; physical += blockDim.x) {
    out[base + physical] = __float2half(0.0f);
  }
  __syncthreads();
  for (int logical = tid; logical < total; logical += blockDim.x) {
    const int physical = logical_to_physical_idx(
        logical, sink_valid, packed_valid, pending_valid, recent_valid,
        sink_physical, packed_physical, pending_physical);
    if (physical >= 0 && physical < T && merged_sum > 0.0f) {
      const float prob = expf(__half2float(scores[base + physical]) - merged_max) / merged_sum;
      out[base + physical] = __float2half(prob);
    }
  }
}

torch::Tensor request_invariant_fixed_split_softmax_cuda(
    torch::Tensor _scores,
    torch::Tensor _total_lens,
    torch::Tensor _sink_lens,
    torch::Tensor _packed_lens,
    torch::Tensor _pending_lens,
    torch::Tensor _recent_lens,
    const int sink_physical,
    const int packed_physical,
    const int pending_physical,
    const int recent_physical,
    const int split_size) {
  TORCH_CHECK(_scores.dim()==4 && _scores.size(2)==1, "scores must be [B,H,1,T]");
  TORCH_CHECK(_scores.scalar_type()==torch::kFloat16, "fixed split softmax currently expects float16 scores");
  TORCH_CHECK(split_size > 0, "split_size must be positive");
  const int B = _scores.size(0);
  const int H = _scores.size(1);
  const int T = _scores.size(3);
  auto scores = _scores.contiguous();
  auto total_lens = _total_lens.to(torch::kInt32).contiguous();
  auto sink_lens = _sink_lens.to(torch::kInt32).contiguous();
  auto packed_lens = _packed_lens.to(torch::kInt32).contiguous();
  auto pending_lens = _pending_lens.to(torch::kInt32).contiguous();
  auto recent_lens = _recent_lens.to(torch::kInt32).contiguous();
  auto out = torch::empty_like(scores);
  dim3 blocks(B * H, 1, 1);
  dim3 threads(128, 1, 1);
  request_invariant_fixed_split_softmax_kernel<<<blocks, threads>>>(
      reinterpret_cast<const half*>(scores.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      reinterpret_cast<const int32_t*>(total_lens.data_ptr<int>()),
      reinterpret_cast<const int32_t*>(sink_lens.data_ptr<int>()),
      reinterpret_cast<const int32_t*>(packed_lens.data_ptr<int>()),
      reinterpret_cast<const int32_t*>(pending_lens.data_ptr<int>()),
      reinterpret_cast<const int32_t*>(recent_lens.data_ptr<int>()),
      B, H, T,
      sink_physical, packed_physical, pending_physical, recent_physical,
      split_size);
  return out;
}

__device__ __forceinline__ int clamp_segment_len(const int requested, const int physical) {
  return max(0, min(requested, physical));
}

__global__ void fp16_tail_value_kernel(
    const half* probs,
    const half* sink_v,
    const half* pending_v,
    const half* recent_v,
    const int64_t* sink_lens,
    const int64_t* pending_lens,
    const int64_t* recent_lens,
    half* out,
    const int B,
    const int H,
    const int Hkv,
    const int T,
    const int D,
    const int sink_physical,
    const int pending_physical,
    const int recent_physical,
    const int sink_offset,
    const int pending_offset,
    const int recent_offset,
    const int num_key_value_groups) {
  const int bh = blockIdx.x;
  const int b = bh / H;
  const int h = bh - b * H;
  const int d = threadIdx.x;
  if (b >= B || h >= H || d >= D) {
    return;
  }
  const int kv = h / num_key_value_groups;
  if (kv >= Hkv) {
    return;
  }

  float acc = 0.0f;
  const size_t prob_base = ((size_t)b * H + h) * (size_t)T;

  const int sink_valid = clamp_segment_len((int)sink_lens[b], sink_physical);
  for (int token = 0; token < sink_valid; ++token) {
    const float p = __half2float(probs[prob_base + (size_t)sink_offset + token]);
    const size_t v_idx = (((size_t)b * Hkv + kv) * (size_t)sink_physical + token) * (size_t)D + d;
    acc += p * __half2float(sink_v[v_idx]);
  }

  const int pending_valid = clamp_segment_len((int)pending_lens[b], pending_physical);
  for (int token = 0; token < pending_valid; ++token) {
    const float p = __half2float(probs[prob_base + (size_t)pending_offset + token]);
    const size_t v_idx = (((size_t)b * Hkv + kv) * (size_t)pending_physical + token) * (size_t)D + d;
    acc += p * __half2float(pending_v[v_idx]);
  }

  const int recent_valid = clamp_segment_len((int)recent_lens[b], recent_physical);
  for (int token = 0; token < recent_valid; ++token) {
    const float p = __half2float(probs[prob_base + (size_t)recent_offset + token]);
    const size_t v_idx = (((size_t)b * Hkv + kv) * (size_t)recent_physical + token) * (size_t)D + d;
    acc += p * __half2float(recent_v[v_idx]);
  }

  const size_t out_idx = (((size_t)b * H + h) * (size_t)D) + d;
  out[out_idx] = __float2half(acc);
}

torch::Tensor fp16_tail_value_forward_cuda(
    torch::Tensor _probs,
    torch::Tensor _sink_v,
    torch::Tensor _pending_v,
    torch::Tensor _recent_v,
    torch::Tensor _sink_lens,
    torch::Tensor _pending_lens,
    torch::Tensor _recent_lens,
    const int sink_offset,
    const int pending_offset,
    const int recent_offset,
    const int num_key_value_groups) {
  TORCH_CHECK(_probs.dim()==4 && _probs.size(2)==1, "probs must be [B,H,1,T]");
  TORCH_CHECK(_probs.scalar_type()==torch::kFloat16, "probs must be float16");
  TORCH_CHECK(_probs.is_cuda(), "probs must be CUDA");
  TORCH_CHECK(_probs.is_contiguous(), "probs must be contiguous");
  TORCH_CHECK(num_key_value_groups > 0, "num_key_value_groups must be positive");

  const int B = _probs.size(0);
  const int H = _probs.size(1);
  const int T = _probs.size(3);
  TORCH_CHECK(H % num_key_value_groups == 0, "H must be divisible by num_key_value_groups");
  const int Hkv = H / num_key_value_groups;

  auto validate_v = [&](const torch::Tensor& value, const char* name) {
    TORCH_CHECK(value.dim()==4, name, " must be [B,Hkv,L,D]");
    TORCH_CHECK(value.scalar_type()==torch::kFloat16, name, " must be float16");
    TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(value.size(0)==B && value.size(1)==Hkv, name, " batch/head shape mismatch");
  };
  validate_v(_sink_v, "sink_v");
  validate_v(_pending_v, "pending_v");
  validate_v(_recent_v, "recent_v");

  const int D = _sink_v.size(3);
  TORCH_CHECK(_pending_v.size(3)==D && _recent_v.size(3)==D, "tail Value head_dim mismatch");
  const int sink_physical = _sink_v.size(2);
  const int pending_physical = _pending_v.size(2);
  const int recent_physical = _recent_v.size(2);
  TORCH_CHECK(sink_offset >= 0 && pending_offset >= 0 && recent_offset >= 0, "segment offsets must be non-negative");
  TORCH_CHECK(sink_offset + sink_physical <= T, "sink segment exceeds probability length");
  TORCH_CHECK(pending_offset + pending_physical <= T, "pending segment exceeds probability length");
  TORCH_CHECK(recent_offset + recent_physical <= T, "recent segment exceeds probability length");

  TORCH_CHECK(_sink_lens.scalar_type()==torch::kInt64, "sink_lens must be int64");
  TORCH_CHECK(_pending_lens.scalar_type()==torch::kInt64, "pending_lens must be int64");
  TORCH_CHECK(_recent_lens.scalar_type()==torch::kInt64, "recent_lens must be int64");
  TORCH_CHECK(_sink_lens.is_cuda() && _pending_lens.is_cuda() && _recent_lens.is_cuda(), "length tensors must be CUDA");
  TORCH_CHECK(_sink_lens.is_contiguous() && _pending_lens.is_contiguous() && _recent_lens.is_contiguous(), "length tensors must be contiguous");
  TORCH_CHECK(_sink_lens.dim()==1 && _sink_lens.size(0)==B, "sink_lens must be [B]");
  TORCH_CHECK(_pending_lens.dim()==1 && _pending_lens.size(0)==B, "pending_lens must be [B]");
  TORCH_CHECK(_recent_lens.dim()==1 && _recent_lens.size(0)==B, "recent_lens must be [B]");

  auto out = torch::empty({B, H, 1, D}, _probs.options());
  dim3 blocks(B * H, 1, 1);
  dim3 threads(std::max(32, D), 1, 1);
  fp16_tail_value_kernel<<<blocks, threads>>>(
      reinterpret_cast<const half*>(_probs.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(_sink_v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(_pending_v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(_recent_v.data_ptr<at::Half>()),
      reinterpret_cast<const int64_t*>(_sink_lens.data_ptr<int64_t>()),
      reinterpret_cast<const int64_t*>(_pending_lens.data_ptr<int64_t>()),
      reinterpret_cast<const int64_t*>(_recent_lens.data_ptr<int64_t>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      B, H, Hkv, T, D,
      sink_physical, pending_physical, recent_physical,
      sink_offset, pending_offset, recent_offset,
      num_key_value_groups);
  return out;
}

torch::Tensor attn_v_forward_cuda_outer_dim_with_base_gqa_v2(
    torch::Tensor _alpha_q,
    torch::Tensor _vq,
    torch::Tensor _vscale,
    torch::Tensor _vzero,
    const int group_size,
    const int nh,
    const int nh_kv,
    torch::Tensor _centroids,
    torch::Tensor _mask_q,
    torch::Tensor _idx_q,
    torch::Tensor _alpha_f,
    torch::Tensor _v_full
){
  TORCH_CHECK(_alpha_q.dim()==3 && _alpha_q.size(1)==1, "alpha_q must be [B*nh,1,K]");
  const int BSnh = _alpha_q.size(0);
  const int K = _alpha_q.size(2);
  TORCH_CHECK(nh > 0 && nh_kv > 0 && nh % nh_kv == 0, "nh must be divisible by nh_kv");
  TORCH_CHECK(nh / nh_kv == 4, "experimental GQA V2 kernel supports ratio=4 only");
  TORCH_CHECK(BSnh % nh == 0, "B*nh must be divisible by nh");
  const int B = BSnh / nh;

  TORCH_CHECK(_centroids.dim()==3 && _centroids.size(0)==nh_kv, "centroids must be [nh_kv,M,OC]");
  const int Mcent = _centroids.size(1);
  const int OC = _centroids.size(2);
  TORCH_CHECK(OC == 128, "experimental GQA V2 kernel supports OC=128 only");
  TORCH_CHECK(group_size == 128, "experimental GQA V2 kernel supports group_size=128 only");
  TORCH_CHECK(Mcent <= MAX_CENTROIDS, "too many centroids");

  constexpr int PACK = 16;
  TORCH_CHECK(_vq.dim()==3 && _vq.size(1) * PACK == OC && _vq.size(2)==K, "vq must be [B*nh_kv,OC/16,K]");
  TORCH_CHECK(_vq.size(0) == B * nh_kv, "vq batch/head mismatch");
  TORCH_CHECK(_vscale.dim()==3 && _vzero.dim()==3, "scale/zero must be [B*nh_kv,OC/group,K]");
  TORCH_CHECK(_vscale.size(0)==B * nh_kv && _vzero.size(0)==B * nh_kv, "scale/zero batch/head mismatch");
  TORCH_CHECK(_vscale.size(2)==K && _vzero.size(2)==K, "scale/zero K mismatch");
  TORCH_CHECK(_mask_q.dim()==3 && _mask_q.size(0)==B && _mask_q.size(1)==nh_kv && _mask_q.size(2)==K, "mask_q must be [B,nh_kv,K]");
  TORCH_CHECK(_idx_q.dim()==3 && _idx_q.size(0)==B && _idx_q.size(1)==nh_kv && _idx_q.size(2)==K, "idx_q must be [B,nh_kv,K]");

  const int Lf = (_alpha_f.numel()==0 || _v_full.numel()==0) ? 0 : _alpha_f.size(1);
  if (Lf > 0) {
    TORCH_CHECK(_v_full.dim()==4 && _v_full.size(0)==B && _v_full.size(1)==nh_kv && _v_full.size(2)==Lf && _v_full.size(3)==OC, "v_full must be [B,nh_kv,Lf,OC]");
  }

  auto options = torch::TensorOptions().dtype(_alpha_q.dtype()).device(_alpha_q.device());
  at::Tensor _out = torch::empty({BSnh, 1, OC}, options);

  const half* alpha_q = reinterpret_cast<const half*>(_alpha_q.data_ptr<at::Half>());
  const uint32_t* vq = reinterpret_cast<const uint32_t*>(_vq.data_ptr<int>());
  const half* vsc = reinterpret_cast<const half*>(_vscale.data_ptr<at::Half>());
  const half* vzr = reinterpret_cast<const half*>(_vzero.data_ptr<at::Half>());
  const half* cent = reinterpret_cast<const half*>(_centroids.data_ptr<at::Half>());
  const uint8_t* mask = reinterpret_cast<const uint8_t*>(_mask_q.data_ptr<uint8_t>());
  const void* idx = static_cast<const void*>(_idx_q.data_ptr());
  const half* alpha_f = (_alpha_f.numel()==0) ? nullptr : reinterpret_cast<const half*>(_alpha_f.data_ptr<at::Half>());
  const half* v_full = (_v_full.numel()==0) ? nullptr : reinterpret_cast<const half*>(_v_full.data_ptr<at::Half>());
  half* outp = reinterpret_cast<half*>(_out.data_ptr<at::Half>());

  const int idx_bytes =
      (_idx_q.dtype() == torch::kUInt8) ? 1 :
      (_idx_q.dtype() == torch::kInt16) ? 2 : 4;

  constexpr int TILE = 128;
  dim3 threads(32, 4, 4);
  dim3 blocks(B * nh_kv, (OC / PACK + threads.y - 1) / threads.y, 1);
  const size_t hist_bytes = (size_t)4 * threads.y * Mcent * sizeof(float);
  const size_t vq_bytes = (size_t)threads.y * TILE * sizeof(uint32_t);
  const size_t scale_bytes = (size_t)threads.y * TILE * sizeof(half);
  const size_t zero_bytes = (size_t)threads.y * TILE * sizeof(half);
  const size_t mask_bytes = (size_t)TILE * sizeof(int);
  const size_t idx_bytes_s = (size_t)TILE * sizeof(int);
  const size_t centroid_bytes = (size_t)Mcent * threads.y * PACK * sizeof(half);
  const size_t shmem = hist_bytes + vq_bytes + scale_bytes + zero_bytes + mask_bytes + idx_bytes_s + centroid_bytes;

  battn_v_kernel_gqa4_v2_with_base<TILE><<<blocks, threads, shmem>>>(
    alpha_q, vq, vsc, vzr, cent, mask, idx, alpha_f, v_full, outp,
    K, OC, Lf, group_size, nh, nh_kv, Mcent, idx_bytes
  );
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
