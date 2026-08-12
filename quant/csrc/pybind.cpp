#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("gemv_forward_cuda", &gemv_forward_cuda);
  m.def("gemv_forward_cuda_outer_dim", &gemv_forward_cuda_outer_dim);
  m.def("gemv_forward_cuda_outer_dim_with_base", &gemv_forward_cuda_outer_dim_with_base);
  m.def("attn_v_forward_cuda_outer_dim_with_base", &attn_v_forward_cuda_outer_dim_with_base);
  m.def("attn_v_forward_cuda_outer_dim_with_base_paged_v2", &attn_v_forward_cuda_outer_dim_with_base_paged_v2);
  m.def("attn_v_forward_cuda_outer_dim_with_base_debug", &attn_v_forward_cuda_outer_dim_with_base_debug);
  m.def("attn_v_forward_cuda_outer_dim_with_base_gqa_v2", &attn_v_forward_cuda_outer_dim_with_base_gqa_v2);
}
