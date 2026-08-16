# Kernel Inventory

Top CUDA kernels by total GPU time:

- `_bi_linear_persistent_kernel`: calls=762 total_gpu_ms=301.498 mean_us=395.666 p50_us=319.059 p95_us=549.544 category=UNKNOWN
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, f`: calls=16512 total_gpu_ms=40.674 mean_us=2.463 p50_us=2.432 p95_us=2.464 category=ELEMENTWISE
- `void page_mixed_pool_value_kernel<2>(__half const*, unsigned int const*, unsigned int const*, __half const*, __half const*, __half const*, _`: calls=256 total_gpu_ms=38.640 mean_us=150.939 p50_us=152.025 p95_us=156.185 category=VALUE_HISTORY
- `ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`: calls=512 total_gpu_ms=24.625 mean_us=48.095 p50_us=48.414 p95_us=51.166 category=PROJECTION
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >(int, at::native::CUDAFun`: calls=16128 total_gpu_ms=23.052 mean_us=1.429 p50_us=1.440 p95_us=1.472 category=ELEMENTWISE
- `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<flo`: calls=16384 total_gpu_ms=21.275 mean_us=1.299 p50_us=1.312 p95_us=1.312 category=ELEMENTWISE
- `void gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStrided`: calls=8 total_gpu_ms=9.479 mean_us=1184.830 p50_us=1185.134 p95_us=1185.806 category=UNKNOWN
- `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __half, __half, __half, float, false, true, true, false, 7, false, cu`: calls=512 total_gpu_ms=6.909 mean_us=13.495 p50_us=13.440 p95_us=13.920 category=PROJECTION
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&):`: calls=1792 total_gpu_ms=6.155 mean_us=3.434 p50_us=3.296 p95_us=4.512 category=VALUE_FP16_TAIL
- `request_invariant_fixed_split_softmax_kernel(__half const*, __half*, int const*, int const*, int const*, int const*, int const*, int, int, i`: calls=256 total_gpu_ms=6.041 mean_us=23.596 p50_us=23.584 p95_us=23.679 category=SOFTMAX
- `void bgemv_kernel_outer_dim_with_base_tiled<2>(__half const*, unsigned int const*, __half const*, __half const*, __half const*, void const*,`: calls=256 total_gpu_ms=5.382 mean_us=21.022 p50_us=21.023 p95_us=21.119 category=QK_HISTORY
- `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::Half, at::native::func_wrapper_t<c10::Half, at::native::sum_functor<c10::Ha`: calls=768 total_gpu_ms=4.387 mean_us=5.712 p50_us=5.472 p95_us=6.272 category=VALUE_FP16_TAIL
- `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<long>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<`: calls=3840 total_gpu_ms=4.355 mean_us=1.134 p50_us=1.120 p95_us=1.152 category=VALUE_FP16_TAIL
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::Half, c10::Half, c10::Half, at`: calls=1536 total_gpu_ms=3.849 mean_us=2.506 p50_us=2.464 p95_us=2.752 category=COPY_CAST_CONTIGUOUS
- `void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<c10::Half, c10::Half, c10::Half, at`: calls=768 total_gpu_ms=3.204 mean_us=4.171 p50_us=4.160 p95_us=4.223 category=VALUE_FP16_TAIL
- `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<long>, at::detail::Array<char*, 3> >(int, at::native::CUDAFunc`: calls=2304 total_gpu_ms=3.188 mean_us=1.384 p50_us=1.376 p95_us=1.440 category=VALUE_FP16_TAIL
- `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<c10::Half, at::native::func_wrapper_t<c10::Half, at::native::sum_functor<c10::Ha`: calls=768 total_gpu_ms=3.150 mean_us=4.101 p50_us=4.256 p95_us=4.416 category=QK_FP16_TAIL
- `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 128, `: calls=1024 total_gpu_ms=3.053 mean_us=2.982 p50_us=2.944 p95_us=3.168 category=CACHE_APPEND
- `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const:`: calls=1040 total_gpu_ms=3.046 mean_us=2.929 p50_us=3.168 p95_us=3.264 category=COPY_CAST_CONTIGUOUS
- `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<long, long, long, at::native::maximum_kernel_cuda(at::TensorIter`: calls=2304 total_gpu_ms=3.031 mean_us=1.316 p50_us=1.312 p95_us=1.344 category=VALUE_FP16_TAIL
