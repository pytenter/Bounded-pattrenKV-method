# KProj Kernel Audit

{
  "events": [
    {
      "cpu_time_total_us": 2982.438,
      "cuda_time_total_us": 0.0,
      "input_shapes": "[[1, 1, 4096], [1024, 4096], []]",
      "key": "aten::linear"
    },
    {
      "cpu_time_total_us": 2915.866,
      "cuda_time_total_us": 0.0,
      "input_shapes": "[[1, 1, 4096], [4096, 1024]]",
      "key": "aten::matmul"
    },
    {
      "cpu_time_total_us": 2867.398,
      "cuda_time_total_us": 0.0,
      "input_shapes": "[[1, 4096], [4096, 1024]]",
      "key": "aten::mm"
    },
    {
      "cpu_time_total_us": 85.83300000000008,
      "cuda_time_total_us": 0.0,
      "input_shapes": "[[2, 1, 4096], [1024, 4096], []]",
      "key": "aten::linear"
    },
    {
      "cpu_time_total_us": 75.63200000000006,
      "cuda_time_total_us": 0.0,
      "input_shapes": "[[2, 1, 4096], [4096, 1024]]",
      "key": "aten::matmul"
    },
    {
      "cpu_time_total_us": 63.85999999999967,
      "cuda_time_total_us": 0.0,
      "input_shapes": "[[2, 4096], [4096, 1024]]",
      "key": "aten::mm"
    },
    {
      "cpu_time_total_us": 0.0,
      "cuda_time_total_us": 0.0,
      "input_shapes": "[]",
      "key": "ampere_fp16_s16816gemm_fp16_64x64_ldg8_f2f_stages_64x6_tn"
    }
  ],
  "profile_succeeded": true
}
