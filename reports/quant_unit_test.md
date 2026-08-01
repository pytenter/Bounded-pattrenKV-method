# Quant Extension Unit Test

Status: PASS
GPU: NVIDIA GeForce RTX 3090 capability [8, 6]
Torch: 2.4.1+cu124 CUDA 12.4

| bits | seq_len | k_code | v_code | qk max_abs | av max_abs |
| --- | --- | --- | --- | --- | --- |
| 2 | 128 | [1, 8, 128, 8] torch.int32 | [1, 8, 128, 8] torch.int32 | 0.03125 | 0.015625 |
| 4 | 128 | [1, 8, 128, 16] torch.int32 | [1, 8, 128, 16] torch.int32 | 0.03125 | 0.0234375 |
| 2 | 256 | [1, 8, 128, 16] torch.int32 | [1, 8, 256, 8] torch.int32 | 0.015625 | 0.03125 |
| 4 | 256 | [1, 8, 128, 32] torch.int32 | [1, 8, 256, 16] torch.int32 | 0.0273438 | 0.03125 |
