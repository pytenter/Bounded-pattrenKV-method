# Environment

Start HEAD: `ea34144ee990c3d06c80b95d205af3b0eb0096b8`
Report directory: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090/reports/system_ragged_decode1_semantic_gate_v1`

Runtime environment:
- `PATTERNKV_CACHE_PATH=segmented`
- `PATTERNKV_CACHE_MODE=segmented_rolling`
- `PATTERNKV_MIXED_V_BACKEND=fused_page`
- `PATTERNKV_RUNTIME_NH=32`
- `PATTERNKV_CENTROID_MAX_SLOTS=4`

GPU:
```text
Fri Aug 14 17:56:17 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:1A:00.0 Off |                  N/A |
| 30%   27C    P8             20W /  350W |   18409MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:1C:00.0 Off |                  N/A |
| 30%   57C    P2            177W /  350W |   21050MiB /  24576MiB |     34%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        Off |   00000000:1D:00.0 Off |                  N/A |
| 30%   30C    P8             20W /  350W |      20MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        Off |   00000000:1E:00.0 Off |                  N/A |
| 30%   32C    P8             22W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA GeForce RTX 3090        Off |   00000000:3E:00.0 Off |                  N/A |
| 30%   31C    P8             23W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA GeForce RTX 3090        Off |   00000000:3F:00.0 Off |                  N/A |
| 30%   28C    P8             21W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA GeForce RTX 3090        Off |   00000000:40:00.0 Off |                  N/A |
| 30%   30C    P8             15W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA GeForce RTX 3090        Off |   00000000:41:00.0 Off |                  N/A |
| 30%   31C    P8             21W /  350W |      18MiB /  24576MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A          201464      C   python                                18380MiB |
|    1   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    1   N/A  N/A         2398877      C   ...mba/envs/patternkv/bin/python      21024MiB |
|    2   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    3   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    4   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    5   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    6   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    7   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```
