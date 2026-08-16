# Environment

- Python: `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`
- Torch: `2.4.1+cu124`
- CUDA: `12.4`
- Pytest: `9.1.1`
- Branch: `sys/causal-v4-25-kernel-v1`
- HEAD: `84d663e970102daedae189c107253563a3427384`
- CUDA_VISIBLE_DEVICES: `1`

## GPU

```text
Sun Aug 16 14:35:24 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:1A:00.0 Off |                  N/A |
| 30%   26C    P8             20W /  350W |   18409MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:1C:00.0 Off |                  N/A |
| 30%   33C    P8             25W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        Off |   00000000:1D:00.0 Off |                  N/A |
| 30%   30C    P8             18W /  350W |      20MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        Off |   00000000:1E:00.0 Off |                  N/A |
| 30%   30C    P8             22W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA GeForce RTX 3090        Off |   00000000:3E:00.0 Off |                  N/A |
| 30%   28C    P8             16W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA GeForce RTX 3090        Off |   00000000:3F:00.0 Off |                  N/A |
| 71%   75C    P2            327W /  350W |   15822MiB /  24576MiB |     88%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA GeForce RTX 3090        Off |   00000000:40:00.0 Off |                  N/A |
| 30%   29C    P8             16W /  350W |      18MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA GeForce RTX 3090        Off |   00000000:41:00.0 Off |                  N/A |
| 71%   72C    P2            324W /  350W |    4404MiB /  24576MiB |    100%      Default |
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
|    2   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    3   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    4   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    5   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    5   N/A  N/A          204073      C   .venv/bin/python                      15796MiB |
|    6   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    7   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    7   N/A  N/A          204156      C   .venv/bin/python                       4378MiB |
+-----------------------------------------------------------------------------------------+
```
