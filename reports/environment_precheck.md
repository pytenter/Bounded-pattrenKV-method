# Environment Precheck

Collected at: 2026-08-01T20:51:21+08:00

## nvidia-smi
Sat Aug  1 20:51:21 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:1A:00.0 Off |                  N/A |
| 30%   26C    P8             20W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:1C:00.0 Off |                  N/A |
| 30%   29C    P8             22W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        Off |   00000000:1D:00.0 Off |                  N/A |
| 30%   29C    P8             19W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        Off |   00000000:1E:00.0 Off |                  N/A |
| 30%   29C    P8             22W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA GeForce RTX 3090        Off |   00000000:3E:00.0 Off |                  N/A |
| 30%   30C    P8             17W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA GeForce RTX 3090        Off |   00000000:3F:00.0 Off |                  N/A |
| 30%   27C    P8             21W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA GeForce RTX 3090        Off |   00000000:40:00.0 Off |                  N/A |
| 30%   29C    P8             16W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA GeForce RTX 3090        Off |   00000000:41:00.0 Off |                  N/A |
| 30%   29C    P8             18W /  350W |      15MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    1   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    2   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    3   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    4   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    5   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    6   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    7   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+

## GPU query
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02
NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02

## nvcc
/bin/bash: line 38: nvcc: command not found
Default CUDA symlink:
lrwxrwxrwx 1 root root 21 Mar 30 18:25 /usr/local/cuda -> /usr/local/cuda-12.4/
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0

## GCC/G++
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

## Conda/Micromamba
/bin/bash: line 42: conda: command not found
/bin/bash: line 42: conda: command not found
2.8.1
  Name  Active  Path                              
────────────────────────────────────────────────────
  base  *       /data/zypan/.local/share/mamba    
                /data/zypan/kvarn-repro/envs/kvarn

## Disk
Filesystem                   Size  Used Avail Use% Mounted on
tmpfs                         26G  4.8M   26G   1% /run
/dev/sdb2                    878G   58G  776G   7% /
tmpfs                        126G     0  126G   0% /dev/shm
tmpfs                        5.0M   36K  5.0M   1% /run/lock
efivarfs                     512K   43K  465K   9% /sys/firmware/efi/efivars
/dev/sdb1                    1.1G  6.2M  1.1G   1% /boot/efi
/dev/mapper/vg_data-lv_data   11T  7.6T  2.7T  74% /data
tmpfs                         26G  100K   26G   1% /run/user/120
tmpfs                         26G  136K   26G   1% /run/user/1011
tmpfs                         26G   88K   26G   1% /run/user/1021
tmpfs                         26G   84K   26G   1% /run/user/1022
tmpfs                         26G   84K   26G   1% /run/user/1003

## Memory
               total        used        free      shared  buff/cache   available
Mem:           251Gi        15Gi        49Gi        15Mi       188Gi       236Gi
Swap:          8.0Gi       1.7Gi       6.3Gi

## Proxy
HTTPS_PROXY=http://127.0.0.1:7897
HTTP_PROXY=http://127.0.0.1:7897
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0HTTP/1.1 200 Connection established

HTTP/2 200 
date: Sat, 01 Aug 2026 12:51:13 GMT
content-type: text/html; charset=utf-8
content-language: en-US
vary: X-PJAX, X-PJAX-Container, Turbo-Visit, Turbo-Frame, X-Requested-With, X-GitHub-Client-Version, Accept-Language, Sec-Fetch-Site,Accept-Encoding, Accept, X-Requested-With
etag: W/"37051af87b3bb2f689bccdab0628f920"
cache-control: max-age=0, private, must-revalidate
strict-transport-security: max-age=31536000; includeSubdomains; preload
x-frame-options: deny
x-content-type-options: nosniff
x-xss-protection: 0
referrer-policy: origin-when-cross-origin, strict-origin-when-cross-origin
content-security-policy: default-src 'none'; base-uri 'self'; child-src github.githubassets.com github.com/assets-cdn/worker/ github.com/assets/ gist.github.com/assets-cdn/worker/; connect-src 'self' uploads.github.com www.githubstatus.com collector.github.com raw.githubusercontent.com api.github.com github-cloud.s3.amazonaws.com github-production-repository-file-5c1aeb.s3.amazonaws.com github-production-upload-manifest-file-7fdce7.s3.amazonaws.com github-production-user-asset-6210df.s3.amazonaws.com *.rel.tunnels.api.visualstudio.com wss://*.rel.tunnels.api.visualstudio.com github.githubassets.com objects-origin.githubusercontent.com copilot-proxy.githubusercontent.com proxy.individual.githubcopilot.com proxy.business.githubcopilot.com proxy.enterprise.githubcopilot.com *.actions.githubusercontent.com wss://*.actions.githubusercontent.com productionresultssa0.blob.core.windows.net productionresultssa1.blob.core.windows.net productionresultssa2.blob.core.windows.net productionresultssa3.blob.core.windows.net productionresultssa4.blob.core.windows.net productionresultssa5.blob.core.windows.net productionresultssa6.blob.core.windows.net productionresultssa7.blob.core.windows.net productionresultssa8.blob.core.windows.net productionresultssa9.blob.core.windows.net productionresultssa10.blob.core.windows.net productionresultssa11.blob.core.windows.net productionresultssa12.blob.core.windows.net productionresultssa13.blob.core.windows.net productionresultssa14.blob.core.windows.net productionresultssa15.blob.core.windows.net productionresultssa16.blob.core.windows.net productionresultssa17.blob.core.windows.net productionresultssa18.blob.core.windows.net productionresultssa19.blob.core.windows.net github-production-repository-image-32fea6.s3.amazonaws.com github-production-release-asset-2e65be.s3.amazonaws.com insights.github.com wss://alive.github.com wss://alive-staging.github.com api.githubcopilot.com api.individual.githubcopilot.com api.business.githubcopilot.com api.enterprise.githubcopilot.com wss://production-copilot-host.webpubsub.azure.com edge.fullstory.com rs.fullstory.com; font-src github.githubassets.com; form-action 'self' github.com gist.github.com copilot-workspace.githubnext.com objects-origin.githubusercontent.com; frame-ancestors 'none'; frame-src viewscreen.githubusercontent.com notebooks.githubusercontent.com www.youtube-nocookie.com; img-src 'self' data: blob: github.githubassets.com media.githubusercontent.com camo.githubusercontent.com identicons.github.com avatars.githubusercontent.com private-avatars.githubusercontent.com github-cloud.s3.amazonaws.com objects.githubusercontent.com release-assets.githubusercontent.com secured-user-images.githubusercontent.com user-images.githubusercontent.com private-user-images.githubusercontent.com opengraph.githubassets.com repository-images.githubusercontent.com marketplace-screenshots.githubusercontent.com copilotprodattachments.blob.core.windows.net/github-production-copilot-attachments/ github-production-user-asset-6210df.s3.amazonaws.com customer-stories-feed.github.com spotlights-feed.github.com explore-feed.github.com objects-origin.githubusercontent.com *.githubusercontent.com images.ctfassets.net/8aevphvgewt8/; manifest-src 'self'; media-src github.com user-images.githubusercontent.com secured-user-images.githubusercontent.com private-user-images.githubusercontent.com github-production-user-asset-6210df.s3.amazonaws.  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
com gist.github.com github.githubassets.com assets.ctfassets.net/8aevphvgewt8/ videos.ctfassets.net/8aevphvgewt8/; script-src github.githubassets.com; style-src 'unsafe-inline' github.githubassets.com; upgrade-insecure-requests; worker-src github.githubassets.com github.com/assets-cdn/worker/ github.com/assets/ gist.github.com/assets-cdn/worker/
server: github.com
accept-ranges: bytes

## Python
/bin/bash: line 50: python: command not found
Python 3.12.3

## Conclusions
- Visible GPUs are NVIDIA GeForce RTX 3090 24GB, not RTX 4090. Compute capability is expected SM86, not SM89.
- Driver 580.173.02 advertises CUDA 13.0 support.
- nvcc is not on PATH, but /usr/local/cuda-12.4/bin/nvcc and /usr/local/cuda-12.8/bin/nvcc exist.
- Shell has no conda command; a user-local micromamba binary exists at /data/zypan/kvarn-repro/tools/bin/micromamba.
- Proxy http://127.0.0.1:7897 is reachable for GitHub during this check.
