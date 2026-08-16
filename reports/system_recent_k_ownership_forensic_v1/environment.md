# Environment

{
  "actual_head": "cc50fdc513181d2137438cc6a7c0dd8322ccf767",
  "branch": "sys/causal-v4-25-kernel-v1",
  "cuda": "12.4",
  "git_status_short": "M bench/run_actual_model_bi_prefill_runtime.py\n M bench/run_bi_vproj_cost_benefit.py\n M bench/run_prefill_projection_mode_policy.py\n M bench/run_ragged_multistep_correctness.py\n M models/llama_patternkv.py\n M quant/batch_invariant_kproj.py\n M reports/system_ragged_multistep_correctness_v1/b2_16step.md\n M reports/system_ragged_multistep_correctness_v1/b2_reorder.md\n M reports/system_ragged_multistep_correctness_v1/b2_reorder_steps.json\n M reports/system_ragged_multistep_correctness_v1/b2_steps.json\n M reports/system_ragged_multistep_correctness_v1/b4_16step.md\n M reports/system_ragged_multistep_correctness_v1/b4_steps.json\n M reports/system_ragged_multistep_correctness_v1/environment.md\n M reports/system_ragged_multistep_correctness_v1/final_gate.json\n M reports/system_ragged_multistep_correctness_v1/free_run.json\n M reports/system_ragged_multistep_correctness_v1/pytest.md\n M reports/system_ragged_multistep_correctness_v1/runtime_counters.json\n M reports/system_ragged_multistep_correctness_v1/semantic_metrics.json\n M tests/test_bi_kproj_prefill_runtime.py\n?? forensics/\n?? reports/centroid_determinism_causal_forensic.md\n?? reports/system_ragged_active_state_forensic_v1/\n?? reports/system_recent_k_ownership_forensic_v1/\n?? reports/system_step1_layer0_kpath_forensic_v1/\n?? scripts/centroid_determinism_causal_forensic.py\n?? scripts/ragged_active_state_forensic.py\n?? scripts/recent_k_ownership_forensic.py\n?? scripts/step1_layer0_k_path_forensic.py",
  "platform": "Linux-7.0.0-28-generic-x86_64-with-glibc2.39",
  "preexisting_dirty_files": [
    "M bench/run_actual_model_bi_prefill_runtime.py",
    " M bench/run_bi_vproj_cost_benefit.py",
    " M bench/run_prefill_projection_mode_policy.py",
    " M bench/run_ragged_multistep_correctness.py",
    " M models/llama_patternkv.py",
    " M quant/batch_invariant_kproj.py",
    " M reports/system_ragged_multistep_correctness_v1/b2_16step.md",
    " M reports/system_ragged_multistep_correctness_v1/b2_reorder.md",
    " M reports/system_ragged_multistep_correctness_v1/b2_reorder_steps.json",
    " M reports/system_ragged_multistep_correctness_v1/b2_steps.json",
    " M reports/system_ragged_multistep_correctness_v1/b4_16step.md",
    " M reports/system_ragged_multistep_correctness_v1/b4_steps.json",
    " M reports/system_ragged_multistep_correctness_v1/environment.md",
    " M reports/system_ragged_multistep_correctness_v1/final_gate.json",
    " M reports/system_ragged_multistep_correctness_v1/free_run.json",
    " M reports/system_ragged_multistep_correctness_v1/pytest.md",
    " M reports/system_ragged_multistep_correctness_v1/runtime_counters.json",
    " M reports/system_ragged_multistep_correctness_v1/semantic_metrics.json",
    " M tests/test_bi_kproj_prefill_runtime.py",
    "?? forensics/",
    "?? reports/centroid_determinism_causal_forensic.md",
    "?? reports/system_ragged_active_state_forensic_v1/",
    "?? reports/system_recent_k_ownership_forensic_v1/",
    "?? reports/system_step1_layer0_kpath_forensic_v1/",
    "?? scripts/centroid_determinism_causal_forensic.py",
    "?? scripts/ragged_active_state_forensic.py",
    "?? scripts/recent_k_ownership_forensic.py",
    "?? scripts/step1_layer0_k_path_forensic.py"
  ],
  "python": "3.10.20",
  "remote_v": "bounded\tgit@github.com:pytenter/Bounded-pattrenKV-method.git (fetch)\nbounded\tgit@github.com:pytenter/Bounded-pattrenKV-method.git (push)\norigin\thttps://github.com/HCOOOH/PatternKV.git (fetch)\norigin\thttps://github.com/HCOOOH/PatternKV.git (push)",
  "start_head": "cc50fdc513181d2137438cc6a7c0dd8322ccf767",
  "torch": "2.4.1+cu124",
  "triton": "3.0.0"
}

```text
Fri Aug 14 20:42:03 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.173.02             Driver Version: 580.173.02     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:1A:00.0 Off |                  N/A |
| 30%   28C    P8             19W /  350W |   19781MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        Off |   00000000:1C:00.0 Off |                  N/A |
| 30%   54C    P2            123W /  350W |    3020MiB /  24576MiB |     24%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 3090        Off |   00000000:1D:00.0 Off |                  N/A |
| 30%   57C    P2            121W /  350W |    3021MiB /  24576MiB |     25%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 3090        Off |   00000000:1E:00.0 Off |                  N/A |
| 30%   54C    P2            118W /  350W |    3020MiB /  24576MiB |     25%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA GeForce RTX 3090        Off |   00000000:3E:00.0 Off |                  N/A |
| 30%   57C    P2            150W /  350W |    3020MiB /  24576MiB |     25%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA GeForce RTX 3090        Off |   00000000:3F:00.0 Off |                  N/A |
| 30%   50C    P2            122W /  350W |    3020MiB /  24576MiB |     26%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA GeForce RTX 3090        Off |   00000000:40:00.0 Off |                  N/A |
| 30%   36C    P8             17W /  350W |      21MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA GeForce RTX 3090        Off |   00000000:41:00.0 Off |                  N/A |
| 30%   33C    P8             17W /  350W |      21MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A          201464      C   python                                18380MiB |
|    0   N/A  N/A         2467109    C+G   policy/pi05/.venv/bin/python            260MiB |
|    0   N/A  N/A         2467112    C+G   policy/pi05/.venv/bin/python            260MiB |
|    0   N/A  N/A         2467115    C+G   policy/pi05/.venv/bin/python            260MiB |
|    0   N/A  N/A         2467118    C+G   policy/pi05/.venv/bin/python            260MiB |
|    0   N/A  N/A         2467121    C+G   policy/pi05/.venv/bin/python            260MiB |
|    1   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    1   N/A  N/A         2467109    C+G   policy/pi05/.venv/bin/python           2970MiB |
|    2   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    2   N/A  N/A         2467112    C+G   policy/pi05/.venv/bin/python           2970MiB |
|    3   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    3   N/A  N/A         2467115    C+G   policy/pi05/.venv/bin/python           2970MiB |
|    4   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    4   N/A  N/A         2467118    C+G   policy/pi05/.venv/bin/python           2970MiB |
|    5   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    5   N/A  N/A         2467121    C+G   policy/pi05/.venv/bin/python           2970MiB |
|    6   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
|    7   N/A  N/A            3731      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```
