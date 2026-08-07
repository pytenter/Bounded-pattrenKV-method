# Wave 1A.2 Sink Recent Manifest

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- HEAD: `242a3a1b7e789a505006d74450f51a45ccfb055c`
- Task manifest: `configs/aime24_wave1_selected_tasks.json`
- Task manifest hash: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- Model: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
- Reused configs: `4`
- Newly run configs: `4`

| GPU | config | source | method | cache | sink | recent | residual | K | V |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `pattern_rolling_k2v2_s0_r128` | `reused_wave1a` | patternkv | segmented_rolling | 0 | 128 | 128 | 2 | 2 |
| 1 | `pattern_rolling_k2v2_s64_r128` | `newly_run_wave1a2` | patternkv | segmented_rolling | 64 | 128 | 128 | 2 | 2 |
| 2 | `pattern_rolling_k2v2_s0_r256` | `newly_run_wave1a2` | patternkv | segmented_rolling | 0 | 256 | 128 | 2 | 2 |
| 3 | `pattern_rolling_k2v2_s64_r256` | `reused_wave1a` | patternkv | segmented_rolling | 64 | 256 | 128 | 2 | 2 |
| 4 | `kivi_rolling_k2v2_s0_r128` | `reused_wave1a` | kivi_official | segmented_rolling | 0 | 128 | 128 | 2 | 2 |
| 5 | `kivi_rolling_k2v2_s64_r128` | `newly_run_wave1a2` | kivi_official | segmented_rolling | 64 | 128 | 128 | 2 | 2 |
| 6 | `kivi_rolling_k2v2_s0_r256` | `newly_run_wave1a2` | kivi_official | segmented_rolling | 0 | 256 | 128 | 2 | 2 |
| 7 | `kivi_rolling_k2v2_s64_r256` | `reused_wave1a` | kivi_official | segmented_rolling | 64 | 256 | 128 | 2 | 2 |
