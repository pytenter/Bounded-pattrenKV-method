# Wave 1A.3 Sink Length Sweep Manifest

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- Starting HEAD: `7ba7662fe49e0580fd2697ed175c96e2801a2f11`
- Task manifest: `configs/aime24_wave1_selected_tasks.json`
- Task manifest hash: `ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e`
- Generation config hash: `a7d6b2f8bab37893b6331c66b3e5eb6a`
- Model: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
- Reuse validation: `passed`
- Reused configs: `4`
- Newly-run configs: `6`

## Logical Configs

| config | method | sink | recent | source | source path |
| --- | --- | ---: | ---: | --- | --- |
| `pattern_rolling_k2v2_s0_r128` | PatternKV | 0 | 128 | reused | `results/aime24_int2_wave1_v100_8gpu_revised_full/wave1a/pattern_rolling_k2v2_s0_r128` |
| `pattern_rolling_k2v2_s16_r128` | PatternKV | 16 | 128 | newly_run | `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3/pattern_rolling_k2v2_s16_r128` |
| `pattern_rolling_k2v2_s32_r128` | PatternKV | 32 | 128 | newly_run | `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3/pattern_rolling_k2v2_s32_r128` |
| `pattern_rolling_k2v2_s64_r128` | PatternKV | 64 | 128 | reused | `results/aime24_int2_wave1_v100_8gpu_wave1a2_sink_recent/wave1a2/pattern_rolling_k2v2_s64_r128` |
| `pattern_rolling_k2v2_s128_r128` | PatternKV | 128 | 128 | newly_run | `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3/pattern_rolling_k2v2_s128_r128` |
| `kivi_rolling_k2v2_s0_r128` | KIVI | 0 | 128 | reused | `results/aime24_int2_wave1_v100_8gpu_revised_full/wave1a/kivi_rolling_k2v2_s0_r128` |
| `kivi_rolling_k2v2_s16_r128` | KIVI | 16 | 128 | newly_run | `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3/kivi_rolling_k2v2_s16_r128` |
| `kivi_rolling_k2v2_s32_r128` | KIVI | 32 | 128 | newly_run | `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3/kivi_rolling_k2v2_s32_r128` |
| `kivi_rolling_k2v2_s64_r128` | KIVI | 64 | 128 | reused | `results/aime24_int2_wave1_v100_8gpu_wave1a2_sink_recent/wave1a2/kivi_rolling_k2v2_s64_r128` |
| `kivi_rolling_k2v2_s128_r128` | KIVI | 128 | 128 | newly_run | `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3/kivi_rolling_k2v2_s128_r128` |

## New Run GPU Mapping

| GPU | config |
| ---: | --- |
| 0 | `pattern_rolling_k2v2_s16_r128` |
| 1 | `pattern_rolling_k2v2_s32_r128` |
| 2 | `pattern_rolling_k2v2_s128_r128` |
| 3 | `kivi_rolling_k2v2_s16_r128` |
| 4 | `kivi_rolling_k2v2_s32_r128` |
| 5 | `kivi_rolling_k2v2_s128_r128` |
