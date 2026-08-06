# Wave 1A Long Smoke

Date: 2026-08-06

Command:

```bash
bash scripts/run_aime24_int2_wave1_8gpu.sh wave1a-long-smoke
```

Status: PASS

- Configs: 6
- Task keys per config: 2
- `max_new_tokens`: 4096
- `PATTERNKV_CACHE_VALIDATE`: enabled
- Runtime errors: 0
- CUDA illegal memory access: 0 observed
- GPU cleanup: no compute processes after completion

| config | rows | stop reason | final sink | final packed | final pending | final recent | assignment tokens | peak reserved bytes | validation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `kivi_k2v2_s0_r128` | 2 | length | 0 | 3968 | 103 | 128 | null | 16506683392 | pass |
| `pattern_k2v2_s0_r128` | 2 | length | 0 | 3968 | 103 | 128 | 3968 | 16546529280 | pass |
| `kivi_k2v2_s64_r256` | 2 | length | 64 | 3840 | 127 | 256 | null | 16552820736 | pass |
| `pattern_k2v2_s64_r256` | 2 | length | 64 | 3840 | 127 | 256 | 3840 | 16584278016 | pass |
| `pattern_k4v2_s0_r128` | 2 | length | 0 | 3968 | 103 | 128 | 3968 | 16489906176 | pass |
| `pattern_k2v4_s0_r128` | 2 | length | 0 | 3968 | 103 | 128 | 3968 | 16489906176 | pass |

## Checks

- Recent rollover occurred repeatedly in every config.
- Pending K history remained FP16 until legal pack boundaries.
- Sink64 configs retained `sink_tokens = 64`.
- Recent128 configs retained `recent_tokens = 128`; Recent256 configs retained `recent_tokens = 256`.
- Pattern assignment token counts matched packed history token counts.
- Pending FP16 overhead was non-zero and recorded.

## Note

PatternKV Wave 1A currently records assignment alignment for packed history and uses the segmented packed cache path for uniform K/V bitwidths. Mixed-key configs remain blocked for Wave 1B.
