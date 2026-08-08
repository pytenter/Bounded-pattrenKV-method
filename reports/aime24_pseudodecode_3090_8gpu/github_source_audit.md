# GitHub Source Audit

| relative path | git blob/hash | purpose | status |
| --- | --- | --- | --- |
| `configs/aime24_wave1_selected_tasks.json` | `22ab8191b6143d26d7ab4233bc7c95ad41aa1ac8` | Frozen 12-task Wave1A cohort | present |
| `bench/aime24_int2_wave1.py` | `7662e6d89dd26c521e6795214e183f9170327809` | Wave1A analysis helpers and stable hashing | present |
| `bench/attention_observer.py` | `d095b0dadeaf26943964f9565bff41331605c927` | Read-only sparse attention metric helpers | present |
| `bench/paper_config.py` | `459dccac9c08ed96294f5a19b73b8755310c7ae1` | Paper and rolling method config defaults | present |
| `bench/patternkv_equivalence_reference.py` | `8a4341a9d415836f47667c8f3aaaa56ec14ba4f9` | PatternKV equivalence reference helpers | present |
| `scripts/run_aime24_int2_wave1_8gpu.sh` | `3086171728d3774ab567ae0faef23b64da90a228` | Prior 8-GPU Wave1A launch semantics | present |
| `scripts/run_aime24_wave1a4_attention_mechanism.sh` | `4639a2a0068f2bf02034dd8238091e10339b2df4` | Wave1A.4 mechanism launcher | present |
| `scripts/run_wave1a4_attention_observer.py` | `9acd9f43549916b387b4cd48d983779a09df52f4` | Wave1A.4 observer runner | present |
| `models/segmented_cache.py` | `ad705c654a5a675ef80744b04940b41432eaa989` | Segmented sink/pending/recent/packed-history cache implementation | present |
| `models/llama_patternkv.py` | `f1893305c50af88511e9c172daad89bb11db99cf` | PatternKV Llama production cache path | present |
| `models/llama_kivi.py` | `32cd0f953ee8693c231cafcc1ea817a1b8b5e5d4` | KIVI Llama production cache path | present |
| `quant/new_pack.py` | `72380af9dcc931547367deb00117dc2cbd5d1ebf` | Quantization pack/depack helpers | present |
| `quant/matmul.py` | `ccf4a34c9ccf10de02574ff09356f81a943936d8` | Quantized matmul helpers | present |
| `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_early_token_mechanism_report.md` | `72092bf188ce7e8c1a17c508bb7e074740658fa9` | Wave1A.4 mechanism report anchor | present |
| `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism_summary.json` | `3f2a796f34d8bb86c8b2b69833fa7b618530dc47` | Wave1A.4 mechanism summary anchor | present |
| `reports/aime24_int2_wave1_v100_8gpu/s128_sink_semantics_resolution.md` | `5156dd24204b83f9ab5511b5470225966fbdbe92` | Absolute-sequence-prefix sink semantics record | present |
