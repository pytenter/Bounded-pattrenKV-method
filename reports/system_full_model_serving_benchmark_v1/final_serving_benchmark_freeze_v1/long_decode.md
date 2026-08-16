# Long Decode

| method | context | batch | decode | scope | physical_gpu | success | oom | tpot_ms | tok_per_s | peak_allocated_bytes | peak_reserved_bytes | prefill_calls_timed | refill_calls_timed | membership_changes_timed | page_batch_pack_calls | historical_fp16_k_materialization | historical_fp16_v_materialization | fallback | serial_request_dispatch | serial_attention_dispatch | status | phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16_FULL_MODEL | 2048 | 1 | 256 | full_model_decode_serving | 2 | True | False | 27.89106329328206 | 35.85359588675742 | 16677680128 | 16930308096 | 0 | 0 | 0 | 0 |  |  | 0 | 0 | 0 | PASS | long_decode |
| CAUSAL_V4_25_FULL_MODEL | 2048 | 1 | 256 | full_model_decode_serving | 2 | True | False | 156.2204092851971 | 6.4012055622328345 | 16513622016 | 16636706816 | 0 | 0 | 0 | 64 | 0 | 0 | 0 | 0 | 0 | PASS | long_decode |
| FP16_FULL_MODEL | 2048 | 1 | 512 | full_model_decode_serving | 2 | True | False | 28.335415765468497 | 35.291425425657955 | 16745313280 | 17014194176 | 0 | 0 | 0 | 0 |  |  | 0 | 0 | 0 | PASS | long_decode |
| CAUSAL_V4_25_FULL_MODEL | 2048 | 1 | 512 | full_model_decode_serving | 2 | True | False | 155.22217463876586 | 6.442374860609848 | 16513622016 | 16651386880 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | PASS | long_decode |

## Profile-Only Boundary Attribution

These profile-only runs are separate from the formal TPOT rows above.

| decode | method | page_batch_pack_calls | page_batch_pack_ms_per_token | pack_window_ms_per_token | cache_append_ms_per_token |
| --- | --- | --- | --- | --- | --- |
| 256 | CAUSAL_V4_25_FULL_MODEL | 64 | 0.4266200000413572 | 5.467604067070484 | 25.868955229991116 |
| 512 | CAUSAL_V4_25_FULL_MODEL | 128 | 0.44451980575434864 | 5.616521221715212 | 25.455363434855826 |
