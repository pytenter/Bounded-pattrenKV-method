# Paper Tables

## Full-Model Batch Scaling

| batch | FP16_status | CAUSAL_status | FP16_tpot_ms | CAUSAL_tpot_ms | FP16_tok_s | CAUSAL_tok_s | CAUSAL_TPOT_over_FP16 | FP16_peak_allocated_bytes | CAUSAL_peak_allocated_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PASS | PASS | 28.041451626146834 | 154.67090412857942 | 35.65918652555947 | 6.465179965655823 | 5.515795194581112 | 16612160512 | 16513622016 |
| 2 | PASS | PASS | 28.045849244032677 | 216.82667679851875 | 71.31436860444889 | 9.226774784058724 | 7.731150335718681 | 17155259392 | 16911032320 |
| 4 | PASS | PASS | 31.456246904175106 | 160.1794648837919 | 127.13991637683745 | 24.978714042362377 | 5.092135287841083 | 18241457152 | 17744289792 |
| 8 | PASS | PASS | 39.62402120426608 | 156.39846688524509 | 201.86439806692866 | 51.14960330568773 | 3.947061962212119 | 20413852672 | 19410804736 |

## Full-Model Context Scaling

| context | FP16_status | CAUSAL_status | FP16_tpot_ms | CAUSAL_tpot_ms | FP16_tok_s | CAUSAL_tok_s | CAUSAL_TPOT_over_FP16 | FP16_peak_allocated_bytes | CAUSAL_peak_allocated_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | PASS | PASS | 28.041451626146834 | 154.67090412857942 | 35.65918652555947 | 6.465179965655823 | 5.515795194581112 | 16612160512 | 16513622016 |
| 4096 | PASS | PASS | 28.39603079094862 | 152.6202732541909 | 35.214725198522395 | 6.552857805283687 | 5.374704457034164 | 17153242112 | 16936968192 |
| 8192 | PASS | PASS | 30.54934024112299 | 152.6505633664783 | 32.729606916829454 | 6.553153112920851 | 4.996853030593203 | 18235405312 | 17834008576 |

## Full-Model Capacity

| method | context | max_success_B | first_OOM_B | tpot_ms_at_max_B | tok_s_at_max_B | peak_allocated_bytes | peak_reserved_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16_FULL_MODEL | 4096 | 4 | 8 | 40.21614062367007 | 99.45046964930178 | 20405783552 | 20786970624 |
| CAUSAL_V4_25_FULL_MODEL | 4096 | 8 | 16 | 204.97080124914646 | 39.02858814821033 | 22432702464 | 23551016960 |

## Long Decode

| method | context | batch | decode | scope | physical_gpu | success | oom | tpot_ms | tok_per_s | peak_allocated_bytes | peak_reserved_bytes | prefill_calls_timed | refill_calls_timed | membership_changes_timed | page_batch_pack_calls | historical_fp16_k_materialization | historical_fp16_v_materialization | fallback | serial_request_dispatch | serial_attention_dispatch | status | phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16_FULL_MODEL | 2048 | 1 | 256 | full_model_decode_serving | 2 | True | False | 27.89106329328206 | 35.85359588675742 | 16677680128 | 16930308096 | 0 | 0 | 0 | 0 |  |  | 0 | 0 | 0 | PASS | long_decode |
| CAUSAL_V4_25_FULL_MODEL | 2048 | 1 | 256 | full_model_decode_serving | 2 | True | False | 156.2204092851971 | 6.4012055622328345 | 16513622016 | 16636706816 | 0 | 0 | 0 | 64 | 0 | 0 | 0 | 0 | 0 | PASS | long_decode |
| FP16_FULL_MODEL | 2048 | 1 | 512 | full_model_decode_serving | 2 | True | False | 28.335415765468497 | 35.291425425657955 | 16745313280 | 17014194176 | 0 | 0 | 0 | 0 |  |  | 0 | 0 | 0 | PASS | long_decode |
| CAUSAL_V4_25_FULL_MODEL | 2048 | 1 | 512 | full_model_decode_serving | 2 | True | False | 155.22217463876586 | 6.442374860609848 | 16513622016 | 16651386880 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | PASS | long_decode |
