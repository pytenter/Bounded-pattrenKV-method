# After Fix Dispatch

After this round, `patternkv_use_bi_kproj` enables BI K projection for both prefill and decode in `bi_k`/`bi_kv` modes. Strict `bi_kv` also routes V projection through the same BI V2 linear projection for prefill and decode. Formal B4 counters: `bi_decode_kproj_calls=800`, `bi_decode_vproj_calls=800`, `normal_decode_kproj_calls=0`, `normal_decode_vproj_calls=0`, serial BI dispatches=0.
