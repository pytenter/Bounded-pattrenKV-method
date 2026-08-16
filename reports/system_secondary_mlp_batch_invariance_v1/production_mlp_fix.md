# Production MLP Fix

`models/llama_patternkv.py` now enables BI linear for decode MLP gate/up/down through `PATTERNKV_DECODE_BI_MLP` default-on production path. `PATTERNKV_BI_MLP_ORACLE` remains debug-only.
