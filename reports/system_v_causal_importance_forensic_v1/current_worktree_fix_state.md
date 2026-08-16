# Current Worktree Fix State

This forensic run intentionally used the dirty S6-B.3.4I worktree. BI K and BI V projection dispatch is active for prefill and decode under `PATTERNKV_PREFILL_PROJ_MODE=bi_kv`; Q projection remains the ordinary `self.q_proj(hidden_states)` path.
