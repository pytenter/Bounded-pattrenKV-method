# Root Cause Evidence

The original K projection root cause is fixed: BI KProj contract M1/M2/reorder/M4 exact, step1 raw K exact, post-RoPE current K exact, and recent_k transition exact. However strict K/V BI did not close the full multistep gate: B2/B4/reorder failed by logit relL2 threshold while top1 stayed matched. Active-state forensic under strict mode localized the next earliest semantic divergence to `v_causal_importance` at request `A`, step `1`, layer `0` with relL2 `0.16538892686367035` and max_abs `0.036240413784980774`. Therefore no commit/push is allowed in this round.
