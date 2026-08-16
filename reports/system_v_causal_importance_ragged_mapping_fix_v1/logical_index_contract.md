# Logical Importance Index Contract

V_CAUSAL_IMPORTANCE_LOGICAL_INDEX_CONTRACT

`v_causal_importance[r][j]` is a request-local logical token index. Logical index 0 is the first retained sink token for request `r`. The logical axis is the concatenation, for that request only, of valid sink, packed historical tokens, pending tokens, and recent tokens in the same semantic order used by K attention. Packed tokens begin after valid sink; pending begins after valid sink + request-local packed length; recent begins after valid sink + request-local packed + request-local pending. Ragged physical padding caused by peer requests is never part of the logical index space.
