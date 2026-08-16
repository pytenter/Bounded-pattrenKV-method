# Production Fix

Decode RMSNorm now uses request-invariant fixed hidden-dim chunk reduction for both input and post-attention RMSNorm. No layer/step special cases.
