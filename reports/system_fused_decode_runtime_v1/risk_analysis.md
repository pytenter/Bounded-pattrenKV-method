# Risk Analysis

The integration is fixed-length and does not implement ragged serving. The old B2/B4 runtime baseline is necessarily an independent B1 aggregate because the legacy production mixed-V backend is B1-only. Multi-step decode reaches a structural blocker at the 128-token flush boundary: dynamic centroid update is cache-global across B, while the golden reference has independent B1 centroid evolution per request.
