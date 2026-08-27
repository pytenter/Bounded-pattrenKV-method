# Request-Local Centroid Audit

CENTROID_SCOPE = REQUEST_LOCAL

The previous scaffold selected initial centroids by reshaping `[B,H,T,D]` into `[H,B*T,D]`, which mixes request states in a shared centroid bank for B2/B4. That is not acceptable for request invariance because a request run alone at B1 can receive different centroids than the same request inside B2/B4.

The compressed backend now selects centroid indices along each request's token axis and returns `[B,H,M,D]`. This preserves request-local centroid banks and matches the segmented cache support for request-local centroid geometry.

Status: PASS_STATIC_FIX. GPU B2/B4 request-invariance still needs full-model closure.
