# Static/Dynamic Centroid Analysis

STATIC_CENTROID_BANK_SHAREABLE=YES

Evidence: the initial centroid bank is passed once to `build_cache_from_prefill` and copied into each active slot. Dynamic centroids are appended per request slot, so static logical indices remain shared while dynamic histories are isolated.
