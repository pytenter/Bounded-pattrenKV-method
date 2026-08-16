# Custom Kernel Capture Compatibility

- QK_INT2_HISTORY = CAPTURE_COMPATIBLE after fixed-split softmax host-read blocker is removed.
- MIXED_V_HISTORY = CAPTURE_COMPATIBLE in the captured sequence.
- FIXED_SPLIT_SOFTMAX = CAPTURE_COMPATIBLE after avoiding pre-dispatch host `.item()`.
- CACHE_APPEND = CAPTURE_COMPATIBLE for captured fixed step shapes, but uses dynamic-address tensor replacement across eager steps.
- CENTROID_OPS = CAPTURE_COMPATIBLE for this B1 decode window; broader dynamic centroid growth remains eligibility-limited.
