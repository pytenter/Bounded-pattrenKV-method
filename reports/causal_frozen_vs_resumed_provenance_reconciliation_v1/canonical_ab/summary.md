# Canonical A/B

The canonicalized A/B used the same physical GPU 1, same model path, same decode-only timing protocol, same selective prefill intent, same active batch cache, same fixed split, same `fused_page` mixed-V backend, and the frozen allocator env. Under that protocol there is no post-freeze CAUSAL runtime regression.
