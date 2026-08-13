# Why V Centroid Hash Failed

`v_centroid_hash` is a byte-exact SHA comparison. A centroid tensor can differ at FP16 bytes while assignment, mask, precision mask, packed payload, and operator output remain semantically equivalent within tolerance.
