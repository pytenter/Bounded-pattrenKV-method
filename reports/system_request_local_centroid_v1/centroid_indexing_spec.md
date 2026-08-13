# Centroid Indexing Spec

Logical indices remain stable: `0..M_static-1` address the copied static bank, and `M_static..count[slot)-1` address that request slot's dynamic centroid history. The same stored assignment index is interpreted against the request-local active bank.
