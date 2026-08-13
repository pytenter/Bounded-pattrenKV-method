# Benchmark Protocol

- Architecture: ASYMMETRIC_KV_RUNTIME.
- K historical layout: tight contiguous production QK path.
- V historical layouts: baseline growing contiguous, fixed capacity, chunked capacity.
- Mixed Value reader: fused compressed-domain V2/V4.
- Algorithm frozen: K INT2, base V INT2, selected V INT4, V4 fraction 25%, sink16, recent128, residual128, group128, selector causal_v4.
- Profile-off matrix: contexts 4096,8192,16384,32768; decode 128,512; backends baseline,fixed_capacity,chunked_capacity; warmup >=1; rounds >=5.
- Same context/decode seeds are reused across backends.
- Profile-on component rows are approximate and not used as final TPOT truth.
