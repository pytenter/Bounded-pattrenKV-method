# Warp Access Mapping

Example: `head_dim=128`, INT2 `pack_factor=16`, logical token pack `packed=0`, capacity `32768`, so `cap_packs=2048`.

In both kernels, a warp lane handles four channel positions: `k = lane * 4 + {0,1,2,3}`.

Tight K layout uses `[B*kv, packed_token, channel]`, so for fixed `packed=0`:

```text
offset = packed * IC + k = k
lane 0 reads offsets 0,1,2,3
lane 1 reads offsets 4,5,6,7
...
lane 31 reads offsets 124,125,126,127
```

STATIC_CODE_EVIDENCE: those 128 int32 words are contiguous across the warp/CTA channel tile.

Strided K capacity layout uses `[B, kv, channel, token_pack]`, so for fixed `packed=0`:

```text
offset = k * cap_packs + packed
lane 0 reads offsets 0,2048,4096,6144
lane 1 reads offsets 8192,10240,12288,14336
...
```

STATIC_CODE_EVIDENCE: lanes traverse channel, but physical memory is token-pack-major within each channel row. Adjacent channel loads are separated by `cap_packs` int32 words, so the tight contiguous channel tile becomes a highly strided load pattern.

HYPOTHESIS: this likely worsens coalescing/global transactions for packed K, but NCU was unavailable, so transaction counters were not measured.
