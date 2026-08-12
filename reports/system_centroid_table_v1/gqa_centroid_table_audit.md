# GQA Centroid Table Audit

`GQA_CENTROID_TABLE_DUPLICATE_LOAD = YES`

The current production grid is organized by query head (`B * nh`) rather than KV-head groups. For Q heads `q0..q3` sharing `kv0`, each query head has distinct attention weights and therefore distinct `SAcc_q[c]`; those accumulators cannot be shared. The centroid table `C_kv0[c, :]`, however, is identical across the four query heads.

- Q heads: 32
- KV heads: 8
- GQA ratio: 4
- Centroid shape per KV head: `[Mcent=16, head_dim=128]`
- Dtype: fp16
- Centroid bytes / KV-head table: `16 * 128 * 2 = 4096` bytes
- Approximate table footprint per Q head per full output vector: 4096 bytes
- Potential footprint for four Q heads sharing one KV head: `4 * 4096 = 16384` bytes
- Ideal reusable table bytes for the group: 4096 bytes
- Theoretical duplicate-load factor: 4x

Candidate B removes intra-warp duplicate table work. It does not remove inter-Q-head duplicate reads. That remaining opportunity is real but needs a GQA-aware kernel organization.
