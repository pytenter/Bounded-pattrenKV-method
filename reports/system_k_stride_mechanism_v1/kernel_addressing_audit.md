# Kernel Addressing Audit

Evidence type: STATIC_CODE_EVIDENCE unless otherwise marked.

## Tight K Reader

Production kernel: `bgemv_kernel_outer_dim_with_base_tiled<2>`.

Tight C++ wrapper receives transposed tight tensors:

- `_kernel`: `[B * nh_kv, ceil(OC / pack), IC]`
- `_scaling_factors`: `[B * nh_kv, ceil(OC / group_size), IC]`
- `_zeros`: `[B * nh_kv, ceil(OC / group_size), IC]`
- `_assignments`: `[B, nh_kv, OC]`

Address equations from `quant/csrc/gemv_cuda.cu`:

```text
batch_kv_flat = b * nh_kv + kv
weight_base = _weight + batch_kv_flat * (OC * IC / pack_factor)
scale_base  = _scale  + batch_kv_flat * (OC * IC / group_size)
zeros_base  = _zeros  + batch_kv_flat * (OC * IC / group_size)
arow_byte   = _assign + ((b * nh_kv + kv) * OC * assign_bytes)

packed K word: weight[packed * IC + k]
scale:         scale[group_idx * IC + k]
zero:          zeros[group_idx * IC + k]
assignment:    arow[oc * assign_bytes]
query:         inputs[k]
centroid:      cbase[m * IC + k]
```

Relevant source:

```cpp
986:   const int batch_kv_flat = b * nh_kv + kv;
987:
988:   const uint32_t* weight  = _weight + (size_t)batch_kv_flat * (OC * IC / pack_factor);
989:   const half*     scale   = _scale  + (size_t)batch_kv_flat * (OC * IC / group_size);
990:   const half*     zeros   = _zeros  + (size_t)batch_kv_flat * (OC * IC / group_size);
991:
992:   // centroids: [nh_kv, M, IC]
993:   const half* cbase = _centroids + (size_t)kv * (M_centroids * IC);
994:   // assignments: [B, nh_kv, OC] （与 OC 对齐）
995:   const char* arow  = reinterpret_cast<const char*>(_assign)
996:                     + ((size_t)b * nh_kv + kv) * (size_t)OC * assign_bytes;
...
1070:       // 载入与该 oc-pack、该 tile 对应的权重&量化参数
1071:       const int w_off  = packed   * IC + k_base + threadIdx.x * 4;
1072:       const int sz_off = group_idx* IC + k_base + threadIdx.x * 4;
1073:
1074:       // 4 份 32-bit 打包权重（每份包含 pack_factor 个子权重的一部分）
1075:       uint32_t qw[4] = {0,0,0,0};
1076:       if (w_off + 0 < (OC * IC / pack_factor)) qw[0] = weight[w_off + 0];
1077:       if (w_off + 1 < (OC * IC / pack_factor)) qw[1] = weight[w_off + 1];
1078:       if (w_off + 2 < (OC * IC / pack_factor)) qw[2] = weight[w_off + 2];
1079:       if (w_off + 3 < (OC * IC / pack_factor)) qw[3] = weight[w_off + 3];
1080:
1081:       // 对应的 scale/zero（与 lane 无关；同一 pack 的 oc 共享一组）
1082:       half sc4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
1083:       half ze4[4] = {__float2half(0.f),__float2half(0.f),__float2half(0.f),__float2half(0.f)};
1084:       if (sz_off + 0 < (OC * IC / group_size)) { sc4[0] = scale[sz_off + 0]; ze4[0] = zeros[sz_off + 0]; }
1085:       if (sz_off + 1 < (OC * IC / group_size)) { sc4[1] = scale[sz_off + 1]; ze4[1] = zeros[sz_off + 1]; }
1086:       if (sz_off + 2 < (OC * IC / group_size)) { sc4[2] = scale[sz_off + 2]; ze4[2] = zeros[sz_off + 2]; }
1087:       if (sz_off + 3 < (OC * IC / group_size)) { sc4[3] = scale[sz_off + 3]; ze4[3] = zeros[sz_off + 3]; }
```

## Strided K Reader

Experimental kernel: `bgemv_kernel_outer_dim_with_base_strided_k<2>`.

Strided wrapper receives logical views over capacity storage:

- `_kernel`: `[B, nh_kv, IC, ceil(logical_OC / pack)]` with physical capacity in stride(2)
- `_scaling_factors`: `[B, nh_kv, IC, ceil(logical_OC / group_size)]`
- `_zeros`: `[B, nh_kv, IC, ceil(logical_OC / group_size)]`
- `_assignments`: `[B, nh_kv, logical_OC]`

Address equations:

```text
logical loop bound OC = _assignments.size(2)
packed K word = _weight[b*w_s0 + kv*w_s1 + k*w_s2 + packed*w_s3]
scale         = _scale [b*sc_s0 + kv*sc_s1 + k*sc_s2 + group_idx*sc_s3]
zero          = _zeros [b*z_s0 + kv*z_s1 + k*z_s2 + group_idx*z_s3]
assignment    = _assign[b*a_s0 + kv*a_s1 + oc*a_s2]
query         = inputs[k]
centroid      = cbase[m * IC + k]
```

Relevant source:

```cpp
1267:   const int nPacked = ceil_div(OC, pack_factor);
1268:   const int start_packed = blockIdx.y * blockDim.y + threadIdx.y;
1269:   const int stride_packed = gridDim.y * blockDim.y;
1270:
1271:   for (int packed = start_packed; packed < nPacked; packed += stride_packed) {
1272:     const int oc_start = packed * pack_factor;
...
1301:       uint32_t qw[4] = {0, 0, 0, 0};
1302:       half sc4[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
1303:       half ze4[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
1304:       const int k0 = k_base + threadIdx.x * 4;
1305:       #pragma unroll
1306:       for (int t = 0; t < 4; ++t) {
1307:         const int kk = k0 + t;
1308:         if (kk < IC) {
1309:           qw[t] = _weight[(int64_t)b * w_s0 + (int64_t)kv * w_s1 + (int64_t)kk * w_s2 + (int64_t)packed * w_s3];
1310:           sc4[t] = _scale[(int64_t)b * sc_s0 + (int64_t)kv * sc_s1 + (int64_t)kk * sc_s2 + (int64_t)group_idx * sc_s3];
...
1340:         float v = warp_reduce_sum_local(psum[i]);
1341:         if (threadIdx.x == 0) {
1342:           const char* abase = reinterpret_cast<const char*>(_assign);
1343:           const int64_t elem_off = (int64_t)b * a_s0 + (int64_t)kv * a_s1 + (int64_t)oc * a_s2;
1344:           int aidx;
1345:           if (assign_bytes == 1)      aidx = *((const uint8_t *)(abase + elem_off));
1346:           else if (assign_bytes == 2) aidx = *((const int16_t *)(abase + elem_off * 2));
1347:           else if (assign_bytes == 4) aidx = *((const int32_t *)(abase + elem_off * 4));
1348:           else                        aidx = static_cast<int>(*((const int64_t *)(abase + elem_off * 8)));
```

## Direct Comparison

- Query and centroid addressing are effectively unchanged.
- Tight K uses a layout-coupled linear expression where `packed * IC + k`; `k` is the fastest varying dimension.
- Strided K must evaluate generic tensor-stride expressions for packed K, scale, zero, and assignment.
- The strided kernel preserves math but turns layout constants into runtime stride operands.
