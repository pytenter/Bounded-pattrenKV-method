# INT2 Unpack / Dequant Audit

## Compressed QK Kernel
- Kernel: `bgemv_kernel_outer_dim_with_base_tiled<2>`.
- Geometry: `group_size=128`, `head_dim=128`, so one quantization group spans one head dimension tile.
- Packed load: each 32-bit word carries 16 INT2 values. The hot loop loads `qw[t]` as `uint32_t`, then extracts values with `(q >> (i * Bit)) & mask`.
- Per 32-bit word for INT2: 1 packed global load, 1 scale load, 1 zero load, then 16 shift+mask extracts and 16 FP32 `fmaf` updates.
- Per decoded element approximation: 1/16 packed load, 1 shift, 1 mask, about 1 FP32 FMA contribution, plus amortized scale/zero and input conversion.
- Scale/zero reuse: current QK tiled kernel loads scale/zero per packed word and K-tile lane, then reuses them across the 16 unpacked INT2 outputs from that word. This exploits `group_size=head_dim=128` partially.

## Mixed V Page Pool Kernel
- Kernel: `page_mixed_pool_value_kernel<2>`.
- Packed load: V2 uses one 32-bit payload load for 16 channels; V4 uses one 32-bit payload load for 8 channels.
- Per token/channel iteration: loads attention, metadata page table/prefix, page table, page offset, payload word, scale, zero, pattern bit, optional assignment, optional centroid, then does shift+mask, half-to-float conversions, affine dequant, optional centroid add, and FP32 accumulation.
- Scale/zero reuse: for `group_size=128`, `oc_group=0` across the current 128-dim head, so one scale/zero pair theoretically serves 128 channels for a token. Current one-CTA-per-output-channel scheduling reloads the same scale/zero for many `oc` CTAs, so theoretical reuse is not fully exploited.

## Classification
Compressed INT2 kernels are not the top full-model discrepancy cause. Within decode-only attention they are secondary and plausibly INT2 unpack / metadata-load bound, but Nsight Compute counters were unavailable, so this is a medium-confidence static+timing classification rather than a measured stall-counter roofline.
