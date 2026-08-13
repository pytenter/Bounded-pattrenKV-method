# Static Operator Audit

| file | symbol | operation | frequency per page/request | possible synchronization | possible allocation | possible kernel launch | suspected cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `quant/page_batch.py` | `pack_mixed_v_pages` | Python B/page loop, .item() from page_precision.sum | B*pages during packing | yes if CUDA sum item | yes | quantize kernels | moderate; outside decode timing unless packing included |
| `quant/page_batch.py` | `patternkv_page_batch_decode` | seq_lens max/min .item, num_pages .item | 3 per decode | yes | no | possible reductions | high sync candidate |
| `quant/page_batch.py` | `patternkv_page_batch_decode` | metadata_pages/v2/v4/counts .item | 6 per logical page | yes | no | none besides sync | high sync candidate |
| `quant/page_batch.py` | `patternkv_page_batch_decode` | page_precision reconstruct | 1 per logical page | no explicit item | yes bool tensor | comparison kernel | moderate |
| `quant/page_batch.py` | `_restore_page_values` | dequantize_v_reference | one per non-empty V2/V4 page | no explicit item | yes page tensor | unpack/elementwise kernels | high |
| `quant/page_batch.py` | `_restore_page_values` | pattern_gather_centroids | one per non-empty V2/V4 page | no explicit item | yes gathered tensor | gather kernel | high |
| `quant/page_batch.py` | `patternkv_page_batch_decode` | boolean attention indexing + contiguous | one per non-empty V2/V4 page | no explicit item | yes compact attn | index/copy kernels | high |
| `quant/page_batch.py` | `_repeat_kv` | expand+reshape GQA replication | one per non-empty V2/V4 page | no | view or copy depending stride | usually no or reshape copy | moderate |
| `quant/page_batch.py` | `patternkv_page_batch_decode` | torch.matmul | one per non-empty V2/V4 page | no | output temp | GEMM/bmm kernel | high fragmentation candidate |
| `quant/page_batch.py` | `patternkv_page_batch_decode` | out slice += part | one per non-empty V2/V4 page | no | no major temp | add/copy kernel | moderate |
| `bench/run_page_batch_mvp_report.py` | `time_cuda_callable` | torch.cuda.synchronize + CUDA events | per timing run | intentional | no | event ops | methodology only |
| `bench/patternkv_page_batch_mvp.py` | `reference_batch_mixed_v` | Python loop over B serial B1 reference | B per reference call | not production path | yes compact streams | legacy CUDA kernels | reference only |
| `quant/patternkv_profile.py` | `profile_snapshot` | torch.cuda.synchronize | snapshot only | intentional | no | none | measurement only |
