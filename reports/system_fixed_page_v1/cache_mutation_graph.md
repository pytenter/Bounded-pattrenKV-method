# Cache Mutation Graph

```text
new K/V token
  -> append to recent_k/recent_v
     current: torch.cat + slice rollover
     fixed-page ABI: RecentRingBuffer write_ptr + overflow segment

recent overflow / pending
  -> pending_k/pending_v
     current: torch.cat into pending
     fixed-page ABI: page/block append or direct 128-token flush input

pending reaches 128 tokens
  -> quantize/pack K and V
     current: quantization math unchanged
     fixed-page ABI: quantization math unchanged

packed historical cache
  -> packed_k, packed_v, packed_v4, scale, zero
     current: torch.cat old + new block
     fixed-page ABI: append_block into page streams

selector metadata
  -> v_precision_mask, pattern mask, assignment/index
     current: torch.cat old + new metadata
     fixed-page ABI: append_block into metadata pages

attention reader
  -> current production CUDA expects contiguous tensors
     S3-1: debug materialize_contiguous only
     next phase: page-native reader consumes descriptors/page pointers
```
