# Operator Implementation

- `quant.page_batch.patternkv_page_batch_decode` is the production-facing MVP API.
- It consumes compact V2/V4 pages plus metadata and does not call the legacy serial B=1 mixed-V kernel.
- It expands only page-local compact payloads during accumulation; full historical Value materialization remains zero.
- Current implementation is Torch/page-local and classified as an operator regression until replaced by a CUDA/Triton page kernel.
