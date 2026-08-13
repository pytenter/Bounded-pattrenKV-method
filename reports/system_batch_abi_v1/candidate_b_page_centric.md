# Candidate B: Page-Centric Dual Stream

## ABI

Page size: `128`.

Each logical page stores or references:

```text
K tight INT2 page/block
V2 independent affine payload page
V4 independent affine payload page
V2 scale/zero
V4 scale/zero
precision bitmap[128]
V2 count
V4 count
V2 compact Pattern metadata
V4 compact Pattern metadata
logical token start
valid tokens
```

Per request:

```text
page_table
num_pages
seq_len
selector_state_ref
```

## Pros

- Natural ragged support.
- Natural continuous batching and future page allocator integration.
- Solves variable V2/V4 lengths page-locally.
- Preserves independent V2/V4 affine streams.
- Keeps K/V asymmetry explicit.
- Fits FlashInfer/vLLM/SGLang-style page-table thinking.

## Cons

- Requires new metadata writer and custom decode reader.
- More metadata design surface than global streams.
- Kernel must either compute page-local prefix counts or read a rank/prefix table.

## Assessment

Recommended ABI. This is the smallest design that is both PatternKV-correct and serving-native.
