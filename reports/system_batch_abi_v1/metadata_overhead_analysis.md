# Metadata Overhead Analysis

Assumptions for concrete estimates:

- `page_size = 128`
- `Hkv = 8`
- `D = 128`
- `group = 128`
- base payload target is about `2.5 bits / KV element` for K INT2 plus mixed V average 2.5 bits at 25% V4.
- Per-token Pattern metadata can be stored per KV head unless compressed further:
  - `pattern_mask`: 1 byte per `(head,token)` today;
  - `assignment_idx`: 4 bytes per `(head,token)` today, but can be `uint8` if centroid count <= 256.

## Candidate A: Ragged Global Stream

Per request stream metadata:

| Metadata | Bytes/request |
|---|---:|
| `seq_len` | 4 |
| `v2_offset`, `v2_length` | 8 |
| `v4_offset`, `v4_length` | 8 |
| K offset/length or block refs | 8-16 |
| selector state ref | 4-8 |

Per token:

| Metadata | Bits/token | Notes |
|---|---:|---|
| precision bitmap | 1 | pack into bits |
| logical-to-physical rank | avoid if rank computed by prefix counts; otherwise 16-32 | rank LUT is expensive |
| Pattern assignment | 8-32 per head | current int32 should become u8/u16 if possible |
| Pattern gate | 1-8 per head | current uint8 is 8x larger than packed bit |

For `Hkv=8`, current uncompressed Pattern metadata is `8 heads * (1+4) = 40 bytes/token = 320 bits/token`, much larger than quantized payload. A serving ABI must compress assignment and gates or store them page-local in compact stream order.

## Candidate B: Page-Centric Dual Stream

Per logical page:

| Metadata | Bytes/page | Bits/token |
|---|---:|---:|
| precision bitmap, 128 bits | 16 | 1 |
| V2 count + V4 count | 4 | 0.25 |
| V2 page id + V4 page id | 8 | 0.5 |
| optional K page id | 4 | 0.25 |
| selector/page flags | 4 | 0.25 |
| page-local rank LUT, optional `uint8[128]` | 128 | 8 |

Without rank LUT: about `36 bytes/page = 2.25 bits/token`. With a `uint8` rank LUT: `164 bytes/page = 10.25 bits/token`.

Per compact stream payload metadata:

- V2/V4 scale/zero are not new overhead; independent affine scales/zeros already exist and must remain.
- V2/V4 compact Pattern metadata in page-local compact order:
  - assignment u8: `Hkv * 128 bytes/page = 8192 bits/page = 64 bits/token`;
  - gate bitpacked: `Hkv * 16 bytes/page = 8 bits/token`;
  - if assignment remains int32: 256 bits/token just for assignment.

The critical optimization is assignment dtype/packing. With u8 assignment and bitpacked gates, Pattern metadata is about `72 bits/token`. That is still significant but bounded, coalesced, and page-local.

## Candidate C: Framework-Native Pool ABI

Per request:

| Metadata | Bytes/request |
|---|---:|
| `seq_len` | 4 |
| `num_pages` or indptr delta | 4 |
| request index / pool index | 4 |

Per page table entry:

| Metadata | Bytes/page ref |
|---|---:|
| K page id | 4 |
| V2 page id | 4 |
| V4 page id | 4 |
| metadata page id | 4 |

For `page_size=128`, 16 bytes/page table overhead is `1 bit/token`. Add precision bitmap and counts (`~1.25 bits/token`) and optional page-local rank LUT (`8 bits/token` if used).

## Relative to 2.5 bits/KV Element

Payload per token for one K+V element pair at 25% V4 is roughly:

```text
K: 2 bits/element
V: 0.75*2 + 0.25*4 = 2.5 bits/element
average K+V = 2.25 bits/element if averaged equally over K and V,
or "about 2.5" when focusing on Value payload under the mixed split.
```

Metadata in bits/token must be normalized by elements/token: `2 * Hkv * D = 2048 elements` for K+V at `Hkv=8,D=128`.

Examples:

- precision bitmap only: `1 bit/token / 2048 = 0.00049 bits/KV element`.
- page tables + counts around `2.25 bits/token / 2048 = 0.0011 bits/KV element`.
- u8 assignment plus bitpacked gates around `72 bits/token / 2048 = 0.035 bits/KV element`.
- int32 assignment plus byte gates around `320 bits/token / 2048 = 0.156 bits/KV element`.

## Conclusion

Page tables and precision bitmaps are cheap. Pattern assignment/gate metadata dominates unless compacted. The recommended serving ABI should:

- bitpack precision bitmap and Pattern gates;
- use `uint8` assignment when centroid count <= 256, `uint16` otherwise;
- keep page tables as int32;
- avoid per-token global rank LUT unless profiling proves prefix-count rank is too expensive.
