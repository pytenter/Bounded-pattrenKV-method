# Risk Analysis

| Risk | Severity | Mitigation |
|---|---|---|
| Metadata overhead dominated by Pattern assignment/gates | high | bitpack gates; use u8/u16 assignments; quantify overhead per page |
| K tight path regresses due to serving pages | high | keep K page layout tight inside pages; do not use generic strided K as primary path |
| V2/V4 rank lookup adds excessive GPU cost | medium | start with prefix counts or page-local LUT; profile both |
| Branch divergence from mixed precision inside pages | medium | page-local compact streams and counts; avoid scattered global lookups |
| SGLang/vLLM integration hides ABI bugs | high | prove standalone fixed-length B=2/B=4 first |
| CUDA graph metadata constraints | medium | defer graph capture until eager decode ABI is stable |
| Selector state isolation bugs | medium | keep selector state per request; add isolation tests |
| Accidental quantization semantic drift | high | final gate pins independent affine V2/V4 and frozen selector |

## Main Architectural Risk

The current global compact V2/V4 representation is easy for B=1 but unsuitable for serving unless redesigned around request/page metadata. Attempting a small patch would create correctness risk larger than implementation effort saved.
