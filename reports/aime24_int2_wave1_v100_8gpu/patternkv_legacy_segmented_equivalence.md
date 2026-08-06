# PatternKV Legacy/Segmented Equivalence Report

## 1. Environment

- Python: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`
- Model: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
- Hardware: `8 x NVIDIA Tesla V100-SXM2 32GB`
- Torch: `2.4.1+cu118`
- CUDA runtime: `11.8`

## 2. Git Commits

- Starting HEAD: `d9a608ca8dba40a3d48394f6810c31f58f1cb21f`
- Parent implementation commit: `c8ee4c564d92341755369e5231fe5595322c5980`
- This report records model-level validation after adding the dual-path harness.

## 3. Selected Equivalence Tasks

- `aime24:p12:s0:seed12042`: problem `12`, sample `0`, seed `12042`, reason `medium_stable_fp16_eos`, FP16 tokens `4299`, legacy PatternKV tokens `10991`.
- `aime24:p14:s0:seed14042`: problem `14`, sample `0`, seed `14042`, reason `longer_quantization_sensitive_fp16_eos`, FP16 tokens `6906`, legacy PatternKV tokens `11347`.

## 4. Teacher Token Provenance

- problem `12`, sample `0`: `4096` tokens, hash `25100907c9fa8b210ffbc877fa24bd27`, source `fp16_greedy_generated`.
- problem `14`, sample `0`: `4096` tokens, hash `97ac6947ed2a362d37ebc800c56052c3`, source `fp16_greedy_generated`.

## 5. Level 1 Synthetic Result

- Status: passed in unit tests from prior commit and rerun subset.
- Covered Chebyshev center, min-max assignment, V gate, assignment alignment, serialization, and bitwidth accounting.

## 6. Level 2 Reference Backend Result

- Status: not run.
- Reason: production backend already found a structural mismatch at the first checkpoint; reference numeric comparison cannot override a structural failure.
- `LEVEL2_REFERENCE_PASS=false`

## 7. Level 2 Production Backend Result

- Checkpoint rows: `12`
- Layer rows: `384`
- First mismatch records: `1128`
- `LEVEL2_PRODUCTION_PASS=false`

## 8. Structural Equivalence

- First structural mismatch: task `aime24:p12:s0:seed12042`, checkpoint `128`, layer `0`.
- Legacy: packed K/V `128/128`, recent `64`, pending `0`, K/V centroid count `33/33`, updates `1/1`.
- Segmented: packed K/V `0/0`, recent `128`, pending `64`, K/V centroid count `32/32`, updates `0/0`.
- Interpretation: with an AIME prompt shorter than 128 tokens, legacy PatternKV packs after the prompt remainder plus generated tokens fill `residual_length`; segmented cache preserves the latest 128 non-sink tokens as recent, so the prompt remains pending at decode position 128. This is a real cache-layout cadence mismatch under the requested comparison.
- `LEVEL2_STRUCTURE_PASS=false`

## 9. Centroid Equivalence

- Not passed because centroid banks are structurally offset at the first checkpoint: legacy has already appended dynamic centroid 33, segmented remains at initial 32.

## 10. Assignment and V Gate Equivalence

- Not passed because segmented has no packed assignment/gate tokens at the first checkpoint while legacy has 128 packed assignment/gate tokens.

## 11. KV Reconstruction Equivalence

- Not evaluated as pass/fail after the structural mismatch. Comparing reconstructed KV after different packed/pending/recent partitioning would mix algorithmic cadence with numeric reconstruction.

## 12. Attention and Logits Equivalence

- Production logits were captured at 12 checkpoints. Top-1 agreement is not sufficient because the structural standard failed.
- Example first checkpoint p12/s0 decode=128: logits cosine `0.98165363073349`, top-1 agreement true, top-5 overlap 3.

## 13. Level 3 Greedy Result

- Status: skipped.
- Reason: Level 2 structural failure prohibits full approval and makes greedy divergence analysis secondary.
- `LEVEL3_PASS=false`

## 14. First-Divergence Analysis

- Greedy first-divergence was not run because Level 3 was skipped.
- Level 2 first mismatch is structural at checkpoint 128, layer 0 for task p12/s0.

## 15. Level 4 Sampling Sanity

- Status: skipped because Level 2/3 did not pass.
- `LEVEL4_SANITY_PASS=false`

## 16. Remaining Differences

- The segmented implementation honors the new `[sink][packed history][pending][recent]` invariant, but this layout is not cadence-equivalent to legacy PatternKV when prompt length is not already aligned to `residual_length`.
- A future equivalence strategy must either compare after a normalized prefill/cache construction or explicitly account for the prompt-remainder offset. No threshold should be relaxed to hide this structural difference.

## 17. Full-Run Decision

- Wave 1A full is not approved.
- The method remains `PatternKV_paper_segmented_candidate`.

LEVEL2_STRUCTURE_PASS=false
LEVEL2_REFERENCE_PASS=false
LEVEL2_PRODUCTION_PASS=false
LEVEL3_PASS=false
LEVEL4_SANITY_PASS=false
FULL_RUN_APPROVED=false
