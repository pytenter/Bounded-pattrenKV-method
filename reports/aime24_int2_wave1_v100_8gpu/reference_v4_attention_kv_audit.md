# Reference v4 Attention/KV Audit

## 1. Purpose

Collect scalar reconstructed K/V and per-layer reference attention metrics needed for final launch approval without saving long-lived activation tensors.

## 2. Git/environment

- Branch: `exp/aime-int2-wave1-v100-8gpu`
- Starting HEAD: `42294b01b9f31743ec82885c560b769bf3bc7f6f`
- Python: `/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python`
- Model: `/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B`
- Torch/CUDA: `2.4.1+cu118` / `11.8`

## 3. Reference v3 inherited evidence

Reference v3 already established no first mismatch, worst logits cosine `0.9999936819076538`, and max teacher-token NLL difference `0.0`.

## 4. Reference v4 metric collection

- Tasks: p12 and p14
- Checkpoints: 128, 256, 512, 1024, 2048, 4096 generated tokens
- Metric layers: 0, 7, 15, 23, 31
- Output directory: `reports/aime24_int2_wave1_v100_8gpu/equivalence_chunked_level2/reference_v4`

## 5. K reconstruction

- Rows: 60
- Worst cosine: `0.999982059002` at aime24:p14:s0:seed14042 @ checkpoint 512 layer 23
- Worst relative MSE: `0`
- Worst max abs: `0`

## 6. V reconstruction

- Rows: 60
- Worst cosine: `0.999967694283` at aime24:p14:s0:seed14042 @ checkpoint 1024 layer 15
- Worst relative MSE: `0`
- Worst max abs: `0`

## 7. Attention score

- Rows: 60
- Worst finite-position cosine: `0.999998807907` at aime24:p14:s0:seed14042 @ checkpoint 1024 layer 7

## 8. Attention probability KL

- Rows: 60
- Worst symmetric KL: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 0

## 9. Attention output

- Rows: 60
- Worst cosine: `0.999998986721` at aime24:p12:s0:seed12042 @ checkpoint 4096 layer 0
- Worst relative MSE: `0`

## 10. Post-o-proj

- Rows: 60
- Worst cosine: `0.999999284744` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 23

## 11. Logits/NLL

- Worst logits cosine: `0.999993681908` at aime24:p12:s0:seed12042 @ checkpoint 128 layer None
- Max teacher-token NLL abs difference: `0`
- Top-1 agreement: `100%`

## 12. Worst-case locations

- worst_k_cosine: `0.999982059002` at aime24:p14:s0:seed14042 @ checkpoint 512 layer 23
- worst_k_relative_mse: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 0
- worst_k_max_abs: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 0
- worst_v_cosine: `0.999967694283` at aime24:p14:s0:seed14042 @ checkpoint 1024 layer 15
- worst_v_relative_mse: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 0
- worst_v_max_abs: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 0
- worst_score_cosine: `0.999998807907` at aime24:p14:s0:seed14042 @ checkpoint 1024 layer 7
- worst_prob_symmetric_kl: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 0
- worst_output_cosine: `0.999998986721` at aime24:p12:s0:seed12042 @ checkpoint 4096 layer 0
- worst_output_relative_mse: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 0
- worst_post_cosine: `0.999999284744` at aime24:p12:s0:seed12042 @ checkpoint 128 layer 23
- worst_logits_cosine: `0.999993681908` at aime24:p12:s0:seed12042 @ checkpoint 128 layer None
- max_nll: `0` at aime24:p12:s0:seed12042 @ checkpoint 128 layer None

## 13. Threshold table

| Metric | Threshold | Worst result | Pass |
| --- | ---: | ---: | --- |
| Structure mismatch | `0` | `0` | PASS |
| K assignment mismatch | `0` | `0` | PASS |
| V assignment mismatch | `0` | `0` | PASS |
| V gate mismatch | `0` | `0` | PASS |
| Reconstructed K cosine | `>=0.9999` | `0.999982059002` | PASS |
| Reconstructed V cosine | `>=0.9999` | `0.999967694283` | PASS |
| Attention score cosine | `>=0.9999` | `0.999998807907` | PASS |
| Attention symmetric KL | `<=1e-4` | `0` | PASS |
| Attention output cosine | `>=0.9999` | `0.999998986721` | PASS |
| Post-o-proj cosine | `>=0.9999` | `0.999999284744` | PASS |
| Logits cosine | `>=0.9999` | `0.999993681908` | PASS |
| Top-1 agreement | `100%` | `100%` | PASS |
| Max teacher NLL abs difference | `<=0.01` | `0` | PASS |
| Rolling smoke errors | `0` | `0` | PASS |
| Rolling long-smoke errors | `0` | `0` | PASS |

## 14. Approval decision

- FULL_RUN_APPROVED=true
- LAUNCH_READINESS_STATUS=approved

## 15. Remaining caveats

- Revised Wave 1A full was not launched in this audit.
- Mixed-Key remains blocked and is not part of Wave 1A.
