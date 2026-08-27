```bash
python scripts/run_gsm8k_selector_truncation_sensitivity.py freeze-union
CUDA_VISIBLE_DEVICES=1 python scripts/run_gsm8k_selector_truncation_sensitivity.py run --phase formal --method causal_v4_25 --physical-gpu-id 1 --output-dir results/gsm8k_selector_truncation_sensitivity_v1/formal --max-new-tokens 8192
python scripts/run_gsm8k_selector_truncation_sensitivity.py summarize
```
