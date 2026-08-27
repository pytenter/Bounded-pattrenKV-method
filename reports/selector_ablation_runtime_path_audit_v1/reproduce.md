# Reproduce

```bash
export CUDA_VISIBLE_DEVICES=""
python scripts/audit_selector_ablation_runtime.py --active-root /data/zypan/Bounded-pattrenKV-pseudodecode-3090
python -m compileall scripts/audit_selector_ablation_runtime.py scripts/run_selector_runtime_diagnostic.py tests/test_selector_ablation_runtime_audit.py
pytest -q tests/test_selector_ablation_runtime_audit.py
git diff --check
```
