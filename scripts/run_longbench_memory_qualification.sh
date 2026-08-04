#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/data/zypan/.local/share/mamba/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_repro_v2/longbench_memory_qualification}"
STATUS_DIR="${STATUS_DIR:-run/paper_repro_v2/longbench_memory_qualification}"
LOG_DIR="${LOG_DIR:-logs/paper_repro_v2/longbench_memory_qualification}"
REPORT_DIR="${REPORT_DIR:-reports/paper_repro_v2/longbench_memory_qualification}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-31500}"
METHODS="${METHOD_FILTER:-fp16 kivi_paper_g128 patternkv_paper}"
TASKS="${TASKS:-narrativeqa gov_report passage_retrieval_en lcc}"
mkdir -p "$OUTPUT_DIR" "$STATUS_DIR" "$LOG_DIR" "$REPORT_DIR"
rc=0
i=0
for method in $METHODS; do
  gpu="$i"
  log="${LOG_DIR}/${method}_gpu${gpu}.log"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" bench/bench_longbench_patternkv.py \
    --method "$method" \
    --tasks $TASKS \
    --num-samples 1 \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --status-dir "$STATUS_DIR" \
    --mode memory_qualification \
    --gpu-id "$gpu" \
    --max-input-length "$MAX_INPUT_LENGTH" \
    --skip-existing \
    > "$log" 2>&1 || rc=1
  i=$((i+1))
done
"$PYTHON_BIN" - <<'PY'
import json, math
from pathlib import Path
methods = "fp16 kivi_paper_g128 patternkv_paper".split()
tasks = "narrativeqa gov_report passage_retrieval_en lcc".split()
base = Path("results/paper_repro_v2/longbench_memory_qualification")
rows = []
for method in methods:
    for task in tasks:
        path = base / method / f"{task}.jsonl"
        recs = []
        if path.exists():
            recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rec = recs[0] if recs else {}
        rows.append({
            "method": method,
            "task": task,
            "success": bool(recs) and not rec.get("error") and bool(str(rec.get("prediction") or "").strip()),
            "raw_input_tokens": rec.get("raw_input_tokens"),
            "effective_input_tokens": rec.get("input_tokens"),
            "generated_tokens": rec.get("output_tokens"),
            "max_new_tokens": rec.get("max_new_tokens"),
            "final_peak_allocated": rec.get("peak_allocated_bytes"),
            "final_peak_reserved": rec.get("peak_reserved_bytes"),
            "oom_stage": "generate_or_prefill" if rec.get("error") and "OutOfMemory" in rec.get("error") else None,
            "exception": rec.get("error"),
            "attention_backend": "model_default_or_flash_if_custom_model",
        })
status = "STRICT_QUALIFICATION_PASS" if rows and all(r["success"] for r in rows) else "STRICT_QUALIFICATION_FAIL"
out = {"status": status, "rows": rows, "note": "Qualification uses representative first samples for requested tasks and tokenizer-based truncation in the runner; failures block strict full run."}
report_dir = Path("reports/paper_repro_v2/longbench_memory_qualification")
report_dir.mkdir(parents=True, exist_ok=True)
(report_dir / "report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
lines = ["# LongBench Memory Qualification", "", f"Status: {status}", "", "| method | task | success | raw input | effective input | generated | peak reserved | oom_stage |", "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
for r in rows:
    lines.append(f"| {r['method']} | {r['task']} | {r['success']} | {r['raw_input_tokens']} | {r['effective_input_tokens']} | {r['generated_tokens']} | {r['final_peak_reserved']} | {r['oom_stage']} |")
lines += ["", "```json", json.dumps(out, indent=2, ensure_ascii=False), "```"]
(report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if status == "STRICT_QUALIFICATION_PASS" else 1)
PY
exit "$rc"
