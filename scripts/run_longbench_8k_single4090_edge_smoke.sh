#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/Bounded-pattrenKV-method}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct}"
cd "$PROJECT_ROOT"
mkdir -p results/paper_repro_v2/longbench_8k_4090_edge_smoke_data/data logs/paper_repro_v2/longbench_8k_4090_edge_smoke reports/paper_repro_v2/longbench_8k_4090_edge_smoke
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer
from bench.bench_longbench_patternkv import build_prompt, sample_id
from bench.longbench_config import SUBTASKS, MAX_NEW_TOKENS
model="/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct"
src=Path("/root/Block-kvcache-experiment/data/LongBench/data")
dst=Path("results/paper_repro_v2/longbench_8k_4090_edge_smoke_data/data")
tok=AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
skip={"trec","triviaqa","samsum","lsht","lcc","repobench-p"}
cands=[]
for task in SUBTASKS:
    for i,line in enumerate((src/f"{task}.jsonl").read_text(encoding="utf-8").splitlines()):
        ex=json.loads(line)
        prompt,_=build_prompt(ex,tok,task,8192,True)
        n=len(tok(prompt, add_special_tokens=task in skip).input_ids)
        if 7800 <= n <= 8187:
            cands.append((abs(8000-n), MAX_NEW_TOKENS[task], task, i, n, ex, sample_id(task,i,ex)))
best=sorted(cands)[0]
_, maxgen, task, idx, n, ex, sid=best
(dst/f"{task}.jsonl").write_text(json.dumps(ex,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
Path("reports/paper_repro_v2/longbench_8k_4090_edge_smoke/selection.json").write_text(json.dumps({"task":task,"index":idx,"sample_id":sid,"input_tokens":n,"max_gen":maxgen},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print(task,idx,sid,n,maxgen)
PY
task="$("$PYTHON_BIN" - <<'PY'
import json
print(json.load(open("reports/paper_repro_v2/longbench_8k_4090_edge_smoke/selection.json"))["task"])
PY
)"
for method in fp16 kivi_paper_g128 patternkv_paper; do
  CUDA_VISIBLE_DEVICES=0 MODEL_PATH="$MODEL_PATH" DATA_DIR=results/paper_repro_v2/longbench_8k_4090_edge_smoke_data OUTPUT_TAG=longbench_8k_4090_edge_smoke OUTPUT_DIR=results/paper_repro_v2/longbench_8k_4090_edge_smoke STATUS_DIR=run/paper_repro_v2/longbench_8k_4090_edge_smoke METHOD_FILTER="$method" TASK_FILTER="$task" RESUME=0 bash scripts/run_longbench_paper_8k_single4090.sh \
    > "logs/paper_repro_v2/longbench_8k_4090_edge_smoke/${method}.log" 2>&1
done
