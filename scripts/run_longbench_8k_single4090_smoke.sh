#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/root/Bounded-pattrenKV-method}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/patternkv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-/root/Block-kvcache-experiment/data/LongBench}"
cd "$PROJECT_ROOT"
mkdir -p results/paper_repro_v2/longbench_8k_4090_smoke_data/data logs/paper_repro_v2/longbench_8k_4090_smoke reports/paper_repro_v2/longbench_8k_4090_smoke
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer
from bench.bench_longbench_patternkv import build_prompt, sample_id
model="/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct"
src=Path("/root/Block-kvcache-experiment/data/LongBench/data")
dst=Path("results/paper_repro_v2/longbench_8k_4090_smoke_data/data")
tok=AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)
skip={"trec","triviaqa","samsum","lsht","lcc","repobench-p"}
for task in ["trec","samsum","passage_count"]:
    best=None
    for i,line in enumerate((src/f"{task}.jsonl").read_text(encoding="utf-8").splitlines()):
        ex=json.loads(line)
        prompt,_=build_prompt(ex,tok,task,8192,True)
        n=len(tok(prompt, add_special_tokens=task in skip).input_ids)
        if n <= 8192 and (best is None or n < best[0]):
            best=(n,i,ex,sample_id(task,i,ex))
    (dst/f"{task}.jsonl").write_text(json.dumps(best[2],ensure_ascii=False,sort_keys=True)+"\n",encoding="utf-8")
    print(task,best[1],best[3],best[0])
PY
for method in fp16 kivi_paper_g128 patternkv_paper; do
  CUDA_VISIBLE_DEVICES=0 MODEL_PATH="$MODEL_PATH" DATA_DIR=results/paper_repro_v2/longbench_8k_4090_smoke_data OUTPUT_TAG=longbench_8k_4090_smoke OUTPUT_DIR=results/paper_repro_v2/longbench_8k_4090_smoke STATUS_DIR=run/paper_repro_v2/longbench_8k_4090_smoke METHOD_FILTER="$method" TASK_FILTER="trec samsum passage_count" RESUME=0 bash scripts/run_longbench_paper_8k_single4090.sh \
    > "logs/paper_repro_v2/longbench_8k_4090_smoke/${method}.log" 2>&1
done
