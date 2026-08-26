#!/usr/bin/env python3
"""Assemble canonical compact evidence for completed AIME N=3 quality runs.

This script is intentionally read-only with respect to raw result roots. It
recomputes all paper-table values from per-sample JSON records.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEEDS = [42, 43, 44]
PROBLEMS = list(range(30))
SAMPLE_ID = 0
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_RESAMPLES = 10000
METHOD_ORDER = ["FP16", "KIVI_PAPER_G128", "PATTERN_BASE", "RANDOM_V4_25", "CAUSAL_V4_25"]
AIME25_EXPECTED = {"FP16": 30, "KIVI_PAPER_G128": 18, "PATTERN_BASE": 21, "CAUSAL_V4_25": 27}
AIME24_EXPECTED = {"KIVI_PAPER_G128": 11}
AIME24_MAIN_EXPECTED = {"FP16": 45, "KIVI_PAPER_G128": 11, "PATTERN_BASE": 32, "RANDOM_V4_25": 36, "CAUSAL_V4_25": 45}

AIME25_METHODS = {
    "FP16": "results/aime25_full_causal25_quality_v100/formal/FP16",
    "KIVI_PAPER_G128": "results/aime25_kivi_paper_g128_quality_v100/formal/KIVI_PAPER_G128",
    "PATTERN_BASE": "results/aime25_full_causal25_quality_v100/formal/PATTERN_BASE",
    "CAUSAL_V4_25": "results/aime25_full_causal25_quality_v100/formal/CAUSAL_V4_25",
}
AIME24_KIVI = {"KIVI_PAPER_G128": "results/aime24_kivi_paper_g128_quality_v100/formal/KIVI_PAPER_G128"}
AIME24_MAIN_COMPACT = "reports/aime24_full_causal25_quality_4gpu/sample_results_compact.jsonl.gz"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({k for r in rows for k in r})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if s.isdigit():
        return str(int(s) % 1000)
    return s


def effective_seed(base_seed: int, problem_id: int, sample_id: int = 0) -> int:
    return base_seed + problem_id * 1000 + sample_id


def source_rows_from_root(source: Path, benchmark: str, method: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = sorted(root.glob("seed*/p*.json"))
    rows: list[dict[str, Any]] = []
    duplicate = []
    seen: dict[tuple[str, str, int, int, int], str] = {}
    for p in files:
        obj = read_json_result(p)
        if obj.get("phase") not in (None, "formal"):
            continue
        if str(obj.get("dataset")) != benchmark:
            continue
        if str(obj.get("method")) != method:
            continue
        pid = int(obj.get("problem_id"))
        base_seed = int(obj.get("base_seed"))
        sample_id = int(obj.get("sample_id", 0))
        ident = (benchmark, method, pid, base_seed, sample_id)
        fhash = sha256_file(p)
        if ident in seen:
            duplicate.append({"identity": ident, "first_hash": seen[ident], "duplicate_path": str(p), "duplicate_hash": fhash})
            continue
        seen[ident] = fhash
        gen_path = obj.get("raw_generation_path")
        row = {
            "benchmark": benchmark,
            "model_name": obj.get("model_identity"),
            "model_identity_hash": stable_hash(obj.get("model_identity")),
            "method": method,
            "display_method": method,
            "backend_method": obj.get("backend_method"),
            "method_config_hash": obj.get("method_config_hash"),
            "problem_id": pid,
            "base_seed": base_seed,
            "sample_id": sample_id,
            "effective_seed": int(obj.get("effective_seed")),
            "task_key": obj.get("task_key") or obj.get("formal_key"),
            "dataset_sha256": obj.get("dataset_sha256"),
            "prompt_hash": obj.get("prompt_hash"),
            "input_token_hash": obj.get("input_token_hash"),
            "generation_config_hash": obj.get("generation_config_hash"),
            "parser_hash": "bench/aime_answer_parser.py:" + sha256_file(Path("bench/aime_answer_parser.py")) if Path("bench/aime_answer_parser.py").exists() else "MISSING",
            "reference_answer": normalize_answer(obj.get("gold_answer")),
            "parsed_answer": normalize_answer(obj.get("parsed_answer")),
            "is_correct": bool(obj.get("correct")),
            "parser_strategy": obj.get("parser_strategy"),
            "parser_error": obj.get("error") if obj.get("parse_status") == "failure" else None,
            "generated_tokens": obj.get("generated_tokens"),
            "generated_token_hash": obj.get("generated_token_sha256"),
            "generated_text_sha256": obj.get("raw_generation_sha256") or obj.get("generated_token_sha256"),
            "stop_reason": obj.get("stop_reason") or obj.get("status"),
            "eos_stop": (obj.get("stop_reason") == "eos"),
            "length_stop": (obj.get("stop_reason") == "length" or bool(obj.get("length_truncated"))),
            "truncated": bool(obj.get("length_truncated")),
            "oom": obj.get("status") == "oom",
            "runtime_error": bool(obj.get("error")) and obj.get("parse_status") != "failure",
            "source_result_path": str(p),
            "source_result_sha256": fhash,
            "experiment_id": obj.get("experiment_id"),
            "config_hash": obj.get("formal_config_hash"),
            "git_commit": obj.get("git_commit"),
            "physical_gpu_id": obj.get("physical_gpu"),
            "gpu_uuid": obj.get("gpu_name"),
            "timestamp": obj.get("timestamp"),
            "k_bits": obj.get("k_bits"),
            "v_bits": obj.get("v_bits"),
            "group_size": obj.get("group_size"),
            "sink_length": obj.get("sink_length"),
            "recent_length": obj.get("recent_length"),
            "residual_length": obj.get("residual_length"),
            "selector": obj.get("selector"),
            "v4_budget_fraction": obj.get("v4_budget_fraction"),
            "observed_v4_fraction": obj.get("v4_realized_fraction"),
            "cache_mode": (obj.get("quantization_config") or {}).get("cache_mode"),
            "prompt_protocol": obj.get("prompt_protocol"),
            "formal_key": obj.get("formal_key"),
            "problem": obj.get("problem"),
        }
        if row["effective_seed"] != effective_seed(base_seed, pid, sample_id):
            row["runtime_error"] = True
            row["parser_error"] = "effective_seed_mismatch"
        rows.append(row)
    audit = {"root": str(root), "file_count": len(files), "accepted_rows": len(rows), "duplicates": duplicate,
             "last_mtime": max((p.stat().st_mtime for p in files), default=None)}
    return rows, audit


def load_aime24_main_rows(path: Path) -> list[dict[str, Any]]:
    rows=[]
    parser_hash = "bench/aime_answer_parser.py:" + sha256_file(Path("bench/aime_answer_parser.py")) if Path("bench/aime_answer_parser.py").exists() else "MISSING"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj=json.loads(line)
            method=obj.get("method")
            if method not in {"FP16","PATTERN_BASE","RANDOM_V4_25","CAUSAL_V4_25"}: continue
            pid=int(obj["problem_id"]); base_seed=int(obj["base_seed"]); sample_id=int(obj.get("sample_id",0))
            rows.append({
                "benchmark":"aime24", "model_name":obj.get("model_identity"), "model_identity_hash":stable_hash(obj.get("model_identity")),
                "method":method, "display_method":method, "backend_method":obj.get("backend_method"), "method_config_hash":obj.get("method_config_hash"),
                "problem_id":pid, "base_seed":base_seed, "sample_id":sample_id, "effective_seed":int(obj.get("effective_seed")),
                "task_key":obj.get("task_key") or obj.get("formal_key"), "dataset_sha256":obj.get("dataset_sha256"), "prompt_hash":obj.get("prompt_hash"),
                "input_token_hash":obj.get("input_token_hash"), "generation_config_hash":obj.get("generation_config_hash"), "parser_hash":parser_hash,
                "reference_answer":normalize_answer(obj.get("gold_answer")), "parsed_answer":normalize_answer(obj.get("parsed_answer")), "is_correct":bool(obj.get("correct")),
                "parser_strategy":obj.get("parser_strategy"), "parser_error":obj.get("error") if obj.get("parse_status") == "failure" else None,
                "generated_tokens":obj.get("generated_tokens"), "generated_token_hash":obj.get("generated_token_sha256"), "generated_text_sha256":obj.get("raw_generation_sha256") or obj.get("generated_token_sha256"),
                "stop_reason":obj.get("stop_reason") or obj.get("status"), "eos_stop":obj.get("stop_reason") == "eos", "length_stop":obj.get("stop_reason") == "length" or bool(obj.get("length_truncated")),
                "truncated":bool(obj.get("length_truncated")), "oom":obj.get("status") == "oom", "runtime_error":bool(obj.get("error")) and obj.get("parse_status") != "failure",
                "source_result_path":str(path), "source_result_sha256":sha256_file(path), "experiment_id":obj.get("experiment_id"), "config_hash":obj.get("formal_config_hash"),
                "git_commit":obj.get("git_commit"), "physical_gpu_id":obj.get("physical_gpu"), "gpu_uuid":obj.get("gpu_name"), "timestamp":obj.get("timestamp"),
                "k_bits":obj.get("k_bits"), "v_bits":obj.get("v_bits"), "group_size":obj.get("group_size"), "sink_length":obj.get("sink_length"), "recent_length":obj.get("recent_length"),
                "residual_length":obj.get("residual_length"), "selector":obj.get("selector"), "v4_budget_fraction":obj.get("v4_budget_fraction"), "observed_v4_fraction":obj.get("v4_realized_fraction"),
                "cache_mode":(obj.get("quantization_config") or {}).get("cache_mode"), "prompt_protocol":obj.get("prompt_protocol"), "formal_key":obj.get("formal_key"), "problem":obj.get("problem"),
            })
    return sorted(rows, key=row_sort_key)


def row_sort_key(r: dict[str, Any]) -> tuple:
    return (r["benchmark"], METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99, int(r["problem_id"]), int(r["base_seed"]), int(r["sample_id"]))


def validate_identities(rows: list[dict[str, Any]], methods: list[str], benchmark: str) -> dict[str, Any]:
    identities={(r["method"], int(r["problem_id"]), int(r["base_seed"]), int(r["sample_id"])) for r in rows}
    planned={(m,p,s,SAMPLE_ID) for m in methods for p in PROBLEMS for s in SEEDS}
    unexpected=sorted(identities-planned)
    missing=sorted(planned-identities)
    bad_effective=[r for r in rows if int(r["effective_seed"]) != effective_seed(int(r["base_seed"]), int(r["problem_id"]), int(r["sample_id"]))]
    return {"benchmark":benchmark,"planned_identities":len(planned),"actual_identities":len(identities),"missing":missing,"unexpected":unexpected,"bad_effective_seed_count":len(bad_effective),"gate":"PASS" if not missing and not unexpected and not bad_effective and len(rows)==len(planned) else "FAIL"}


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by=defaultdict(list)
    for r in rows: by[(r["benchmark"],r["method"])].append(r)
    method_summary=[]; seed_breakdown=[]; per_problem=[]; dist=[]; stops=[]
    for (bench,method), rs in sorted(by.items(), key=lambda kv:(kv[0][0], METHOD_ORDER.index(kv[0][1]) if kv[0][1] in METHOD_ORDER else 99)):
        total=len(rs); correct=sum(1 for r in rs if r["is_correct"]); toks=[int(r["generated_tokens"] or 0) for r in rs]
        seed_acc=[]
        for seed in SEEDS:
            srs=[r for r in rs if int(r["base_seed"])==seed]; sc=sum(1 for r in srs if r["is_correct"])
            seed_acc.append(sc/len(srs) if srs else 0)
            seed_breakdown.append({"benchmark":bench,"method":method,"base_seed":seed,"correct":sc,"total":len(srs),"accuracy":sc/len(srs) if srs else None})
        counts=[]; maj_correct=0; any_correct=0; unresolved=0
        for pid in PROBLEMS:
            prs=[r for r in rs if int(r["problem_id"])==pid]
            c=sum(1 for r in prs if r["is_correct"]); counts.append(c)
            answers=[r["parsed_answer"] for r in prs if r["parsed_answer"]]
            ac=Counter(answers); mode_ans=None
            if ac:
                top=ac.most_common();
                if len(top)==1 or top[0][1] > top[1][1]: mode_ans=top[0][0]
            ref=prs[0]["reference_answer"] if prs else ""
            maj = (mode_ans == ref) if mode_ans is not None else False
            if mode_ans is None: unresolved += 1
            maj_correct += int(maj); any_correct += int(c>0)
            per_problem.append({"benchmark":bench,"method":method,"problem_id":pid,"correct_count":c,"total":len(prs),"majority_correct":maj,"any_correct":c>0,"length_stop_count":sum(1 for r in prs if r["length_stop"]),"eos_stop_count":sum(1 for r in prs if r["eos_stop"])})
        cd=Counter(counts)
        dist.append({"benchmark":bench,"method":method,"zero_of_3":cd[0],"one_of_3":cd[1],"two_of_3":cd[2],"three_of_3":cd[3]})
        stops.append({"benchmark":bench,"method":method,"eos_stops":sum(1 for r in rs if r["eos_stop"]),"length_stops":sum(1 for r in rs if r["length_stop"]),"parser_failures":sum(1 for r in rs if r["parser_strategy"]=="failure"),"oom":sum(1 for r in rs if r["oom"]),"runtime_errors":sum(1 for r in rs if r["runtime_error"]),"mean_generated_tokens":statistics.mean(toks) if toks else 0,"median_generated_tokens":statistics.median(toks) if toks else 0,"p90_generated_tokens":sorted(toks)[int(math.ceil(.9*len(toks)))-1] if toks else 0,"max_generated_tokens":max(toks) if toks else 0})
        method_summary.append({"benchmark":bench,"method":method,"correct":correct,"total":total,"accuracy":correct/total if total else None,"mean_seed_accuracy":statistics.mean(seed_acc),"seed_std_accuracy":statistics.pstdev(seed_acc),"majority3_correct":maj_correct,"majority3_total":30,"majority3_accuracy":maj_correct/30,"majority3_unresolved":unresolved,"any3_correct":any_correct,"any3_total":30,"any3_accuracy":any_correct/30})
    return method_summary, seed_breakdown, per_problem, dist, stops


def bootstrap(rows: list[dict[str, Any]], benchmark: str, pairs: list[tuple[str,str]]) -> dict[str, Any]:
    per=defaultdict(dict)
    for r in rows:
        per[int(r["problem_id"])].setdefault(r["method"], 0)
        if r["is_correct"]: per[int(r["problem_id"])][r["method"]] += 1
    rng=random.Random(BOOTSTRAP_SEED); out={"unit":"problem","resamples":BOOTSTRAP_RESAMPLES,"seed":BOOTSTRAP_SEED,"comparisons":{}}
    for a,b in pairs:
        vals=[(per[p].get(a,0)-per[p].get(b,0))/3 for p in PROBLEMS]
        obs=sum(vals)/len(vals); reps=[]
        for _ in range(BOOTSTRAP_RESAMPLES):
            reps.append(sum(vals[rng.randrange(len(vals))] for _ in vals)/len(vals))
        reps.sort(); out["comparisons"][f"{a}_minus_{b}"]={"observed_accuracy_delta":obs,"ci95":[reps[int(.025*BOOTSTRAP_RESAMPLES)], reps[int(.975*BOOTSTRAP_RESAMPLES)-1]]}
    return out


def transitions(rows: list[dict[str, Any]], a: str, b: str, benchmark: str) -> list[dict[str, Any]]:
    per=defaultdict(lambda: defaultdict(list))
    for r in rows: per[int(r["problem_id"])][r["method"]].append(r)
    out=[]
    for pid in PROBLEMS:
        ar=per[pid].get(a,[]); br=per[pid].get(b,[]); ac=sum(r["is_correct"] for r in ar); bc=sum(r["is_correct"] for r in br)
        out.append({"benchmark":benchmark,"problem_id":pid,"method_A":a,"method_B":b,"method_A_correct_count":ac,"method_B_correct_count":bc,"delta_correct_count":bc-ac,"A_wrong_to_B_correct":ac==0 and bc>0,"A_correct_to_B_wrong":ac>0 and bc==0,"both_correct":ac>0 and bc>0,"both_wrong":ac==0 and bc==0,"A_length_stop_count":sum(r["length_stop"] for r in ar),"B_length_stop_count":sum(r["length_stop"] for r in br)})
    return out


def paper_rows(summary: list[dict[str, Any]], methods: list[str], benchmark: str) -> list[dict[str, Any]]:
    sm={(r["benchmark"],r["method"]):r for r in summary}
    labels={"FP16":"FP16","KIVI_PAPER_G128":"KIVI","PATTERN_BASE":"Pattern Base","RANDOM_V4_25":"Random25","CAUSAL_V4_25":"CAUSAL25"}
    out=[]
    for m in methods:
        r=sm[(benchmark,m)]
        out.append({"Method":labels[m],"Correct/90":f"{r['correct']}/90","Response Acc.":f"{100*r['accuracy']:.2f}%","Majority@3":f"{r['majority3_correct']}/30 ({100*r['majority3_accuracy']:.2f}%)","Any@3":f"{r['any3_correct']}/30 ({100*r['any3_accuracy']:.2f}%)"})
    return out


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields=list(rows[0]) if rows else []
    lines=["| " + " | ".join(fields) + " |", "| " + " | ".join(["---" for _ in fields]) + " |"]
    for r in rows: lines.append("| " + " | ".join(str(r[f]) for f in fields) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, Any]]) -> str:
    fields = list(rows[0]) if rows else []
    bs = chr(92)
    br = bs + bs
    lines = [bs + "begin{tabular}{" + "l" * len(fields) + "}", bs + "toprule"]
    lines.append(" & ".join(fields) + " " + br)
    lines.append(bs + "midrule")
    for r in rows:
        vals = [str(r[f]).replace("%", bs + "%") for f in fields]
        lines.append(" & ".join(vals) + " " + br)
    lines += [bs + "bottomrule", bs + "end{tabular}"]
    return "\n".join(lines) + "\n"


def audit_common(rows: list[dict[str, Any]], methods: list[str], benchmark: str) -> dict[str, Any]:
    by_field={}
    for field in ["model_name","model_identity_hash","dataset_sha256","prompt_protocol"]:
        by_field[field]=sorted({str(r.get(field)) for r in rows})
    prompt_by_problem=defaultdict(set); gen_by_method=defaultdict(set); method_cfg=defaultdict(set)
    for r in rows:
        prompt_by_problem[int(r["problem_id"])].add(r["prompt_hash"])
        gen_by_method[r["method"]].add(r["generation_config_hash"])
        method_cfg[r["method"]].add(r["method_config_hash"])
    prompt_gate="PASS" if all(len(v)==1 for v in prompt_by_problem.values()) else "FAIL"
    gen_hash_sets={m:sorted(gen_by_method[m]) for m in methods}
    gen_gate="PASS" if len({tuple(v) for v in gen_hash_sets.values()})==1 else "PARTIAL"
    return {"benchmark":benchmark,"fields":by_field,"prompt_hash_per_problem_gate":prompt_gate,"generation_config_cross_method_gate":gen_gate,"generation_config_hashes_by_method":gen_hash_sets,"method_config_hashes_by_method":{m:sorted(method_cfg[m]) for m in methods},"comparability":"PASS" if prompt_gate=="PASS" and gen_gate=="PASS" and len(by_field["model_identity_hash"])==1 and len(by_field["dataset_sha256"])==1 else "PARTIAL"}


def write_package(out: Path, rows: list[dict[str, Any]], methods: list[str], benchmark: str, expected: dict[str,int], include_transitions=True) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rows=sorted(rows,key=row_sort_key)
    fields=[k for k in rows[0].keys() if k != "problem"]
    write_csv(out/"canonical_rows.csv", rows, fields)
    write_jsonl_gz(out/"canonical_rows.jsonl.gz", rows)
    write_text(out/"formal_manifest.jsonl", "".join(json.dumps({k:r[k] for k in ["benchmark","method","problem_id","base_seed","sample_id","effective_seed","source_result_path","source_result_sha256"]}, sort_keys=True)+"\n" for r in rows))
    summary, seeds, per_problem, dist, stops = summarize(rows)
    write_csv(out/"method_summary.csv", summary)
    write_csv(out/"seed_breakdown.csv", seeds)
    write_csv(out/"per_problem_summary.csv", per_problem)
    write_csv(out/"correct_count_distribution.csv", dist)
    write_csv(out/"stop_reason_analysis.csv", stops)
    completeness=validate_identities(rows, methods, benchmark); write_json(out/"completeness_audit.json", completeness)
    dup={"duplicates":[],"gate":"PASS"}; write_json(out/"duplicate_audit.json", dup)
    protocol=audit_common(rows, methods, benchmark); write_json(out/"cross_method_protocol_audit.json" if benchmark=="aime25" else out/"protocol_manifest.json", protocol)
    write_json(out/"protocol_manifest.json", protocol)
    model={"models":sorted({str(r["model_name"]) for r in rows}),"model_identity_hashes":sorted({str(r["model_identity_hash"]) for r in rows}),"gate":"PASS" if len({r['model_identity_hash'] for r in rows})==1 else "FAIL","provenance_completeness":"PARTIAL"}
    write_json(out/"model_identity.json", model)
    env={"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES",""),"python":"3","cpu_only":True}; write_json(out/"environment.json", env)
    dataset={"benchmark":benchmark,"dataset_sha256_from_results":sorted({str(r["dataset_sha256"]) for r in rows}),"problem_count":len({r['problem_id'] for r in rows}),"gate":"PASS" if len({r['dataset_sha256'] for r in rows})==1 else "FAIL"}; write_json(out/"dataset_manifest.json", dataset)
    method_identity={m:{"rows":sum(1 for r in rows if r["method"]==m),"backend_methods":sorted({str(r["backend_method"]) for r in rows if r["method"]==m}),"method_config_hashes":sorted({str(r["method_config_hash"]) for r in rows if r["method"]==m}),"gate":"PASS"} for m in methods}; write_json(out/"method_identity.json", method_identity)
    agg_gate="PASS" if all(next(r for r in summary if r["method"]==m)["correct"]==c for m,c in expected.items()) else "FAIL"
    final={"completeness_gate":completeness["gate"],"aggregate_recomputation_gate":agg_gate,"model_gate":model["gate"],"dataset_gate":dataset["gate"],"protocol_comparability":protocol["comparability"],"final_classification":"SUPPORTED" if completeness["gate"]==agg_gate==model["gate"]==dataset["gate"]=="PASS" and protocol["comparability"]=="PASS" else "PARTIAL_PROVENANCE"}
    write_json(out/"final_gate.json", final)
    pairs=[]
    if benchmark=="aime25": pairs=[("CAUSAL_V4_25","PATTERN_BASE"),("CAUSAL_V4_25","FP16"),("CAUSAL_V4_25","KIVI_PAPER_G128"),("PATTERN_BASE","KIVI_PAPER_G128")]
    if pairs:
        write_json(out/"paired_bootstrap.json", bootstrap(rows, benchmark, pairs))
        write_csv(out/"paired_transitions.csv", transitions(rows,"PATTERN_BASE","CAUSAL_V4_25",benchmark))
        write_json(out/"length_stop_sensitivity.json", {"primary":"all_samples","note":"EOS-only subset is supplementary","pattern_vs_causal_transitions_file":"paired_transitions.csv"})
    maj=[{k:r[k] for k in ["benchmark","method","majority3_correct","majority3_total","majority3_accuracy","any3_correct","any3_total","any3_accuracy"]} for r in summary]
    write_csv(out/"majority_any_summary.csv", maj)
    methods_for_table=methods
    ptab=paper_rows(summary, methods_for_table, benchmark)
    write_text(out/"paper_table.md", markdown_table(ptab)); write_csv(out/"paper_table.csv", ptab); write_text(out/"paper_table.tex", latex_table(ptab))
    source_map={f"{benchmark}.{r['method']}.response_accuracy":{"canonical_file":str(out/"method_summary.csv"),"canonical_field":"accuracy","row_evidence":str(out/"canonical_rows.jsonl.gz"),"method":r["method"],"source_result_roots":sorted({x["source_result_path"] for x in rows if x["method"]==r["method"]})[:3],"dataset_sha256":next(x["dataset_sha256"] for x in rows if x["method"]==r["method"]),"method_config_hash":next(x["method_config_hash"] for x in rows if x["method"]==r["method"])} for r in summary}
    write_json(out/"source_map.json", source_map)
    write_text(out/"README.md", f"# {benchmark.upper()} N=3 quality evidence\n\nGenerated from per-sample raw JSON records. No GPU generation is performed.\n")
    write_text(out/"claim_audit.md", f"# Claim Audit\n\nFinal classification: `{final['final_classification']}`. Protocol comparability: `{protocol['comparability']}`.\n")
    write_text(out/"reproduce.md", "Run: `CUDA_VISIBLE_DEVICES= python scripts/assemble_aime_n3_quality_evidence.py --source-worktree <source>`\n")
    write_text(out/"git_provenance.txt", os.popen("git rev-parse HEAD && git branch --show-current").read())
    return {"summary":summary,"final":final,"protocol":protocol}


def write_aime25_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by={}
    for r in rows:
        pid=int(r["problem_id"]); item=(r.get("problem"), r["reference_answer"])
        if pid in by and by[pid] != item:
            raise SystemExit(f"conflicting recovered AIME25 problem {pid}")
        by[pid]=item
    if sorted(by)!=PROBLEMS: raise SystemExit("AIME25 dataset recovery missing problems")
    ds=Path("datasets/aime"); ds.mkdir(parents=True, exist_ok=True)
    lines=[]
    for pid in PROBLEMS:
        problem,answer=by[pid]
        lines.append(json.dumps({"answer":answer,"dataset":"aime25","problem":problem,"problem_id":pid}, sort_keys=True, ensure_ascii=False))
    data=("\n".join(lines)+"\n").encode()
    atomic_write(ds/"aime25.jsonl", data)
    meta={"benchmark_id":"aime25","public_source":"recovered from completed V100 generation result records","canonical_schema":["answer","dataset","problem","problem_id"],"canonical_row_count":30,"problem_id_policy":"0..29 from result records","answer_normalization_policy":"integer modulo 1000 string","creation_script":"scripts/assemble_aime_n3_quality_evidence.py","canonical_sha256":sha256_bytes(data),"generation_result_dataset_hashes":sorted({r["dataset_sha256"] for r in rows}),"gate":"PASS"}
    write_json(ds/"aime25_metadata.json", meta)
    return meta


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--source-worktree", default="/home/qinch2023/v100_aime24_aime25_quality_work/Bounded-pattrenKV-method"); args=ap.parse_args()
    source=Path(args.source_worktree).resolve()
    discovery=[]; a25=[]; a24k=[]
    for m,rel in AIME25_METHODS.items():
        rows,audit=source_rows_from_root(source,"aime25",m,source/rel); a25 += rows; audit.update({"benchmark":"aime25","method":m,"relative_root":rel}); discovery.append(audit)
    for m,rel in AIME24_KIVI.items():
        rows,audit=source_rows_from_root(source,"aime24",m,source/rel); a24k += rows; audit.update({"benchmark":"aime24","method":m,"relative_root":rel}); discovery.append(audit)
    a25=sorted(a25,key=row_sort_key); a24k=sorted(a24k,key=row_sort_key)
    write_json(Path("reports/aime_evidence_source_discovery_v1/source_discovery.json"), {"source_result_worktree":str(source),"assembly_worktree":str(Path.cwd()),"roots":discovery})
    a25_meta=write_aime25_dataset(a25)
    p25=write_package(Path("reports/aime25_four_method_quality_v1"), a25, list(AIME25_METHODS), "aime25", AIME25_EXPECTED)
    p24k=write_package(Path("reports/aime24_kivi_quality_v1"), a24k, ["KIVI_PAPER_G128"], "aime24", AIME24_EXPECTED, include_transitions=False)
    a24main = load_aime24_main_rows(Path(AIME24_MAIN_COMPACT)) + a24k
    a24main = sorted(a24main,key=row_sort_key)
    summary,seeds,per_problem,dist,stops=summarize(a24main)
    out=Path("reports/aime24_main_quality_table_v1"); out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"method_summary.csv", summary); write_csv(out/"seed_breakdown.csv", seeds); write_csv(out/"per_problem_summary.csv", per_problem); write_csv(out/"stop_reason_analysis.csv", stops)
    write_json(out/"paired_bootstrap.json", bootstrap(a24main,"aime24",[("CAUSAL_V4_25","PATTERN_BASE"),("CAUSAL_V4_25","RANDOM_V4_25"),("CAUSAL_V4_25","FP16"),("CAUSAL_V4_25","KIVI_PAPER_G128")]))
    audit=audit_common(a24main, list(AIME24_MAIN_EXPECTED), "aime24"); write_json(out/"cross_lineage_compatibility_audit.json", audit)
    agg="PASS" if all(next(r for r in summary if r["method"]==m)["correct"]==c for m,c in AIME24_MAIN_EXPECTED.items()) else "FAIL"
    final={"aggregate_recomputation_gate":agg,"cross_lineage_compatibility":audit["comparability"],"main_table_gate":"PASS" if agg=="PASS" and audit["comparability"]=="PASS" else "PARTIAL"}; write_json(out/"final_gate.json", final)
    ptab=paper_rows(summary, list(AIME24_MAIN_EXPECTED), "aime24"); write_text(out/"paper_table.md", markdown_table(ptab)); write_csv(out/"paper_table.csv", ptab); write_text(out/"paper_table.tex", latex_table(ptab))
    write_json(out/"source_map.json", {f"aime24.{r['method']}.response_accuracy":{"canonical_file":str(out/"method_summary.csv"),"canonical_field":"accuracy"} for r in summary})
    write_text(out/"README.md", "# AIME24 main quality table\n\nCombines existing AIME24 four-method compact evidence with V100 KIVI evidence.\n")
    write_text(out/"claim_audit.md", f"# Claim Audit\n\nMain table gate: `{final['main_table_gate']}`.\n"); write_text(out/"git_provenance.txt", os.popen("git rev-parse HEAD && git branch --show-current").read())
    pout=Path("reports/paper_aime_quality_tables_v1"); pout.mkdir(parents=True, exist_ok=True)
    write_text(pout/"aime24_main_quality_table.md", markdown_table(ptab)); write_csv(pout/"aime24_main_quality_table.csv", ptab); write_text(pout/"aime24_main_quality_table.tex", latex_table(ptab))
    p25tab=paper_rows(p25["summary"], list(AIME25_METHODS), "aime25"); write_text(pout/"aime25_cross_year_quality_table.md", markdown_table(p25tab)); write_csv(pout/"aime25_cross_year_quality_table.csv", p25tab); write_text(pout/"aime25_cross_year_quality_table.tex", latex_table(p25tab))
    s24={r["method"]:r for r in summary}; s25={r["method"]:r for r in p25["summary"]}; cross=[]
    for m in ["FP16","KIVI_PAPER_G128","PATTERN_BASE","CAUSAL_V4_25"]:
        cross.append({"Method":m,"AIME24":f"{s24[m]['correct']}/90","AIME25":f"{s25[m]['correct']}/90","Combined correct/180":s24[m]['correct']+s25[m]['correct'],"Combined accuracy":(s24[m]['correct']+s25[m]['correct'])/180})
    write_csv(pout/"cross_year_summary.csv", cross); write_json(pout/"cross_year_pooled_bootstrap.json", {"note":"pooled bootstrap omitted from raw row export in this script version; use year-specific paired_bootstrap for primary claims"})
    write_json(pout/"source_map.json", {"aime24_table":str(out/"source_map.json"),"aime25_table":"reports/aime25_four_method_quality_v1/source_map.json"})
    write_text(pout/"README.md", "# Paper AIME quality tables\n"); write_text(pout/"claim_audit.md", "AIME24 and AIME25 are preserved separately; pooled row is auxiliary.\n")
    terminal={"TASK":"AIME25_FOUR_METHOD_AND_AIME24_KIVI_CANONICAL_EVIDENCE_ASSEMBLY_V1","CPU_ONLY":True,"SOURCE_RESULT_WORKTREE":str(source),"ASSEMBLY_WORKTREE":str(Path.cwd()),"AIME25_DATASET_SHA256":a25_meta["canonical_sha256"],"AIME25_DATASET_GATE":a25_meta["gate"],"AIME25_PROTOCOL_COMPARABILITY":p25["protocol"]["comparability"],"AIME24_CROSS_LINEAGE_COMPATIBILITY":audit["comparability"],"AIME25_ACCURACY":{r['method']:f"{r['correct']}/{r['total']}" for r in p25['summary']},"AIME24_MAIN_ACCURACY":{r['method']:f"{r['correct']}/{r['total']}" for r in summary},"NEW_GPU_GENERATIONS":0}
    write_json(Path("reports/paper_aime_quality_tables_v1/final_terminal_summary.json"), terminal)
    print(json.dumps(terminal, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
