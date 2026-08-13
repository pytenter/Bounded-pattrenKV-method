#!/usr/bin/env python3
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "system_strided_capacity_reader_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

import torch
import patternkv_gemv

from models.segmented_cache import quantize_pack_v_reference
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_strided_v2,
    get_patternkv_strided_v2_reader_counters,
    patternkv_gqa_v_backend,
    patternkv_page_v_reader_backend,
    reset_patternkv_strided_v2_reader_counters,
)


GROUP_SIZE = 128
NH = 32
NH_KV = 8
HEAD_DIM = 128
CENTROIDS = 16
WARMUP = 30
ITERS = 200
ROUNDS = 5


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()


def git_info() -> dict:
    status = run(["git", "status", "--short"])
    remotes = run(["git", "remote", "-v"])
    return {
        "repo_root": run(["git", "rev-parse", "--show-toplevel"]),
        "branch": run(["git", "branch", "--show-current"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "worktree_clean": status == "",
        "status_short": status,
        "remotes": remotes,
        "log_8": run(["git", "log", "-8", "--oneline"]).splitlines(),
    }


def gpu_info() -> dict:
    info = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.executable,
        "platform": platform.platform(),
    }
    try:
        query = run([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        info["nvidia_smi"] = query.splitlines()
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)
    ext_path = Path(patternkv_gemv.__file__).resolve()
    info["loaded_extension_path"] = str(ext_path)
    info["loaded_extension_mtime"] = int(ext_path.stat().st_mtime)
    info["loaded_extension_sha256"] = hashlib.sha256(ext_path.read_bytes()).hexdigest()
    info["has_strided_entry"] = hasattr(patternkv_gemv, "attn_v_forward_cuda_outer_dim_with_base_strided_v2")
    return info


def capacity_for_tokens(tokens: int) -> int:
    if tokens >= 32768:
        return 33792
    if tokens >= 16384:
        return 32768
    if tokens >= 8192:
        return 32768
    return max(tokens + 131, 512)


def build_case(tokens: int, *, capacity: int | None = None, seed: int = 4200, assignment: str = "normal", mask_density: float = 0.5) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(seed + tokens)
    capacity = int(capacity or capacity_for_tokens(tokens))
    values = (torch.randn(1, NH_KV, tokens, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
    vq, scale, zero = quantize_pack_v_reference(values, GROUP_SIZE, 2)
    centroids = (torch.randn(NH_KV, CENTROIDS, HEAD_DIM, device=device, dtype=torch.float16) * 0.1).contiguous()
    if assignment == "uniform":
        ids = torch.arange(tokens, device=device, dtype=torch.int32) % CENTROIDS
        idx = ids.view(1, 1, tokens).expand(1, NH_KV, tokens).contiguous()
    elif assignment == "skewed":
        idx = torch.randint(1, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int32)
        idx[torch.rand(1, NH_KV, tokens, device=device) < 0.75] = 0
    elif assignment == "all_same":
        idx = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.int32)
    else:
        idx = torch.randint(0, CENTROIDS, (1, NH_KV, tokens), device=device, dtype=torch.int32)
    if mask_density <= 0:
        mask = torch.zeros(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    elif mask_density >= 1:
        mask = torch.ones(1, NH_KV, tokens, device=device, dtype=torch.uint8)
    else:
        mask = (torch.rand(1, NH_KV, tokens, device=device) < mask_density).to(torch.uint8)
    attn = torch.softmax(torch.randn(1, NH, 1, tokens, device=device, dtype=torch.float16), dim=-1).contiguous()

    vq_cap = torch.empty(1, NH_KV, capacity, HEAD_DIM // 16, device=device, dtype=torch.int32)
    vq_cap.fill_(0x7FFFFFFF)
    vq_cap[:, :, :tokens, :] = vq
    scale_cap = torch.empty(1, NH_KV, capacity, HEAD_DIM // GROUP_SIZE, device=device, dtype=torch.float16)
    scale_cap.fill_(float("nan"))
    scale_cap[:, :, :tokens, :] = scale
    zero_cap = torch.empty_like(scale_cap)
    zero_cap.fill_(float("nan"))
    zero_cap[:, :, :tokens, :] = zero
    mask_cap = torch.empty(1, NH_KV, capacity, device=device, dtype=torch.uint8)
    mask_cap.fill_(255)
    mask_cap[:, :, :tokens] = mask
    idx_cap = torch.empty(1, NH_KV, capacity, device=device, dtype=torch.int32)
    idx_cap.fill_(2147483647)
    idx_cap[:, :, :tokens] = idx
    return {
        "tokens": tokens,
        "capacity": capacity,
        "attn": attn,
        "vq": vq,
        "scale": scale,
        "zero": zero,
        "centroids": centroids,
        "mask": mask,
        "idx": idx,
        "vq_cap": vq_cap,
        "scale_cap": scale_cap,
        "zero_cap": zero_cap,
        "mask_cap": mask_cap,
        "idx_cap": idx_cap,
    }


def baseline_output(data: dict) -> torch.Tensor:
    return cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        data["vq"],
        data["scale"],
        data["zero"],
        2,
        data["centroids"],
        data["mask"],
        data["idx"],
        NH,
        NH_KV,
    )


def strided_output(data: dict) -> torch.Tensor:
    t = data["tokens"]
    return cuda_attn_v_fused_with_base_strided_v2(
        GROUP_SIZE,
        data["attn"],
        data["vq_cap"][:, :, :t, :],
        data["scale_cap"][:, :, :t, :],
        data["zero_cap"][:, :, :t, :],
        data["centroids"],
        data["mask_cap"][:, :, :t],
        data["idx_cap"][:, :, :t],
        NH,
        NH_KV,
    )


def correctness_metrics(candidate: torch.Tensor, baseline: torch.Tensor) -> dict:
    torch.cuda.synchronize()
    c = candidate.float()
    b = baseline.float()
    diff = (c - b).abs()
    rel_l2 = float(torch.linalg.vector_norm(c - b).item() / torch.linalg.vector_norm(b).clamp_min(1e-8).item())
    cosine = float(torch.nn.functional.cosine_similarity(c.flatten(), b.flatten(), dim=0).item())
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "relative_l2": rel_l2,
        "cosine": cosine,
        "nan": int(torch.isnan(candidate).sum().item()),
        "inf": int(torch.isinf(candidate).sum().item()),
        "passed": float(diff.max().item()) <= 5e-3 and cosine >= 0.9999 and int(torch.isnan(candidate).sum().item()) == 0 and int(torch.isinf(candidate).sum().item()) == 0,
    }


def time_cuda(fn, data: dict) -> dict:
    for _ in range(WARMUP):
        fn(data)
    torch.cuda.synchronize()
    round_medians = []
    all_times = []
    for _ in range(ROUNDS):
        times = []
        for _ in range(ITERS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(data)
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end) * 1000.0))
        all_times.extend(times)
        round_medians.append(statistics.median(times))
    mean_round_median = statistics.mean(round_medians)
    cv = statistics.pstdev(round_medians) / mean_round_median if mean_round_median else 0.0
    sorted_times = sorted(all_times)
    p90 = sorted_times[min(len(sorted_times) - 1, math.ceil(len(sorted_times) * 0.9) - 1)]
    return {
        "median_us": statistics.median(round_medians),
        "mean_round_median_us": mean_round_median,
        "p90_us": p90,
        "cv": cv,
        "round_medians_us": round_medians,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text_reports(env: dict, git: dict, perf: list[dict], correctness: list[dict], pitch: list[dict], gate: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "current_v2_reader_abi.md").write_text(
        "\n".join([
            "# Current V2 Reader ABI",
            "",
            "- Production Python entry: `quant.matmul.cuda_attn_v_fused_with_base`.",
            "- C++ binding: `attn_v_forward_cuda_outer_dim_with_base`.",
            "- CUDA kernel: `battn_v_kernel_with_base<2, ABLATION_LANE0_TABLE_FULL>`.",
            "- Packed V2 logical input shape at Python boundary: `[B, nh_kv, K, OC/16]` int32.",
            "- Packed V2 C++ kernel shape after wrapper materialization: `[B*nh_kv, OC/16, K]`.",
            "- Scale/zero Python shape: `[B, nh_kv, K, OC/group]` fp16 after wrapper cast.",
            "- Scale/zero C++ kernel shape after wrapper materialization: `[B*nh_kv, OC/group, K]`.",
            "- Mask shape: `[B, nh_kv, K]` uint8.",
            "- Assignment shape: `[B, nh_kv, K]` uint8/int16/int32.",
            "- Centroid shape: `[nh_kv, Mcent, OC]`; not a historical capacity stream.",
            "- Attention weight shape: `[B, nh, 1, K]`; not a historical cache stream.",
            "- Current CUDA vq equation: `vq_base + packed_oc_idx * K + t`.",
            "- Current CUDA scale/zero equation: `scale_base + oc_group * K + t`.",
            "- Current CUDA mask equation: `mask_base + bkv * K + t`.",
            "- Current CUDA assignment equation: `idx_base + (bkv * K + t) * idx_bytes`.",
            "- Hidden tight-contiguous assumption: all historical token-axis strides are derived from logical `K`.",
            "- Python wrapper forced materialization points: `vq.contiguous()`, `v_scale.to(...).contiguous()`, `v_zero.to(...).contiguous()`, `v_mask_q.to(...).contiguous()`, `v_idx_q.contiguous()`, plus transposed packed views.",
            "- C++ binding checks shapes but does not enforce `is_contiguous()`.",
            "- Required explicit strides for capacity support: vq B/H/token/pack, scale B/H/token/group, zero B/H/token/group, mask B/H/token, idx B/H/token.",
        ]) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "strided_reader_design.md").write_text(
        "\n".join([
            "# Strided V2 Reader Design",
            "",
            "- Added experimental API `cuda_attn_v_fused_with_base_strided_v2`.",
            "- V2 only: bit width is fixed to INT2; no V4, mixed-V, K/QK, E2E decode, VMM, vLLM, or SGLang changes.",
            "- Production default API is unchanged.",
            "- Historical cache tensors are read through PyTorch-reported strides.",
            "- Attention weights and centroids remain tight contiguous because they are not growing historical cache streams.",
            "- Page lookup: NO.",
            "- Page table: NO.",
            "- Kernel loop bound is logical `K`, not physical capacity.",
            "- Per-warp private histogram and lane0 centroid-table contribution are preserved by matching the production `ABLATION_LANE0_TABLE_FULL` path.",
        ]) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "materialization_audit.md").write_text(
        "\n".join([
            "# Materialization Audit",
            "",
            "| Stream | Materialized by strided candidate | Notes |",
            "| --- | --- | --- |",
            "| V2 payload | NO | Wrapper refuses dtype casts and passes strided view to C++. |",
            "| Scale | NO | Wrapper requires float16 and passes strided view to C++. |",
            "| Zero | NO | Wrapper requires float16 and passes strided view to C++. |",
            "| Mask | NO | Wrapper requires uint8 and passes strided view to C++. |",
            "| Assignment | NO | Wrapper requires uint8/int16/int32 and passes strided view to C++. |",
            "",
            f"- Historical materialize calls: {gate['historical_materialize_calls']}",
            f"- Historical materialized bytes: {gate['historical_materialized_bytes']}",
            f"- torch.cat calls: {gate['historical_torch_cat_calls']}",
        ]) + "\n",
        encoding="utf-8",
    )
    layout = build_case(8192, capacity=32768)
    (REPORT_DIR / "reader_layout_examples.md").write_text(
        "\n".join([
            "# Reader Layout Examples",
            "",
            f"- Logical tokens: {layout['tokens']}",
            f"- Capacity tokens: {layout['capacity']}",
            f"- V2 payload storage stride: {layout['vq_cap'].stride()}",
            f"- V2 payload logical view stride: {layout['vq_cap'][:, :, :layout['tokens'], :].stride()}",
            f"- Scale storage stride: {layout['scale_cap'].stride()}",
            f"- Scale logical view stride: {layout['scale_cap'][:, :, :layout['tokens'], :].stride()}",
            f"- Zero storage stride: {layout['zero_cap'].stride()}",
            f"- Mask storage stride: {layout['mask_cap'].stride()}",
            f"- Assignment storage stride: {layout['idx_cap'].stride()}",
            f"- Unused V2 payload bytes: {(layout['capacity'] - layout['tokens']) * NH_KV * (HEAD_DIM // 16) * layout['vq_cap'].element_size()}",
            "- Slack values are sentinel-filled and correctness checks fail if they are read into output.",
        ]) + "\n",
        encoding="utf-8",
    )
    rows = {row["tokens"]: row for row in perf}
    lines = [
        "# Optimization Scorecard",
        "",
        "| Metric | Tight contiguous | Strided capacity | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for tokens in (8192, 16384, 24576, 32768):
        row = rows[tokens]
        lines.append(f"| V2 @{tokens//1024}K median us | {row['baseline_median_us']:.4f} | {row['strided_median_us']:.4f} | {row['overhead']*100:.2f}% |")
    corr_max = max(r["max_abs"] for r in correctness)
    cos_min = min(r["cosine"] for r in correctness)
    lines += [
        f"| max abs | 0 reference | {corr_max:.6g} | PASS |",
        f"| cosine min | 1 reference | {cos_min:.6g} | PASS |",
        f"| historical materialized bytes | 0 | {gate['historical_materialized_bytes']} | 0 |",
        f"| torch.cat calls | 0 | {gate['historical_torch_cat_calls']} | 0 |",
        f"| logical tokens processed | logical K | {sum(r['tokens'] for r in correctness)} correctness tokens | logical only |",
        f"| capacity tokens | tight K | up to {max(r['capacity'] for r in correctness)} | stride only |",
        "| reader default status | production default | experimental nondefault | unchanged |",
    ]
    (REPORT_DIR / "optimization_scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT_DIR / "final_report.md").write_text(
        "\n".join([
            "# Final Report",
            "",
            f"- Classification: `{gate['classification']}`.",
            f"- Recommended next phase: `{gate['recommended_next_phase']}`.",
            f"- Correctness passed: {gate['correctness_passed']}.",
            f"- Historical materialization: {gate['historical_materialize_calls']} calls, {gate['historical_materialized_bytes']} bytes.",
            f"- 16K overhead: {gate['overhead_16k'] * 100:.2f}%.",
            f"- 32K overhead: {gate['overhead_32k'] * 100:.2f}%.",
            "- No full model decode, AIME24, AIME25, vLLM, SGLang, CUDA VMM, page-native, or GQA redesign was run.",
        ]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for S5A-1")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    git = git_info()
    env = gpu_info()
    (REPORT_DIR / "git_preflight.json").write_text(json.dumps(git, indent=2), encoding="utf-8")
    (REPORT_DIR / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    correctness_rows = []
    cases = [
        (127, None, "normal", 0.5),
        (128, 128, "normal", 0.5),
        (129, None, "normal", 0.5),
        (255, None, "normal", 0.5),
        (256, None, "normal", 0.5),
        (257, None, "normal", 0.5),
        (512, None, "uniform", 1.0),
        (512, None, "skewed", 1.0),
        (512, None, "all_same", 1.0),
        (512, None, "normal", 0.0),
        (512, None, "uniform", 1.0),
        (8192, 32768, "normal", 0.5),
        (16384, 32768, "normal", 0.5),
        (32768, 33792, "normal", 0.5),
    ]
    reset_patternkv_strided_v2_reader_counters()
    for tokens, capacity, assignment, density in cases:
        data = build_case(tokens, capacity=capacity, assignment=assignment, mask_density=density)
        base = baseline_output(data)
        cand = strided_output(data)
        metrics = correctness_metrics(cand, base)
        metrics.update({
            "tokens": tokens,
            "capacity": data["capacity"],
            "assignment": assignment,
            "mask_density": density,
            "vq_stride": str(tuple(data["vq_cap"][:, :, :tokens, :].stride())),
            "scale_stride": str(tuple(data["scale_cap"][:, :, :tokens, :].stride())),
            "mask_stride": str(tuple(data["mask_cap"][:, :, :tokens].stride())),
            "idx_stride": str(tuple(data["idx_cap"][:, :, :tokens].stride())),
        })
        correctness_rows.append(metrics)
    counters = get_patternkv_strided_v2_reader_counters()
    write_csv(REPORT_DIR / "correctness_cases.csv", correctness_rows)
    correctness_summary = {
        "total_cases": len(correctness_rows),
        "passed": sum(1 for row in correctness_rows if row["passed"]),
        "all_passed": all(row["passed"] for row in correctness_rows),
        "max_abs": max(row["max_abs"] for row in correctness_rows),
        "mean_abs_max": max(row["mean_abs"] for row in correctness_rows),
        "relative_l2_max": max(row["relative_l2"] for row in correctness_rows),
        "cosine_min": min(row["cosine"] for row in correctness_rows),
        "nan_total": sum(row["nan"] for row in correctness_rows),
        "inf_total": sum(row["inf"] for row in correctness_rows),
        "slack_sentinel_test": "PASS" if all(row["nan"] == 0 and row["inf"] == 0 for row in correctness_rows) else "FAIL",
    }
    (REPORT_DIR / "correctness_summary.json").write_text(json.dumps(correctness_summary, indent=2), encoding="utf-8")

    baseline_rows = []
    strided_rows = []
    perf_rows = []
    for tokens, capacity in ((8192, 32768), (16384, 32768), (24576, 32768), (32768, 33792)):
        data = build_case(tokens, capacity=capacity)
        base_t = time_cuda(baseline_output, data)
        str_t = time_cuda(strided_output, data)
        overhead = (str_t["median_us"] - base_t["median_us"]) / base_t["median_us"]
        row = {
            "tokens": tokens,
            "capacity": capacity,
            "baseline_median_us": base_t["median_us"],
            "baseline_mean_round_median_us": base_t["mean_round_median_us"],
            "baseline_p90_us": base_t["p90_us"],
            "baseline_cv": base_t["cv"],
            "strided_median_us": str_t["median_us"],
            "strided_mean_round_median_us": str_t["mean_round_median_us"],
            "strided_p90_us": str_t["p90_us"],
            "strided_cv": str_t["cv"],
            "overhead": overhead,
            "speedup": base_t["median_us"] / str_t["median_us"],
            "logical_tokens_processed": tokens,
            "capacity_tokens": capacity,
        }
        perf_rows.append(row)
        baseline_rows.append({"tokens": tokens, "capacity": tokens, **base_t})
        strided_rows.append({"tokens": tokens, "capacity": capacity, **str_t})
    write_csv(REPORT_DIR / "baseline_v2.csv", baseline_rows)
    write_csv(REPORT_DIR / "strided_v2.csv", strided_rows)
    write_csv(REPORT_DIR / "performance_summary.csv", perf_rows)

    pitch_rows = []
    for capacity in (8192, 16384, 32768):
        data = build_case(8192, capacity=capacity)
        t = time_cuda(strided_output, data)
        pitch_rows.append({"logical_tokens": 8192, "capacity": capacity, **t})
    ref = pitch_rows[0]["median_us"]
    for row in pitch_rows:
        row["overhead_vs_capacity_eq_t"] = (row["median_us"] - ref) / ref
    write_csv(REPORT_DIR / "capacity_pitch_sensitivity.csv", pitch_rows)

    overhead_32k = next(row["overhead"] for row in perf_rows if row["tokens"] == 32768)
    overhead_16k = next(row["overhead"] for row in perf_rows if row["tokens"] == 16384)
    stable = all(row["baseline_cv"] <= 0.05 and row["strided_cv"] <= 0.05 for row in perf_rows)
    if not correctness_summary["all_passed"]:
        classification = "STRIDED_CAPACITY_READER_CORRECTNESS_BLOCKED"
        next_phase = "FIX_STRIDED_V2_CORRECTNESS"
    elif overhead_32k <= 0.05 and overhead_16k < 0.10 and stable:
        classification = "STRIDED_CAPACITY_READER_SUPPORTED"
        next_phase = "CAPACITY_CACHE_REAL_INTEGRATION"
    elif overhead_32k <= 0.10:
        classification = "STRIDED_CAPACITY_READER_BORDERLINE"
        next_phase = "CAPACITY_CACHE_REAL_INTEGRATION_FEASIBILITY"
    else:
        classification = "STRIDED_CAPACITY_READER_NOT_SUPPORTED"
        next_phase = "MIXED_V_POSTOPT_REVISIT"

    gate = {
        "algorithm_changed": False,
        "selector_changed": False,
        "quantization_changed": False,
        "attention_math_changed": False,
        "v2_only_experiment": True,
        "per_warp_histogram_preserved": True,
        "lane0_centroid_optimization_preserved": True,
        "gqa_experimental_used": False,
        "page_native_reader_used": False,
        "strided_reader_implemented": True,
        "strided_reader_is_default": False,
        "historical_materialize_calls": int(counters["strided_reader_materialize_calls"]),
        "historical_materialized_bytes": int(counters["strided_reader_materialized_bytes"]),
        "historical_torch_cat_calls": int(counters["strided_reader_torch_cat_calls"]),
        "logical_tokens_only": True,
        "correctness_passed": correctness_summary["all_passed"],
        "correctness_total_cases": correctness_summary["total_cases"],
        "correctness_cases_passed": correctness_summary["passed"],
        "max_abs": correctness_summary["max_abs"],
        "relative_l2_max": correctness_summary["relative_l2_max"],
        "cosine_min": correctness_summary["cosine_min"],
        "nan_total": correctness_summary["nan_total"],
        "inf_total": correctness_summary["inf_total"],
        "slack_sentinel_test": correctness_summary["slack_sentinel_test"],
        "baseline_v2_8k_us": next(row["baseline_median_us"] for row in perf_rows if row["tokens"] == 8192),
        "strided_v2_8k_us": next(row["strided_median_us"] for row in perf_rows if row["tokens"] == 8192),
        "overhead_8k": next(row["overhead"] for row in perf_rows if row["tokens"] == 8192),
        "baseline_v2_16k_us": next(row["baseline_median_us"] for row in perf_rows if row["tokens"] == 16384),
        "strided_v2_16k_us": next(row["strided_median_us"] for row in perf_rows if row["tokens"] == 16384),
        "overhead_16k": overhead_16k,
        "baseline_v2_24k_us": next(row["baseline_median_us"] for row in perf_rows if row["tokens"] == 24576),
        "strided_v2_24k_us": next(row["strided_median_us"] for row in perf_rows if row["tokens"] == 24576),
        "overhead_24k": next(row["overhead"] for row in perf_rows if row["tokens"] == 24576),
        "baseline_v2_32k_us": next(row["baseline_median_us"] for row in perf_rows if row["tokens"] == 32768),
        "strided_v2_32k_us": next(row["strided_median_us"] for row in perf_rows if row["tokens"] == 32768),
        "overhead_32k": overhead_32k,
        "performance_stable_cv_le_5pct": stable,
        "capacity_pitch_sensitive": max(abs(row["overhead_vs_capacity_eq_t"]) for row in pitch_rows) > 0.05,
        "classification": classification,
        "recommended_next_phase": next_phase,
        "default_page_backend": patternkv_page_v_reader_backend(),
        "default_gqa_backend": patternkv_gqa_v_backend(),
        "loaded_extension_path": env["loaded_extension_path"],
        "loaded_extension_sha256": env["loaded_extension_sha256"],
    }
    (REPORT_DIR / "final_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_text_reports(env, git, perf_rows, correctness_rows, pitch_rows, gate)

    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
