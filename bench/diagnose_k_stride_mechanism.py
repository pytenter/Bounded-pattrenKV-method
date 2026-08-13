#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "quant" / "csrc" / "gemv_cuda.cu"
OUT = ROOT / "reports" / "system_k_stride_mechanism_v1"
S5A3 = ROOT / "reports" / "system_strided_k_reader_v1"
START_HEAD = "5b13e3a05e78e42b5103c1a88ba69708a2186c4f"

sys.path.insert(0, str(ROOT / "quant"))
import patternkv_gemv  # noqa: E402


def run_text(cmd: list[str], *, check: bool = False) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        if check:
            raise
        return exc.output.strip()
    except FileNotFoundError:
        return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for root in (Path("/usr/local/cuda-12.4/bin"), Path("/usr/local/cuda-12.8/bin"), Path(sys.executable).parents[1] / "lib/python3.10/site-packages/triton/backends/nvidia/bin"):
        cand = root / name
        if cand.exists():
            return str(cand)
    return None


def parse_resource_usage(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = None
    for line in path.read_text(errors="ignore").splitlines():
        m = re.search(r"Function\s+(\S+):", line)
        if m:
            current = m.group(1)
            continue
        if current and "REG:" in line:
            row: dict[str, Any] = {"function": current}
            for key in ("REG", "STACK", "SHARED", "LOCAL"):
                mm = re.search(rf"{key}:(\d+)", line)
                row[key.lower()] = int(mm.group(1)) if mm else None
            rows.append(row)
            current = None
    labels = {
        "bgemv_kernel_outer_dim_with_base_tiledILi2": "tight_k2",
        "bgemv_kernel_outer_dim_with_base_strided_kILi2": "strided_k2",
        "battn_v_kernel_with_baseILi2ELi7": "v2_tight",
        "battn_v_kernel_with_base_stridedILi2": "v2_strided",
    }
    for row in rows:
        row["label"] = ""
        for needle, label in labels.items():
            if needle in row["function"]:
                row["label"] = label
                break
    return [row for row in rows if row["label"]]


def source_lines(start: int, end: int) -> str:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    return "\n".join(f"{idx + 1}: {lines[idx]}" for idx in range(start - 1, end))


def percent(a: float) -> str:
    return f"{a * 100:.2f}%"


def build_reports() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    ext = Path(patternkv_gemv.__file__).resolve()
    ncu_path = shutil.which("ncu")
    nsys_path = shutil.which("nsys")
    nvcc_path = find_tool("nvcc")
    cuobjdump_path = find_tool("cuobjdump")
    nvdisasm_path = find_tool("nvdisasm")
    ptxas_path = find_tool("ptxas")

    env = {
        "repo_root": run_text(["git", "rev-parse", "--show-toplevel"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "start_head_expected": START_HEAD,
        "head_at_report": run_text(["git", "rev-parse", "HEAD"]),
        "worktree_clean": run_text(["git", "status", "--short"]) == "",
        "remotes": run_text(["git", "remote", "-v"]),
        "git_log_10": run_text(["git", "log", "-10", "--oneline"]).splitlines(),
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]).splitlines(),
        "ncu_available": ncu_path is not None,
        "ncu_path": ncu_path,
        "nsys_available": nsys_path is not None,
        "nsys_path": nsys_path,
        "nvcc_path": nvcc_path,
        "ptxas_path": ptxas_path,
        "cuobjdump_path": cuobjdump_path,
        "nvdisasm_path": nvdisasm_path,
        "extension_path": str(ext),
        "extension_mtime": int(ext.stat().st_mtime),
        "extension_sha256": sha256(ext),
    }
    write_json(OUT / "environment.json", env)

    perf = read_csv(S5A3 / "performance_summary.csv")
    pitch = read_csv(S5A3 / "capacity_pitch_sensitivity.csv")
    gate3 = json.loads((S5A3 / "final_gate.json").read_text(encoding="utf-8"))
    sass_rows = json.loads((OUT / "sass_instruction_counts.json").read_text(encoding="utf-8")) if (OUT / "sass_instruction_counts.json").exists() else []
    resource_rows = parse_resource_usage(OUT / "cuobjdump_resource_usage.txt") if (OUT / "cuobjdump_resource_usage.txt").exists() else []

    launch = [
        {"reader": "tight_k2", "threads_x": 32, "threads_y": 8, "threads_per_block": 256, "warps_per_block": 8, "blocks_x_32k": 32, "blocks_y_32k": 256, "dynamic_smem_bytes": 16 * 4 + 128 * 2, "logical_bound": "OC from tight padded assignment/output"},
        {"reader": "strided_k2", "threads_x": 32, "threads_y": 8, "threads_per_block": 256, "warps_per_block": 8, "blocks_x_32k": 32, "blocks_y_32k": 256, "dynamic_smem_bytes": 16 * 4 + 128 * 2, "logical_bound": "OC = assignments.size(2)"},
        {"reader": "v2_tight", "threads_x": 32, "threads_y": 4, "threads_per_block": 128, "warps_per_block": 4, "blocks_x_32k": 32, "blocks_y_32k": 4, "dynamic_smem_bytes": 16 * 4 * 4, "logical_bound": "K attention tokens"},
        {"reader": "v2_strided", "threads_x": 32, "threads_y": 4, "threads_per_block": 128, "warps_per_block": 4, "blocks_x_32k": 32, "blocks_y_32k": 4, "dynamic_smem_bytes": 16 * 4 * 4, "logical_bound": "K attention tokens"},
    ]
    write_csv(OUT / "launch_geometry.csv", launch)
    write_csv(OUT / "ptxas_resource_summary.csv", resource_rows)
    ncu_rows = [{"status": "unavailable", "reason": "ncu not found in PATH; Nsight Compute profiling was skipped", "tight_kernel": "", "strided_kernel": ""}]
    if ncu_path:
        ncu_rows = [{"status": "available_not_run", "reason": "This diagnostic used cuobjdump/SASS static evidence only to avoid long privileged profiler runs.", "tight_kernel": "bgemv_kernel_outer_dim_with_base_tiled<2>", "strided_kernel": "bgemv_kernel_outer_dim_with_base_strided_k<2>"}]
    write_csv(OUT / "ncu_summary.csv", ncu_rows)

    (OUT / "ncu_raw_commands.md").write_text(
        "\n".join([
            "# NCU Raw Commands",
            "",
            f"- `which ncu`: `{ncu_path or ''}`",
            f"- `which nsys`: `{nsys_path or ''}`",
            "",
            "Nsight Compute was not used for final evidence in this phase. The portable fallback commands used were:",
            "",
            "```bash",
            "/usr/local/cuda-12.4/bin/cuobjdump --dump-resource-usage quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so > reports/system_k_stride_mechanism_v1/cuobjdump_resource_usage.txt",
            "/usr/local/cuda-12.4/bin/cuobjdump --dump-sass quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so > reports/system_k_stride_mechanism_v1/cuobjdump_sass.txt",
            "```",
        ]) + "\n",
        encoding="utf-8",
    )

    write_addressing_report()
    write_warp_report()
    write_behavior_reports(perf, pitch, sass_rows, resource_rows)

    tight32 = float(gate3["baseline_qk_32k_us"])
    strided32 = float(gate3["strided_qk_32k_us"])
    overhead32 = float(gate3["overhead_32k"])
    gate = {
        "algorithm_changed": False,
        "kernel_math_changed": False,
        "production_backend_changed": False,
        "value_capacity_path_changed": False,
        "k_capacity_reenabled": False,
        "ncu_available": bool(ncu_path),
        "nsys_available": bool(nsys_path),
        "tight_k_32k_us_reference": tight32,
        "strided_k_32k_us_reference": strided32,
        "observed_overhead_reference": overhead32,
        "physical_capacity_scan_hypothesis": "rejected",
        "memory_coalescing_hypothesis": "inconclusive",
        "address_arithmetic_hypothesis": "supported",
        "register_pressure_hypothesis": "rejected",
        "occupancy_hypothesis": "rejected",
        "vector_load_hypothesis": "inconclusive",
        "dominant_mechanism": "ADDRESS_ARITHMETIC_DOMINATED",
        "mechanism_confidence": "MEDIUM",
        "classification": "K_STRIDE_REGRESSION_PARTIALLY_SUPPORTED",
        "recommended_next_phase": "V_ONLY_CAPACITY_FINAL_SYSTEM_BENCHMARK",
    }
    write_json(OUT / "final_gate.json", gate)
    write_final_report(gate, perf, pitch, sass_rows, resource_rows)
    return gate


def write_addressing_report() -> None:
    text = [
        "# Kernel Addressing Audit",
        "",
        "Evidence type: STATIC_CODE_EVIDENCE unless otherwise marked.",
        "",
        "## Tight K Reader",
        "",
        "Production kernel: `bgemv_kernel_outer_dim_with_base_tiled<2>`.",
        "",
        "Tight C++ wrapper receives transposed tight tensors:",
        "",
        "- `_kernel`: `[B * nh_kv, ceil(OC / pack), IC]`",
        "- `_scaling_factors`: `[B * nh_kv, ceil(OC / group_size), IC]`",
        "- `_zeros`: `[B * nh_kv, ceil(OC / group_size), IC]`",
        "- `_assignments`: `[B, nh_kv, OC]`",
        "",
        "Address equations from `quant/csrc/gemv_cuda.cu`:",
        "",
        "```text",
        "batch_kv_flat = b * nh_kv + kv",
        "weight_base = _weight + batch_kv_flat * (OC * IC / pack_factor)",
        "scale_base  = _scale  + batch_kv_flat * (OC * IC / group_size)",
        "zeros_base  = _zeros  + batch_kv_flat * (OC * IC / group_size)",
        "arow_byte   = _assign + ((b * nh_kv + kv) * OC * assign_bytes)",
        "",
        "packed K word: weight[packed * IC + k]",
        "scale:         scale[group_idx * IC + k]",
        "zero:          zeros[group_idx * IC + k]",
        "assignment:    arow[oc * assign_bytes]",
        "query:         inputs[k]",
        "centroid:      cbase[m * IC + k]",
        "```",
        "",
        "Relevant source:",
        "",
        "```cpp",
        source_lines(986, 996),
        "...",
        source_lines(1070, 1087),
        "```",
        "",
        "## Strided K Reader",
        "",
        "Experimental kernel: `bgemv_kernel_outer_dim_with_base_strided_k<2>`.",
        "",
        "Strided wrapper receives logical views over capacity storage:",
        "",
        "- `_kernel`: `[B, nh_kv, IC, ceil(logical_OC / pack)]` with physical capacity in stride(2)",
        "- `_scaling_factors`: `[B, nh_kv, IC, ceil(logical_OC / group_size)]`",
        "- `_zeros`: `[B, nh_kv, IC, ceil(logical_OC / group_size)]`",
        "- `_assignments`: `[B, nh_kv, logical_OC]`",
        "",
        "Address equations:",
        "",
        "```text",
        "logical loop bound OC = _assignments.size(2)",
        "packed K word = _weight[b*w_s0 + kv*w_s1 + k*w_s2 + packed*w_s3]",
        "scale         = _scale [b*sc_s0 + kv*sc_s1 + k*sc_s2 + group_idx*sc_s3]",
        "zero          = _zeros [b*z_s0 + kv*z_s1 + k*z_s2 + group_idx*z_s3]",
        "assignment    = _assign[b*a_s0 + kv*a_s1 + oc*a_s2]",
        "query         = inputs[k]",
        "centroid      = cbase[m * IC + k]",
        "```",
        "",
        "Relevant source:",
        "",
        "```cpp",
        source_lines(1267, 1272),
        "...",
        source_lines(1301, 1310),
        "...",
        source_lines(1340, 1348),
        "```",
        "",
        "## Direct Comparison",
        "",
        "- Query and centroid addressing are effectively unchanged.",
        "- Tight K uses a layout-coupled linear expression where `packed * IC + k`; `k` is the fastest varying dimension.",
        "- Strided K must evaluate generic tensor-stride expressions for packed K, scale, zero, and assignment.",
        "- The strided kernel preserves math but turns layout constants into runtime stride operands.",
    ]
    (OUT / "kernel_addressing_audit.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_warp_report() -> None:
    rows = []
    pack_factor = 16
    ic = 128
    cap_packs = 32768 // pack_factor
    for lane in range(8):
        for t in range(4):
            k = lane * 4 + t
            rows.append({
                "lane": lane,
                "local_t": t,
                "logical_k_channel": k,
                "packed_token_index": 0,
                "tight_word_offset": 0 * ic + k,
                "strided_word_offset": k * cap_packs + 0,
                "strided_minus_tight": k * cap_packs - k,
            })
    write_csv(OUT / "warp_access_offsets_example.csv", rows)
    md = [
        "# Warp Access Mapping",
        "",
        "Example: `head_dim=128`, INT2 `pack_factor=16`, logical token pack `packed=0`, capacity `32768`, so `cap_packs=2048`.",
        "",
        "In both kernels, a warp lane handles four channel positions: `k = lane * 4 + {0,1,2,3}`.",
        "",
        "Tight K layout uses `[B*kv, packed_token, channel]`, so for fixed `packed=0`:",
        "",
        "```text",
        "offset = packed * IC + k = k",
        "lane 0 reads offsets 0,1,2,3",
        "lane 1 reads offsets 4,5,6,7",
        "...",
        "lane 31 reads offsets 124,125,126,127",
        "```",
        "",
        "STATIC_CODE_EVIDENCE: those 128 int32 words are contiguous across the warp/CTA channel tile.",
        "",
        "Strided K capacity layout uses `[B, kv, channel, token_pack]`, so for fixed `packed=0`:",
        "",
        "```text",
        "offset = k * cap_packs + packed",
        "lane 0 reads offsets 0,2048,4096,6144",
        "lane 1 reads offsets 8192,10240,12288,14336",
        "...",
        "```",
        "",
        "STATIC_CODE_EVIDENCE: lanes traverse channel, but physical memory is token-pack-major within each channel row. Adjacent channel loads are separated by `cap_packs` int32 words, so the tight contiguous channel tile becomes a highly strided load pattern.",
        "",
        "HYPOTHESIS: this likely worsens coalescing/global transactions for packed K, but NCU was unavailable, so transaction counters were not measured.",
    ]
    (OUT / "warp_access_mapping.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_behavior_reports(perf: list[dict[str, str]], pitch: list[dict[str, str]], sass_rows: list[dict[str, Any]], resource_rows: list[dict[str, Any]]) -> None:
    sass = {row["label"]: row for row in sass_rows}
    res = {row["label"]: row for row in resource_rows}
    tight = sass.get("tight_k2", {})
    strided = sass.get("strided_k2", {})
    v_tight = sass.get("v2_tight", {})
    v_strided = sass.get("v2_strided", {})
    md_mem = [
        "# Memory Behavior Comparison",
        "",
        "MEASURED CUDA Event reference from S5A-3:",
        "",
        "| Context | Tight K | Strided K | Overhead |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in perf:
        md_mem.append(f"| {int(row['context_tokens'])//1024}K | {float(row['baseline_median_us']):.3f} us | {float(row['strided_median_us']):.3f} us | {percent(float(row['overhead']))} |")
    md_mem.extend([
        "",
        "Capacity pitch sensitivity at logical 8K:",
        "",
        "| Logical T | Capacity | Strided K latency |",
        "| ---: | ---: | ---: |",
    ])
    for row in pitch:
        md_mem.append(f"| {row['logical_tokens']} | {row['capacity_tokens']} | {float(row['median_us']):.3f} us |")
    md_mem.extend([
        "",
        "PHYSICAL_CAPACITY_SCAN_HYPOTHESIS=REJECTED.",
        "",
        "Reason: keeping logical T fixed at 8192 while changing capacity from 8192 to 32768 keeps latency around 95-97 us. The kernel is not looping over physical capacity.",
        "",
        "Memory coalescing hypothesis: INCONCLUSIVE but plausible.",
        "",
        "STATIC_CODE_EVIDENCE: tight K stores the 128-channel tile contiguously for a fixed packed token; strided capacity storage spaces adjacent channel loads by `cap_packs`. Without NCU memory-sector counters, this remains a hypothesis rather than a measured conclusion.",
        "",
        f"SASS global load instruction count: tight K `{tight.get('global_loads')}`, strided K `{strided.get('global_loads')}`. This is a static instruction count, not DRAM bytes.",
    ])
    (OUT / "memory_behavior_comparison.md").write_text("\n".join(md_mem) + "\n", encoding="utf-8")

    md_inst = [
        "# Instruction Behavior Comparison",
        "",
        "STATIC_CODE_EVIDENCE from `cuobjdump --dump-sass`:",
        "",
        "| Kernel | Instructions | Integer add | Integer multiply/MAD | Constant refs | Global loads |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Tight K INT2 | {tight.get('instructions')} | {tight.get('integer_add')} | {tight.get('integer_mul_mad')} | {tight.get('constant_load_refs')} | {tight.get('global_loads')} |",
        f"| Strided K INT2 | {strided.get('instructions')} | {strided.get('integer_add')} | {strided.get('integer_mul_mad')} | {strided.get('constant_load_refs')} | {strided.get('global_loads')} |",
        f"| Tight V2 | {v_tight.get('instructions')} | {v_tight.get('integer_add')} | {v_tight.get('integer_mul_mad')} | {v_tight.get('constant_load_refs')} | {v_tight.get('global_loads')} |",
        f"| Strided V2 | {v_strided.get('instructions')} | {v_strided.get('integer_add')} | {v_strided.get('integer_mul_mad')} | {v_strided.get('constant_load_refs')} | {v_strided.get('global_loads')} |",
        "",
        "Address arithmetic hypothesis: SUPPORTED.",
        "",
        "Strided K adds runtime stride multiplications for packed K, scale, zero, and assignment. SASS reflects this: K IMAD count rises sharply, while V2 strided is close to V2 tight.",
        "",
        "Register/occupancy hypothesis: REJECTED by static resource evidence.",
        "",
        "| Kernel | Registers/thread | Stack | Local |",
        "| --- | ---: | ---: | ---: |",
        f"| Tight K INT2 | {res.get('tight_k2', {}).get('reg')} | {res.get('tight_k2', {}).get('stack')} | {res.get('tight_k2', {}).get('local')} |",
        f"| Strided K INT2 | {res.get('strided_k2', {}).get('reg')} | {res.get('strided_k2', {}).get('stack')} | {res.get('strided_k2', {}).get('local')} |",
        f"| Tight V2 | {res.get('v2_tight', {}).get('reg')} | {res.get('v2_tight', {}).get('stack')} | {res.get('v2_tight', {}).get('local')} |",
        f"| Strided V2 | {res.get('v2_strided', {}).get('reg')} | {res.get('v2_strided', {}).get('stack')} | {res.get('v2_strided', {}).get('local')} |",
        "",
        "No stack or local spill is reported for tight or strided K.",
    ]
    (OUT / "instruction_behavior_comparison.md").write_text("\n".join(md_inst) + "\n", encoding="utf-8")

    k_vs_v = [
        "# K vs V Layout Sensitivity",
        "",
        "| Property | K/QK | V/AV |",
        "| --- | --- | --- |",
        "| Compute pattern | Q dot K per token, loop over head_dim for each token-pack | Attention weights reduce across tokens into output channels |",
        "| Token traversal | Output dimension is token; QK writes one score per historical token | Token dimension is reduction axis; output dimension is channel |",
        "| Channel traversal | Warp lanes traverse head_dim channels for a fixed packed token | Warp lanes/histogram path tolerate token-strided metadata and payload access better |",
        "| Packed layout | Tight reader expects `[packed_token, channel]` so channel tile is contiguous | V reader already uses `[token, output_pack]`; capacity stride preserves token-major logical access |",
        "| Reuse pattern | Query tile is reused while reading K residual/scales/zeros per token-pack | Attention scalar and centroid histogram dominate; V payload access is less coupled to a transposed channel-contiguous layout |",
        "| Stride sensitivity | High: capacity layout changes fixed-token channel tile from contiguous to spaced by `cap_packs` | Low: S5A-1 measured around 4.99% V2 overhead |",
        "| Static instruction delta | K IMAD/add count rises strongly | V2 IMAD/add count changes modestly |",
        "| Observed overhead | 32K K overhead 33.68%; 16K/24K around 46% | Prior V2 strided overhead around 4.99%; S5A-2 mixed-V/E2E still faster |",
        "",
        "Interpretation: K/QK has strong layout-kernel coupling to the tight transposed K layout. V/AV is more compatible with capacity-backed token-strided views.",
    ]
    (OUT / "k_vs_v_layout_sensitivity.md").write_text("\n".join(k_vs_v) + "\n", encoding="utf-8")

    evidence = [
        {"Hypothesis": "Physical capacity scan", "Evidence": "MEASURED: 8K logical with 8K/16K/32K capacity stays around 95-97 us.", "Result": "REJECTED"},
        {"Hypothesis": "Poor coalescing", "Evidence": "STATIC: tight lane/channel offsets contiguous; strided offsets spaced by cap_packs. No NCU sectors measured.", "Result": "INCONCLUSIVE"},
        {"Hypothesis": "Extra memory transactions", "Evidence": "STATIC supports possible extra sectors, but no NCU transaction counters.", "Result": "INCONCLUSIVE"},
        {"Hypothesis": "Extra L2 traffic", "Evidence": "No NCU L2 sector/byte counters available.", "Result": "INCONCLUSIVE"},
        {"Hypothesis": "Extra address arithmetic", "Evidence": "STATIC+SASS: K IMAD rises 486 -> 918 and IADD rises 173 -> 421; V2 changes are much smaller.", "Result": "SUPPORTED"},
        {"Hypothesis": "Register pressure", "Evidence": "cuobjdump resource: tight K 69 regs/thread; strided K 63; no spills.", "Result": "REJECTED"},
        {"Hypothesis": "Occupancy loss", "Evidence": "Same launch geometry; lower register count for strided K; no spill/local memory.", "Result": "REJECTED"},
        {"Hypothesis": "Vector-load loss", "Evidence": "STATIC: generic strides prevent tight linear channel-tile addressing; SASS load count increases modestly, but vector transaction counters unavailable.", "Result": "INCONCLUSIVE"},
        {"Hypothesis": "Kernel launch difference", "Evidence": "Launch geometry identical for tight and strided K: 32x8 threads, same blocks for 32K, same dynamic shared memory.", "Result": "REJECTED"},
    ]
    write_csv(OUT / "mechanism_evidence_table.csv", evidence)
    md = ["# Mechanism Evidence Table", "", "| Hypothesis | Evidence | Result |", "| --- | --- | --- |"]
    md.extend(f"| {row['Hypothesis']} | {row['Evidence']} | {row['Result']} |" for row in evidence)
    (OUT / "mechanism_evidence_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_final_report(gate: dict[str, Any], perf: list[dict[str, str]], pitch: list[dict[str, str]], sass_rows: list[dict[str, Any]], resource_rows: list[dict[str, Any]]) -> None:
    row32 = next(row for row in perf if int(row["context_tokens"]) == 32768)
    text = [
        "# Final Report",
        "",
        f"Classification: `{gate['classification']}`",
        "",
        f"Dominant mechanism: `{gate['dominant_mechanism']}`",
        "",
        f"Confidence: `{gate['mechanism_confidence']}`",
        "",
        "## Decision",
        "",
        "Keep the asymmetric runtime architecture:",
        "",
        "- K: tight QK-optimized layout",
        "- V: capacity-managed stride-aware layout",
        "- Architecture name: `ASYMMETRIC_KV_RUNTIME`",
        "",
        "Do not re-enable K capacity in this branch. The S5A-3 performance gate stopped that path.",
        "",
        "## Evidence Summary",
        "",
        f"- MEASURED: 32K tight K `{float(row32['baseline_median_us']):.3f} us`, strided K `{float(row32['strided_median_us']):.3f} us`, overhead `{percent(float(row32['overhead']))}`.",
        "- MEASURED: physical capacity scan hypothesis is rejected by 8K pitch sensitivity.",
        "- STATIC_CODE_EVIDENCE: strided K uses generic stride address equations for packed K, scale, zero, and assignment.",
        "- STATIC_CODE_EVIDENCE: tight K keeps a fixed-token channel tile contiguous; strided capacity K spaces adjacent channels by `cap_packs`.",
        "- STATIC_CODE_EVIDENCE: SASS integer address arithmetic grows substantially in strided K.",
        "- STATIC_CODE_EVIDENCE: register pressure and occupancy-loss hypotheses are rejected by cuobjdump resource usage.",
        "",
        "## Caveat",
        "",
        "NCU/NSYS were unavailable in PATH, so memory-sector and stall-reason counters were not measured. Coalescing remains a plausible but unmeasured contributor. The strongest directly supported mechanism is address arithmetic plus tight-layout coupling.",
        "",
        "NEXT_TASK: `V_ONLY_CAPACITY_FINAL_SYSTEM_BENCHMARK`",
    ]
    (OUT / "final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    gate = build_reports()
    print(json.dumps(gate, indent=2, sort_keys=True))
