#!/usr/bin/env python3
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/paper_system_table_and_figure_assembly_v1"
SRC = ROOT / "reports/paper_baseline_system_comparison_v1_reconciled"

METHOD_ORDER = [
    "FP16_FULL_MODEL",
    "KIVI_PAPER_G128_FULL_MODEL",
    "PATTERNKV_PAPER_FULL_MODEL",
    "CAUSAL_V4_25_FULL_MODEL",
]

DISPLAY = {
    "FP16_FULL_MODEL": "FP16",
    "KIVI_PAPER_G128_FULL_MODEL": "KIVI",
    "PATTERNKV_PAPER_FULL_MODEL": "PatternKV",
    "CAUSAL_V4_25_FULL_MODEL": "CAUSAL-V4@25%",
}

EFFECTIVE_BITS = {
    "FP16_FULL_MODEL": "16-bit KV",
    "KIVI_PAPER_G128_FULL_MODEL": "2.25-bit quantized-region effective storage",
    "PATTERNKV_PAPER_FULL_MODEL": "2.25-bit quantized-region effective storage",
    "CAUSAL_V4_25_FULL_MODEL": "~2.50 effective KV bits",
}

EXPECTED_FINAL = {
    "FP16_FULL_MODEL": {
        "c2048_b1_tpot_ms": 30.604021856561303,
        "c2048_b4_tpot_ms": 31.553811364574358,
        "c2048_b4_throughput_tokens_s": 126.74733629692822,
        "c4096_max_success_B": "4",
        "c4096_first_failure_B": "8",
        "capacity_ratio_vs_FP16": "1.0",
        "long_decode_c4096_b1_d256_tpot_ms": 28.663047660302254,
    },
    "KIVI_PAPER_G128_FULL_MODEL": {
        "c2048_b1_tpot_ms": 68.9668008708395,
        "c2048_b4_tpot_ms": 62.327605875907466,
        "c2048_b4_throughput_tokens_s": 64.17142884541053,
        "c4096_max_success_B": "8",
        "c4096_first_failure_B": "16",
        "capacity_ratio_vs_FP16": "2.0",
        "long_decode_c4096_b1_d256_tpot_ms": 56.831604874787445,
    },
    "PATTERNKV_PAPER_FULL_MODEL": {
        "c2048_b1_tpot_ms": 162.75336360558867,
        "c2048_b4_tpot_ms": 158.92638938385062,
        "c2048_b4_throughput_tokens_s": 25.16789399783627,
        "c4096_max_success_B": "8",
        "c4096_first_failure_B": "16",
        "capacity_ratio_vs_FP16": "2.0",
        "long_decode_c4096_b1_d256_tpot_ms": 154.64627086657856,
    },
    "CAUSAL_V4_25_FULL_MODEL": {
        "c2048_b1_tpot_ms": 165.16838225652464,
        "c2048_b4_tpot_ms": 165.7085524930153,
        "c2048_b4_throughput_tokens_s": 24.137850552291873,
        "c4096_max_success_B": "8",
        "c4096_first_failure_B": "16",
        "capacity_ratio_vs_FP16": "2.0",
        "long_decode_c4096_b1_d256_tpot_ms": 159.53037909730483,
    },
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def gib(bytes_value):
    return float(bytes_value) / (2**30)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out) + "\n"


def tex_escape(text):
    return str(text).replace("%", r"\%")


def tex_table(headers, rows, caption, label):
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def load_sources():
    final_gate = json.loads((SRC / "final_gate.json").read_text())
    final_rows = {
        row["method"]: row
        for row in final_gate["FINAL_PAPER_SYSTEM_NUMBERS_V1"]
    }
    for method in METHOD_ORDER:
        assert method in final_rows
        for key, expected in EXPECTED_FINAL[method].items():
            actual = final_rows[method][key]
            if isinstance(expected, float):
                assert abs(float(actual) - expected) < 1e-12, (method, key, actual, expected)
            else:
                assert str(actual) == expected, (method, key, actual, expected)
    return {
        "final_gate": final_gate,
        "final_rows": final_rows,
        "batch": read_csv(SRC / "batch_scaling/batch_scaling_summary.csv"),
        "context": read_csv(SRC / "context_scaling/context_scaling_summary.csv"),
        "capacity": read_csv(SRC / "capacity_summary.csv"),
        "memory": read_csv(SRC / "matched_memory_c4096_b4.csv"),
        "long": read_csv(SRC / "long_decode/long_decode_summary.csv"),
    }


def by_method(rows):
    return {row["method"]: row for row in rows}


def write_tables(data):
    tables = OUT / "tables"
    figdata = OUT / "figure_data"
    tables.mkdir(parents=True, exist_ok=True)
    figdata.mkdir(parents=True, exist_ok=True)

    final = data["final_rows"]
    memory = by_method(data["memory"])
    capacity = by_method(data["capacity"])
    long_rows = by_method(data["long"])

    canonical_rows = []
    for method in METHOD_ORDER:
        row = final[method]
        mem = memory[method]
        canonical_rows.append({
            "method": method,
            "display_method": DISPLAY[method],
            "effective_kv_bits": EFFECTIVE_BITS[method],
            "c2048_b1_tpot_ms": row["c2048_b1_tpot_ms"],
            "c2048_b4_tpot_ms": row["c2048_b4_tpot_ms"],
            "c2048_b4_throughput_tokens_s": row["c2048_b4_throughput_tokens_s"],
            "c4096_b4_peak_allocated_bytes": mem["peak_allocated_bytes"],
            "c4096_b4_peak_allocated_gib": gib(mem["peak_allocated_bytes"]),
            "c4096_b4_peak_reserved_bytes": mem["peak_reserved_bytes"],
            "c4096_b4_peak_reserved_gib": gib(mem["peak_reserved_bytes"]),
            "c4096_max_success_B": row["c4096_max_success_B"],
            "c4096_first_failure_B": row["c4096_first_failure_B"],
            "capacity_ratio_vs_FP16": row["capacity_ratio_vs_FP16"],
            "long_decode_c4096_b1_d256_tpot_ms": row["long_decode_c4096_b1_d256_tpot_ms"],
        })
    write_csv(
        OUT / "system_numbers_canonical.csv",
        list(canonical_rows[0].keys()),
        canonical_rows,
    )
    (OUT / "system_numbers_canonical.json").write_text(json.dumps(canonical_rows, indent=2) + "\n")

    main_rows = []
    for row in canonical_rows:
        max_b = row["c4096_max_success_B"]
        cap = f"{float(row['capacity_ratio_vs_FP16']):.1f}x"
        main_rows.append([
            row["display_method"],
            row["effective_kv_bits"],
            fmt(row["c2048_b1_tpot_ms"]),
            fmt(row["c2048_b4_throughput_tokens_s"]),
            fmt(row["c4096_b4_peak_allocated_gib"], 2),
            max_b if max_b == "4" else f"**{max_b}**",
            cap,
            fmt(row["long_decode_c4096_b1_d256_tpot_ms"]),
        ])
    main_headers_md = [
        "Method",
        "Effective KV bits",
        "C2048 B1 TPOT ↓",
        "C2048 B4 Throughput ↑",
        "C4096 B4 Peak Memory ↓",
        "C4096 Max Batch ↑",
        "Capacity vs FP16",
        "D256 TPOT ↓",
    ]
    (tables / "system_main_table.md").write_text(
        "# Main System Table\n\n"
        + md_table(main_headers_md, main_rows)
        + "\nPeak memory is full-model full-lifecycle peak allocated memory in GiB under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.\n"
    )
    main_rows_tex = [
        [
            r"\textbf{" + cell[2:-2] + "}" if cell.startswith("**") and cell.endswith("**") else cell
            for cell in row
        ]
        for row in main_rows
    ]
    (tables / "system_main_table.tex").write_text(tex_table(
        [
            "Method",
            "Effective KV bits",
            "C2048 B1 TPOT $\\downarrow$",
            "C2048 B4 tok/s $\\uparrow$",
            "C4096 B4 GiB $\\downarrow$",
            "C4096 max B $\\uparrow$",
            "Capacity vs FP16",
            "D256 TPOT $\\downarrow$",
        ],
        main_rows_tex,
        "Frozen full-model system results on a single RTX 3090.",
        "tab:system-main",
    ))

    mem_rows = []
    fp16_alloc = float(memory["FP16_FULL_MODEL"]["peak_allocated_bytes"])
    for method in METHOD_ORDER:
        mem = memory[method]
        reduction = 100.0 * (1.0 - float(mem["peak_allocated_bytes"]) / fp16_alloc)
        mem_rows.append([
            DISPLAY[method],
            fmt(gib(mem["peak_allocated_bytes"]), 2),
            fmt(gib(mem["peak_reserved_bytes"]), 2),
            fmt(reduction, 1),
        ])
    (tables / "matched_memory_table.md").write_text(
        "# Matched Memory Table\n\n"
        + md_table(["Method", "Peak allocated GiB ↓", "Peak reserved GiB ↓", "Allocated reduction vs FP16 (%) ↑"], mem_rows)
    )
    (tables / "matched_memory_table.tex").write_text(tex_table(
        ["Method", "Peak allocated GiB $\\downarrow$", "Peak reserved GiB $\\downarrow$", "Reduction vs FP16 (\\%) $\\uparrow$"],
        mem_rows,
        "Full-model peak memory at C4096/B4/D8.",
        "tab:matched-memory",
    ))

    cap_rows = []
    for method in METHOD_ORDER:
        cap = capacity[method]
        max_b = cap["max_success_B"]
        cap_rows.append([
            DISPLAY[method],
            max_b if max_b == "4" else f"**{max_b}**",
            cap["first_OOM_B"],
            f"{float(cap['capacity_ratio_vs_FP16']):.1f}x",
        ])
    (tables / "capacity_table.md").write_text(
        "# Capacity Table\n\n"
        + md_table(["Method", "Max successful batch ↑", "First OOM batch", "Capacity vs FP16"], cap_rows)
    )
    cap_rows_tex = [[cell.replace("**", "") if i == 1 else cell for i, cell in enumerate(row)] for row in cap_rows]
    for row in cap_rows_tex:
        if row[1] == "8":
            row[1] = r"\textbf{8}"
    (tables / "capacity_table.tex").write_text(tex_table(
        ["Method", "Max successful batch $\\uparrow$", "First OOM batch", "Capacity vs FP16"],
        cap_rows_tex,
        "C4096 capacity under the reconciled allocator protocol.",
        "tab:capacity",
    ))

    context_rows = []
    context_by_key = {(row["method"], row["context_length"]): row for row in data["context"]}
    for method in METHOD_ORDER:
        context_rows.append([
            DISPLAY[method],
            fmt(context_by_key[(method, "2048")]["median_tpot_ms"]),
            fmt(context_by_key[(method, "4096")]["median_tpot_ms"]),
            fmt(context_by_key[(method, "8192")]["median_tpot_ms"]),
        ])
    (tables / "context_scaling_table.md").write_text(
        "# Context Scaling Table\n\n"
        + md_table(["Method", "C2048 TPOT ↓", "C4096 TPOT ↓", "C8192 TPOT ↓"], context_rows)
    )
    (tables / "context_scaling_table.tex").write_text(tex_table(
        ["Method", "C2048 TPOT $\\downarrow$", "C4096 TPOT $\\downarrow$", "C8192 TPOT $\\downarrow$"],
        context_rows,
        "Decode TPOT over the tested 2K-8K context range at B1/D8.",
        "tab:context-scaling",
    ))

    ld_rows = []
    for method in METHOD_ORDER:
        row = long_rows[method]
        ld_rows.append([DISPLAY[method], row["context_length"], row["batch_size"], row["decode_length"], fmt(row["median_tpot_ms"]), fmt(row["throughput_tokens_s"])])
    (tables / "long_decode_table.md").write_text(
        "# Long Decode Table\n\n"
        + md_table(["Method", "Context", "Batch", "Decode", "TPOT ↓", "Throughput ↑"], ld_rows)
    )
    (tables / "long_decode_table.tex").write_text(tex_table(
        ["Method", "Context", "Batch", "Decode", "TPOT $\\downarrow$", "Throughput $\\uparrow$"],
        ld_rows,
        "Long-decode C4096/B1/D256 full-model results.",
        "tab:long-decode",
    ))

    write_csv(figdata / "throughput_c2048_b4.csv", ["method", "throughput_tokens_s"], [
        {"method": row["display_method"], "throughput_tokens_s": row["c2048_b4_throughput_tokens_s"]}
        for row in canonical_rows
    ])
    write_csv(figdata / "capacity_c4096.csv", ["method", "max_success_B", "first_OOM_B", "capacity_vs_FP16"], [
        {"method": DISPLAY[method], "max_success_B": capacity[method]["max_success_B"], "first_OOM_B": capacity[method]["first_OOM_B"], "capacity_vs_FP16": capacity[method]["capacity_ratio_vs_FP16"]}
        for method in METHOD_ORDER
    ])
    write_csv(figdata / "memory_c4096_b4.csv", ["method", "peak_allocated_bytes", "peak_allocated_gib", "peak_reserved_bytes", "peak_reserved_gib", "allocated_reduction_vs_FP16_percent"], [
        {
            "method": DISPLAY[method],
            "peak_allocated_bytes": memory[method]["peak_allocated_bytes"],
            "peak_allocated_gib": gib(memory[method]["peak_allocated_bytes"]),
            "peak_reserved_bytes": memory[method]["peak_reserved_bytes"],
            "peak_reserved_gib": gib(memory[method]["peak_reserved_bytes"]),
            "allocated_reduction_vs_FP16_percent": 100.0 * (1.0 - float(memory[method]["peak_allocated_bytes"]) / fp16_alloc),
        }
        for method in METHOD_ORDER
    ])
    write_csv(figdata / "context_scaling.csv", ["method", "context_length", "tpot_ms"], [
        {"method": DISPLAY[row["method"]], "context_length": row["context_length"], "tpot_ms": row["median_tpot_ms"]}
        for method in METHOD_ORDER
        for row in data["context"]
        if row["method"] == method
    ])
    write_csv(figdata / "long_decode_d256.csv", ["method", "tpot_ms", "throughput_tokens_s"], [
        {"method": DISPLAY[method], "tpot_ms": long_rows[method]["median_tpot_ms"], "throughput_tokens_s": long_rows[method]["throughput_tokens_s"]}
        for method in METHOD_ORDER
    ])
    write_csv(figdata / "tradeoff_system_side.csv", ["method", "effective_kv_bits", "throughput_c2048_b4", "tpot_c2048_b4", "max_batch_c4096", "matched_peak_memory_c4096_b4"], [
        {
            "method": row["display_method"],
            "effective_kv_bits": row["effective_kv_bits"],
            "throughput_c2048_b4": row["c2048_b4_throughput_tokens_s"],
            "tpot_c2048_b4": row["c2048_b4_tpot_ms"],
            "max_batch_c4096": row["c4096_max_success_B"],
            "matched_peak_memory_c4096_b4": row["c4096_b4_peak_allocated_gib"],
        }
        for row in canonical_rows
    ])


def write_reports(data):
    final = data["final_rows"]
    memory = by_method(data["memory"])
    fp16_tput = final["FP16_FULL_MODEL"]["c2048_b4_throughput_tokens_s"]
    kivi_tput = final["KIVI_PAPER_G128_FULL_MODEL"]["c2048_b4_throughput_tokens_s"]
    pattern_tput = final["PATTERNKV_PAPER_FULL_MODEL"]["c2048_b4_throughput_tokens_s"]
    causal_tput = final["CAUSAL_V4_25_FULL_MODEL"]["c2048_b4_throughput_tokens_s"]
    pattern_d256 = final["PATTERNKV_PAPER_FULL_MODEL"]["long_decode_c4096_b1_d256_tpot_ms"]
    causal_d256 = final["CAUSAL_V4_25_FULL_MODEL"]["long_decode_c4096_b1_d256_tpot_ms"]
    fp16_alloc = float(memory["FP16_FULL_MODEL"]["peak_allocated_bytes"])
    reductions = {
        method: 100.0 * (1.0 - float(memory[method]["peak_allocated_bytes"]) / fp16_alloc)
        for method in METHOD_ORDER
    }
    metrics = {
        "KIVI_vs_FP16_throughput": kivi_tput / fp16_tput,
        "PatternKV_vs_FP16_throughput": pattern_tput / fp16_tput,
        "CAUSAL_vs_FP16_throughput": causal_tput / fp16_tput,
        "PatternKV_vs_KIVI_throughput": pattern_tput / kivi_tput,
        "CAUSAL_vs_KIVI_throughput": causal_tput / kivi_tput,
        "CAUSAL_vs_PatternKV_throughput": causal_tput / pattern_tput,
        "CAUSAL_extra_overhead_vs_PatternKV_B4": 100.0 * (1.0 - causal_tput / pattern_tput),
        "CAUSAL_extra_overhead_vs_PatternKV_D256": 100.0 * (causal_d256 / pattern_d256 - 1.0),
        "memory_reductions": reductions,
    }
    (OUT / "computed_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    source_manifest = """# Source Manifest

Primary system numbers are derived only from the reconciled canonical report. Superseded pre-reconciliation data in `reports/paper_baseline_system_comparison_v1/` is excluded from primary tables and figures.

## Canonical Inputs

- `reports/paper_baseline_system_comparison_v1_reconciled/final_gate.json`: `FINAL_PAPER_SYSTEM_NUMBERS_V1`, classification, allocator protocol.
- `reports/paper_baseline_system_comparison_v1_reconciled/paper_table.md`: rounded paper-facing cross-check of final system rows.
- `reports/paper_baseline_system_comparison_v1_reconciled/batch_scaling/`: C2048/B1 and C2048/B4 TPOT and throughput provenance.
- `reports/paper_baseline_system_comparison_v1_reconciled/context_scaling/`: B1/D8 context-scaling summary used by `system_context_scaling`.
- `reports/paper_baseline_system_comparison_v1_reconciled/capacity/`: capacity raw stop probes and capacity report provenance.
- `reports/paper_baseline_system_comparison_v1_reconciled/capacity_summary.csv`: max successful batch, first OOM batch, and capacity ratios.
- `reports/paper_baseline_system_comparison_v1_reconciled/long_decode/`: C4096/B1/D256 long-decode summary.
- `reports/paper_baseline_system_comparison_v1_reconciled/matched_memory_c4096_b4.csv`: matched C4096/B4 full-model peak allocated and reserved memory.

## Asset Mapping

- Main system table: `final_gate.json` plus `matched_memory_c4096_b4.csv`.
- Matched memory table and figure: `matched_memory_c4096_b4.csv`.
- Capacity table and figure: `final_gate.json` and `capacity_summary.csv`.
- Context-scaling table and figure: `context_scaling/context_scaling_summary.csv`.
- Long-decode table and figure: `final_gate.json` and `long_decode/long_decode_summary.csv`.
- Pairwise metrics and narrative: `FINAL_PAPER_SYSTEM_NUMBERS_V1` in `final_gate.json`.
"""
    (OUT / "source_manifest.md").write_text(source_manifest)

    story = (
        "In our tested RTX3090 / DeepSeek-R1-Distill-Llama-8B setup, FP16 provides the highest matched-B throughput. "
        "KIVI is slower than FP16 but substantially faster than PatternKV and CAUSAL. "
        "KIVI, PatternKV, and CAUSAL all reach B8 at C4096, compared with FP16 B4. "
        f"CAUSAL is close to PatternKV at matched throughput, with {metrics['CAUSAL_extra_overhead_vs_PatternKV_B4']:.1f}% lower C2048/B4 throughput and {metrics['CAUSAL_extra_overhead_vs_PatternKV_D256']:.1f}% higher C4096/B1/D256 TPOT."
    )
    paper_safe = (
        "These system results position CAUSAL-V4@25% as a quality-oriented selective-precision runtime built on a PatternKV-like compressed system path, "
        "not as a throughput optimization over KIVI or FP16. Quality benefit should be cited from the frozen quality evidence rather than recomputed here."
    )
    (OUT / "summary.md").write_text(
        "# Summary\n\n"
        + "PAPER_SYSTEM_TABLE_AND_FIGURE_ASSEMBLY_V1_SUPPORTED. System experiments remain frozen; no new GPU experiments were run and no experimental numbers were modified.\n\n"
        + story + "\n\n" + paper_safe + "\n"
    )
    (OUT / "README.md").write_text(
        "# Paper System Table and Figure Assembly V1\n\n"
        + "This directory contains paper-safe tables, figures, figure data, LaTeX assets, provenance, and claim audit artifacts assembled from the reconciled frozen system evidence.\n\n"
        + "- Run `python reports/paper_system_table_and_figure_assembly_v1/scripts/generate_system_tables.py` to regenerate tables and figure data.\n"
        + "- Run `python reports/paper_system_table_and_figure_assembly_v1/scripts/generate_system_figures.py` to regenerate PDF and PNG figures.\n"
        + "- Primary numbers trace to `reports/paper_baseline_system_comparison_v1_reconciled/final_gate.json`.\n"
    )

    paper_md = f"""# Paper System Results

### System Setup

We evaluate full-model decode serving on a single RTX3090 using DeepSeek-R1-Distill-Llama-8B. Measurements use decode-only timing, true batch execution, subprocess isolation, the same model/harness/protocol across all methods, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Throughput

At C2048/B4/D8, the throughput ranking is FP16 > KIVI >> PatternKV ~= CAUSAL. FP16 reaches {fp16_tput:.3f} tokens/s, KIVI reaches {kivi_tput:.3f} tokens/s, PatternKV reaches {pattern_tput:.3f} tokens/s, and CAUSAL-V4@25% reaches {causal_tput:.3f} tokens/s. Relative to FP16, KIVI is {metrics['KIVI_vs_FP16_throughput']:.3f}x, PatternKV is {metrics['PatternKV_vs_FP16_throughput']:.3f}x, and CAUSAL is {metrics['CAUSAL_vs_FP16_throughput']:.3f}x.

### Capacity and Memory

At C4096 under the reconciled allocator protocol, FP16 reaches maximum successful batch B4 and first OOM at B8. KIVI, PatternKV, and CAUSAL all reach B8 and first OOM at B16, doubling the observed maximum successful batch size versus FP16 in this tested setup. At matched C4096/B4/D8, full-model peak allocated memory is {gib(memory['FP16_FULL_MODEL']['peak_allocated_bytes']):.2f} GiB for FP16, {gib(memory['KIVI_PAPER_G128_FULL_MODEL']['peak_allocated_bytes']):.2f} GiB for KIVI, {gib(memory['PATTERNKV_PAPER_FULL_MODEL']['peak_allocated_bytes']):.2f} GiB for PatternKV, and {gib(memory['CAUSAL_V4_25_FULL_MODEL']['peak_allocated_bytes']):.2f} GiB for CAUSAL. These are full-model memory measurements, not KV-cache-only memory.

### CAUSAL Overhead over PatternKV

CAUSAL adds a small system overhead relative to PatternKV: {metrics['CAUSAL_extra_overhead_vs_PatternKV_B4']:.1f}% lower throughput at C2048/B4 and {metrics['CAUSAL_extra_overhead_vs_PatternKV_D256']:.1f}% higher TPOT at C4096/B1/D256. This is the measured runtime cost of CAUSAL's selective heterogeneous V2/V4 mechanism in this harness.

### Interpretation

CAUSAL-V4@25% should be positioned as quality-oriented selective precision built on a PatternKV-like compressed runtime, not as a throughput optimization over KIVI or FP16. Quality benefit must be referenced from the frozen quality evidence.
"""
    (OUT / "paper_system_results.md").write_text(paper_md)
    (OUT / "paper_system_results.tex").write_text(
        "\\subsection{System Setup}\n"
        "We evaluate full-model decode serving on a single RTX3090 using DeepSeek-R1-Distill-Llama-8B. Measurements use decode-only timing, true batch execution, subprocess isolation, the same model/harness/protocol across all methods, and \\texttt{PYTORCH\\_CUDA\\_ALLOC\\_CONF=expandable\\_segments:True}.\n\n"
        "\\subsection{Throughput}\n"
        f"At C2048/B4/D8, the throughput ranking is FP16 $>$ KIVI $\\gg$ PatternKV $\\approx$ CAUSAL. FP16 reaches {fp16_tput:.3f} tokens/s, KIVI reaches {kivi_tput:.3f} tokens/s, PatternKV reaches {pattern_tput:.3f} tokens/s, and CAUSAL-V4@25\\% reaches {causal_tput:.3f} tokens/s.\n\n"
        "\\subsection{Capacity and Memory}\n"
        "At C4096 under the reconciled allocator protocol, KIVI, PatternKV, and CAUSAL all reach B8, doubling the maximum successful batch size of FP16 (B4). Matched C4096/B4 memory is full-model peak allocated memory, not KV-cache-only memory.\n\n"
        "\\subsection{CAUSAL Overhead over PatternKV}\n"
        f"CAUSAL has {metrics['CAUSAL_extra_overhead_vs_PatternKV_B4']:.1f}\\% lower throughput than PatternKV at C2048/B4 and {metrics['CAUSAL_extra_overhead_vs_PatternKV_D256']:.1f}\\% higher TPOT at C4096/B1/D256. This is the measured runtime cost of CAUSAL's selective heterogeneous V2/V4 mechanism in this harness.\n\n"
        "\\subsection{Interpretation}\n"
        "CAUSAL-V4@25\\% should be positioned as quality-oriented selective precision built on a PatternKV-like compressed runtime, not as a throughput optimization over KIVI or FP16. Quality benefit must be referenced from the frozen quality evidence.\n"
    )

    (OUT / "claim_audit.md").write_text("""# Claim Audit

## Supported

- Same-GPU four-method comparison.
- True batch execution.
- Zero fallback.
- Reconciled allocator protocol.
- CAUSAL 2x FP16 max successful B in the tested C4096 setup.
- PatternKV 2x FP16 max successful B in the tested C4096 setup.
- KIVI 2x FP16 max successful B in the tested C4096 setup.
- CAUSAL ~4% throughput overhead vs PatternKV at C2048/B4.
- CAUSAL ~3% TPOT overhead vs PatternKV at D256.
- Nearly context-flat PatternKV/CAUSAL TPOT over the tested 2K-8K context range.

## Not Supported

- CAUSAL full-model speedup over FP16.
- CAUSAL full-model speedup over KIVI.
- Universal 2x capacity.
- Whole-GPU 2.5-bit memory.
- 84% full-model memory reduction.
- Production vLLM/SGLang integration.
""")
    (OUT / "final_gate.json").write_text(json.dumps({
        "classification": "PAPER_SYSTEM_TABLE_AND_FIGURE_ASSEMBLY_V1_SUPPORTED",
        "source_system_checkpoint": "dc2da40f8bcf5a494e81a423c2f2a0a2aaea665c",
        "source_final_gate": "reports/paper_baseline_system_comparison_v1_reconciled/final_gate.json",
        "primary_system_numbers": "FINAL_PAPER_SYSTEM_NUMBERS_V1",
        "system_experiments_frozen": True,
        "new_gpu_experiments_run": False,
        "experimental_numbers_modified": False,
        "allocator_protocol": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "method_display_names": DISPLAY,
        "computed_pairwise_metrics": metrics,
        "source_final_classification": data["final_gate"]["classification"],
    }, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_sources()
    write_tables(data)
    write_reports(data)


if __name__ == "__main__":
    main()
