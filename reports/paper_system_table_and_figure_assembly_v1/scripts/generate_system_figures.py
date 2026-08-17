#!/usr/bin/env python3
from pathlib import Path
import csv
import math

import cairo

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports/paper_system_table_and_figure_assembly_v1"
FIGDATA = OUT / "figure_data"
FIGURES = OUT / "figures"

METHOD_ORDER = ["FP16", "KIVI", "PatternKV", "CAUSAL-V4@25%"]
COLORS = {
    "FP16": (0.000, 0.447, 0.698),
    "KIVI": (0.000, 0.620, 0.451),
    "PatternKV": (0.902, 0.624, 0.000),
    "CAUSAL-V4@25%": (0.835, 0.369, 0.000),
}
BLACK = (0.10, 0.10, 0.10)
GRAY = (0.82, 0.82, 0.82)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def set_rgb(ctx, color):
    ctx.set_source_rgb(*color)


def text(ctx, x, y, value, size=8, align="center", color=BLACK):
    ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(size)
    set_rgb(ctx, color)
    xb, yb, width, height, _xa, _ya = ctx.text_extents(value)
    if align == "center":
        tx = x - width / 2 - xb
    elif align == "right":
        tx = x - width - xb
    else:
        tx = x - xb
    ctx.move_to(tx, y - yb - height)
    ctx.show_text(value)


def rotated_text(ctx, x, y, value, angle_degrees=-18, size=8):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(math.radians(angle_degrees))
    text(ctx, 0, 0, value, size=size, align="center")
    ctx.restore()


def render(name, width, height, draw_fn):
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf_surface = cairo.PDFSurface(str(FIGURES / f"{name}.pdf"), width, height)
    pdf_ctx = cairo.Context(pdf_surface)
    draw_fn(pdf_ctx, width, height)
    pdf_ctx.show_page()
    pdf_surface.finish()

    scale = 3
    png_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width * scale), int(height * scale))
    png_ctx = cairo.Context(png_surface)
    png_ctx.scale(scale, scale)
    draw_fn(png_ctx, width, height)
    png_surface.write_to_png(str(FIGURES / f"{name}.png"))


def axes(ctx, x0, y0, plot_w, plot_h, y_max, ylabel, yticks=4):
    set_rgb(ctx, (1, 1, 1))
    ctx.paint()
    set_rgb(ctx, BLACK)
    ctx.set_line_width(0.8)
    ctx.move_to(x0, y0)
    ctx.line_to(x0, y0 + plot_h)
    ctx.line_to(x0 + plot_w, y0 + plot_h)
    ctx.stroke()
    for i in range(yticks + 1):
        value = y_max * i / yticks
        y = y0 + plot_h - plot_h * i / yticks
        set_rgb(ctx, GRAY)
        ctx.set_line_width(0.45)
        ctx.move_to(x0, y)
        ctx.line_to(x0 + plot_w, y)
        ctx.stroke()
        text(ctx, x0 - 6, y + 3, f"{value:.0f}", size=7, align="right")
    ctx.save()
    ctx.translate(11, y0 + plot_h / 2)
    ctx.rotate(-math.pi / 2)
    text(ctx, 0, 0, ylabel, size=9, align="center")
    ctx.restore()


def bar_chart(name, rows, value_key, ylabel, annotation_fn, y_max=None):
    values = [float(rows[method][value_key]) for method in METHOD_ORDER]
    y_max = y_max or max(values) * 1.2

    def draw(ctx, width, height):
        x0, y0, plot_w, plot_h = 40, 16, width - 52, height - 54
        axes(ctx, x0, y0, plot_w, plot_h, y_max, ylabel)
        gap = plot_w / len(METHOD_ORDER)
        bar_w = gap * 0.62
        for idx, method in enumerate(METHOD_ORDER):
            value = float(rows[method][value_key])
            bar_h = plot_h * value / y_max
            x = x0 + idx * gap + (gap - bar_w) / 2
            y = y0 + plot_h - bar_h
            set_rgb(ctx, COLORS[method])
            ctx.rectangle(x, y, bar_w, bar_h)
            ctx.fill()
            annotation_lines = annotation_fn(value, rows[method]).split("\n")
            for line_idx, line in enumerate(annotation_lines):
                offset = 4 + (len(annotation_lines) - 1 - line_idx) * 8
                text(ctx, x + bar_w / 2, y - offset, line, size=7)
            rotated_text(ctx, x + bar_w / 2, y0 + plot_h + 17, method, size=7.2)

    render(name, 250, 172, draw)


def throughput():
    rows = {row["method"]: row for row in read_csv(FIGDATA / "throughput_c2048_b4.csv")}
    bar_chart(
        "system_throughput_c2048_b4",
        rows,
        "throughput_tokens_s",
        "Throughput (tokens/s)",
        lambda value, _row: f"{value:.1f}",
    )


def capacity():
    rows = {row["method"]: row for row in read_csv(FIGDATA / "capacity_c4096.csv")}
    bar_chart(
        "system_capacity_c4096",
        rows,
        "max_success_B",
        "Maximum successful batch size",
        lambda _value, row: f"{float(row['capacity_vs_FP16']):.0f}x",
        y_max=10,
    )


def memory():
    rows = {row["method"]: row for row in read_csv(FIGDATA / "memory_c4096_b4.csv")}
    bar_chart(
        "system_memory_c4096_b4",
        rows,
        "peak_allocated_gib",
        "Peak allocated memory (GiB)",
        lambda value, row: (
            f"{value:.1f}\nref."
            if row["method"] == "FP16"
            else f"{value:.1f}\n-{float(row['allocated_reduction_vs_FP16_percent']):.1f}%"
        ),
        y_max=22,
    )


def context_scaling():
    rows = read_csv(FIGDATA / "context_scaling.csv")
    by_method = {method: [] for method in METHOD_ORDER}
    for row in rows:
        by_method[row["method"]].append((int(row["context_length"]), float(row["tpot_ms"])))

    def draw(ctx, width, height):
        x0, y0, plot_w, plot_h = 42, 13, width - 58, height - 42
        y_max = 180
        axes(ctx, x0, y0, plot_w, plot_h, y_max, "TPOT (ms/token)")
        contexts = [2048, 4096, 8192]
        for idx, context in enumerate(contexts):
            x = x0 + plot_w * idx / (len(contexts) - 1)
            text(ctx, x, y0 + plot_h + 13, str(context), size=7.5)
        text(ctx, x0 + plot_w / 2, height - 5, "Context length", size=9)
        for method in METHOD_ORDER:
            points = sorted(by_method[method])
            set_rgb(ctx, COLORS[method])
            ctx.set_line_width(1.6)
            for idx, (context, value) in enumerate(points):
                x = x0 + plot_w * contexts.index(context) / (len(contexts) - 1)
                y = y0 + plot_h - plot_h * value / y_max
                if idx == 0:
                    ctx.move_to(x, y)
                else:
                    ctx.line_to(x, y)
            ctx.stroke()
            for context, value in points:
                x = x0 + plot_w * contexts.index(context) / (len(contexts) - 1)
                y = y0 + plot_h - plot_h * value / y_max
                ctx.arc(x, y, 2.3, 0, 2 * math.pi)
                ctx.fill()
        legend_x, legend_y = x0 + 78, y0 + 50
        for idx, method in enumerate(METHOD_ORDER):
            y = legend_y + idx * 10
            set_rgb(ctx, COLORS[method])
            ctx.rectangle(legend_x, y, 6, 6)
            ctx.fill()
            text(ctx, legend_x + 10, y + 6, method, size=7, align="left")

    render("system_context_scaling", 250, 178, draw)


def long_decode():
    rows = {row["method"]: row for row in read_csv(FIGDATA / "long_decode_d256.csv")}
    bar_chart(
        "system_long_decode_d256",
        rows,
        "tpot_ms",
        "TPOT (ms/token)",
        lambda value, _row: f"{value:.1f}",
        y_max=185,
    )


def main():
    throughput()
    capacity()
    memory()
    context_scaling()
    long_decode()


if __name__ == "__main__":
    main()
