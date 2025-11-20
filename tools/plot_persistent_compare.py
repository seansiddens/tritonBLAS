"""Plot GFLOP/s vs arithmetic intensity from persistent schedule benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


DIM_COLOR = "#b3b3b3"


DTYPE_BYTES = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.half": 2,
    "torch.float32": 4,
    "torch.bfloat": 2,
    "torch.bf16": 2,
    "torch.f16": 2,
    "torch.f32": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate arithmetic-intensity scatter plots from persistent_schedule_compare CSV output."
    )
    parser.add_argument("csv_path", type=Path, help="CSV produced by persistent_schedule_compare.py")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output image file (default: <csv>_ai_scatter.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Optional plot title (defaults to CSV stem)",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Only plot rows matching this category value",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    """Return a filesystem-friendly version of value."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")
    return slug or "category"


def dtype_num_bytes(dtype_str: str) -> int:
    if dtype_str not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Update DTYPE_BYTES mapping.")
    return DTYPE_BYTES[dtype_str]


def arithmetic_intensity(m: int, n: int, k: int, bytes_per_elem: int) -> float:
    flops = 2 * m * n * k
    # Approximate bytes moved: read A (m*k) + read B (k*n) + read+write C (2*m*n)
    bytes_moved = (m * k + k * n + 2 * m * n) * bytes_per_elem
    if bytes_moved == 0:
        return float("nan")
    return flops / bytes_moved


def load_points(
    csv_path: Path,
    category_filter: str | None = None,
    collect_stats: bool = False,
) -> tuple[dict[str, dict[str, list]], int, dict | None]:
    def empty_point() -> dict[str, list]:
        return {"ai": [], "tflops": [], "highlight": []}

    points = {
        "mall": empty_point(),
        "random": empty_point(),
        "l2": empty_point(),
    }
    total_rows = 0
    stats = None
    if collect_stats:
        stats = {
            "overall_sum": 0.0,
            "overall_count": 0,
            "per_category": {},
            "categories": set(),
            "perf_totals": {
                "mall": {"overall": {"sum": 0.0, "count": 0}, "per_category": {}},
                "random": {"overall": {"sum": 0.0, "count": 0}, "per_category": {}},
                "l2": {"overall": {"sum": 0.0, "count": 0}, "per_category": {}},
            },
        }
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if category_filter and "category" not in (reader.fieldnames or []):
            raise ValueError("CSV missing required 'category' column for filtering.")
        for row in reader:
            if category_filter and row.get("category") != category_filter:
                continue
            total_rows += 1
            try:
                m = int(row["m"])
                n = int(row["n"])
                k = int(row["k"])
                in_dtype = row["in_dtype"]
                comparator_gflops_str = row.get("comparator_gflops") or row.get("default_gflops")
                mall_gflops = float(comparator_gflops_str) if comparator_gflops_str else float("nan")
                workgroup_shuffle_gflops = float(row.get("workgroup_shuffle_gflops", "nan"))
                l2_gflops = float(row.get("shuffled_gflops", "nan"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid row encountered: {row}") from exc

            bytes_per_elem = dtype_num_bytes(in_dtype)
            ai = arithmetic_intensity(m, n, k, bytes_per_elem)
            if math.isnan(ai):
                continue

            category = row.get("category", "unknown")
            bytes_threshold = 256 * 1024 * 1024  # 256 MiB
            bytes_loaded = (m * k + n * k) * bytes_per_elem
            highlight = bytes_loaded > bytes_threshold

            if not math.isnan(mall_gflops):
                points["mall"]["ai"].append(ai)
                points["mall"]["tflops"].append(mall_gflops / 1000)  # convert to TFLOP/s
                points["mall"]["highlight"].append(highlight)
            if not math.isnan(workgroup_shuffle_gflops):
                points["random"]["ai"].append(ai)
                points["random"]["tflops"].append(workgroup_shuffle_gflops / 1000)
                points["random"]["highlight"].append(highlight)
            if not math.isnan(l2_gflops):
                points["l2"]["ai"].append(ai)
                points["l2"]["tflops"].append(l2_gflops / 1000)
                points["l2"]["highlight"].append(highlight)
            if (
                collect_stats
            ):
                stats["categories"].add(category)

                def record_perf(strategy: str, value: float) -> None:
                    if math.isnan(value):
                        return
                    totals = stats["perf_totals"][strategy]
                    totals["overall"]["sum"] += value
                    totals["overall"]["count"] += 1
                    cat_totals = totals["per_category"].setdefault(category, {"sum": 0.0, "count": 0})
                    cat_totals["sum"] += value
                    cat_totals["count"] += 1

                record_perf("mall", mall_gflops)
                record_perf("random", workgroup_shuffle_gflops)
                record_perf("l2", l2_gflops)

                if not math.isnan(mall_gflops) and not math.isnan(l2_gflops) and l2_gflops != 0:
                    speedup = mall_gflops / l2_gflops
                    stats["overall_sum"] += speedup
                    stats["overall_count"] += 1
                    cat_sum, cat_count = stats["per_category"].get(category, (0.0, 0))
                    stats["per_category"][category] = (cat_sum + speedup, cat_count + 1)
    return points, total_rows, stats


def plot(
    points: dict[str, dict[str, list]],
    title: str,
    output_path: Path,
    axis_points: dict[str, dict[str, list]] | None = None,
    *,
    highlight_mode: bool = False,
    series: list[str] | None = None,
    dim_series: set[str] | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    colors = {"mall": "tab:blue", "random": "tab:red", "l2": "tab:orange"}
    labels = {
        "mall": "L2 Aware, MALL Aware",
        "random": "L2 Random, MALL Random",
        "l2": "L2 Aware, MALL Random",
    }
    dim_series = dim_series or set()
    ordered_series = series or list(points.keys())
    for label in ordered_series:
        data = points[label]
        x_vals = data["ai"]
        y_vals = data["tflops"]
        highlights = data["highlight"]
        base_color = colors[label]
        if not x_vals:
            continue
        if highlight_mode:
            hi_x = [x for x, keep in zip(x_vals, highlights) if keep]
            hi_y = [y for y, keep in zip(y_vals, highlights) if keep]
            dim_x = [x for x, keep in zip(x_vals, highlights) if not keep]
            dim_y = [y for y, keep in zip(y_vals, highlights) if not keep]
            if dim_x:
                plt.scatter(
                    dim_x,
                    dim_y,
                    alpha=0.35,
                    edgecolors="none",
                    s=40,
                    color=DIM_COLOR,
                )
            if hi_x:
                plt.scatter(
                    hi_x,
                    hi_y,
                    label=labels[label],
                    alpha=0.9,
                    edgecolors="none",
                    s=40,
                    color=base_color,
                )
        else:
            is_dimmed = label in dim_series
            plt.scatter(
                x_vals,
                y_vals,
                label=labels[label],
                alpha=0.4 if is_dimmed else 0.75,
                edgecolors="none",
                s=40,
                color=DIM_COLOR if is_dimmed else base_color,
            )

    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("TFLOP/s")
    plt.title(title)
    plt.legend()
    source = axis_points or points
    flat_x = [x for vals in source.values() for x in vals["ai"]]
    flat_y = [y for vals in source.values() for y in vals["tflops"]]
    if flat_x:
        plt.xlim(min(flat_x), max(flat_x))
    if flat_y:
        plt.ylim(min(flat_y), max(flat_y))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Wrote scatter plot to {output_path}")


def tflops_range(point_map: dict[str, dict[str, list]]) -> tuple[float | None, float | None]:
    values = [
        val
        for series in point_map.values()
        for val in series["tflops"]
        if not math.isnan(val)
    ]
    if not values:
        return None, None
    return min(values), max(values)


def plot_colormap_series(
    data: dict[str, list],
    label: str,
    title: str,
    output_path: Path,
    axis_points: dict[str, dict[str, list]],
    norm: Normalize,
    *,
    grey_non_highlight: bool = False,
) -> None:
    x_vals = data["ai"]
    y_vals = data["tflops"]
    highlights = data["highlight"]
    if not x_vals:
        return
    plt.figure(figsize=(10, 6))
    if grey_non_highlight:
        hi_x = [x for x, keep in zip(x_vals, highlights) if keep]
        hi_y = [y for y, keep in zip(y_vals, highlights) if keep]
        dim_x = [x for x, keep in zip(x_vals, highlights) if not keep]
        dim_y = [y for y, keep in zip(y_vals, highlights) if not keep]
        if dim_x:
            plt.scatter(
                dim_x,
                dim_y,
                color=DIM_COLOR,
                alpha=0.35,
                s=40,
                edgecolors="none",
            )
        scatter = plt.scatter(
            hi_x,
            hi_y,
            c=hi_y,
            cmap="viridis",
            norm=norm,
            s=40,
            edgecolors="none",
        )
    else:
        scatter = plt.scatter(
            x_vals,
            y_vals,
            c=y_vals,
            cmap="viridis",
            norm=norm,
            s=40,
            edgecolors="none",
        )
    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("TFLOP/s")
    plt.title(title)
    source = axis_points
    flat_x = [x for vals in source.values() for x in vals["ai"]]
    flat_y = [y for vals in source.values() for y in vals["tflops"]]
    if flat_x:
        plt.xlim(min(flat_x), max(flat_x))
    if flat_y:
        plt.ylim(min(flat_y), max(flat_y))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    cbar = plt.colorbar(scatter)
    cbar.set_label("TFLOP/s")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Wrote {label} heat scatter to {output_path}")


def plot_bar_chart(stats: dict, title: str, output_path: Path) -> None:
    perf_totals = stats.get("perf_totals")
    if not perf_totals:
        return

    def avg_from_entry(entry: dict | None) -> float:
        if not entry or entry["count"] == 0:
            return float("nan")
        return entry["sum"] / entry["count"]

    def get_avg(strategy: str, label: str) -> float:
        totals = perf_totals[strategy]
        if label == "overall":
            return avg_from_entry(totals["overall"])
        return avg_from_entry(totals["per_category"].get(label))

    strategies = ["l2", "mall"]
    labels = ["overall"] + sorted(stats.get("categories", []))
    valid_labels: list[str] = []
    values: dict[str, list[float]] = {s: [] for s in strategies}

    for label in labels:
        random_avg = get_avg("random", label)
        if math.isnan(random_avg) or random_avg == 0:
            continue
        valid_labels.append(label)
        for strategy in strategies:
            avg = get_avg(strategy, label)
            if math.isnan(avg):
                ratio = float("nan")
            else:
                ratio = avg / random_avg
            values[strategy].append(ratio)

    if not valid_labels:
        print("Skipping bar chart due to insufficient data.")
        return

    num_groups = len(valid_labels)
    indices = list(range(num_groups))
    width = 0.25
    offsets = [(-width), 0, width]
    plt.figure(figsize=(max(8, num_groups * 0.8), 6))
    colors = {"mall": "tab:blue", "l2": "tab:orange"}
    labels_map = {"mall": "L2 + MALL aware", "l2": "L2 aware"}

    for idx, strategy in enumerate(strategies):
        bar_positions = [pos + offsets[idx] for pos in indices]
        plt.bar(
            bar_positions,
            values[strategy],
            width,
            label=labels_map[strategy],
            color=colors[strategy],
            alpha=0.8,
        )

    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xticks(indices, valid_labels, rotation=45, ha="right")
    plt.ylabel("Relative TFLOP/s vs random")
    plt.title(f"{title} - Average Performance Breakdown")
    plt.legend()
    all_vals = [val for series in values.values() for val in series if not math.isnan(val)]
    if all_vals:
        ymax = max(1.0, max(all_vals)) * 1.1
    else:
        ymax = 1.1
    plt.ylim(1.0, ymax)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Wrote bar chart to {output_path}")


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path
    if not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")

    category_filter = args.category
    suffix = f"_{slugify(category_filter)}" if category_filter else ""
    default_filename = f"{csv_path.stem}{suffix}_ai_scatter.png"
    if args.output:
        if args.output.suffix:
            base_filename = args.output.name
            base_dir = args.output.parent
        else:
            base_filename = default_filename
            base_dir = args.output
    else:
        base_filename = default_filename
        base_dir = Path(".")
    output_dir = base_dir / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / base_filename

    full_points, _, stats = load_points(csv_path, None, collect_stats=True)
    if category_filter:
        points, _, _ = load_points(csv_path, category_filter)
        axis_points = full_points
    else:
        points = full_points
        axis_points = None

    if not any(points[label]["ai"] for label in points):
        raise SystemExit("No valid data points to plot.")

    if category_filter:
        default_title = f"Random vs Mall Aware Grid Schedules - {category_filter}"
    else:
        default_title = "Random vs Mall Aware Grid Schedules"
    title = args.title or default_title

    axis_source = axis_points or points
    plot(points, title, output_path, axis_points=axis_source)

    highlight_output = output_path.with_name(f"{output_path.stem}_large_inputs{output_path.suffix}")
    highlight_title = f"{title} (>256 MiB inputs)"
    plot(points, highlight_title, highlight_output, axis_points=axis_source, highlight_mode=True)

    if stats:
        bar_output = output_path.with_name(f"{output_path.stem}_relative_perf{output_path.suffix}")
        plot_bar_chart(stats, title, bar_output)

    tflop_min, tflop_max = tflops_range(axis_source)
    if tflop_min is not None and tflop_max is not None:
        if tflop_max <= tflop_min:
            tflop_max = tflop_min + 1e-6
        norm = Normalize(vmin=tflop_min, vmax=tflop_max)
        heat_variants = [
            ("random", "random_heat", "Random"),
            ("l2", "l2_heat", "L2 aware"),
            ("mall", "mall_heat", "L2 + MALL aware"),
        ]
        for label, suffix, pretty in heat_variants:
            series_data = points[label]
            if not series_data["ai"]:
                continue
            heat_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
            plot_colormap_series(
                series_data,
                label,
                f"{title} - {pretty} Heatmap",
                heat_path,
                axis_source,
                norm,
            )
        mall_series = points["mall"]
        if mall_series["ai"]:
            grey_heat_path = output_path.with_name(
                f"{output_path.stem}_mall_heat_large_inputs{output_path.suffix}"
            )
            plot_colormap_series(
                mall_series,
                "mall",
                f"{title} - L2 + MALL aware Heatmap (>256 MiB)",
                grey_heat_path,
                axis_source,
                norm,
                grey_non_highlight=True,
            )

    random_only_path = output_path.with_name(f"{output_path.stem}_random{output_path.suffix}")
    plot(
        points,
        f"{title} - Random Only",
        random_only_path,
        axis_points=axis_source,
        series=["random"],
    )

    l2_random_path = output_path.with_name(f"{output_path.stem}_l2_vs_random{output_path.suffix}")
    plot(
        points,
        f"{title} - L2 vs Random",
        l2_random_path,
        axis_points=axis_source,
        series=["l2", "random"],
        dim_series={"random"},
    )

    mall_focus_path = output_path.with_name(f"{output_path.stem}_mall_focus{output_path.suffix}")
    plot(
        points,
        f"{title} - Mall Focus",
        mall_focus_path,
        axis_points=axis_source,
        series=["mall", "l2", "random"],
        dim_series={"l2", "random"},
    )

    if stats and stats["overall_count"] > 0:
        avg_speedup = stats["overall_sum"] / stats["overall_count"]
        avg_pct = (avg_speedup - 1) * 100
        print(
            f"Average MALL-aware speedup across all categories: "
            f"{avg_speedup:.4f}x ({avg_pct:+.2f}%) over {stats['overall_count']} rows"
        )
        per_category = stats["per_category"]
        for category in sorted(per_category):
            cat_sum, cat_count = per_category[category]
            if cat_count == 0:
                continue
            cat_avg = cat_sum / cat_count
            cat_pct = (cat_avg - 1) * 100
            print(f"  {category}: {cat_avg:.4f}x ({cat_pct:+.2f}%) over {cat_count} rows")


if __name__ == "__main__":
    main()
