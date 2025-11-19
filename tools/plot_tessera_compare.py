"""Plot GFLOP/s vs arithmetic intensity for Tessera benchmark results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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


def dtype_num_bytes(dtype_str: str) -> int:
    if dtype_str not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Update DTYPE_BYTES mapping.")
    return DTYPE_BYTES[dtype_str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tessera vs baseline arithmetic intensity scatter plots.")
    parser.add_argument("csv_path", type=Path, help="CSV produced by tessera_benchmark.py")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output image file (default: <csv>_tessera_ai_scatter.png)",
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


def arithmetic_intensity(m: int, n: int, k: int, bytes_per_elem: int) -> float:
    flops = 2 * m * n * k
    bytes_moved = (m * k + k * n + 2 * m * n) * bytes_per_elem
    if bytes_moved == 0:
        return float("nan")
    return flops / bytes_moved


def load_rows(csv_path: Path, category_filter: str | None = None) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = {
        "categories": set(),
        "baseline": {"overall": {"sum": 0.0, "count": 0}, "per_category": {}},
        "tessera": {"overall": {"sum": 0.0, "count": 0}, "per_category": {}},
    }
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if category_filter and row.get("category") != category_filter:
                continue
            try:
                m = int(row["m"])
                n = int(row["n"])
                k = int(row["k"])
                in_dtype = row["in_dtype"]
                ai = arithmetic_intensity(m, n, k, dtype_num_bytes(in_dtype))
                if math.isnan(ai):
                    continue
                baseline = float(row["baseline_gflops"])
                tessera = float(row["tessera_gflops"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid row encountered: {row}") from exc
            category = row.get("category", "unknown")
            stats["categories"].add(category)

            rows.append(
                {
                    "category": category,
                    "ai": ai,
                    "baseline": baseline / 1000,
                    "tessera": tessera / 1000,
                    "m": m,
                    "n": n,
                    "k": k,
                }
            )
            for strategy, value in (("baseline", baseline), ("tessera", tessera)):
                entry = stats[strategy]
                entry["overall"]["sum"] += value
                entry["overall"]["count"] += 1
                cat_entry = entry["per_category"].setdefault(category, {"sum": 0.0, "count": 0})
                cat_entry["sum"] += value
                cat_entry["count"] += 1
    return rows, stats


def scatter(points: list[tuple[float, float]], label: str, color: str) -> None:
    if not points:
        return
    xs, ys = zip(*points)
    plt.scatter(
        xs,
        ys,
        label=label,
        alpha=0.75,
        edgecolors="none",
        s=40,
        color=color,
    )


def scatter_heat(points: list[tuple[float, float]], values: list[float], title: str, output_path: Path) -> None:
    if not points:
        return
    xs, ys = zip(*points)
    norm = Normalize(vmin=min(values), vmax=max(values)) if values else None
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(xs, ys, c=values, cmap="viridis", norm=norm, s=40, edgecolors="none")
    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("TFLOP/s")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    cbar = plt.colorbar(scatter)
    cbar.set_label("TFLOP/s")
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote heat scatter to {output_path}")


def plot(rows: list[dict], title: str, output_path: Path) -> None:
    tessera_pts: list[tuple[float, float]] = []
    baseline_pts: list[tuple[float, float]] = []
    for row in rows:
        tessera_pts.append((row["ai"], row["tessera"]))
        baseline_pts.append((row["ai"], row["baseline"]))
    plt.figure(figsize=(10, 6))
    scatter(tessera_pts, "Tessera", "tab:blue")
    scatter(baseline_pts, "Baseline", "tab:orange")
    all_x = [pt[0] for pt in tessera_pts + baseline_pts]
    all_y = [pt[1] for pt in tessera_pts + baseline_pts]
    if all_x:
        plt.xlim(min(all_x), max(all_x))
    if all_y:
        plt.ylim(min(all_y), max(all_y))
    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("TFLOP/s")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote scatter plot to {output_path}")


def average_from_entry(entry: dict | None) -> float:
    if not entry or entry["count"] == 0:
        return float("nan")
    return entry["sum"] / entry["count"]


def plot_relative_bar(stats: dict, title: str, output_path: Path) -> None:
    category_names = sorted(stats.get("categories", []))
    if len(category_names) > 1:
        categories = ["overall"] + category_names
    else:
        categories = category_names

    def get_avg(strategy: str, category: str) -> float:
        if category == "overall":
            return average_from_entry(stats[strategy]["overall"])
        return average_from_entry(stats[strategy]["per_category"].get(category))

    values: list[tuple[str, float]] = []
    for cat in categories:
        base_avg = get_avg("baseline", cat)
        tess_avg = get_avg("tessera", cat)
        if math.isnan(base_avg) or base_avg == 0 or math.isnan(tess_avg):
            continue
        ratio = tess_avg / base_avg
        values.append((cat, ratio))

    if not values:
        print("Skipping relative performance bar chart due to insufficient data.")
        return

    labels = [cat for cat, _ in values]
    ratios = [ratio for _, ratio in values]
    num_groups = len(labels)
    indices = list(range(num_groups))
    width = 0.35

    plt.figure(figsize=(max(8, num_groups * 0.8), 6))
    plt.bar([i - width / 2 for i in indices], [1.0] * num_groups, width, label="Baseline", color="tab:orange", alpha=0.7)
    plt.bar(
        [i + width / 2 for i in indices],
        ratios,
        width,
        label="Tessera",
        color="tab:blue",
        alpha=0.85,
    )
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xticks(indices, labels, rotation=45, ha="right")
    plt.ylabel("Relative TFLOP/s vs baseline")
    plt.title(f"{title} - Tessera Relative Performance")
    plt.legend()

    all_vals = [1.0] + ratios
    min_val = min(all_vals)
    max_val = max(all_vals)
    ymin = min(0.9, min_val * 0.9)
    ymax = max(1.0, max_val * 1.1)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Wrote relative performance bar chart to {output_path}")


def main() -> None:
    args = parse_args()
    if not args.csv_path.is_file():
        raise SystemExit(f"CSV file not found: {args.csv_path}")
    title = args.title or f"Tessera vs Baseline - {args.csv_path.stem}"
    default_filename = f"{args.csv_path.stem}_tessera_ai_scatter.png"
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
    output_dir = base_dir / args.csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / base_filename

    rows, stats = load_rows(args.csv_path, args.category)
    if not rows:
        raise SystemExit("No valid rows to plot.")

    plot(rows, title, output_path)

    # Heatmaps per series
    tessera_points = [(row["ai"], row["tessera"]) for row in rows]
    tessera_values = [row["tessera"] for row in rows]
    scatter_heat(
        tessera_points,
        tessera_values,
        f"{title} - Tessera Heatmap",
        output_path.with_name(f"{output_path.stem}_tessera_heat{output_path.suffix}"),
    )

    baseline_points = [(row["ai"], row["baseline"]) for row in rows]
    baseline_values = [row["baseline"] for row in rows]
    scatter_heat(
        baseline_points,
        baseline_values,
        f"{title} - Baseline Heatmap",
        output_path.with_name(f"{output_path.stem}_baseline_heat{output_path.suffix}"),
    )

    plot_relative_bar(
        stats,
        title,
        output_path.with_name(f"{output_path.stem}_relative_perf{output_path.suffix}"),
    )


if __name__ == "__main__":
    main()
