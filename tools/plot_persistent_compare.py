"""Plot GFLOP/s vs arithmetic intensity from persistent schedule benchmarks."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


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
    csv_path: Path, category_filter: str | None = None
) -> tuple[dict[str, tuple[list[float], list[float]]], int]:
    points = {
        "default": ([], []),
        "shuffled": ([], []),
    }
    total_rows = 0
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
                comparator_gflops_str = row.get("default_gflops") or row.get("comparator_gflops")
                default_gflops = float(comparator_gflops_str) if comparator_gflops_str else float("nan")
                shuffled_gflops = float(row.get("shuffled_gflops", "nan"))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid row encountered: {row}") from exc

            bytes_per_elem = dtype_num_bytes(in_dtype)
            ai = arithmetic_intensity(m, n, k, bytes_per_elem)
            if math.isnan(ai):
                continue

            if not math.isnan(default_gflops):
                points["default"][0].append(ai)
                points["default"][1].append(default_gflops / 1000)  # convert to TFLOP/s
            if not math.isnan(shuffled_gflops):
                points["shuffled"][0].append(ai)
                points["shuffled"][1].append(shuffled_gflops / 1000)
    return points, total_rows


def plot(
    points: dict[str, tuple[list[float], list[float]]],
    title: str,
    output_path: Path,
    axis_points: dict[str, tuple[list[float], list[float]]] | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    colors = {"default": "tab:blue", "shuffled": "tab:orange"}
    labels = {"default": "MALL aware", "shuffled": "random"}

    for label, (x_vals, y_vals) in points.items():
        if not x_vals:
            continue
        plt.scatter(
            x_vals,
            y_vals,
            label=labels[label],
            alpha=0.75,
            edgecolors="none",
            s=40,
            color=colors[label],
        )

    plt.xlabel("Arithmetic Intensity")
    plt.ylabel("TFLOP/s")
    plt.title(title)
    plt.legend()
    source = axis_points or points
    flat_x = [x for vals in source.values() for x in vals[0]]
    flat_y = [y for vals in source.values() for y in vals[1]]
    if flat_x:
        plt.xlim(min(flat_x), max(flat_x))
    if flat_y:
        plt.ylim(min(flat_y), max(flat_y))
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Wrote scatter plot to {output_path}")


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path
    if not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")

    category_filter = args.category
    if args.output:
        output_path = args.output
    else:
        suffix = f"_{slugify(category_filter)}" if category_filter else ""
        output_path = Path(f"{csv_path.stem}{suffix}_ai_scatter.png")

    full_points, total_rows_all = load_points(csv_path, None)
    if category_filter:
        points, _ = load_points(csv_path, category_filter)
        axis_points = full_points
    else:
        points = full_points
        axis_points = None

    if not any(points[label][0] for label in points):
        raise SystemExit("No valid data points to plot.")

    if category_filter:
        default_title = f"Random vs Mall Aware Grid Schedules - {category_filter}"
    else:
        default_title = "Random vs Mall Aware Grid Schedules"
    title = args.title or default_title

    plot(points, title, output_path, axis_points=axis_points)


if __name__ == "__main__":
    main()
