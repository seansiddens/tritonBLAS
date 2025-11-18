"""Sample N GEMM problems from a given category and emit them as YAML."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_IN_DTYPE = "bfloat16"
DEFAULT_OUT_DTYPE = "bfloat16"
DEFAULT_TRANSA = "N"
DEFAULT_TRANSB = "T"

BF16_BYTES = 2
BYTES_PER_GB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick N random problems from the requested category (or every category if omitted) "
            "and write a YAML list."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to categorized CSV input file")
    parser.add_argument(
        "--category",
        help="Category name to sample from (omit to sample from every category)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        required=True,
        help="Number of rows to sample",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Optional output filename (default auto-generated in script directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible sampling",
    )
    parser.add_argument(
        "--gb",
        type=float,
        default=2.0,
        help="Maximum bf16 operand size in gigabytes (default: 2)",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required_columns = {"m", "n", "k", "category", "batch_count"}
    missing = required_columns.difference(fieldnames)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    return rows


def filter_batch_count(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        try:
            batch_count = int(row.get("batch_count", 0))
        except (TypeError, ValueError):
            continue
        if batch_count == 1:
            filtered.append(row)
    return filtered


def max_elements_from_gb(gigabytes: float) -> int:
    if gigabytes <= 0:
        raise ValueError("--gb must be positive")
    bytes_limit = gigabytes * BYTES_PER_GB
    return int(bytes_limit // BF16_BYTES)


def filter_category(rows: list[dict[str, str]], category: str) -> list[dict[str, str]]:
    category_lower = category.lower()
    return [row for row in rows if row.get("category", "").lower() == category_lower]


def filter_matrix_size(rows: list[dict[str, str]], max_elements: int) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        try:
            m = int(row["m"])
            n = int(row["n"])
            k = int(row["k"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unable to parse m/n/k as integers for row: {row}") from exc

        if (
            m * k < max_elements
            and k * n < max_elements
            and m * n < max_elements
        ):
            filtered.append(row)
    return filtered


def filter_min_grid(rows: list[dict[str, str]], min_tiles: int) -> list[dict[str, str]]:
    threshold = 256 * min_tiles
    return [
        row
        for row in rows
        if int(row["m"]) > threshold and int(row["n"]) > threshold
    ]


def select_rows(rows: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows but only {len(rows)} available after filtering")
    return rng.sample(rows, count)


def select_rows_per_category(
    groups: dict[str, list[dict[str, str]]], count: int, rng: random.Random
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for category, rows in sorted(groups.items()):
        if len(rows) < count:
            raise ValueError(
                f"Category '{category}' only has {len(rows)} rows available after filtering"
            )
        selected.extend(rng.sample(rows, count))
    return selected


def build_problem(row: dict[str, str]) -> dict[str, int | str]:
    m = int(row["m"])
    n = int(row["n"])
    k = int(row["k"])
    return {
        "in_dtype": DEFAULT_IN_DTYPE,
        "out_dtype": DEFAULT_OUT_DTYPE,
        "transA": DEFAULT_TRANSA,
        "transB": DEFAULT_TRANSB,
        "m": m,
        "n": n,
        "k": k,
        "category": row.get("category", "unknown"),
    }


def format_yaml(problems: list[dict[str, int | str]]) -> str:
    lines: list[str] = []
    for problem in problems:
        lines.append(f"- in_dtype: {problem['in_dtype']}")
        lines.append(f"  out_dtype: {problem['out_dtype']}")
        lines.append(f"  transA: {problem['transA']}")
        lines.append(f"  transB: {problem['transB']}")
        lines.append(f"  m: {problem['m']}")
        lines.append(f"  n: {problem['n']}")
        lines.append(f"  k: {problem['k']}")
        lines.append(f"  category: {problem.get('category', 'unknown')}")
    return "\n".join(lines) + "\n"


def format_gb_suffix(gb_value: float) -> str:
    value_str = f"{gb_value:g}"
    sanitized = value_str.replace(".", "p")
    return f"{sanitized}gb"


def determine_output_path(
    script_dir: Path,
    category: str | None,
    count: int,
    override: str | None,
    gb_suffix: str,
) -> Path:
    if override:
        return script_dir / override
    if category:
        safe_category = "".join(ch if ch.isalnum() else "_" for ch in category.lower()).strip("_")
        filename = f"{safe_category or 'problems'}_{count}_{gb_suffix}.yaml"
    else:
        filename = f"all_categories_{count}_{gb_suffix}.yaml"
    return script_dir / filename


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = load_rows(args.csv_path)
    rows = filter_batch_count(rows)
    if not rows:
        raise SystemExit("No rows with batch_count == 1 found in the CSV.")

    try:
        max_elements = max_elements_from_gb(args.gb)
    except ValueError as exc:
        raise SystemExit(str(exc))

    rows = filter_matrix_size(rows, max_elements)
    if not rows:
        raise SystemExit(
            f"No rows satisfy bf16 matrix < {args.gb}GB constraints after batch filtering."
        )

    rows = filter_min_grid(rows, 5)
    if not rows:
        raise SystemExit("No rows satisfy the minimum grid requirement (m/256 > 5 and n/256 > 5).")

    if args.category:
        filtered = filter_category(rows, args.category)
        if not filtered:
            raise SystemExit(f"No rows found for category '{args.category}' after filtering")
        selected = select_rows(filtered, args.count, rng)
    else:
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            cat = row["category"]
            groups.setdefault(cat, []).append(row)
        if not groups:
            raise SystemExit("No categories remain after filtering.")
        selected = select_rows_per_category(groups, args.count, rng)

    problems = [build_problem(row) for row in selected]

    script_dir = Path(__file__).resolve().parent
    gb_suffix = format_gb_suffix(args.gb)
    output_path = determine_output_path(
        script_dir, args.category, args.count, args.output, gb_suffix
    )
    contents = format_yaml(problems)
    output_path.write_text(contents)
    print(f"Wrote {len(problems)} problems to {output_path}")


if __name__ == "__main__":
    main()
