"""Utility to sample matrix multiplication problems from a categorized CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


DEFAULT_IN_DTYPE = "bfloat16"
DEFAULT_OUT_DTYPE = "bfloat16"
DEFAULT_TRANSA = "N"
DEFAULT_TRANSB = "T"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the categorized CSV (e.g. datasets/Guillermo_Large_1_categorized.csv)",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Problem category to sample from (case insensitive)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        required=True,
        help="How many problems to sample",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Optional output filename (written next to this script)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible sampling",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        required = {"m", "n", "k", "category"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        return rows


def filter_by_category(rows: list[dict[str, str]], category: str) -> list[dict[str, str]]:
    category_lower = category.lower()
    return [row for row in rows if row.get("category", "").lower() == category_lower]


def select_rows(rows: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if len(rows) < count:
        raise ValueError(f"Requested {count} problems but only found {len(rows)} in the category")
    return rng.sample(rows, count)


def build_problem(row: dict[str, str]) -> dict[str, int | str]:
    try:
        m = int(row["m"])
        n = int(row["n"])
        k = int(row["k"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to convert m/n/k to integers for row: {row}") from exc

    return {
        "in_dtype": DEFAULT_IN_DTYPE,
        "out_dtype": DEFAULT_OUT_DTYPE,
        "transA": DEFAULT_TRANSA,
        "transB": DEFAULT_TRANSB,
        "m": m,
        "n": n,
        "k": k,
    }


def format_yaml(problems: list[dict[str, int | str]]) -> str:
    lines: list[str] = []
    for problem in problems:
        lines.append("- in_dtype: {}".format(problem["in_dtype"]))
        lines.append("  out_dtype: {}".format(problem["out_dtype"]))
        lines.append("  transA: {}".format(problem["transA"]))
        lines.append("  transB: {}".format(problem["transB"]))
        lines.append("  m: {}".format(problem["m"]))
        lines.append("  n: {}".format(problem["n"]))
        lines.append("  k: {}".format(problem["k"]))
    return "\n".join(lines) + "\n"


def determine_output_path(script_dir: Path, category: str, count: int, override: str | None) -> Path:
    if override:
        return script_dir / override
    safe_category = "".join(ch if ch.isalnum() else "_" for ch in category.lower()).strip("_")
    filename = f"{safe_category or 'problems'}_{count}.yaml"
    return script_dir / filename


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    rows = load_rows(args.csv_path)
    filtered = filter_by_category(rows, args.category)
    if not filtered:
        raise ValueError(f"No rows found for category '{args.category}'")

    selected = select_rows(filtered, args.count, rng)
    problems = [build_problem(row) for row in selected]

    script_dir = Path(__file__).resolve().parent
    output_path = determine_output_path(script_dir, args.category, args.count, args.output)
    yaml_contents = format_yaml(problems)
    output_path.write_text(yaml_contents)
    print(f"Wrote {len(problems)} problems to {output_path}")


if __name__ == "__main__":
    main()
