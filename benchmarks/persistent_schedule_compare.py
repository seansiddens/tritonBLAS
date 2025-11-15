#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import torch
import triton
import triton.testing
import yaml

import tritonblas


def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.replace("torch.", "")
    try:
        return getattr(torch, dtype_str)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype string '{dtype_str}'") from exc


def randn_like(size, dtype):
    tmp = torch.randn(size, device="cuda", dtype=torch.float32)
    return tmp.to(dtype)


def count_l2_tiles(num_pid_m: int, num_pid_n: int, tile_dim: int) -> int:
    if tile_dim <= 0:
        return 0
    quantized_m = (num_pid_m // tile_dim) * tile_dim
    quantized_n = (num_pid_n // tile_dim) * tile_dim
    tiles_per_row = quantized_n // tile_dim
    tiles_per_col = quantized_m // tile_dim
    return tiles_per_row * tiles_per_col


def matmul_time(A, B, C, selector, schedule: str, shuffle_seed: int | None):
    def run():
        tritonblas.matmul_lt(
            A,
            B,
            C,
            selector,
            enable_streamk=False,
            workgroup_schedule=schedule,
            shuffle_seed=shuffle_seed,
        )

    ms = triton.testing.do_bench(run, warmup=20, rep=100)
    return ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark persistent matmul default vs shuffled scheduling"
    )
    parser.add_argument("input_yaml", type=str, help="Path to YAML file of problems.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output CSV filename. Defaults to <input>_persistent_compare.csv",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed used when benchmarking the shuffled schedule.",
    )
    parser.add_argument(
        "--shuffle-order",
        action="store_true",
        help="Shuffle order of benchmark cases before running.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-case benchmark information."
    )
    args = parser.parse_args()

    with open(args.input_yaml, "r") as f:
        problems = yaml.safe_load(f)

    if args.shuffle_order:
        import random

        random.shuffle(problems)

    output_csv = (
        args.output_csv
        if args.output_csv
        else f"{Path(args.input_yaml).stem}_persistent_compare.csv"
    )

    rows = []
    comparisons: list[tuple[float, float]] = []
    for case in problems:
        m = case["m"]
        n = case["n"]
        k = case["k"]
        in_dtype = str_to_dtype(case["in_dtype"])
        out_dtype = str_to_dtype(case["out_dtype"])
        transA = case.get("transA", "T")
        transB = case.get("transB", "T")

        if transA == "T":
            A_size = (m, k)
        else:
            A_size = (k, m)

        if transB == "T":
            B_size = (k, n)
        else:
            B_size = (n, k)

        A = randn_like(A_size, in_dtype)
        B = randn_like(B_size, in_dtype)

        if transA == "N":
            A = A.T
        if transB == "N":
            B = B.T

        selector = tritonblas.MatmulHeuristicResult(m, n, k, in_dtype, in_dtype, out_dtype)
        config = selector.get_config()
        gsize_m = config[3]
        total_blocks_m = int(triton.cdiv(m, config[0]))
        total_blocks_n = int(triton.cdiv(n, config[1]))
        num_l2_tiles = count_l2_tiles(total_blocks_m, total_blocks_n, gsize_m)

        C_default = torch.zeros((m, n), device="cuda", dtype=out_dtype)
        default_ms = matmul_time(A, B, C_default, selector, "default", None)

        shuffled_ms = float("nan")
        shuffled_gflops = float("nan")
        if num_l2_tiles > 1:
            C_random = torch.zeros((m, n), device="cuda", dtype=out_dtype)
            shuffled_ms = matmul_time(
                A, B, C_random, selector, "random", args.shuffle_seed
            )
            shuffled_gflops = (
                2 * m * n * k * 1e-9 / (shuffled_ms * 1e-3) if shuffled_ms > 0 else float("nan")
            )

        default_gflops = 2 * m * n * k * 1e-9 / (default_ms * 1e-3)

        row = {
            "m": m,
            "n": n,
            "k": k,
            "in_dtype": str(in_dtype),
            "out_dtype": str(out_dtype),
            "transA": transA,
            "transB": transB,
            "macro_tile": f"{config[0]}x{config[1]}x{config[2]}",
            "group_size_m": gsize_m,
            "num_l2_tiles": num_l2_tiles,
            "default_us": default_ms / 1000,
            "default_gflops": default_gflops,
            "shuffled_us": (shuffled_ms / 1000) if num_l2_tiles > 1 else float("nan"),
            "shuffled_gflops": shuffled_gflops,
        }
        rows.append(row)
        if num_l2_tiles > 1 and not math.isnan(shuffled_gflops):
            comparisons.append((default_gflops, shuffled_gflops))

        if args.verbose:
            msg = (
                f"m={m}, n={n}, k={k}, in={in_dtype}, out={out_dtype}, "
                f"default={default_gflops:.2f} GF/s"
            )
            if num_l2_tiles > 1:
                msg += f", shuffled={shuffled_gflops:.2f} GF/s"
            else:
                msg += ", shuffled=N/A (insufficient L2 tiles)"
            print(msg)

    fieldnames = [
        "m",
        "n",
        "k",
        "in_dtype",
        "out_dtype",
        "transA",
        "transB",
        "macro_tile",
        "group_size_m",
        "num_l2_tiles",
        "default_us",
        "default_gflops",
        "shuffled_us",
        "shuffled_gflops",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark rows to {output_csv}")
    if comparisons:
        num_cases = len(comparisons)
        default_better = sum(1 for d, s in comparisons if d > s)
        shuffled_better = sum(1 for d, s in comparisons if d < s)
        pct_diffs = [((d - s) / s) * 100 for d, s in comparisons if s > 0]
        avg_pct_diff = sum(pct_diffs) / len(pct_diffs) if pct_diffs else float("nan")
        print(
            "Summary (default vs shuffled, GFLOP/s): "
            f"avg delta {avg_pct_diff:+.2f}% over {num_cases} comparable cases; "
            f"default faster in {default_better} ({default_better / num_cases * 100:.1f}%), "
            f"shuffled faster in {shuffled_better} ({shuffled_better / num_cases * 100:.1f}%)."
        )
    else:
        print("Summary: no comparable cases with shuffled schedule were collected.")


if __name__ == "__main__":
    main()
