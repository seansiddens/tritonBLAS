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


DEFAULT_HIERARCHICAL_CONFIG = tritonblas.HierarchicalPersistentConfig(
    ordering0=1,
    ordering1=1,
    ordering2=1,
    L3Y=2,
    L3X=4,
    L2Y=8,
    L2X=4,
)


def parse_hierarchical_config(arg: str | None):
    if arg is None:
        return DEFAULT_HIERARCHICAL_CONFIG
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != 7:
        raise ValueError(
            "hierarchical-config must provide 7 comma-separated ints: "
            "ordering0,ordering1,ordering2,L3Y,L3X,L2Y,L2X."
        )
    ints = list(map(int, parts))
    return tritonblas.HierarchicalPersistentConfig(
        ordering0=ints[0],
        ordering1=ints[1],
        ordering2=ints[2],
        L3Y=ints[3],
        L3X=ints[4],
        L2Y=ints[5],
        L2X=ints[6],
    )


MIN_GRID_TILES = 16


def matmul_time(A, B, C, selector, schedule: str, hierarchical_config=None):
    def run():
        tritonblas.matmul_lt(
            A,
            B,
            C,
            selector,
            enable_streamk=False,
            workgroup_schedule=schedule,
            shuffle_seed=None,
            hierarchical_config=hierarchical_config,
        )

    ms = triton.testing.do_bench(run, warmup=20, rep=100)
    return ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark tessera vs baseline persistent matmul schedules")
    parser.add_argument("input_yaml", type=str, help="Path to YAML file of problems.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional output CSV filename. Defaults to <input>_tessera_compare.csv",
    )
    parser.add_argument(
        "--shuffle-order",
        action="store_true",
        help="Shuffle order of benchmark cases before running.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case benchmark information.",
    )
    parser.add_argument(
        "--hierarchical-config",
        type=str,
        default=None,
        help=(
            "Comma-separated ordering0,ordering1,ordering2,L3Y,L3X,L2Y,L2X "
            "values for the hierarchical schedule. If omitted, the default tessera config is used."
        ),
    )
    args = parser.parse_args()

    with open(args.input_yaml, "r") as f:
        problems = yaml.safe_load(f)

    if args.shuffle_order:
        import random

        random.shuffle(problems)

    output_csv = (
        args.output_csv if args.output_csv else f"{Path(args.input_yaml).stem}_tessera_compare.csv"
    )

    hierarchical_config = parse_hierarchical_config(args.hierarchical_config)

    fieldnames = [
        "category",
        "m",
        "n",
        "k",
        "in_dtype",
        "out_dtype",
        "transA",
        "transB",
        "macro_tile",
        "group_size_m",
        "grid_tiles_m",
        "grid_tiles_n",
        "baseline_us",
        "baseline_gflops",
        "tessera_us",
        "tessera_gflops",
    ]
    rows = []
    errors: list[str] = []
    skipped = 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, case in enumerate(problems):
            m = case["m"]
            n = case["n"]
            k = case["k"]
            transA = case.get("transA", "T")
            transB = case.get("transB", "T")
            category = case.get("category", "unknown")
            try:
                in_dtype = str_to_dtype(case["in_dtype"])
                out_dtype = str_to_dtype(case["out_dtype"])

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

                selector = tritonblas.MatmulHeuristicResult(
                    m, n, k, in_dtype, in_dtype, out_dtype
                )
                config = selector.get_config()
                gsize_m = config[3]
                total_blocks_m = int(triton.cdiv(m, config[0]))
                total_blocks_n = int(triton.cdiv(n, config[1]))

                if total_blocks_m < MIN_GRID_TILES or total_blocks_n < MIN_GRID_TILES:
                    skipped += 1
                    if args.verbose:
                        print(
                            f"[{idx+1}/{len(problems)}] Skipping category={category} (grid too small "
                            f"{total_blocks_m}x{total_blocks_n})."
                        )
                    continue

                problem_flops = 2 * m * n * k * 1e-9

                C_baseline = torch.zeros((m, n), device="cuda", dtype=out_dtype)
                baseline_ms = matmul_time(
                    A,
                    B,
                    C_baseline,
                    selector,
                    schedule="baseline",
                    hierarchical_config=None,
                )
                baseline_gflops = problem_flops / (baseline_ms * 1e-3) if baseline_ms > 0 else float("nan")

                C_tessera = torch.zeros((m, n), device="cuda", dtype=out_dtype)
                tessera_ms = matmul_time(
                    A,
                    B,
                    C_tessera,
                    selector,
                    schedule="hierarchical",
                    hierarchical_config=hierarchical_config,
                )
                tessera_gflops = problem_flops / (tessera_ms * 1e-3) if tessera_ms > 0 else float("nan")

                row = {
                    "category": category,
                    "m": m,
                    "n": n,
                    "k": k,
                    "in_dtype": str(in_dtype),
                    "out_dtype": str(out_dtype),
                    "transA": transA,
                    "transB": transB,
                    "macro_tile": f"{config[0]}x{config[1]}x{config[2]}",
                    "group_size_m": gsize_m,
                    "grid_tiles_m": total_blocks_m,
                    "grid_tiles_n": total_blocks_n,
                    "baseline_us": baseline_ms / 1000,
                    "baseline_gflops": baseline_gflops,
                    "tessera_us": tessera_ms / 1000,
                    "tessera_gflops": tessera_gflops,
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)

                if args.verbose:
                    print(
                        f"[{idx+1}/{len(problems)}] category={category}, m={m}, n={n}, k={k}, "
                        f"in={in_dtype}, out={out_dtype}, baseline={baseline_gflops:.2f} GF/s, "
                        f"tessera={tessera_gflops:.2f} GF/s"
                    )
            except Exception as exc:
                err_msg = (
                    f"[{idx+1}/{len(problems)}] ERROR for category={category}, m={m}, n={n}, k={k}, "
                    f"transA={transA}, transB={transB}: {exc}"
                )
                print(err_msg)
                errors.append(err_msg)
                continue

    print(f"Wrote {len(rows)} benchmark rows to {output_csv}")
    if skipped:
        print(f"Skipped {skipped} cases due to insufficient grid size for tessera schedule.")
    if errors:
        print(f"{len(errors)} cases failed and were skipped; see messages above.")

    if rows:
        deltas = [
            ((row["tessera_gflops"] - row["baseline_gflops"]) / row["baseline_gflops"]) * 100
            for row in rows
            if row["baseline_gflops"] > 0
        ]
        avg_delta = sum(deltas) / len(deltas) if deltas else float("nan")
        print(f"Tessera vs baseline average delta: {avg_delta:+.2f}% over {len(deltas)} comparable rows.")


if __name__ == "__main__":
    main()
