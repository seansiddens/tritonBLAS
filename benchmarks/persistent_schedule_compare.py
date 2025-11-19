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


def matmul_time(A, B, C, selector, schedule: str, shuffle_seed: int | None, hierarchical_config=None):
    def run():
        tritonblas.matmul_lt(
            A,
            B,
            C,
            selector,
            enable_streamk=False,
            workgroup_schedule=schedule,
            shuffle_seed=shuffle_seed,
            hierarchical_config=hierarchical_config,
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
    parser.add_argument(
        "--hierarchical-config",
        type=str,
        default=None,
        help=(
            "Comma-separated ordering0,ordering1,ordering2,L3Y,L3X,L2Y,L2X "
            "values for the hierarchical schedule. If omitted, the schedule is skipped."
        ),
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
        "num_l2_tiles",
        "grid_tiles_m",
        "grid_tiles_n",
        "comparator",
        "comparator_us",
        "comparator_gflops",
        "workgroup_shuffle_us",
        "workgroup_shuffle_gflops",
        "shuffled_us",
        "shuffled_gflops",
    ]
    rows = []
    random_comparisons: list[tuple[str, float, float]] = []
    workgroup_comparisons: list[tuple[str, float, float]] = []
    errors: list[str] = []
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
                num_l2_tiles = count_l2_tiles(total_blocks_m, total_blocks_n, gsize_m)

                use_hierarchical = (
                    hierarchical_config is not None
                    and total_blocks_m >= MIN_GRID_TILES
                    and total_blocks_n >= MIN_GRID_TILES
                )
                comparator_label = "hierarchical" if use_hierarchical else "baseline"
                comparator_schedule = "hierarchical" if use_hierarchical else "default"
                comparator_cfg = hierarchical_config if use_hierarchical else None

                problem_flops = 2 * m * n * k * 1e-9

                C_comp = torch.zeros((m, n), device="cuda", dtype=out_dtype)
                comparator_ms = matmul_time(
                    A,
                    B,
                    C_comp,
                    selector,
                    comparator_schedule,
                    None,
                    comparator_cfg,
                )
                comparator_gflops = problem_flops / (comparator_ms * 1e-3)

                C_workgroup = torch.zeros((m, n), device="cuda", dtype=out_dtype)
                workgroup_shuffle_ms = matmul_time(
                    A, B, C_workgroup, selector, "workgroup_shuffle", args.shuffle_seed, None
                )
                workgroup_shuffle_gflops = (
                    problem_flops / (workgroup_shuffle_ms * 1e-3)
                    if workgroup_shuffle_ms > 0
                    else float("nan")
                )

                if num_l2_tiles <= 1:
                    raise ValueError(
                        "Shuffled workgroup schedule requires at least two full L2 tiles in the quantized region."
                    )
                C_random = torch.zeros((m, n), device="cuda", dtype=out_dtype)
                shuffled_ms = matmul_time(
                    A, B, C_random, selector, "random", args.shuffle_seed, None
                )
                shuffled_gflops = (
                    problem_flops / (shuffled_ms * 1e-3) if shuffled_ms > 0 else float("nan")
                )

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
                    "num_l2_tiles": num_l2_tiles,
                    "grid_tiles_m": total_blocks_m,
                    "grid_tiles_n": total_blocks_n,
                    "comparator": comparator_label,
                    "comparator_us": comparator_ms / 1000,
                    "comparator_gflops": comparator_gflops,
                    "workgroup_shuffle_us": workgroup_shuffle_ms / 1000,
                    "workgroup_shuffle_gflops": workgroup_shuffle_gflops,
                    "shuffled_us": shuffled_ms / 1000,
                    "shuffled_gflops": shuffled_gflops,
                }
                writer.writerow(row)
                f.flush()
                rows.append(row)
                if not math.isnan(workgroup_shuffle_gflops):
                    workgroup_comparisons.append(
                        (comparator_label, comparator_gflops, workgroup_shuffle_gflops)
                    )
                if not math.isnan(shuffled_gflops):
                    random_comparisons.append((comparator_label, comparator_gflops, shuffled_gflops))

                if args.verbose:
                    msg = (
                        f"[{idx+1}/{len(problems)}] category={category}, m={m}, n={n}, k={k}, "
                        f"in={in_dtype}, out={out_dtype}, comparator({comparator_label})={comparator_gflops:.2f} GF/s, "
                        f"workgroup_shuffle={workgroup_shuffle_gflops:.2f} GF/s, "
                        f"shuffled={shuffled_gflops:.2f} GF/s"
                    )
                    print(msg)
            except Exception as exc:
                err_msg = (
                    f"[{idx+1}/{len(problems)}] ERROR for category={category}, m={m}, n={n}, k={k}, "
                    f"transA={transA}, transB={transB}: {exc}"
                )
                print(err_msg)
                errors.append(err_msg)
                continue

    print(f"Wrote {len(rows)} benchmark rows to {output_csv}")
    if rows:
        print(f"Wrote {len(rows)} benchmark rows to {output_csv}")
    else:
        print("No benchmark rows were written due to errors.")

    if errors:
        print(f"{len(errors)} cases failed and were skipped; see messages above.")

    def print_comparison_summary(name: str, data: list[tuple[str, float, float]]):
        if not data:
            print(f"Summary: no comparable cases with {name} schedule were collected.")
            return
        num_cases = len(data)
        comparator_better = sum(1 for _, comp, shuf in data if comp > shuf)
        shuffled_better = sum(1 for _, comp, shuf in data if comp < shuf)
        pct_diffs = [((comp - shuf) / shuf) * 100 for _, comp, shuf in data if shuf > 0]
        avg_pct_diff = sum(pct_diffs) / len(pct_diffs) if pct_diffs else float("nan")
        mode_counts: dict[str, int] = {"baseline": 0, "hierarchical": 0}
        for mode, _, _ in data:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        print(
            f"Summary (comparator vs {name}, GFLOP/s): "
            f"avg delta {avg_pct_diff:+.2f}% over {num_cases} comparable cases; "
            f"comparator faster in {comparator_better} ({comparator_better / num_cases * 100:.1f}%), "
            f"{name} faster in {shuffled_better} ({shuffled_better / num_cases * 100:.1f}%)."
        )
        print(
            f"Comparisons by mode: baseline={mode_counts.get('baseline',0)}, "
            f"hierarchical={mode_counts.get('hierarchical',0)}"
        )

    print_comparison_summary("workgroup_shuffle", workgroup_comparisons)
    print_comparison_summary("random", random_comparisons)


if __name__ == "__main__":
    main()
