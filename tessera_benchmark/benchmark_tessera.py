#!/usr/bin/env python3
"""
Benchmark Tessera schedules against baseline sweep results.

For every baseline sweep JSON in the provided directory, this script:
  * extracts the matrix problem metadata,
  * benchmarks the Tessera kernel with a fixed schedule,
  * compares measured TFLOPS against the best baseline schedule,
  * saves JSON/CSV outputs to the requested results directory.
"""

import argparse
import copy
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

BENCHMARK_TIMEOUT_SECONDS = 10

ORDERING_NAMES = {
    0: "ROW_MAJOR",
    1: "COLUMN_MAJOR",
    2: "SNAKE",
    3: "SPIRAL",
    4: "GILBERT",
}

REFERENCE_BASELINE_KEY = "no_chunking_row_major_1"


class TesseraStrategy:
    """Base class for Tessera schedule selection strategies."""

    name: str = "unnamed"

    def select(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Return scheduling parameters derived from problem metadata."""
        raise NotImplementedError


class DefaultStrategy(TesseraStrategy):
    """Use CLI-provided defaults for Tessera scheduling."""

    def __init__(
        self, ordering0: int, ordering1: int, wgm: int, wgn: int, dtype: str, chunk_size: int
    ):
        self.name = "default"
        self.ordering0 = ordering0
        self.ordering1 = ordering1
        self.wgm = wgm
        self.wgn = wgn
        self.dtype = dtype
        self.chunk_size = chunk_size

    def select(self, metadata: Dict[str, Any]) -> Dict[str, Any]:  # pylint: disable=unused-argument
        return {
            "ordering0": self.ordering0,
            "ordering1": self.ordering1,
            "wgm": self.wgm,
            "wgn": self.wgn,
            "dtype": self.dtype,
            "chunk_size": self.chunk_size,
        }


class HardwareL2Strategy(TesseraStrategy):
    """Placeholder for hardware/L2 driven heuristic."""

    def __init__(self):
        self.name = "HW_L2"

    def select(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide Tessera parameters based on matrix metadata.
        """

        # Temporary placeholder that mirrors the default schedule.
        matrix_dims = metadata.get("matrix_dimensions", {})
        m = matrix_dims.get("m")
        n = matrix_dims.get("n")
        k = matrix_dims.get("k")
        block_dims = metadata.get("block_dimensions", {})
        BLK_M = block_dims.get("BLK_M")
        BLK_N = block_dims.get("BLK_N")
        grid_dims = metadata.get("grid", {})
        grid_m = grid_dims.get("num_pid_m")
        grid_n = grid_dims.get("num_pid_n")
        arch = metadata.get("arch")
        category = metadata.get("category")

        # For now, have the L2 level ordering just use column major
        ordering1 = 1 # COLUMN_MAJOR

        if category in {"Large_N_Block_Panel", "Large_N_K_Panel_Matrix"}:
            # Grid is wider than it is tall, temporal timestep in column-major.
            ordering0 = 1  
            wgm = min(8, grid_m)
            wgn = min(4, grid_n)
        else: 
            ordering0 = 0
            wgm = min(4, grid_m)
            wgn = min(8, grid_n)


        # Set chunk size to area of L2 tile.
        chunk_size = wgm * wgn


        # For now, fall back to a safe default.
        return {
            "ordering0": ordering0,
            "ordering1": ordering1,
            "wgm": wgm,
            "wgn": wgn,
            "dtype": metadata.get("dtype", "bfloat16"),
            "chunk_size": chunk_size
        }

class QuantizedGridStrategy(TesseraStrategy):
    """Placeholder for hardware/L2 driven heuristic."""

    def __init__(self):
        self.name = "L2_QUANTIZED_GRID"

    def select(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide Tessera parameters based on matrix metadata.
        """

        # Temporary placeholder that mirrors the default schedule.
        matrix_dims = metadata.get("matrix_dimensions", {})
        m = matrix_dims.get("m")
        n = matrix_dims.get("n")
        k = matrix_dims.get("k")
        block_dims = metadata.get("block_dimensions", {})
        BLK_M = block_dims.get("BLK_M")
        BLK_N = block_dims.get("BLK_N")
        grid_dims = metadata.get("grid", {})
        grid_m = grid_dims.get("num_pid_m")
        grid_n = grid_dims.get("num_pid_n")
        arch = metadata.get("arch")
        category = metadata.get("category")

        # For now, have the L2 level ordering just use column major
        ordering1 = 1 # COLUMN_MAJOR

        WIDE_CATS = {"Large_N_Block_Panel", "Large_N_K_Panel_Matrix"}         # grid wider than tall
        TALL_CATS = {"Large_M_Panel_Block", "Large_M_K_Matrix_Panel"}       # grid taller than wide

        def choose_axis_tile(dim_len: int, max_tile: int = 8, target_remainders=(8, 4)) -> int:
            """
            Primary axis tile chooser:
            1) exact divisor (largest first)
            2) remainder in {8,4} (prefer 8, then 4; then larger tile)
            3) remainder closest to {8,4} (then larger tile)
            fallback -> 1
            """
            max_tile = min(max_tile, dim_len)

            # 1) exact divisor (largest first)
            for w in range(max_tile, 1, -1):
                if dim_len % w == 0:
                    return w

            # 2) exact target remainder (prefer 8, then 4; then larger w)
            pref_index = {r: i for i, r in enumerate(target_remainders)}
            exact_hits = [w for w in range(max_tile, 1, -1) if (dim_len % w) in target_remainders]
            if exact_hits:
                exact_hits.sort(key=lambda w: (pref_index.get(dim_len % w, 999), -w))
                return exact_hits[0]

            # 3) closest to {8,4}; then larger w
            def remainder_distance(w):
                r = dim_len % w
                return min(abs(r - tr) for tr in target_remainders)

            candidates = list(range(max_tile, 1, -1))
            if candidates:
                return min(candidates, key=lambda w: (remainder_distance(w), -w))

            return 1  # degenerate fallback


        def choose_partner_tile(primary_tile: int, partner_len: int | None = None,
                                max_tile: int = 8, target_area: int = 32) -> int:
            """
            Partner axis tile chooser to hit area≈target_area and keep near-square.
            If partner_len is given, prefer divisors of partner_len; if none exist ≤ max_tile,
            fall back to 1..max_tile.
            """
            divisors = [c for c in range(1, max_tile + 1) if (partner_len is None or partner_len % c == 0)]
            candidates = divisors if divisors else list(range(1, max_tile + 1))
            return min(candidates, key=lambda c: (abs(primary_tile * c - target_area), abs(c - primary_tile)))


        if category in WIDE_CATS:
            # Grid is wider than tall; timestep in column-major.
            ordering0 = 1
            max_primary = 8

            # Primary along height (M): choose WGM first with divisor/remainder logic.
            wgm = choose_axis_tile(grid_m, max_tile=max_primary, target_remainders=(8, 4))
            # Partner along width (N): choose WGN to reach area≈32 and stay near-square; prefer divisors of N.
            wgn = choose_partner_tile(wgm, partner_len=grid_n, max_tile=8, target_area=32)

        elif category in TALL_CATS:
            # Grid is taller than wide; timestep in row-major.
            ordering0 = 0
            max_primary = 8

            # Primary along width (N): choose WGN first with the exact same divisor/remainder logic.
            wgn = choose_axis_tile(grid_n, max_tile=max_primary, target_remainders=(8, 4))
            # Partner along height (M): choose WGM to reach area≈32 and stay near-square; prefer divisors of M.
            wgm = choose_partner_tile(wgn, partner_len=grid_m, max_tile=8, target_area=32)
        elif category in {"Very_Large_Matrix_Matrix"}:
            if grid_m > grid_n:
                ordering0 = 0
                wgn = choose_axis_tile(grid_n, max_tile=8, target_remainders=(8, 4))
                wgm = choose_partner_tile(wgn, partner_len=grid_m, max_tile=8, target_area=32)
            else:
                ordering0 = 1
                wgm = choose_axis_tile(grid_m, max_tile=8, target_remainders=(8, 4))
                wgn = choose_partner_tile(wgm, partner_len=grid_n, max_tile=8, target_area=32)

        else:
            ordering0 = 0
            wgm = min(4, grid_m)
            wgn = min(8, grid_n)


        # Set chunk size to area of L2 tile.
        chunk_size = wgm * wgn


        # For now, fall back to a safe default.
        return {
            "ordering0": ordering0,
            "ordering1": ordering1,
            "wgm": wgm,
            "wgn": wgn,
            "dtype": metadata.get("dtype", "bfloat16"),
            "chunk_size": chunk_size
        }


def build_strategy_registry(args) -> Dict[str, TesseraStrategy]:
    """Construct available strategy instances keyed by name."""
    default_strategy = DefaultStrategy(
        args.ordering0,
        args.ordering1,
        args.wgm,
        args.wgn,
        args.dtype,
        args.chunk_size,
    )
    hw_l2_strategy = HardwareL2Strategy()
    quantized_grid_strategy = QuantizedGridStrategy()
    return {
        # default_strategy.name: default_strategy,
        hw_l2_strategy.name: hw_l2_strategy,
        quantized_grid_strategy.name: quantized_grid_strategy
    }


class BenchmarkError(RuntimeError):
    """Raised when a benchmark invocation fails."""


def run_tessera_benchmark(
    m: int,
    n: int,
    k: int,
    ordering0: int,
    ordering1: int,
    wgm: int,
    wgn: int,
    dtype: str,
    chunk_size: int,
    warmup_ms: int,
    rep_ms: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Invoke run_benchmark.py and return the generated JSON payload."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        "run_benchmark.py",
        str(m),
        str(n),
        str(k),
        str(ordering0),
        str(ordering1),
        str(wgm),
        str(wgn),
        "--dtype",
        dtype,
        "--warmup",
        str(warmup_ms),
        "--rep",
        str(rep_ms),
        "--chunk-size",
        str(chunk_size),
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=base_dir,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkError(
            f"Tessera benchmark timed out after {timeout_seconds}s "
            f"(M={m}, N={n}, K={k})"
        ) from exc

    if completed.returncode != 0:
        raise BenchmarkError(
            f"Tessera benchmark failed with return code {completed.returncode} "
            f"(M={m}, N={n}, K={k})"
        )

    results_path = os.path.join(base_dir, "benchmark_results.json")
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise BenchmarkError(
            "run_benchmark.py did not produce benchmark_results.json"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(
            "Unable to parse benchmark_results.json produced by run_benchmark.py"
        ) from exc

    return data


def calculate_tcc_hit_rate(csv_file: str, kernel_name: str = "persistent_matmul_tessera"):
    """
    Calculate TCC hit rate from rocprof CSV output.
    Hit Rate = (100 * TCC_HIT_sum) / (TCC_HIT_sum + TCC_MISS_sum)
    """
    try:
        df = pd.read_csv(csv_file)
        kernel_df = df[df["Kernel_Name"] == kernel_name]
        if len(kernel_df) == 0:
            print(f"Warning: No data found for kernel '{kernel_name}'")
            return None

        hit_rates = []
        for dispatch_id in kernel_df["Dispatch_Id"].unique():
            dispatch_data = kernel_df[kernel_df["Dispatch_Id"] == dispatch_id]
            hit_sum = None
            miss_sum = None
            for _, row in dispatch_data.iterrows():
                if row["Counter_Name"] == "TCC_HIT_sum":
                    hit_sum = row["Counter_Value"]
                elif row["Counter_Name"] == "TCC_MISS_sum":
                    miss_sum = row["Counter_Value"]

            if hit_sum is not None and miss_sum is not None:
                total_accesses = hit_sum + miss_sum
                hit_rate = (100 * hit_sum) / total_accesses if total_accesses != 0 else 0
                hit_rates.append(
                    {
                        "Dispatch_Id": dispatch_id,
                        "TCC_HIT_sum": hit_sum,
                        "TCC_MISS_sum": miss_sum,
                        "Total_Accesses": total_accesses,
                        "Hit_Rate_pct": hit_rate,
                    }
                )

        if not hit_rates:
            print("Warning: No valid hit rate data found")
            return None

        results_df = pd.DataFrame(hit_rates)
        hit_rates_pct = results_df["Hit_Rate_pct"].dropna()
        if len(hit_rates_pct) == 0:
            return None

        return {
            "tcc_hits": int(results_df["TCC_HIT_sum"].sum()),
            "tcc_misses": int(results_df["TCC_MISS_sum"].sum()),
            "total_accesses": int(results_df["Total_Accesses"].sum()),
            "l2_hit_rate": hit_rates_pct.mean() / 100.0,
            "hit_rate_pct": hit_rates_pct.mean(),
            "min_hit_rate_pct": hit_rates_pct.min(),
            "max_hit_rate_pct": hit_rates_pct.max(),
            "std_hit_rate_pct": hit_rates_pct.std(),
            "num_dispatches": len(hit_rates_pct),
        }

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error calculating TCC hit rate: {exc}")
        return None


def run_tessera_profiler(
    m: int,
    n: int,
    k: int,
    ordering0: int,
    ordering1: int,
    wgm: int,
    wgn: int,
    dtype: str,
    chunk_size: int,
    warmup_ms: int,
    rep_ms: int,
    timeout_seconds: int,
) -> Optional[Dict[str, Any]]:
    """Profile Tessera benchmark with rocprof to gather L2 metrics."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure stale profiler output does not leak into new run.
    pmc_dir = os.path.join(base_dir, "pmc_1")
    if os.path.isdir(pmc_dir):
        shutil.rmtree(pmc_dir, ignore_errors=True)

    rocprof_cmd = [
        "rocprofv3",
        "-i",
        "counters.txt",
        "-o",
        "tessera_benchmark",
        "--",
        sys.executable,
        "run_benchmark.py",
        str(m),
        str(n),
        str(k),
        str(ordering0),
        str(ordering1),
        str(wgm),
        str(wgn),
        "--dtype",
        dtype,
        "--warmup",
        str(warmup_ms),
        "--rep",
        str(rep_ms),
        "--chunk-size",
        str(chunk_size),
    ]

    try:
        completed = subprocess.run(
            rocprof_cmd,
            cwd=base_dir,
            check=False,
            timeout=timeout_seconds,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkError(
            f"rocprof benchmark timed out after {timeout_seconds}s "
            f"(M={m}, N={n}, K={k})"
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip() if completed.stderr else ""
        raise BenchmarkError(
            f"rocprof benchmark failed with return code {completed.returncode} "
            f"(M={m}, N={n}, K={k}): {stderr}"
        )

    csv_file = os.path.join(base_dir, "pmc_1", "tessera_benchmark_counter_collection.csv")
    if not os.path.exists(csv_file):
        print(f"Warning: rocprof CSV file not found: {csv_file}")
        return None

    return calculate_tcc_hit_rate(csv_file, "persistent_matmul_tessera")


def _extract_best_candidate(label: str, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a normalized baseline candidate dictionary from a raw entry."""
    if not isinstance(entry, dict):
        return None

    if "optimal_tflops" in entry:
        tflops = entry.get("optimal_tflops")
        ms = entry.get("optimal_ms")
        wgm = entry.get("optimal_wgm")
        chunk_size = entry.get("optimal_chunk_size")
        row_major = entry.get("optimal_row_major", entry.get("row_major"))
    else:
        tflops = entry.get("tflops")
        ms = entry.get("ms")
        wgm = entry.get("wgm")
        chunk_size = entry.get("chunk_size")
        row_major = entry.get("row_major")

    if tflops is None:
        return None

    return {
        "label": label,
        "tflops": tflops,
        "ms": ms,
        "wgm": wgm,
        "chunk_size": chunk_size,
        "row_major": row_major,
    }


def find_best_baseline(baseline_data: Any) -> Optional[Dict[str, Any]]:
    """Return the best-performing baseline schedule extracted from metadata."""
    if baseline_data is None:
        return None

    best: Optional[Dict[str, Any]] = None

    def consider(candidate: Optional[Dict[str, Any]]):
        nonlocal best
        if not candidate:
            return
        current_best = best.get("tflops") if best else None
        candidate_tflops = candidate.get("tflops")
        if candidate_tflops is None:
            return
        if current_best is None or candidate_tflops > current_best:
            best = candidate

    if isinstance(baseline_data, dict):
        consider(_extract_best_candidate("baseline", baseline_data))
        for key, value in baseline_data.items():
            if isinstance(value, dict):
                consider(_extract_best_candidate(str(key), value))
            elif isinstance(value, list):
                for idx, item in enumerate(value, start=1):
                    consider(_extract_best_candidate(f"{key}[{idx}]", item))
        runs = baseline_data.get("baseline_runs")
        if isinstance(runs, list):
            for idx, item in enumerate(runs, start=1):
                consider(_extract_best_candidate(f"baseline_run[{idx}]", item))

    elif isinstance(baseline_data, list):
        for idx, entry in enumerate(baseline_data, start=1):
            consider(_extract_best_candidate(f"baseline[{idx}]", entry))

    return best


def extract_reference_prediction(
    baseline_data: Any, strategy_key: str = REFERENCE_BASELINE_KEY
) -> Optional[Dict[str, Any]]:
    """Return the predicted baseline entry for the given strategy key."""
    if not isinstance(baseline_data, dict):
        return None

    entry = baseline_data.get(strategy_key)
    if not isinstance(entry, dict):
        return None

    predicted_tflops = entry.get("predicted_tflops")
    if predicted_tflops is None:
        return None

    return {
        "label": strategy_key,
        "tflops": predicted_tflops,
        "ms": entry.get("predicted_ms"),
        "wgm": entry.get("predicted_wgm"),
        "chunk_size": entry.get("predicted_chunk_size"),
        "row_major": entry.get("predicted_row_major"),
    }


def build_output_filenames(
    results_dir: str,
    m: int,
    n: int,
    k: int,
    block_dims: Dict[str, Any],
    arch: Optional[str],
) -> (str, str):
    """Construct JSON/CSV filenames mirroring the sweep naming style."""
    blk_m = block_dims.get("BLK_M", "unknown")
    blk_n = block_dims.get("BLK_N", "unknown")
    blk_k = block_dims.get("BLK_K", "unknown")
    arch_suffix = arch if arch is not None else "unknown"

    base_name = (
        f"tessera_benchmark_results_m{m}_n{n}_k{k}_"
        f"mt{blk_m}_nt{blk_n}_kt{blk_k}_{arch_suffix}"
    )
    json_path = os.path.join(results_dir, f"{base_name}.json")
    csv_path = os.path.join(results_dir, f"{base_name}.csv")
    return json_path, csv_path


def save_results(
    json_path: str,
    csv_path: str,
    metadata: Dict[str, Any],
    tessera_entries: List[Dict[str, Any]],
    baseline_best: Optional[Dict[str, Any]],
    baseline_predicted: Optional[Dict[str, Any]],
) -> None:
    """Persist JSON and CSV outputs for a single problem."""
    baseline_tflops = baseline_best.get("tflops") if baseline_best else None
    predicted_tflops = baseline_predicted.get("tflops") if baseline_predicted else None
    enriched_entries: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for entry in tessera_entries:
        entry_copy = copy.deepcopy(entry)
        tessera_tflops = entry_copy.get("tflops")
        speedup = (
            tessera_tflops / baseline_tflops
            if baseline_tflops and tessera_tflops is not None
            else None
        )
        speedup_predicted = (
            tessera_tflops / predicted_tflops
            if predicted_tflops and tessera_tflops is not None
            else None
        )

        entry_copy["speedup_vs_baseline"] = speedup
        entry_copy["speedup_vs_baseline_reference"] = speedup_predicted
        entry_copy["baseline_label"] = baseline_best.get("label") if baseline_best else None
        entry_copy["baseline_reference_label"] = (
            baseline_predicted.get("label") if baseline_predicted else None
        )

        enriched_entries.append(entry_copy)
        csv_rows.append(
            {
                "category": metadata.get("category", ""),
                "strategy_name": entry_copy.get("strategy_name"),
                "m": metadata.get("matrix_dimensions", {}).get("m"),
                "n": metadata.get("matrix_dimensions", {}).get("n"),
                "k": metadata.get("matrix_dimensions", {}).get("k"),
                "ordering_0": entry_copy.get("ordering_0"),
                "ordering_1": entry_copy.get("ordering_1"),
                "WGM": entry_copy.get("WGM"),
                "WGN": entry_copy.get("WGN"),
                "chunk_size": entry_copy.get("chunk_size"),
                "dtype": entry_copy.get("dtype"),
                "tessera_l2_hit_rate_pct": (entry_copy.get("profiler_data") or {}).get(
                    "hit_rate_pct"
                ),
                "tessera_tflops": tessera_tflops,
                "tessera_ms": entry_copy.get("ms"),
                "baseline_label": entry_copy.get("baseline_label"),
                "baseline_tflops": baseline_tflops,
                "baseline_ms": baseline_best.get("ms") if baseline_best else None,
                "baseline_reference_label": entry_copy.get("baseline_reference_label"),
                "baseline_reference_tflops": predicted_tflops,
                "baseline_reference_ms": baseline_predicted.get("ms")
                if baseline_predicted
                else None,
                "speedup_vs_baseline": speedup,
                "speedup_vs_baseline_reference": speedup_predicted,
            }
        )

    best_entry = None
    if enriched_entries:
        best_entry = max(
            enriched_entries,
            key=lambda e: e.get("tflops") if e.get("tflops") is not None else float("-inf"),
        )

    results_payload = {
        "metadata": metadata,
        "tessera_results": enriched_entries,
        "baseline_best": baseline_best,
        "baseline_reference": baseline_predicted,
    }
    if best_entry:
        results_payload["best_strategy"] = {
            "strategy_name": best_entry.get("strategy_name"),
            "tflops": best_entry.get("tflops"),
            "ms": best_entry.get("ms"),
        }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    csv_fields = [
        "category",
        "strategy_name",
        "m",
        "n",
        "k",
        "ordering_0",
        "ordering_1",
        "WGM",
        "WGN",
        "chunk_size",
        "dtype",
        "tessera_l2_hit_rate_pct",
        "tessera_tflops",
        "tessera_ms",
        "baseline_label",
        "baseline_tflops",
        "baseline_ms",
        "baseline_reference_label",
        "baseline_reference_tflops",
        "baseline_reference_ms",
        "speedup_vs_baseline",
        "speedup_vs_baseline_reference",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)


def load_baseline_metadata(path: str) -> Dict[str, Any]:
    """Read baseline sweep JSON and return its parsed contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Tessera schedules using baseline sweep outputs."
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Directory containing baseline_sweep_results_*.json files.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Output directory for Tessera benchmark comparisons.",
    )
    parser.add_argument(
        "--ordering0",
        type=int,
        default=0,
        help="Tessera ordering0 value (default: 0 / ROW_MAJOR).",
    )
    parser.add_argument(
        "--ordering1",
        type=int,
        default=0,
        help="Tessera ordering1 value (default: 0 / ROW_MAJOR).",
    )
    parser.add_argument(
        "--wgm",
        type=int,
        default=1,
        help="Tessera workgroup size in M dimension (default: 1).",
    )
    parser.add_argument(
        "--wgn",
        type=int,
        default=1,
        help="Tessera workgroup size in N dimension (default: 1).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Data type passed to run_benchmark.py (default: bfloat16).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=-1,
        help="Chunk size forwarded to Tessera matmul (default: -1, chiplet remap).",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated list of strategy names to evaluate (default: all registered strategies).",
    )
    parser.add_argument(
        "--bench-warmup-ms",
        type=int,
        default=50,
        help="Warmup duration (ms) for run_benchmark.py (default: 50).",
    )
    parser.add_argument(
        "--bench-rep-ms",
        type=int,
        default=1000,
        help="Repeat duration (ms) for run_benchmark.py (default: 1000).",
    )
    parser.add_argument(
        "--prof-warmup-ms",
        type=int,
        default=20,
        help="Warmup duration (ms) for rocprof profiling run (default: 20).",
    )
    parser.add_argument(
        "--prof-rep-ms",
        type=int,
        default=100,
        help="Repeat duration (ms) for rocprof profiling run (default: 20).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=BENCHMARK_TIMEOUT_SECONDS,
        help="Timeout for each benchmark invocation (default: 10 seconds).",
    )
    parser.add_argument(
        "--start-problem",
        type=int,
        default=1,
        help="1-based index to start processing baseline files from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_dir = os.path.abspath(args.baseline_dir)
    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(baseline_dir):
        raise SystemExit(f"Baseline directory does not exist: {baseline_dir}")

    strategy_registry = build_strategy_registry(args)
    if args.strategies is None or args.strategies.strip().lower() == "all":
        requested_strategy_names = list(strategy_registry.keys())
    else:
        requested_strategy_names = [
            name.strip() for name in args.strategies.split(",") if name.strip()
        ]
    if not requested_strategy_names:
        raise SystemExit("No strategies specified.")

    strategies: List[TesseraStrategy] = []
    for name in requested_strategy_names:
        strategy = strategy_registry.get(name)
        if strategy is None:
            available = ", ".join(sorted(strategy_registry.keys()))
            raise SystemExit(
                f"Unknown strategy '{name}'. Available strategies: {available}"
            )
        strategies.append(strategy)

    json_paths = sorted(glob.glob(os.path.join(baseline_dir, "*.json")))
    if not json_paths:
        raise SystemExit(f"No baseline JSON files found in {baseline_dir}")

    if args.start_problem < 1:
        raise SystemExit("--start-problem must be >= 1")

    total_files = len(json_paths)
    processed = 0

    for idx, json_path in enumerate(json_paths, start=1):
        if idx < args.start_problem:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        print("=" * 80)
        print(f"Processing baseline file {idx}/{total_files}: {os.path.basename(json_path)}")

        baseline_payload = load_baseline_metadata(json_path)
        metadata = baseline_payload.get("metadata", {})
        if metadata.get("dtype") is None:
            print(
                "Warning: Baseline metadata missing 'dtype'; defaulting to bfloat16 for Tessera runs."
            )
            metadata["dtype"] = "bfloat16"

        matrix_dims = metadata.get("matrix_dimensions", {})

        m = int(matrix_dims.get("m"))
        n = int(matrix_dims.get("n"))
        k = int(matrix_dims.get("k"))
        arch = metadata.get("arch")
        block_dims = metadata.get("block_dimensions", {})

        baseline_data = metadata.get("baseline_data")
        baseline_best = find_best_baseline(baseline_data)
        if not baseline_best:
            print("Warning: Unable to identify a best baseline schedule; skipping.")
            continue
        baseline_reference = extract_reference_prediction(baseline_data)

        tessera_entries: List[Dict[str, Any]] = []

        for strategy in strategies:
            schedule = strategy.select(metadata)
            ordering0 = int(schedule.get("ordering0", args.ordering0))
            ordering1 = int(schedule.get("ordering1", args.ordering1))
            wgm = int(schedule.get("wgm", args.wgm))
            wgn = int(schedule.get("wgn", args.wgn))
            dtype = schedule.get("dtype", args.dtype)
            sched_chunk_size = schedule.get("chunk_size", args.chunk_size)
            chunk_size = int(sched_chunk_size if sched_chunk_size is not None else args.chunk_size)

            print(
                f"  Strategy '{strategy.name}': ordering=({ordering0},{ordering1}), "
                f"WGM={wgm}, WGN={wgn}, dtype={dtype}, chunk_size={chunk_size}"
            )

            try:
                tessera_results = run_tessera_benchmark(
                    m,
                    n,
                    k,
                    ordering0,
                    ordering1,
                    wgm,
                    wgn,
                    dtype,
                    chunk_size,
                    args.bench_warmup_ms,
                    args.bench_rep_ms,
                    args.timeout_seconds,
                )
            except BenchmarkError as exc:
                print(f"Warning: {exc}")
                continue

            try:
                profiler_data = run_tessera_profiler(
                    m,
                    n,
                    k,
                    ordering0,
                    ordering1,
                    wgm,
                    wgn,
                    dtype,
                    chunk_size,
                    args.prof_warmup_ms,
                    args.prof_rep_ms,
                    args.timeout_seconds,
                )
            except BenchmarkError as exc:
                print(f"Warning: {exc}")
                profiler_data = None

            tessera_entry = {
                "strategy_name": strategy.name,
                "ordering_0": ORDERING_NAMES.get(ordering0, f"UNKNOWN_{ordering0}"),
                "ordering_1": ORDERING_NAMES.get(ordering1, f"UNKNOWN_{ordering1}"),
                "ordering0": ordering0,
                "ordering1": ordering1,
                "WGM": wgm,
                "WGN": wgn,
                "dtype": dtype,
                "chunk_size": chunk_size,
                "tflops": tessera_results.get("tflops"),
                "ms": tessera_results.get("ms"),
                "transA": tessera_results.get("transA"),
                "transB": tessera_results.get("transB"),
                "init_type": tessera_results.get("init_type"),
                "profiler_data": profiler_data,
                "strategy_schedule": schedule,
            }
            if profiler_data:
                tessera_entry["l2_hit_rate"] = profiler_data.get("l2_hit_rate")
                tessera_entry["l2_hit_rate_pct"] = profiler_data.get("hit_rate_pct")

            tessera_entries.append(tessera_entry)

        if not tessera_entries:
            print("Warning: All strategies failed for this problem; skipping result save.")
            continue

        metadata_copy = copy.deepcopy(metadata)
        metadata_copy["baseline_source"] = os.path.abspath(json_path)
        metadata_copy["strategy_request"] = {
            "requested_names": requested_strategy_names,
            "cli_default_schedule": {
                "ordering0": args.ordering0,
                "ordering1": args.ordering1,
                "wgm": args.wgm,
                "wgn": args.wgn,
                "dtype": args.dtype,
                "warmup_ms": args.bench_warmup_ms,
                "rep_ms": args.bench_rep_ms,
                "prof_warmup_ms": args.prof_warmup_ms,
                "prof_rep_ms": args.prof_rep_ms,
            },
        }
        metadata_copy["strategies_used"] = [entry["strategy_name"] for entry in tessera_entries]

        json_path_out, csv_path_out = build_output_filenames(
            results_dir,
            m,
            n,
            k,
            block_dims,
            arch,
        )

        save_results(
            json_path_out,
            csv_path_out,
            metadata_copy,
            tessera_entries,
            baseline_best,
            baseline_reference,
        )
        for entry in tessera_entries:
            print(
                f"Strategy {entry['strategy_name']}: TFLOPS={entry.get('tflops')} "
                f"(ordering=({entry['ordering0']},{entry['ordering1']}), "
                f"WGM={entry['WGM']}, WGN={entry['WGN']})"
            )
        print(f"Best baseline TFLOPS: {baseline_best.get('tflops')}")
        if baseline_reference:
            print(
                f"Baseline reference ({baseline_reference.get('label')}): "
                f"{baseline_reference.get('tflops')}"
            )
        print(f"Results saved to: {json_path_out}")
        processed += 1

    print("=" * 80)
    print(f"Benchmark complete. Processed {processed} problems.")


if __name__ == "__main__":
    main()
