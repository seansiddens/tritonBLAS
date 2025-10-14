#!/usr/bin/env python3
"""
Sweep script for tessera matmul across different configurations.
Reads matrix problems from CSV and sweeps through orderings and workgroup sizes.
"""

import argparse
import copy
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime
import torch
import tritonblas
import pandas as pd
import numpy as np
import math
import time

BENCHMARK_TIMEOUT_SECONDS = 10
PROCESS_GROUP_GRACE_SECONDS = 2


class BenchmarkTimeoutError(RuntimeError):
    def __init__(self, cmd, timeout, stdout=None, stderr=None):
        cmd_str = " ".join(shlex.quote(str(part)) for part in cmd)
        super().__init__(f"Timed out after {timeout}s: {cmd_str}")
        self.cmd = cmd
        self.timeout = timeout
        self.stdout = stdout
        self.stderr = stderr


def terminate_process_group(proc):
    """Send SIGTERM/SIGKILL to an entire process group."""
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + PROCESS_GROUP_GRACE_SECONDS
    while proc.poll() is None and time.time() < deadline:
        time.sleep(0.1)

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_subprocess_with_timeout(
    cmd,
    cwd,
    timeout=BENCHMARK_TIMEOUT_SECONDS,
    capture_output=True,
    text=True,
    env=None,
):
    """Run a subprocess with timeout and ensure child processes are terminated."""
    stdout_pipe = subprocess.PIPE if capture_output else None
    stderr_pipe = subprocess.PIPE if capture_output else None

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=stdout_pipe,
        stderr=stderr_pipe,
        text=text,
        env=env,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        raise BenchmarkTimeoutError(cmd, timeout, stdout=stdout, stderr=stderr)

    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

MISCOPE_GPU_ID = "3"
MISCOPE_COLUMNS = [
    "curr_gfxclk",
    "curr_socclk",
    "curr_uclk",
    "curr_power",
    "pcie_bandwidth_inst",
] + [f"curr_gfxclks{i}" for i in range(8)]

MISCOPE_MEAN_KEYS = [f"{col}_mean" for col in MISCOPE_COLUMNS]
MISCOPE_OUTPUT_DIR = "miscope_metrics"

STRATEGY_LABELS = {
    "no_chunking_row_major_1": "No chunking / Row Major 1",
    "no_chunking_row_major_0": "No chunking / Row Major 0",
    "chiplet_chunk_row_major_1": "Chiplet chunk / Row Major 1",
    "chiplet_chunk_row_major_0": "Chiplet chunk / Row Major 0",
}

STRATEGY_LABEL_ORDER = [
    "no_chunking_row_major_1",
    "no_chunking_row_major_0",
    "chiplet_chunk_row_major_1",
    "chiplet_chunk_row_major_0",
]

REFERENCE_BASELINE_KEY = "no_chunking_row_major_1"


def compute_sweep_summary(metadata, sweep_results):
    """Build summary information for sweep results."""
    summary = {
        "best_schedule": None,
        "best_schedule_index": None,
        "best_tflops": None,
        "best_schedules_by_strategy": {},
        "speedup_vs_predicted_no_chunking_rm1": None,
        "speedup_vs_best_no_chunking_rm1": None,
        "reference_baseline_key": REFERENCE_BASELINE_KEY,
        "num_sweep_results": len(sweep_results or [])
    }

    if not sweep_results:
        return summary

    def tflops_key(entry):
        value = entry.get("tflops")
        return value if value is not None else float("-inf")

    best_idx = None
    best_entry = None
    best_entry_tflops = float("-inf")

    strategy_bests = {}

    for idx, entry in enumerate(sweep_results, start=1):
        entry_tflops = tflops_key(entry)

        if best_entry is None or entry_tflops > best_entry_tflops:
            best_entry = entry
            best_entry_tflops = entry_tflops
            best_idx = idx

        chunking_strategy = entry.get("chunking_strategy")
        row_major = entry.get("row_major")
        if chunking_strategy is not None and row_major is not None:
            strategy_key = f"{chunking_strategy}_row_major_{row_major}"
            current_best = strategy_bests.get(strategy_key)
            if current_best is None or entry_tflops > tflops_key(current_best["schedule"]):
                strategy_bests[strategy_key] = {
                    "schedule": copy.deepcopy(entry),
                    "schedule_index": idx
                }

    # Ensure all known strategy keys are present (even if None)
    for strategy_key in STRATEGY_LABELS.keys():
        strategy_bests.setdefault(strategy_key, None)

    baseline_data = (metadata or {}).get("baseline_data", {}) or {}
    best_tflops = best_entry.get("tflops") if best_entry else None

    summary.update({
        "best_schedule_index": best_idx,
        "best_tflops": best_tflops,
        "best_schedule": copy.deepcopy(best_entry) if best_entry else None,
        "best_schedules_by_strategy": strategy_bests
    })

    reference_key = summary["reference_baseline_key"]
    reference_data = None
    if isinstance(baseline_data, dict):
        reference_data = baseline_data.get(reference_key)

    if reference_data:
        predicted_tflops = reference_data.get("predicted_tflops")
        optimal_tflops = reference_data.get("optimal_tflops")

        if best_tflops is not None and predicted_tflops and predicted_tflops > 0:
            summary["speedup_vs_predicted_no_chunking_rm1"] = best_tflops / predicted_tflops

        if best_tflops is not None and optimal_tflops and optimal_tflops > 0:
            summary["speedup_vs_best_no_chunking_rm1"] = best_tflops / optimal_tflops

    return summary


def sanitize_for_filename(value):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))


def build_miscope_prefix(arch, m, n, k, ordering0, ordering1, wgm, wgn, dtype):
    problem_folder = os.path.join(
        MISCOPE_OUTPUT_DIR,
        f"m{m}_n{n}_k{k}"
    )

    ordering_name_0 = sanitize_for_filename(get_ordering_name(ordering0))
    ordering_name_1 = sanitize_for_filename(get_ordering_name(ordering1))

    parts = [
        f"arch{sanitize_for_filename(arch)}",
        f"o{ordering_name_0}_{ordering_name_1}",
        f"wgm{wgm}",
        f"wgn{wgn}",
        f"dtype{sanitize_for_filename(dtype)}"
    ]
    filename = "_".join(parts)
    return os.path.join(problem_folder, filename)


def build_baseline_miscope_prefix(arch, m, n, k, wgm, dtype):
    problem_folder = os.path.join(
        MISCOPE_OUTPUT_DIR,
        f"m{m}_n{n}_k{k}"
    )

    parts = [
        f"arch{sanitize_for_filename(arch)}",
        "baseline",
        f"wgm{wgm}",
        f"dtype{sanitize_for_filename(dtype)}"
    ]
    filename = "_".join(parts)
    return os.path.join(problem_folder, filename)

def get_all_wgm_wgn_combinations(max_wgm=8, max_wgn=8, num_pid_m=None, num_pid_n=None):
    """Generate all WGM and WGN combinations from 1 to max values, constrained by grid dimensions."""
    combinations = []
    
    # Apply grid dimension constraints if provided
    actual_max_wgm = max_wgm
    actual_max_wgn = max_wgn
    
    if num_pid_m is not None:
        actual_max_wgm = min(max_wgm, num_pid_m)
    if num_pid_n is not None:
        actual_max_wgn = min(max_wgn, num_pid_n)
    
    for wgm in range(1, actual_max_wgm + 1):
        for wgn in range(1, actual_max_wgn + 1):
            combinations.append((wgm, wgn))
    return combinations

def get_ordering_name(ordering):
    """Convert ordering number to name."""
    names = {0: "ROW_MAJOR", 1: "COLUMN_MAJOR", 2: "SNAKE", 3: "SPIRAL", 4: "GILBERT"}
    return names.get(ordering, f"UNKNOWN_{ordering}")

def calculate_tcc_hit_rate(csv_file, kernel_name='persistent_matmul_tessera'):
    """
    Calculate TCC hit rate from rocprof CSV output.
    Hit Rate = (100 * TCC_HIT_sum) / (TCC_HIT_sum + TCC_MISS_sum)
    """
    try:
        # Load CSV data
        df = pd.read_csv(csv_file)
        
        # Filter for the specific kernel
        kernel_df = df[df['Kernel_Name'] == kernel_name]
        
        if len(kernel_df) == 0:
            print(f"Warning: No data found for kernel '{kernel_name}'")
            return None
        
        print(f"Found {len(kernel_df)} rows for kernel '{kernel_name}'")
        print(f"Unique dispatches: {kernel_df['Dispatch_Id'].nunique()}")
        
        # Group by dispatch ID to get HIT and MISS values for each dispatch
        hit_rates = []
        
        for dispatch_id in kernel_df['Dispatch_Id'].unique():
            dispatch_data = kernel_df[kernel_df['Dispatch_Id'] == dispatch_id]
            
            # Get TCC_HIT_sum and TCC_MISS_sum for this dispatch
            hit_sum = None
            miss_sum = None
            
            for _, row in dispatch_data.iterrows():
                if row['Counter_Name'] == 'TCC_HIT_sum':
                    hit_sum = row['Counter_Value']
                elif row['Counter_Name'] == 'TCC_MISS_sum':
                    miss_sum = row['Counter_Value']
            
            # Calculate hit rate
            if hit_sum is not None and miss_sum is not None:
                total_accesses = hit_sum + miss_sum
                if total_accesses != 0:
                    hit_rate = (100 * hit_sum) / total_accesses
                else:
                    hit_rate = 0
                    
                hit_rates.append({
                    'Dispatch_Id': dispatch_id,
                    'TCC_HIT_sum': hit_sum,
                    'TCC_MISS_sum': miss_sum,
                    'Total_Accesses': total_accesses,
                    'Hit_Rate_pct': hit_rate
                })
            else:
                print(f"Warning: Missing counter data for dispatch {dispatch_id}")
        
        if not hit_rates:
            print("Warning: No valid hit rate data found")
            return None
            
        # Convert to DataFrame
        results_df = pd.DataFrame(hit_rates)
        
        # Calculate statistics
        hit_rates_pct = results_df['Hit_Rate_pct'].dropna()
        if len(hit_rates_pct) == 0:
            return None
            
        return {
            'tcc_hits': int(results_df['TCC_HIT_sum'].sum()),
            'tcc_misses': int(results_df['TCC_MISS_sum'].sum()),
            'total_accesses': int(results_df['Total_Accesses'].sum()),
            'l2_hit_rate': hit_rates_pct.mean() / 100.0,
            'hit_rate_pct': hit_rates_pct.mean(),
            'min_hit_rate_pct': hit_rates_pct.min(),
            'max_hit_rate_pct': hit_rates_pct.max(),
            'std_hit_rate_pct': hit_rates_pct.std(),
            'num_dispatches': len(hit_rates_pct)
        }
        
    except Exception as e:
        print(f"Error calculating TCC hit rate: {e}")
        return None


def run_benchmark_with_miscope(
    bench_cmd,
    base_dir,
    metrics_prefix="metrics",
    gpu_ids="0",
    timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
):
    """Run the benchmark through miscope and return process result.

    Note: We purposefully avoid parsing/aggregating any MiScope output here.
    """
    bench_cmd_str = " ".join(shlex.quote(str(arg)) for arg in bench_cmd)

    prefix_dir = os.path.dirname(metrics_prefix)
    if prefix_dir:
        os.makedirs(os.path.join(base_dir, prefix_dir), exist_ok=True)

    metrics_candidates = [
        os.path.join(base_dir, f"{metrics_prefix}_0"),
        os.path.join(base_dir, f"{metrics_prefix}.csv_0"),
    ]

    for path in metrics_candidates:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    miscope_script = os.path.join(base_dir, "miscope", "miscope.py")
    miscope_cmd = [
        sys.executable,
        miscope_script,
        "--cmd",
        bench_cmd_str,
        "--gpus",
        gpu_ids,
        "--prefix",
        metrics_prefix,
    ]

    print(f"Running with miscope: {' '.join(shlex.quote(str(arg)) for arg in miscope_cmd)}")

    result = run_subprocess_with_timeout(
        miscope_cmd,
        cwd=base_dir,
        timeout=timeout_seconds,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"miscope benchmark failed: {result.stderr}")
        return None, None

    # Do not parse or aggregate MiScope results; just return the process result.
    return result, None

def run_tessera_benchmark(
    m,
    n,
    k,
    ordering0,
    ordering1,
    wgm,
    wgn,
    arch,
    dtype="bfloat16",
    bench_warmup_ms=10,
    bench_rep_ms=10,
    prof_warmup_ms=20,
    prof_rep_ms=20,
):
    """Run a single benchmark with rocprof profiling and return results."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Create input file for rocprof
        with open(os.path.join(base_dir, "input.txt"), "w") as f:
            f.write("pmc: TCC_HIT_sum TCC_MISS_sum\n")

        # Run the benchmark script with rocprof
        bench_cmd = [
            sys.executable, "run_benchmark.py",
            str(m), str(n), str(k),
            str(ordering0), str(ordering1),
            str(wgm), str(wgn),
            "--dtype", dtype,
            "--warmup", str(bench_warmup_ms),
            "--rep", str(bench_rep_ms)
        ]

        # Benchmark first without rocprof, wrapped with miscope for metrics capture
        metrics_prefix = build_miscope_prefix(arch, m, n, k, ordering0, ordering1, wgm, wgn, dtype)
        try:
            miscope_result, _ = run_benchmark_with_miscope(
                bench_cmd,
                base_dir,
                metrics_prefix=metrics_prefix,
                gpu_ids=MISCOPE_GPU_ID,
                timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
            )
        except BenchmarkTimeoutError as exc:
            print(
                f"[Timeout] MiScope benchmark timed out for m={m}, n={n}, k={k}: {exc}"
            )
            return None
        if miscope_result is None:
            return None

        # Read the JSON results from run_benchmark.py
        benchmark_data = None
        try:
            with open(os.path.join(base_dir, "benchmark_results.json"), "r") as f:
                benchmark_data = json.load(f)
        except Exception as e:
            print(f"Error reading benchmark results: {e}")
            return None

        rocprof_cmd = [
            "rocprofv3", "-i", "counters.txt", "-o", "tessera_benchmark", "--",
            sys.executable, "run_benchmark.py",
            str(m), str(n), str(k),
            str(ordering0), str(ordering1),
            str(wgm), str(wgn),
            "--dtype", dtype,
            "--warmup", str(prof_warmup_ms),
            "--rep", str(prof_rep_ms)
        ]
        
        print(f"Running: {' '.join(rocprof_cmd)}")
        try:
            rocprof_result = run_subprocess_with_timeout(
                rocprof_cmd,
                cwd=base_dir,
                timeout=BENCHMARK_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
        except BenchmarkTimeoutError as exc:
            print(
                f"[Timeout] rocprof benchmark timed out for m={m}, n={n}, k={k}: {exc}"
            )
            return None
        
        if rocprof_result.returncode != 0:
            print(f"Benchmark failed: {rocprof_result.stderr}")
            return None
        
        # Analyze rocprof results to get TCC hit rate
        profiler_data = None
        csv_file = os.path.join(base_dir, "pmc_1", "tessera_benchmark_counter_collection.csv")
        if os.path.exists(csv_file):
            profiler_data = calculate_tcc_hit_rate(csv_file, 'persistent_matmul_tessera')
        else:
            print(f"Warning: rocprof CSV file not found: {csv_file}")

        # Combine benchmark and profiler data
        if benchmark_data and profiler_data:
            combined_benchmark_data = {
                "ordering_name_0": get_ordering_name(ordering0),
                "ordering_name_1": get_ordering_name(ordering1),
                "wgm": wgm,
                "wgn": wgn,
                "dtype": dtype,
                "tflops": benchmark_data.get('tflops', 0),
                "ms": benchmark_data.get('ms', 0),
                "transA": benchmark_data["transA"],
                "transB": benchmark_data["transB"],
                "init_type": benchmark_data["init_type"]
            }
            # Intentionally exclude aggregated MiScope metrics from saved results

            return {
                "profiler_data": profiler_data,
                "benchmark_data": combined_benchmark_data
            }
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None

def run_baseline_benchmark(
    m,
    n,
    k,
    wgm,
    arch,
    dtype="bfloat16",
    bench_warmup_ms=10,
    bench_rep_ms=10,
    prof_warmup_ms=20,
    prof_rep_ms=20,
    chunk_size=-1,
    row_major=1,
):
    """Run a single benchmark with rocprof profiling and return results."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Create input file for rocprof
        with open(os.path.join(base_dir, "input.txt"), "w") as f:
            f.write("pmc: TCC_HIT_sum TCC_MISS_sum\n")

        # Run the benchmark script with rocprof
        bench_cmd = [
            sys.executable, "run_benchmark.py",
            str(m), str(n), str(k),
            "0", "0",
            str(wgm), "1",
            "--dtype", dtype,
            "--warmup", str(bench_warmup_ms),
            "--rep", str(bench_rep_ms),
            "--baseline",
            "--chunk-size", str(chunk_size),
            "--row-major", str(row_major)
        ]

        # Benchmark first without rocprof, wrapped with miscope for metrics capture
        metrics_prefix = build_baseline_miscope_prefix(arch, m, n, k, wgm, dtype)
        try:
            miscope_result, _ = run_benchmark_with_miscope(
                bench_cmd,
                base_dir,
                metrics_prefix=metrics_prefix,
                gpu_ids=MISCOPE_GPU_ID,
                timeout_seconds=BENCHMARK_TIMEOUT_SECONDS,
            )
        except BenchmarkTimeoutError as exc:
            print(
                f"[Timeout] MiScope baseline benchmark timed out for m={m}, n={n}, k={k}: {exc}"
            )
            return None
        if miscope_result is None:
            return None

        # Read the JSON results from run_benchmark.py
        benchmark_data = None
        try:
            with open(os.path.join(base_dir, "benchmark_results.json"), "r") as f:
                benchmark_data = json.load(f)
            print("loaded baseline data")
        except Exception as e:
            print(f"Error reading benchmark results: {e}")
            return None

        rocprof_cmd = [
            "rocprofv3", "-i", "counters.txt", "-o", "tessera_benchmark", "--",
            sys.executable, "run_benchmark.py",
            str(m), str(n), str(k),
            "0", "0",
            str(wgm), "1",
            "--dtype", dtype,
            "--warmup", str(prof_warmup_ms),
            "--rep", str(prof_rep_ms),
            "--baseline",
            "--chunk-size", str(chunk_size),
            "--row-major", str(row_major)
        ]
        
        print("Running with rocprof...") 
        print(f"Running: {' '.join(rocprof_cmd)}")
        try:
            rocprof_result = run_subprocess_with_timeout(
                rocprof_cmd,
                cwd=base_dir,
                timeout=BENCHMARK_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
        except BenchmarkTimeoutError as exc:
            print(
                f"[Timeout] rocprof baseline benchmark timed out for m={m}, n={n}, k={k}: {exc}"
            )
            return None
        
        if rocprof_result.returncode != 0:
            print(f"Benchmark failed: {rocprof_result.stderr}")
            return None

        # Analyze rocprof results to get TCC hit rate
        profiler_data = None
        csv_file = os.path.join(base_dir, "pmc_1", "tessera_benchmark_counter_collection.csv")
        if os.path.exists(csv_file):
            profiler_data = calculate_tcc_hit_rate(csv_file, 'persistent_matmul')
            print(json.dumps(profiler_data, indent=4))
        else:
            print(f"Warning: rocprof CSV file not found: {csv_file}")
        
        # Combine benchmark and profiler data
        if benchmark_data and profiler_data:
            combined_benchmark_data = {
                "wgm": wgm,
                "tflops": benchmark_data.get('tflops', 0),
                "ms": benchmark_data.get('ms', 0),
                "transA": benchmark_data["transA"],
                "transB": benchmark_data["transB"],
                "init_type": benchmark_data["init_type"],
                "dtype": dtype,
                "chunk_size": chunk_size,
                "row_major": row_major
            }
            # Intentionally exclude aggregated MiScope metrics from saved results

            return {
                "profiler_data": profiler_data,
                "benchmark_data": combined_benchmark_data
            }
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None

def save_progressive_results(results, csv_data, json_path, csv_path, baseline_sweep=False):
    """Save results progressively to avoid data loss."""
    metadata = results.get("metadata", {})
    sweep_results = results.get("sweep_results", [])
    results["summary"] = compute_sweep_summary(metadata, sweep_results)

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    if csv_data:
        with open(csv_path, 'w', newline='') as f:
            if baseline_sweep:
                fieldnames = ["category", "WGM", "tflops", "ms", "chunk_size", "chunking_strategy", "row_major", "l2_hit_rate_pct"]
            else:
                fieldnames = [
                    "category", "ordering_0", "ordering_1", "WGM", "WGN",
                    "tflops", "ms", "number_of_errors", "l2_hit_rate_pct"
                ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

def sweep_matrix_problem(
    m,
    n,
    k,
    arch,
    dtype="bfloat16",
    max_wgm=16,
    max_wgn=16,
    results_dir="results",
    input_csv=None,
    bench_warmup_ms=10,
    bench_rep_ms=10,
    prof_warmup_ms=20,
    prof_rep_ms=20,
    baseline_sweep=False,
    problem_category=None,
):
    """Sweep all configurations for a single matrix problem with progressive saving."""
    print(f"\nSweeping matrix problem: M={m}, N={n}, K={k}")
    if problem_category:
        print(f"  Category: {problem_category}")
    category_value = problem_category if problem_category is not None else ""
    
    # Get block dimensions from selector
    selector = tritonblas.MatmulHeuristicResult(m, n, k, 
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32,
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32,
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32)
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    
    # Get all workgroup combinations (constrained by grid dimensions)
    num_pid_m = math.ceil(m / BLK_M)
    num_pid_n = math.ceil(n / BLK_N)
    
    if baseline_sweep:
        # For baseline sweep, only test WGM values (WGN is always 1 for baseline)
        wgm_values = list(range(1, min(max_wgm, num_pid_m) + 1))
        wgm_wgn_combinations = [(wgm, 1) for wgm in wgm_values]
        ordering_combinations = [(0, 0)]  # Only ROW_MAJOR, ROW_MAJOR for baseline
        print(f"Baseline sweep: testing {len(wgm_values)} WGM values with two chunking strategies")
    else:
        wgm_wgn_combinations = get_all_wgm_wgn_combinations(max_wgm, max_wgn, num_pid_m, num_pid_n)
        # All ordering combinations
        orderings = [0, 1, 2, 3]  # ROW_MAJOR, COLUMN_MAJOR, SNAKE, SPIRAL
        ordering_combinations = [(o0, o1) for o0 in orderings for o1 in orderings]
        print(f"Testing orderings: {[get_ordering_name(ord) for ord in orderings]}")
    
    if baseline_sweep:
        # For baseline sweep, we test each WGM with 4 combinations (2 chunking strategies × 2 row_major values)
        total_combinations = len(wgm_wgn_combinations) * len(ordering_combinations) * 4
    else:
        total_combinations = len(wgm_wgn_combinations) * len(ordering_combinations)
    
    print(f"  Block sizes: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
    print(f"  Grid: {num_pid_m} x {num_pid_n}")
    print(f"  WGM/WGN combinations: {len(wgm_wgn_combinations)}")
    if not baseline_sweep:
        print(f"  Ordering combinations: {len(ordering_combinations)}")
    else:
        print(f"  Combinations: 4 (2 chunking strategies × 2 row_major values)")
    print(f"  Total combinations: {total_combinations}")
    
    # Generate filenames
    if baseline_sweep:
        json_filename = f"baseline_sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}_{arch}.json"
        csv_filename = f"baseline_sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}_{arch}.csv"
    else:
        json_filename = f"sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}_{arch}.json"
        csv_filename = f"sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}_{arch}.csv"
    json_path = os.path.join(results_dir, json_filename)
    csv_path = os.path.join(results_dir, csv_filename)


    # Get optimal baseline:
    baseline_results = []
    # Use max_wgm for baseline computation, but ensure we don't exceed grid constraints.
    # Always include the heuristic WGM even if it exceeds the grid so we can
    # collect comparable baseline data for the predicted configuration.
    baseline_wgm_values = {
        x for x in range(1, min(max_wgm, num_pid_m) + 1)
    }
    if gsize_m is not None and gsize_m > 0:
        if gsize_m > num_pid_m:
            print(
                f"Heuristic WGM {gsize_m} exceeds grid dimension {num_pid_m}; including it for baseline coverage."
            )
        if gsize_m > max_wgm:
            print(
                f"Heuristic WGM {gsize_m} exceeds configured max_wgm {max_wgm}; including it for baseline coverage."
            )
        baseline_wgm_values.add(gsize_m)
    baseline_wgm_values = sorted(baseline_wgm_values)
    print(f"Computing baseline perf for WGMs: {baseline_wgm_values}...")
    
    if baseline_sweep:
        # For baseline sweep, test all 4 combinations of chunking strategy + row_major
        combinations = [
            (-1, "no_chunking", 1),
            (-1, "no_chunking", 0),
            (None, "chiplet_chunk", 1),  # None means calculate as wgm*wgm
            (None, "chiplet_chunk", 0)
        ]
        for chunk_size, strategy_name, row_major in combinations:
            print(f"Testing baseline with {strategy_name} chunking strategy, row_major={row_major}...")
            for wgm in baseline_wgm_values:
                actual_chunk_size = chunk_size if chunk_size != None else wgm * wgm
                baseline_result = run_baseline_benchmark(
                    m, n, k, wgm, arch, dtype=dtype,
                    bench_warmup_ms=bench_warmup_ms, bench_rep_ms=bench_rep_ms,
                    prof_warmup_ms=prof_warmup_ms, prof_rep_ms=prof_rep_ms,
                    chunk_size=actual_chunk_size, row_major=row_major
                )
                if baseline_result is not None:
                    baseline_results.append(baseline_result)
                else: 
                    sys.exit(1)
    else:
        # For regular sweep, use normal chunking
        for wgm in baseline_wgm_values:
            baseline_result = run_baseline_benchmark(
                m, n, k, wgm, arch, dtype=dtype,
                bench_warmup_ms=bench_warmup_ms, bench_rep_ms=bench_rep_ms,
                prof_warmup_ms=prof_warmup_ms, prof_rep_ms=prof_rep_ms,
                chunk_size=-1
            )
            if baseline_result is not None:
                baseline_results.append(baseline_result)
            else: 
                sys.exit(1)

    baseline_runs = []
    baseline_data = {}

    if baseline_sweep:
        # For baseline sweep, process each combination of chunking strategy + row_major separately
        combinations = [
            (-1, "no_chunking", 1),
            (-1, "no_chunking", 0),
            (None, "chiplet_chunk", 1),
            (None, "chiplet_chunk", 0)
        ]
        
        for chunk_size, strategy_name, row_major in combinations:
            strategy_results = []
            optimal_l2_hit_rate = -1
            optimal_tflops = -1
            optimal_ms = -1
            optimal_wgm = -1
            heuristic_wgm = gsize_m
            predicted_wgm_value = gsize_m
            predicted_tflops = None
            predicted_l2_hit_rate = None
            predicted_ms = None
            
            # Filter results for this combination
            for res in baseline_results:
                benchmark_data = res["benchmark_data"]
                actual_chunk_size = chunk_size if chunk_size != None else benchmark_data["wgm"] * benchmark_data["wgm"]
                
                if (benchmark_data.get("chunk_size") == actual_chunk_size and 
                    benchmark_data.get("row_major") == row_major):
                    profiler_data = res["profiler_data"]
                    
                    baseline_entry = {
                        "wgm": benchmark_data.get("wgm"),
                        "tflops": benchmark_data.get("tflops"),
                        "ms": benchmark_data.get("ms"),
                        "dtype": benchmark_data.get("dtype", dtype),
                        "l2_hit_rate": profiler_data.get("l2_hit_rate") if profiler_data else None,
                        "hit_rate_pct": profiler_data.get("hit_rate_pct") if profiler_data else None,
                        "chunk_size": actual_chunk_size,
                        "chunking_strategy": strategy_name,
                        "row_major": row_major
                    }
                    strategy_results.append(baseline_entry)
                    
                    if benchmark_data["wgm"] == gsize_m:
                        predicted_tflops = benchmark_data["tflops"]
                        predicted_l2_hit_rate = profiler_data["l2_hit_rate"]
                        predicted_ms = benchmark_data["ms"]
                    
                    if benchmark_data["tflops"] > optimal_tflops:
                        optimal_wgm = benchmark_data["wgm"]
                        optimal_tflops = benchmark_data["tflops"]
                        optimal_l2_hit_rate = profiler_data["l2_hit_rate"]
                        optimal_ms = benchmark_data["ms"]
            
            # Check if predicted values were found for this combination
            if predicted_tflops is None:
                available_wgms = [res["wgm"] for res in strategy_results]
                if not available_wgms:
                    raise RuntimeError(
                        f"No valid baseline results found for {strategy_name} chunking, row_major={row_major} - all baseline runs failed"
                    )
                fallback_wgm = min(available_wgms, key=lambda w: abs(w - gsize_m))
                print(
                    f"Warning: baseline data missing for heuristic WGM={gsize_m} "
                    f"with {strategy_name} chunking, row_major={row_major}; "
                    f"using closest available WGM={fallback_wgm}."
                )
                fallback_entry = next(
                    (res for res in strategy_results if res["wgm"] == fallback_wgm),
                    None,
                )
                if fallback_entry is None:
                    raise RuntimeError(
                        f"Could not resolve fallback baseline result for WGM={fallback_wgm}"
                    )
                predicted_wgm_value = fallback_entry["wgm"]
                predicted_tflops = fallback_entry["tflops"]
                predicted_l2_hit_rate = fallback_entry["l2_hit_rate"]
                predicted_ms = fallback_entry["ms"]
            if optimal_tflops == -1:
                raise RuntimeError(f"No valid baseline results found for {strategy_name} chunking, row_major={row_major} - all baseline runs failed")
            
            # Calculate chunk sizes for predicted and optimal values
            predicted_chunk_size = chunk_size if chunk_size != None else predicted_wgm_value * predicted_wgm_value
            optimal_chunk_size = chunk_size if chunk_size != None else optimal_wgm * optimal_wgm
            
            combination_key = f"{strategy_name}_row_major_{row_major}"
            baseline_data[combination_key] = {
                "heuristic_wgm": heuristic_wgm,
                "predicted_wgm": predicted_wgm_value, 
                "predicted_chunk_size": predicted_chunk_size,
                "predicted_row_major": row_major,
                "predicted_tflops": predicted_tflops,
                "predicted_l2_hit_rate": predicted_l2_hit_rate, 
                "predicted_ms": predicted_ms, 
                "optimal_wgm": optimal_wgm,
                "optimal_chunk_size": optimal_chunk_size,
                "optimal_row_major": row_major,
                "optimal_tflops": optimal_tflops,
                "optimal_ms": optimal_ms,
                "optimal_l2_hit_rate": optimal_l2_hit_rate,
                "baseline_runs": strategy_results,
            }
            baseline_runs.extend(strategy_results)
    else:
        # For regular sweep, use normal processing
        optimal_l2_hit_rate = -1
        optimal_tflops = -1
        optimal_ms = -1
        optimal_wgm = -1
        heuristic_wgm = gsize_m
        predicted_wgm_value = gsize_m
        predicted_tflops = None
        predicted_l2_hit_rate = None
        predicted_ms = None

        for res in baseline_results:
            profiler_data = res["profiler_data"]
            benchmark_data = res["benchmark_data"]

            baseline_entry = {
                "wgm": benchmark_data.get("wgm"),
                "tflops": benchmark_data.get("tflops"),
                "ms": benchmark_data.get("ms"),
                "dtype": benchmark_data.get("dtype", dtype),
                "l2_hit_rate": profiler_data.get("l2_hit_rate") if profiler_data else None,
                "hit_rate_pct": profiler_data.get("hit_rate_pct") if profiler_data else None
            }
            baseline_runs.append(baseline_entry)

            if benchmark_data["wgm"] == gsize_m:
                predicted_tflops = benchmark_data["tflops"]
                predicted_l2_hit_rate = profiler_data["l2_hit_rate"]
                predicted_ms = benchmark_data["ms"]
            
            if benchmark_data["tflops"] > optimal_tflops:
                optimal_wgm = benchmark_data["wgm"]
                optimal_tflops = benchmark_data["tflops"]
                optimal_l2_hit_rate = profiler_data["l2_hit_rate"]
                optimal_ms = benchmark_data["ms"]

        # Check if predicted values were found
        if predicted_tflops is None:
            available_wgms = [res["benchmark_data"]["wgm"] for res in baseline_results]
            fallback_wgm = min(available_wgms, key=lambda w: abs(w - gsize_m)) if available_wgms else None
            if fallback_wgm is None:
                raise RuntimeError("No valid baseline results found - all baseline runs failed")
            print(
                f"Warning: baseline data missing for heuristic WGM={gsize_m}; using closest available WGM={fallback_wgm}."
            )
            predicted_wgm_value = fallback_wgm
            fallback_entry = next(
                (res for res in baseline_results if res["benchmark_data"]["wgm"] == fallback_wgm),
                None,
            )
            profiler_data = fallback_entry["profiler_data"] if fallback_entry else None
            benchmark_data = fallback_entry["benchmark_data"] if fallback_entry else None
            if benchmark_data is None:
                raise RuntimeError(
                    f"Could not resolve fallback baseline result for WGM={fallback_wgm}"
                )
            predicted_tflops = benchmark_data["tflops"]
            predicted_l2_hit_rate = profiler_data["l2_hit_rate"] if profiler_data else None
            predicted_ms = benchmark_data["ms"]
        if optimal_tflops == -1:
            raise RuntimeError("No valid baseline results found - all baseline runs failed")

        baseline_data = {
            "heuristic_wgm": heuristic_wgm,
            "predicted_wgm": predicted_wgm_value, 
            "predicted_tflops": predicted_tflops,
            "predicted_l2_hit_rate": predicted_l2_hit_rate, 
            "predicted_ms": predicted_ms, 
            "optimal_wgm": optimal_wgm,
            "optimal_tflops": optimal_tflops,
            "optimal_ms": optimal_ms,
            "optimal_l2_hit_rate": optimal_l2_hit_rate,
            "baseline_runs": baseline_runs,
        }

    # Exclude aggregated MiScope metrics from baseline data

    print("Baseline results: ")
    print(json.dumps(baseline_data, indent=4))
    
    # Create metadata
    metadata = {
        "matrix_dimensions": {
            "m": m,
            "n": n,
            "k": k
        },
        "block_dimensions": {
            "BLK_M": BLK_M,
            "BLK_N": BLK_N,
            "BLK_K": BLK_K
        },
        "grid": {
            "num_pid_m": math.ceil(m / BLK_M),
            "num_pid_n": math.ceil(n / BLK_N),
            "k_tiles": math.ceil(k / 64)
        },
        "arch": arch,
        "dtype": dtype,
        "total_combinations": total_combinations,
        "baseline_data": baseline_data
    }

    if problem_category is not None:
        metadata["category"] = problem_category

    if input_csv:
        metadata["input_csv"] = os.path.abspath(input_csv)
    
    # Add orderings info only for non-baseline sweeps
    if not baseline_sweep:
        metadata["orderings_tested"] = [get_ordering_name(o) for o in orderings]
    else:
        metadata["baseline_sweep"] = True
        metadata["wgm_values_tested"] = wgm_values

    print(json.dumps(metadata, indent=4))

            
    
    # Prepare results structure
    sweep_results = []
    csv_data = []
    
    # Run all combinations with progressive saving
    combination_count = 0
    save_interval = 1  # Save after every combination to persist progress more frequently
    
    for wgm, wgn in wgm_wgn_combinations:
        for ordering0, ordering1 in ordering_combinations:
            if baseline_sweep:
                # For baseline sweep, test all 4 combinations of chunking strategy + row_major
                combinations = [
                    (-1, "no_chunking", 1),
                    (-1, "no_chunking", 0),
                    (wgm * wgm, "chiplet_chunk", 1),
                    (wgm * wgm, "chiplet_chunk", 0)
                ]
                for chunk_size, strategy_name, row_major in combinations:
                    combination_count += 1
                    print(f"  [{combination_count}/{total_combinations}] WGM={wgm} (baseline, {strategy_name}, chunk_size={chunk_size}, row_major={row_major})")
                    
                    result = run_baseline_benchmark(
                        m, n, k, wgm, arch, dtype,
                        bench_warmup_ms, bench_rep_ms,
                        prof_warmup_ms, prof_rep_ms,
                        chunk_size=chunk_size, row_major=row_major
                    )
                    
                    if result is not None:
                        # Extract benchmark and profiler data
                        benchmark_data = result["benchmark_data"]
                        profiler_data = result["profiler_data"]
                        
                        # Add to sweep results
                        sweep_result = {
                            "WGM": wgm,
                            "tflops": benchmark_data.get("tflops", 0),
                            "ms": benchmark_data.get("ms", 0),
                            "transA": benchmark_data["transA"],
                            "transB": benchmark_data["transB"],
                            "init_type": benchmark_data["init_type"],
                            "chunk_size": chunk_size,
                            "chunking_strategy": strategy_name,
                            "row_major": row_major,
                            "profiler_data": profiler_data
                        }
                        sweep_results.append(sweep_result)
                        
                        # Add to CSV data
                        csv_row = {
                            "category": category_value,
                            "WGM": wgm,
                            "tflops": benchmark_data.get("tflops", 0),
                            "ms": benchmark_data.get("ms", 0),
                            "chunk_size": chunk_size,
                            "chunking_strategy": strategy_name,
                            "row_major": row_major,
                            "l2_hit_rate_pct": profiler_data.get("hit_rate_pct", 0) if profiler_data else 0
                        }
                        csv_data.append(csv_row)

                        print(json.dumps(result, indent=4))
                    else:
                        print(f"    Failed to get results")
            else:
                combination_count += 1
                print(f"  [{combination_count}/{total_combinations}] Ordering=({ordering0},{ordering1}), WGM={wgm}, WGN={wgn}")

                # Run benchmark
                result = run_tessera_benchmark(
                    m,
                    n,
                    k,
                    ordering0,
                    ordering1,
                    wgm,
                    wgn,
                    arch,
                    dtype,
                    bench_warmup_ms,
                    bench_rep_ms,
                    prof_warmup_ms,
                    prof_rep_ms,
                )
            
                if result is not None:
                    # Extract benchmark and profiler data
                    benchmark_data = result["benchmark_data"]
                    profiler_data = result["profiler_data"]
                    
                    # Add to sweep results
                    sweep_result = {
                        "ordering_0": get_ordering_name(ordering0),
                        "ordering_1": get_ordering_name(ordering1),
                        "WGM": wgm,
                        "WGN": wgn,
                        "tflops": benchmark_data.get("tflops", 0),
                        "ms": benchmark_data.get("ms", 0),
                        "number_of_errors": benchmark_data.get("number_of_errors", 0),
                        "transA": benchmark_data["transA"],
                        "transB": benchmark_data["transB"],
                        "init_type": benchmark_data["init_type"],
                        "profiler_data": profiler_data
                    }
                    sweep_results.append(sweep_result)
                    
                    # Add to CSV data
                    csv_row = {
                        "category": category_value,
                        "ordering_0": get_ordering_name(ordering0),
                        "ordering_1": get_ordering_name(ordering1),
                        "WGM": wgm,
                        "WGN": wgn,
                        "tflops": benchmark_data.get("tflops", 0),
                        "ms": benchmark_data.get("ms", 0),
                        "number_of_errors": benchmark_data.get("number_of_errors", 0),
                        "l2_hit_rate_pct": profiler_data.get("hit_rate_pct", 0) if profiler_data else 0
                    }
                    csv_data.append(csv_row)

                    print(json.dumps(result, indent=4))
                else:
                    print(f"    Failed to get results")
            
            # Progressive save
            if combination_count % save_interval == 0 or combination_count == total_combinations:
                results = {
                    "metadata": metadata,
                    "sweep_results": sweep_results
                }
                save_progressive_results(results, csv_data, json_path, csv_path, baseline_sweep)
                print(f"    Progress saved: {combination_count}/{total_combinations} ({100*combination_count/total_combinations:.1f}%)")
    
    # Final save
    results = {
        "metadata": metadata,
        "sweep_results": sweep_results
    }
    save_progressive_results(results, csv_data, json_path, csv_path, baseline_sweep)

    summary = results.get("summary", {})
    if summary:
        def format_float(value):
            return f"{value:.3f}" if isinstance(value, (int, float)) else "N/A"

        print("\nSummary of best schedules:")
        best_schedules = summary.get("best_schedules_by_strategy") or {}
        any_strategy_printed = False
        for strategy_key in STRATEGY_LABEL_ORDER:
            label = STRATEGY_LABELS.get(strategy_key, strategy_key)
            strategy_info = best_schedules.get(strategy_key)
            if strategy_info and strategy_info.get("schedule"):
                schedule = strategy_info["schedule"]
                tflops = schedule.get("tflops")
                ms = schedule.get("ms")
                wgm = schedule.get("WGM") or schedule.get("wgm")
                wgn = schedule.get("WGN") or schedule.get("wgn")
                chunking_strategy = schedule.get("chunking_strategy")
                row_major = schedule.get("row_major")

                details = []
                if wgm is not None:
                    details.append(f"WGM={wgm}")
                if wgn is not None:
                    details.append(f"WGN={wgn}")
                if chunking_strategy is not None:
                    details.append(f"chunking={chunking_strategy}")
                if row_major is not None:
                    details.append(f"row_major={row_major}")

                detail_str = ", ".join(details)
                print(f"  {label}: TFLOPS={format_float(tflops)}, Time={format_float(ms)} ms ({detail_str})")
                any_strategy_printed = True
            else:
                print(f"  {label}: no valid schedule")
                any_strategy_printed = True

        if not any_strategy_printed:
            print("  No strategy-specific schedules recorded.")

        best_schedule = summary.get("best_schedule")
        if best_schedule:
            best_tflops = best_schedule.get("tflops")
            best_ms = best_schedule.get("ms")
            wgm = best_schedule.get("WGM") or best_schedule.get("wgm")
            wgn = best_schedule.get("WGN") or best_schedule.get("wgn")
            chunking_strategy = best_schedule.get("chunking_strategy")
            row_major = best_schedule.get("row_major")

            details = []
            if wgm is not None:
                details.append(f"WGM={wgm}")
            if wgn is not None:
                details.append(f"WGN={wgn}")
            if chunking_strategy is not None:
                details.append(f"chunking={chunking_strategy}")
            if row_major is not None:
                details.append(f"row_major={row_major}")

            detail_str = ", ".join(details)
            print(f"\n  Overall best schedule: TFLOPS={format_float(best_tflops)}, Time={format_float(best_ms)} ms ({detail_str})")
        else:
            print("\n  Overall best schedule: none")

        speedup_pred = summary.get("speedup_vs_predicted_no_chunking_rm1")
        speedup_best = summary.get("speedup_vs_best_no_chunking_rm1")
        reference_label = STRATEGY_LABELS.get(summary.get("reference_baseline_key", REFERENCE_BASELINE_KEY), summary.get("reference_baseline_key", REFERENCE_BASELINE_KEY))

        if speedup_pred:
            print(f"  Speedup vs predicted ({reference_label}): {format_float(speedup_pred)}x")
        if speedup_best:
            print(f"  Speedup vs best ({reference_label}): {format_float(speedup_best)}x")
        if not speedup_pred and not speedup_best:
            print(f"  Speedup metrics unavailable for reference ({reference_label}).")
    
    return results, csv_data

def run_single_configuration(
    m, n, k, wgm, wgn, arch, dtype="bfloat16",
    ordering0=0, ordering1=0, baseline_only=False,
    results_dir="results",
    bench_warmup_ms=50, bench_rep_ms=1000,
    prof_warmup_ms=50, prof_rep_ms=100
):
    """Run a single configuration with profiling and save results."""
    print(f"\nRunning single configuration:")
    print(f"  Dimensions: M={m}, N={n}, K={k}")
    print(f"  Workgroup: WGM={wgm}, WGN={wgn}")
    print(f"  Ordering: ({ordering0}, {ordering1})")
    print(f"  Architecture: {arch}")
    print(f"  Data type: {dtype}")
    print(f"  Baseline only: {baseline_only}")
    print()
    
    # Get block dimensions from selector
    selector = tritonblas.MatmulHeuristicResult(m, n, k, 
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32,
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32,
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32)
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    
    print(f"  Block sizes: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
    print(f"  Grid dimensions: {math.ceil(m/BLK_M)} x {math.ceil(n/BLK_N)}")
    print()
    
    # Generate filenames
    if baseline_only:
        json_filename = f"single_config_baseline_m{m}_n{n}_k{k}_wgm{wgm}_{arch}.json"
        csv_filename = f"single_config_baseline_m{m}_n{n}_k{k}_wgm{wgm}_{arch}.csv"
    else:
        json_filename = f"single_config_tessera_m{m}_n{n}_k{k}_o{ordering0}_{ordering1}_wgm{wgm}_wgn{wgn}_{arch}.json"
        csv_filename = f"single_config_tessera_m{m}_n{n}_k{k}_o{ordering0}_{ordering1}_wgm{wgm}_wgn{wgn}_{arch}.csv"
    
    json_path = os.path.join(results_dir, json_filename)
    csv_path = os.path.join(results_dir, csv_filename)
    
    results = []
    csv_data = []
    
    if baseline_only:
        # Run baseline benchmark
        print("Running baseline benchmark...")
        result = run_baseline_benchmark(
            m, n, k, wgm, arch, dtype,
            bench_warmup_ms, bench_rep_ms,
            prof_warmup_ms, prof_rep_ms
        )
        
        if result is not None:
            profiler_data = result["profiler_data"]
            benchmark_data = result["benchmark_data"]
            
            # Create result entry
            result_entry = {
                "wgm": benchmark_data.get("wgm"),
                "tflops": benchmark_data.get("tflops"),
                "ms": benchmark_data.get("ms"),
                "transA": benchmark_data["transA"],
                "transB": benchmark_data["transB"],
                "init_type": benchmark_data["init_type"],
                "profiler_data": profiler_data
            }
            results.append(result_entry)
            
            # Create CSV row
            csv_row = {
                "wgm": benchmark_data.get("wgm"),
                "tflops": benchmark_data.get("tflops"),
                "ms": benchmark_data.get("ms"),
                "l2_hit_rate_pct": profiler_data.get("hit_rate_pct", 0) if profiler_data else 0
            }
            csv_data.append(csv_row)
            
            print(f"Baseline results:")
            print(f"  TFLOPS: {benchmark_data.get('tflops', 0):.3f}")
            print(f"  Time: {benchmark_data.get('ms', 0):.3f} ms")
            if profiler_data:
                print(f"  L2 Hit Rate: {profiler_data.get('hit_rate_pct', 0):.2f}%")
        else:
            print("Baseline benchmark failed!")
            return None, None
    
    else:
        # Run tessera benchmark
        print("Running tessera benchmark...")
        result = run_tessera_benchmark(
            m, n, k, ordering0, ordering1, wgm, wgn, arch, dtype,
            bench_warmup_ms, bench_rep_ms, prof_warmup_ms, prof_rep_ms
        )
        
        if result is not None:
            profiler_data = result["profiler_data"]
            benchmark_data = result["benchmark_data"]
            
            # Create result entry
            result_entry = {
                "ordering_0": get_ordering_name(ordering0),
                "ordering_1": get_ordering_name(ordering1),
                "WGM": wgm,
                "WGN": wgn,
                "tflops": benchmark_data.get("tflops", 0),
                "ms": benchmark_data.get("ms", 0),
                "transA": benchmark_data["transA"],
                "transB": benchmark_data["transB"],
                "init_type": benchmark_data["init_type"],
                "profiler_data": profiler_data
            }
            results.append(result_entry)
            
            # Create CSV row
            csv_row = {
                "ordering_0": get_ordering_name(ordering0),
                "ordering_1": get_ordering_name(ordering1),
                "WGM": wgm,
                "WGN": wgn,
                "tflops": benchmark_data.get("tflops", 0),
                "ms": benchmark_data.get("ms", 0),
                "l2_hit_rate_pct": profiler_data.get("hit_rate_pct", 0) if profiler_data else 0
            }
            csv_data.append(csv_row)
            
            print(f"Tessera results:")
            print(f"  TFLOPS: {benchmark_data.get('tflops', 0):.3f}")
            print(f"  Time: {benchmark_data.get('ms', 0):.3f} ms")
            if profiler_data:
                print(f"  L2 Hit Rate: {profiler_data.get('hit_rate_pct', 0):.2f}%")
        else:
            print("Tessera benchmark failed!")
            return None, None
    
    # Create metadata
    metadata = {
        "matrix_dimensions": {"m": m, "n": n, "k": k},
        "block_dimensions": {"BLK_M": BLK_M, "BLK_N": BLK_N, "BLK_K": BLK_K},
        "grid": {
            "num_pid_m": math.ceil(m / BLK_M),
            "num_pid_n": math.ceil(n / BLK_N),
            "k_tiles": math.ceil(k / 64)
        },
        "arch": arch,
        "dtype": dtype,
        "single_config": True,
        "baseline_only": baseline_only
    }
    
    if not baseline_only:
        metadata.update({
            "ordering0": ordering0,
            "ordering1": ordering1,
            "wgm": wgm,
            "wgn": wgn
        })
    else:
        metadata.update({
            "wgm": wgm
        })
    
    # Save results
    final_results = {
        "metadata": metadata,
        "results": results
    }
    
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save CSV
    if csv_data:
        with open(csv_path, 'w', newline='') as f:
            if baseline_only:
                fieldnames = ["wgm", "tflops", "ms"]
            else:
                fieldnames = ["ordering_0", "ordering_1", "WGM", "WGN", "tflops", "ms"]
            fieldnames.append("l2_hit_rate_pct")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    
    return final_results, csv_data

def main():
    parser = argparse.ArgumentParser(description="Sweep tessera matmul configurations")
    parser.add_argument("csv_file", nargs='?', help="CSV file with matrix problems (m,n,k) - required for sweep mode")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--max-wgm", type=int, default=8, help="Maximum WGM value")
    parser.add_argument("--max-wgn", type=int, default=8, help="Maximum WGN value")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--bench-warmup-ms", type=int, default=50, help="Warmup duration (ms) for miscope (non-rocprof) benchmark runs")
    parser.add_argument("--bench-rep-ms", type=int, default=1000, help="Measurement duration (ms) for miscope (non-rocprof) benchmark runs")
    parser.add_argument("--prof-warmup-ms", type=int, default=50, help="Warmup duration (ms) for rocprof benchmark runs")
    parser.add_argument("--prof-rep-ms", type=int, default=100, help="Measurement duration (ms) for rocprof benchmark runs")
    parser.add_argument("--chunk-size", type=int, default=-1, help="Chunk size for matmul operation (only used for non-baseline-sweep mode)")
    parser.add_argument("--start-problem", type=int, default=1, help="1-based index of the problem in the CSV to start processing from")
    
    # Single configuration mode arguments
    parser.add_argument("--single-config", action="store_true", help="Run single configuration instead of sweep")
    parser.add_argument("--m", type=int, help="Matrix M dimension (required for single config)")
    parser.add_argument("--n", type=int, help="Matrix N dimension (required for single config)")
    parser.add_argument("--k", type=int, help="Matrix K dimension (required for single config)")
    parser.add_argument("--wgm", type=int, help="Workgroup M dimension (required for single config)")
    parser.add_argument("--wgn", type=int, help="Workgroup N dimension (optional for single config, defaults to 1)")
    parser.add_argument("--ordering0", type=int, choices=[0,1,2,3], default=0, help="Ordering0 (default: 0)")
    parser.add_argument("--ordering1", type=int, choices=[0,1,2,3], default=0, help="Ordering1 (default: 0)")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline (no tessera) for single config")
    parser.add_argument("--baseline-sweep", action="store_true", help="Run baseline sweep (only WGM values, no tessera orderings) for sweep mode. Results saved with 'baseline' prefix.")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Handle single configuration mode
    if args.single_config:
        # Validate required arguments for single config
        if args.m is None or args.n is None or args.k is None or args.wgm is None:
            print("Error: For single configuration mode, --m, --n, --k, and --wgm are required")
            sys.exit(1)
        
        # Set default wgn if not provided
        wgn = args.wgn if args.wgn is not None else 1
        
        try:
            # Run single configuration
            results, csv_data = run_single_configuration(
                args.m, args.n, args.k, args.wgm, wgn, args.arch, args.dtype,
                args.ordering0, args.ordering1, args.baseline_only,
                args.results_dir, args.bench_warmup_ms, args.bench_rep_ms,
                args.prof_warmup_ms, args.prof_rep_ms
            )
            
            if results is not None:
                print(f"\nSingle configuration completed successfully!")
                if results["results"]:
                    result = results["results"][0]
                    print(f"Final results:")
                    print(f"  TFLOPS: {result.get('tflops', 0):.3f}")
                    print(f"  Time: {result.get('ms', 0):.3f} ms")
                    if result.get("profiler_data"):
                        print(f"  L2 Hit Rate: {result['profiler_data'].get('hit_rate_pct', 0):.2f}%")
            else:
                print("Single configuration failed!")
                sys.exit(1)
                
        except KeyboardInterrupt:
            print(f"\nInterrupted during single configuration run.")
            sys.exit(1)
        except Exception as e:
            print(f"Error running single configuration: {e}")
            sys.exit(1)
    
    else:
        # Handle sweep mode
        if args.csv_file is None:
            print("Error: CSV file is required for sweep mode")
            sys.exit(1)
        
        # Read matrix problems from CSV
        matrix_problems = []
        with open(args.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter out rows where batch_count is not 1
                if 'batch_count' in row and int(row['batch_count']) != 1:
                    continue
                
                m = int(row['m'])
                n = int(row['n'])
                k = int(row['k'])
                category = row.get('category')
                if category is not None:
                    category = category.strip()
                if category == "":
                    category = None
                matrix_problems.append({"m": m, "n": n, "k": k, "category": category})
        
        total_problems = len(matrix_problems)
        print(f"Found {total_problems} matrix problems in {args.csv_file}")
        if args.baseline_sweep:
            print("Running in BASELINE SWEEP mode - only testing WGM values with baseline approach")
        
        if args.start_problem < 1:
            print(f"Error: --start-problem must be >= 1 (received {args.start_problem})")
            sys.exit(1)
        if args.start_problem > total_problems:
            print(f"Error: --start-problem ({args.start_problem}) exceeds total problems ({total_problems})")
            sys.exit(1)
        if args.start_problem > 1:
            print(f"Resuming sweep from problem {args.start_problem}. Earlier problems will be skipped.")
        
        # Process each matrix problem
        for problem_idx, problem in enumerate(matrix_problems, start=1):
            if problem_idx < args.start_problem:
                print(f"Skipping problem {problem_idx}/{total_problems}: already processed.")
                continue
            m = problem["m"]
            n = problem["n"]
            k = problem["k"]
            category = problem.get("category")
            print(f"\n{'='*80}")
            print(f"Processing problem {problem_idx}/{total_problems}: M={m}, N={n}, K={k}")
            if category:
                print(f"Category: {category}")
            print(f"{'='*80}")
            
            try:
                # Run sweep with progressive saving
                results, csv_data = sweep_matrix_problem(
                    m,
                    n,
                    k,
                    args.arch,
                    args.dtype,
                    args.max_wgm,
                    args.max_wgn,
                    args.results_dir,
                    args.csv_file,
                    args.bench_warmup_ms,
                    args.bench_rep_ms,
                    args.prof_warmup_ms,
                    args.prof_rep_ms,
                    args.baseline_sweep,
                    problem_category=category,
                )
                
                # Print summary
                sweep_results = results["sweep_results"]
                # successful_runs = len([r for r in sweep_results if r["number_of_errors"] == 0])
                # print(f"Summary: {successful_runs}/{len(sweep_results)} runs successful (0 errors)")
                
                if sweep_results:
                    best_tflops = max(r["tflops"] for r in sweep_results)
                    best_config = next(r for r in sweep_results if r["tflops"] == best_tflops)
                    if args.baseline_sweep:
                        print(f"Best TFLOPS: {best_tflops:.3f} (WGM={best_config['WGM']}, chunk_size={best_config['chunk_size']}, strategy={best_config['chunking_strategy']}, row_major={best_config['row_major']})")
                    else:
                        print(f"Best TFLOPS: {best_tflops:.3f} (Ordering=({best_config['ordering_0']},{best_config['ordering_1']}), WGM={best_config['WGM']}, WGN={best_config['WGN']})")
                
                print(f"Problem {problem_idx} completed successfully!")
                
            except KeyboardInterrupt:
                print(f"\nInterrupted during problem {problem_idx}. Partial results saved.")
                break
            except Exception as e:
                print(f"Error processing problem {problem_idx}: {e}")
                sys.exit(1)
        
        print(f"\nSweep completed! Results saved to {args.results_dir}/")

if __name__ == "__main__":
    main()
