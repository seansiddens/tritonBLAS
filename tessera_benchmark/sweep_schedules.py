#!/usr/bin/env python3
"""
Sweep script for tessera matmul across different configurations.
Reads matrix problems from CSV and sweeps through orderings and workgroup sizes.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
import torch
import tritonblas
import pandas as pd
import numpy as np
import math

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

def run_tessera_benchmark(m, n, k, ordering0, ordering1, wgm, wgn, dtype="float16"):
    """Run a single benchmark with rocprof profiling and return results."""
    try:
        # Create input file for rocprof
        with open("input.txt", "w") as f:
            f.write("pmc: TCC_HIT_sum TCC_MISS_sum\n")

        # Run the benchmark script with rocprof
        bench_cmd = [
            sys.executable, "run_benchmark.py",
            str(m), str(n), str(k),
            str(ordering0), str(ordering1),
            str(wgm), str(wgn),
            "--dtype", dtype,
            "--warmup", "10",
            "--rep", "10"
        ]

        # Benchmark first without 
        print(f"Running: {' '.join(bench_cmd)}")
        bench_result = subprocess.run(bench_cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

        # Read the JSON results from run_benchmark.py
        benchmark_data = None
        try:
            with open("benchmark_results.json", "r") as f:
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
            "--warmup", "20",
            "--rep", "20"
        ]
        
        print(f"Running: {' '.join(rocprof_cmd)}")
        rocprof_result = subprocess.run(rocprof_cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if rocprof_result.returncode != 0:
            print(f"Benchmark failed: {rocprof_result.stderr}")
            return None
        
        # Analyze rocprof results to get TCC hit rate
        profiler_data = None
        csv_file = "pmc_1/tessera_benchmark_counter_collection.csv"
        if os.path.exists(csv_file):
            profiler_data = calculate_tcc_hit_rate(csv_file, 'persistent_matmul_tessera')
        else:
            print(f"Warning: rocprof CSV file not found: {csv_file}")
        
        # Combine benchmark and profiler data
        if benchmark_data and profiler_data:
            return {
                "profiler_data": profiler_data,
                "benchmark_data": {
                    "ordering_name_0": get_ordering_name(ordering0),
                    "ordering_name_1": get_ordering_name(ordering1),
                    "wgm": wgm,
                    "wgn": wgn,
                    "tflops": benchmark_data.get('tflops', 0),
                    "ms": benchmark_data.get('ms', 0),
                    "transA": benchmark_data["transA"],
                    "transB": benchmark_data["transB"],
                    "init_type": benchmark_data["init_type"]
                }
            }
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None

def run_baseline_benchmark(m, n, k, wgm, dtype="bfloat16"):
    """Run a single benchmark with rocprof profiling and return results."""
    try:
        # Create input file for rocprof
        with open("input.txt", "w") as f:
            f.write("pmc: TCC_HIT_sum TCC_MISS_sum\n")

        # Run the benchmark script with rocprof
        bench_cmd = [
            sys.executable, "run_benchmark.py",
            str(m), str(n), str(k),
            "0", "0",
            str(wgm), "1",
            "--dtype", dtype,
            "--warmup", "20",
            "--rep", "20",
            "--baseline"            
        ]

        # Benchmark first without 
        print(f"Running: {' '.join(bench_cmd)}")
        bench_result = subprocess.run(bench_cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))

        # Read the JSON results from run_benchmark.py
        benchmark_data = None
        try:
            with open("benchmark_results.json", "r") as f:
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
            "--warmup", "20",
            "--rep", "20",
            "--baseline"
        ]
        
        print("Running with rocprof...") 
        print(f"Running: {' '.join(rocprof_cmd)}")
        rocprof_result = subprocess.run(rocprof_cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if rocprof_result.returncode != 0:
            print(f"Benchmark failed: {rocprof_result.stderr}")
            return None
        
        # Analyze rocprof results to get TCC hit rate
        profiler_data = None
        csv_file = "pmc_1/tessera_benchmark_counter_collection.csv"
        if os.path.exists(csv_file):
            profiler_data = calculate_tcc_hit_rate(csv_file, 'persistent_matmul')
            print(json.dumps(profiler_data, indent=4))
        else:
            print(f"Warning: rocprof CSV file not found: {csv_file}")
        
        # Combine benchmark and profiler data
        if benchmark_data and profiler_data:
            return {
                "profiler_data": profiler_data,
                "benchmark_data": {
                    "wgm": wgm,
                    "tflops": benchmark_data.get('tflops', 0),
                    "ms": benchmark_data.get('ms', 0),
                    "transA": benchmark_data["transA"],
                    "transB": benchmark_data["transB"],
                    "init_type": benchmark_data["init_type"]
                }
            }
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None

def save_progressive_results(results, csv_data, json_path, csv_path):
    """Save results progressively to avoid data loss."""
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    if csv_data:
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                "ordering_0", "ordering_1", "WGM", "WGN", 
                "tflops", "ms", "number_of_errors", 
                "l2_hit_rate_pct"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

def sweep_matrix_problem(m, n, k, arch, dtype="float16", max_wgm=16, max_wgn=16, results_dir="results"):
    """Sweep all configurations for a single matrix problem with progressive saving."""
    print(f"\nSweeping matrix problem: M={m}, N={n}, K={k}")
    
    # Get block dimensions from selector
    selector = tritonblas.MatmulHeuristicResult(m, n, k, 
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32,
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32,
                                               torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32)
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    
    # Get all workgroup combinations (constrained by grid dimensions)
    num_pid_m = math.ceil(m / BLK_M)
    num_pid_n = math.ceil(n / BLK_N)
    wgm_wgn_combinations = get_all_wgm_wgn_combinations(max_wgm, max_wgn, num_pid_m, num_pid_n)
    print(f"Num wgm/wgn combinations to test: {len(wgm_wgn_combinations)}")
    
    # All ordering combinations
    orderings = [0, 1, 2, 3]  # ROW_MAJOR, COLUMN_MAJOR, SNAKE, SPIRAL
    print(f"Testing orderings: {[get_ordering_name(ord) for ord in orderings]}")
    ordering_combinations = [(o0, o1) for o0 in orderings for o1 in orderings]
    
    total_combinations = len(wgm_wgn_combinations) * len(ordering_combinations)
    print(f"  Block sizes: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
    print(f"  Grid: {num_pid_m} x {num_pid_n}")
    print(f"  WGM/WGN combinations: {len(wgm_wgn_combinations)}")
    print(f"  Ordering combinations: {len(ordering_combinations)}")
    print(f"  Total combinations: {total_combinations}")
    
    # Generate filenames
    json_filename = f"sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}_{arch}.json"
    csv_filename = f"sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}_{arch}.csv"
    json_path = os.path.join(results_dir, json_filename)
    csv_path = os.path.join(results_dir, csv_filename)


    # Get optimal baseline:
    baseline_results = []
    baseline_wgm_values = [1, 2, 4, 6, 8, 16]
    print(f"Computing baseline perf for WGMs: {baseline_wgm_values}...")
    for wgm in baseline_wgm_values:
        # if wgm > num_pid_m or wgm > num_pid_n:
        #     break
        baseline_results.append(run_baseline_benchmark(m, n, k, wgm))

    optimal_l2_hit_rate = -1
    optimal_tflops = -1
    optimal_ms = -1
    optimal_wgm = -1
    
    # Initialize predicted values
    predicted_tflops = None
    predicted_l2_hit_rate = None
    predicted_ms = None
    
    for res in baseline_results:
        profiler_data = res["profiler_data"]
        benchmark_data = res["benchmark_data"]
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
        raise RuntimeError(f"Could not find baseline result for predicted WGM={gsize_m}. Available WGMs: {[res['benchmark_data']['wgm'] for res in baseline_results]}")
    if optimal_tflops == -1:
        raise RuntimeError("No valid baseline results found - all baseline runs failed")

    baseline_data = {
        "predicted_wgm": gsize_m, 
        "predicted_tflops": predicted_tflops,
        "predicted_l2_hit_rate": predicted_l2_hit_rate, 
        "predicted_ms": predicted_ms, 
        "optimal_wgm": optimal_wgm,
        "optimal_tflops": optimal_tflops,
        "optimal_ms": optimal_ms,
        "optimal_l2_hit_rate": optimal_l2_hit_rate,
    }

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
            "num_pid_m": m // BLK_M,
            "num_pid_n": n // BLK_N
        },
        "arch": arch,
        "total_combinations": total_combinations,
        "orderings_tested": [get_ordering_name(o) for o in orderings],
        "baseline_data": baseline_data
    }

            
    
    # Prepare results structure
    sweep_results = []
    csv_data = []
    
    # Run all combinations with progressive saving
    combination_count = 0
    save_interval = max(1, total_combinations // 20)  # Save every 5% of progress
    
    for wgm, wgn in wgm_wgn_combinations:
        for ordering0, ordering1 in ordering_combinations:
            combination_count += 1
            print(f"  [{combination_count}/{total_combinations}] Ordering=({ordering0},{ordering1}), WGM={wgm}, WGN={wgn}")

            
            # Run benchmark
            result = run_tessera_benchmark(m, n, k, ordering0, ordering1, wgm, wgn, dtype)
            
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
                save_progressive_results(results, csv_data, json_path, csv_path)
                print(f"    Progress saved: {combination_count}/{total_combinations} ({100*combination_count/total_combinations:.1f}%)")
    
    # Final save
    results = {
        "metadata": metadata,
        "sweep_results": sweep_results
    }
    save_progressive_results(results, csv_data, json_path, csv_path)
    
    return results, csv_data

def main():
    parser = argparse.ArgumentParser(description="Sweep tessera matmul configurations")
    parser.add_argument("csv_file", help="CSV file with matrix problems (m,n,k)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--max-wgm", type=int, default=8, help="Maximum WGM value")
    parser.add_argument("--max-wgn", type=int, default=8, help="Maximum WGN value")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--arch", type=str, required=True)
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Read matrix problems from CSV
    matrix_problems = []
    with open(args.csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row['m'])
            n = int(row['n'])
            k = int(row['k'])
            matrix_problems.append((m, n, k))
    
    print(f"Found {len(matrix_problems)} matrix problems in {args.csv_file}")
    
    # Process each matrix problem
    for problem_idx, (m, n, k) in enumerate(matrix_problems):
        print(f"\n{'='*80}")
        print(f"Processing problem {problem_idx+1}/{len(matrix_problems)}: M={m}, N={n}, K={k}")
        print(f"{'='*80}")
        
        try:
            # Run sweep with progressive saving
            results, csv_data = sweep_matrix_problem(m, n, k, args.arch, args.dtype, args.max_wgm, args.max_wgn, args.results_dir)
            
            # Print summary
            sweep_results = results["sweep_results"]
            successful_runs = len([r for r in sweep_results if r["number_of_errors"] == 0])
            print(f"Summary: {successful_runs}/{len(sweep_results)} runs successful (0 errors)")
            
            if sweep_results:
                best_tflops = max(r["tflops"] for r in sweep_results)
                best_config = next(r for r in sweep_results if r["tflops"] == best_tflops)
                print(f"Best TFLOPS: {best_tflops:.3f} (Ordering=({best_config['ordering_0']},{best_config['ordering_1']}), WGM={best_config['WGM']}, WGN={best_config['WGN']})")
            
            print(f"Problem {problem_idx+1} completed successfully!")
            
        except KeyboardInterrupt:
            print(f"\nInterrupted during problem {problem_idx+1}. Partial results saved.")
            break
        except Exception as e:
            print(f"Error processing problem {problem_idx+1}: {e}")
            sys.exit(1)
    
    print(f"\nSweep completed! Results saved to {args.results_dir}/")

if __name__ == "__main__":
    main()
