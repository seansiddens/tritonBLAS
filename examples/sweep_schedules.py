import torch
import math
import triton
import tritonblas
import argparse
import time
import json
import subprocess
import os
import glob
import pandas as pd
import sys
from datetime import datetime, timedelta

ORDERINGS = {
    "ROW_MAJOR": 0,
    "COLUMN_MAJOR": 1,
    "SPIRAL": 2,
    "DIAGONAL": 3,
    "SNAKE": 4,
}

def analyze_profiler_data(csv_file, kernel_name='persistent_matmul'):
    try:
        # Load CSV data
        df = pd.read_csv(csv_file)
        
        # Filter for the specified kernel
        kernel_df = df[df['Kernel_Name'] == kernel_name]
        
        if len(kernel_df) == 0:
            return {"error": f"No data found for kernel '{kernel_name}'"}
        
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
        
        if not hit_rates:
            return {"error": "No valid hit rate data found"}
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(hit_rates)
        
        # Calculate overall statistics
        avg_hit_rate = results_df['Hit_Rate_pct'].mean()
        min_hit_rate = results_df['Hit_Rate_pct'].min()
        max_hit_rate = results_df['Hit_Rate_pct'].max()
        std_hit_rate = results_df['Hit_Rate_pct'].std()
        
        # Return comprehensive results
        return {
            "tcc_hits": int(results_df['TCC_HIT_sum'].sum()),
            "tcc_misses": int(results_df['TCC_MISS_sum'].sum()),
            "total_accesses": int(results_df['Total_Accesses'].sum()),
            "l2_hit_rate": float(avg_hit_rate / 100.0),  # Convert percentage to decimal
            "hit_rate_pct": float(avg_hit_rate),
            "min_hit_rate_pct": float(min_hit_rate),
            "max_hit_rate_pct": float(max_hit_rate),
            "std_hit_rate_pct": float(std_hit_rate),
            "num_dispatches": int(len(hit_rates)),
        }
        
    except Exception as e:
        return {"error": f"Failed to analyze profiler data: {str(e)}"}

def run_benchmark_with_profiling(m, n, k, ord0, ord1, wgm, wgn, run_id):
    """Run the benchmark script with profiling and return results"""
    print(f"Running benchmark: ord0={ord0}, ord1={ord1}, wgm={wgm}, wgn={wgn}")
    
    # Create profiling input file
    with open("input.txt", "w") as f:
        f.write("pmc: TCC_HIT_sum TCC_MISS_sum\n")
    
    # Run the benchmark with rocprof
    output_file = f"pmc_{run_id}"
    cmd = [
        "rocprofv3", "-i", "input.txt", "-o", output_file,
        "--", "python3", "run_benchmark.py",
        "--m", str(m), "--n", str(n), "--k", str(k),
        "--ord0", str(ord0), "--ord1", str(ord1),
        "--wgm", str(wgm), "--wgn", str(wgn)
    ]
    
    try:
        print(f"  Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  Benchmark completed successfully for run {run_id}")
        
        # Analyze profiler data
        csv_file = f"pmc_1/{output_file}_counter_collection.csv"
        if os.path.exists(csv_file):
            profiler_data = analyze_profiler_data(csv_file)
            if "error" not in profiler_data:
                print(f"  L2 hit rate: {profiler_data['hit_rate_pct']:.2f}% "
                      f"(min: {profiler_data['min_hit_rate_pct']:.2f}%, "
                      f"max: {profiler_data['max_hit_rate_pct']:.2f}%)")
                print(f"  TCC hits: {profiler_data['tcc_hits']}, "
                      f"misses: {profiler_data['tcc_misses']}, "
                      f"total: {profiler_data['total_accesses']}")
            else:
                print(f"  Profiler data error: {profiler_data['error']}")
        else:
            profiler_data = {"error": "Profiler CSV file not found"}
            print(f"  Profiler CSV file not found: {csv_file}")
        
        # Read benchmark results
        if os.path.exists("bench_result.json"):
            with open("bench_result.json", "r") as f:
                bench_data = json.load(f)
            if "error" not in bench_data:
                print(f"  Performance: {bench_data.get('tflops', 'N/A'):.2f} TFLOPS")
            else:
                print(f"  Benchmark data error: {bench_data['error']}")
        else:
            bench_data = {"error": "Benchmark results file not found"}
            print(f"  Benchmark results file not found")
        
        # # Clean up temporary files
        # if os.path.exists("input.txt"):
        #     os.remove("input.txt")
        # if os.path.exists("bench_result.json"):
        #     os.remove("bench_result.json")
        
        return {
            "profiler_data": profiler_data,
            "benchmark_data": bench_data
        }
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Benchmark failed for run {run_id}: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return {
            "profiler_data": {"error": f"Benchmark execution failed: {str(e)}"},
            "benchmark_data": {"error": f"Benchmark execution failed: {str(e)}"}
        }
    except FileNotFoundError as e:
        print(f"  ❌ Command not found: {e}")
        return {
            "profiler_data": {"error": f"Command not found: {str(e)}"},
            "benchmark_data": {"error": f"Command not found: {str(e)}"}
        }

def load_schedules_from_json(json_file_path):
    """Load schedules from JSON file"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        required_fields = ['metadata', 'sweep_results']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        metadata = data['metadata']
        required_metadata = ['matrix_dimensions', 'block_dimensions', 'grid']
        for field in required_metadata:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: {field}")
        
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error loading JSON file: {e}")

def main(json_file_path):
    # Load schedules from JSON file
    print(f"Loading schedules from: {json_file_path}")
    data = load_schedules_from_json(json_file_path)
    
    # Extract metadata
    metadata = data['metadata']
    matrix_dims = metadata['matrix_dimensions']
    block_dims = metadata['block_dimensions']
    grid_info = metadata['grid']
    sweep_results = data['sweep_results']
    
    m = matrix_dims['m']
    n = matrix_dims['n']
    k = matrix_dims['k']
    BLK_M = block_dims['BLK_M']
    BLK_N = block_dims['BLK_N']
    BLK_K = block_dims['BLK_K']
    num_pid_m = grid_info['num_pid_m']
    num_pid_n = grid_info['num_pid_n']
    
    total_combinations = len(sweep_results)
    
    print(f"Matrix dimensions: m={m}, n={n}, k={k}")
    print(f"Block dimensions: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
    print(f"Number of PIDs: num_pid_m={num_pid_m}, num_pid_n={num_pid_n}")
    print(f"Total combinations to test: {total_combinations}")
    
    start_time = datetime.now()
    current_combination = 0
    run_id = 1

    # Initialize results structure
    all_results = {
        "metadata": {
            "source_json": json_file_path,
            "sweep_start_time": start_time.isoformat(),
            "matrix_dimensions": matrix_dims,
            "block_dimensions": block_dims,
            "grid": grid_info,
            "total_combinations": total_combinations,
            "orderings_tested": list(ORDERINGS.keys())
        },
        "sweep_results": []
    }
    
    print(f"\nStarting sweep at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total combinations to test: {total_combinations}")
    
    # Process each schedule from the JSON file
    for schedule in sweep_results:
        current_combination += 1
        elapsed_time = datetime.now() - start_time
        
        # Extract schedule information
        ordering_name_0 = schedule['ordering_0']
        ordering_name_1 = schedule['ordering_1']

        if ordering_name_0 == "DIAGONAL" or ordering_name_1 == "DIAGONAL":
            # THIS DOENS"T WORK RN!!
            continue
        
        WGM = schedule['WGM']
        WGN = schedule['WGN']
        combination_id = schedule.get('combination_id', current_combination)
        
        # Get ordering values
        ordering_value_0 = ORDERINGS[ordering_name_0]
        ordering_value_1 = ORDERINGS[ordering_name_1]
        
        # Calculate progress and estimated time remaining
        progress = current_combination / total_combinations
        if current_combination > 1:
            avg_time_per_run = elapsed_time / (current_combination - 1)
            remaining_runs = total_combinations - current_combination
            estimated_remaining = avg_time_per_run * remaining_runs
            eta = datetime.now() + estimated_remaining
        else:
            eta = "Calculating..."
            estimated_remaining = "Calculating..."
        
        print(f"\n[{current_combination}/{total_combinations}] "
              f"Progress: {progress:.1%} | "
              f"Elapsed: {str(elapsed_time).split('.')[0]} | "
              f"ETA: {eta} | "
              f"Estimated remaining time: {estimated_remaining}")
        print(f"Testing {ordering_name_0} + {ordering_name_1} WGM={WGM} WGN={WGN} (ID: {combination_id})")
        
        # Run benchmark with profiling
        result = run_benchmark_with_profiling(m, n, k, ordering_value_0, ordering_value_1, WGM, WGN, run_id)
        
        # Store results
        sweep_result = {
            "ordering_0": ordering_name_0,
            "ordering_1": ordering_name_1,
            "WGM": WGM,
            "WGN": WGN,
            "combination_id": combination_id,
            "run_id": run_id,
            "profiler_data": result["profiler_data"],
            "benchmark_data": result["benchmark_data"]
        }
        
        all_results["sweep_results"].append(sweep_result)
        run_id += 1
        
        # Save current state after each test
        output_filename = f"sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}.json"
        with open(output_filename, "w") as f:
            json.dump(all_results, f, indent=2)
    
    total_execution_time = datetime.now() - start_time
    print(f"\nSweep completed!")
    print(f"Total combinations tested: {total_combinations}")
    print(f"Total execution time: {str(total_execution_time).split('.')[0]}")
    print(f"Final results saved to: sweep_results_m{m}_n{n}_k{k}_mt{BLK_M}_nt{BLK_N}_kt{BLK_K}.json")
    
    # Display summary of best performing configurations
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    # Filter out failed runs
    successful_runs = [r for r in all_results["sweep_results"] 
                      if "error" not in r["benchmark_data"] and "error" not in r["profiler_data"]]
    
    if successful_runs:
        # Sort by TFLOPS performance
        successful_runs.sort(key=lambda x: x["benchmark_data"].get("tflops", 0), reverse=True)
        
        print(f"\nTop 5 configurations by TFLOPS:")
        for i, run in enumerate(successful_runs[:5]):
            bench = run["benchmark_data"]
            prof = run["profiler_data"]
            print(f"{i+1}. {run['ordering_0']} + {run['ordering_1']} "
                  f"(WGM={run['WGM']}, WGN={run['WGN']})")
            print(f"   TFLOPS: {bench.get('tflops', 'N/A'):.2f}, "
                  f"L2 Hit Rate: {prof.get('hit_rate_pct', 'N/A'):.2f}%")
        
        # Sort by L2 hit rate
        successful_runs.sort(key=lambda x: x["profiler_data"].get("hit_rate_pct", 0), reverse=True)
        
        print(f"\nTop 5 configurations by L2 Hit Rate:")
        for i, run in enumerate(successful_runs[:5]):
            bench = run["benchmark_data"]
            prof = run["profiler_data"]
            print(f"{i+1}. {run['ordering_0']} + {run['ordering_1']} "
                  f"(WGM={run['WGM']}, WGN={run['WGN']})")
            print(f"   TFLOPS: {bench.get('tflops', 'N/A'):.2f}, "
                  f"L2 Hit Rate: {prof.get('hit_rate_pct', 'N/A'):.2f}%")
        
        # Overall statistics
        tflops_values = [r["benchmark_data"].get("tflops", 0) for r in successful_runs]
        l2_hit_rates = [r["profiler_data"].get("hit_rate_pct", 0) for r in successful_runs]
        
        print(f"\nOverall Statistics:")
        print(f"  Average TFLOPS: {sum(tflops_values)/len(tflops_values):.2f}")
        print(f"  Best TFLOPS: {max(tflops_values):.2f}")
        print(f"  Average L2 Hit Rate: {sum(l2_hit_rates)/len(l2_hit_rates):.2f}%")
        print(f"  Best L2 Hit Rate: {max(l2_hit_rates):.2f}%")
    else:
        print("\nNo successful runs to analyze.")
    
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmarks for specific schedules loaded from a JSON file with profiling."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to JSON file containing schedules to test"
    )
    args = parser.parse_args()
    
    # Check if JSON file exists
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        sys.exit(1)
    
    main(args.json_file)