#!/usr/bin/env python3
"""
Benchmark script for tritonBLAS tessera matmul function.
Measures performance and correctness against reference implementation.
"""

import argparse
import torch
import triton
import tritonblas
import time
import json
import os

def calculate_tflops(ms, m, n, k):
    """Calculate TFLOPS given elapsed time in ms and matrix dimensions."""
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)

def benchmark_tessera_matmul(m, n, k, ordering0, ordering1, wgm, wgn, dtype=torch.float16, warmup=10, rep=100):
    """
    Benchmark tessera matmul and compare with reference implementation.
    
    Args:
        m, n, k: Matrix dimensions
        ordering0, ordering1: Tessera ordering parameters (0=row_major, 1=col_major, 2=snake, 3=gilbert)
        wgm, wgn: Tessera workgroup dimensions
        dtype: Data type for matrices
        warmup: Number of warmup iterations
        rep: Number of benchmark iterations
    
    Returns:
        dict: Benchmark results including performance and correctness info
    """
    print(f"Benchmarking tessera matmul:")
    print(f"  Dimensions: M={m}, N={n}, K={k}")
    print(f"  Ordering: ({ordering0}, {ordering1})")
    print(f"  Workgroup: WGM={wgm}, WGN={wgn}")
    print(f"  Data type: {dtype}")
    print()
    
    # Create input matrices
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    
    # Allocate output tensors
    C_tessera = torch.zeros((m, n), device="cuda", dtype=dtype)
    C_reference = torch.zeros((m, n), device="cuda", dtype=dtype)
    
    # Create selector
    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C_tessera.dtype)
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    
    print(f"  Block sizes: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
    print(f"  Grid dimensions: {m//BLK_M} x {n//BLK_N}")
    print()
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        C_tessera.fill_(0)
        tritonblas.matmul_lt_tessera(A, B, C_tessera, selector, ordering0, ordering1, wgm, wgn)
    
    # Benchmark tessera
    print("Benchmarking tessera matmul...")
    def tessera_fn():
        C_tessera.fill_(0)
        tritonblas.matmul_lt_tessera(A, B, C_tessera, selector, ordering0, ordering1, wgm, wgn)
        return C_tessera
    
    tessera_ms = triton.testing.do_bench(tessera_fn, warmup=0, rep=rep)
    tessera_tflops = calculate_tflops(tessera_ms, m, n, k)
    
    # Benchmark reference
    print("Benchmarking reference matmul...")
    def reference_fn():
        C_reference.fill_(0)
        tritonblas.matmul_lt(A, B, C_reference, selector)
        return C_reference
    
    reference_ms = triton.testing.do_bench(reference_fn, warmup=0, rep=rep)
    reference_tflops = calculate_tflops(reference_ms, m, n, k)
    
    # Correctness check
    print("Checking correctness...")
    C_tessera.fill_(0)
    C_reference.fill_(0)
    
    tritonblas.matmul_lt_tessera(A, B, C_tessera, selector, ordering0, ordering1, wgm, wgn)
    tritonblas.matmul_lt(A, B, C_reference, selector)
    
    # Calculate error
    diff = torch.abs(C_tessera - C_reference)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    # Check if results are close
    try:
        torch.testing.assert_close(C_tessera, C_reference, atol=1e-3, rtol=1e-3)
        correctness = "PASS"
    except AssertionError:
        correctness = "FAIL"
    
    # Calculate number of errors (elements that differ significantly)
    error_threshold = 1e-3
    significant_errors = torch.sum(torch.abs(C_tessera - C_reference) > error_threshold).item()
    
    # Results for JSON output
    json_results = {
        'ordering0': ordering0,
        'ordering1': ordering1,
        'wgm': wgm,
        'wgn': wgn,
        'dtype': str(dtype),
        'tflops': tessera_tflops,
        'tflops_ref': reference_tflops,
        'ms': tessera_ms,
        'ms_ref': reference_ms,
        'number_of_errors': significant_errors
    }
    
    # Results for console output
    results = {
        'tessera_ms': tessera_ms,
        'tessera_tflops': tessera_tflops,
        'reference_ms': reference_ms,
        'reference_tflops': reference_tflops,
        'speedup': reference_ms / tessera_ms,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'correctness': correctness
    }
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Tessera MatMul:")
    print(f"  Time:     {tessera_ms:.3f} ms")
    print(f"  TFLOPS:   {tessera_tflops:.3f}")
    print()
    print(f"Reference MatMul:")
    print(f"  Time:     {reference_ms:.3f} ms")
    print(f"  TFLOPS:   {reference_tflops:.3f}")
    print()
    print(f"Performance:")
    print(f"  Speedup:  {results['speedup']:.3f}x")
    print(f"  Tessera is {'faster' if results['speedup'] > 1 else 'slower'}")
    print()
    print(f"Correctness:")
    print(f"  Status:   {correctness}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff:{mean_diff:.2e}")
    print("="*60)
    
    # Save results to JSON file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark tritonBLAS tessera matmul")
    parser.add_argument("m", type=int, help="Matrix M dimension")
    parser.add_argument("n", type=int, help="Matrix N dimension") 
    parser.add_argument("k", type=int, help="Matrix K dimension")
    parser.add_argument("ordering0", type=int, choices=[0,1,2,3], help="Ordering0 (0=row_major, 1=col_major, 2=snake, 3=gilbert)")
    parser.add_argument("ordering1", type=int, choices=[0,1,2,3], help="Ordering1 (0=row_major, 1=col_major, 2=snake, 3=gilbert)")
    parser.add_argument("wgm", type=int, help="Workgroup M dimension")
    parser.add_argument("wgn", type=int, help="Workgroup N dimension")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # Run benchmark
    results = benchmark_tessera_matmul(
        args.m, args.n, args.k,
        args.ordering0, args.ordering1,
        args.wgm, args.wgn,
        dtype=dtype,
        warmup=args.warmup,
        rep=args.rep
    )
    
    if results is None:
        return 1
    
    return 0 if results['correctness'] == "PASS" else 1

if __name__ == "__main__":
    exit(main())
