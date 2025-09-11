import torch
import math
import triton
import tritonblas
import argparse
import time
import json

ORDERINGS = {
    "row_major": 0,
    "column_major": 1,
    "spiral": 2,
    "diagonal": 3,
    "snake": 4,
}

def get_ordering_name(ord):
    for name, value in ORDERINGS.items():
        if value == ord:
            return name
    raise ValueError(f"Invalid ordering value: {ord}")

TEST_ITERS = 100

def main(m, n, k, ord0, ord1, wgm, wgn):
    # Allocate Tensors
    A = torch.randn(m, k, device="cuda", dtype=torch.float16)
    B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
    C = torch.zeros((m, n), device="cuda", dtype=torch.float16)


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    config = selector.get_config()

    print(f"Origami selected config:")
    print(f"BLK_M: {config[0]}, BLK_N: {config[1]}, BLK_K: {config[2]}, gsize_m: {config[3]}")


    # Compute performance metrics
    flops = lambda: 2 * m * n * k * 1e-12
    gflops = lambda ms: 2 * m * n * k * 1e-9 / (ms * 1e-3)
    bytes_fn = lambda: (A.element_size() * ((m * k) + (n * k))) + (
        (m * n) * C.element_size()
    )

    warmup_iters = 5
    bench_iters = 50

    # Compute baseline
    print("Computing baseline...")
    times_ref = []
    for _ in range(warmup_iters):
        ref = torch.matmul(A, B)

    for _ in range(bench_iters):
        start_event.record()
        torch_result = torch.matmul(A, B)
        torch_result = torch_result.contiguous()
        end_event.record()
        torch.cuda.synchronize()
        times_ref.append(start_event.elapsed_time(end_event))

    ms_ref = sum(times_ref) / len(times_ref)
    perf_ref_gflops = gflops(ms_ref)

    print(f"Testing {get_ordering_name(ord0)} {get_ordering_name(ord1)} wgm={wgm} wgn={wgn}")

    times = []
    for _ in range(warmup_iters):
        tritonblas.matmul_lt(A, B, C, selector, ord0, ord1, wgm, wgn)
    
    for _ in range(bench_iters):
        start_event.record()
        tritonblas.matmul_lt(A, B, C, selector, ord0, ord1, wgm, wgn)
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    ms = sum(times) / len(times)
    perf_gflops = gflops(ms)

    bench_results = {
        "ordering_name_0": get_ordering_name(ord0),
        "ordering_name_1": get_ordering_name(ord1),
        "wgm": wgm,
        "wgn": wgn,
        "ms": ms,
        "tflops": perf_gflops / 1000,
        "ms_ref": ms_ref,
        "tflops_ref": perf_ref_gflops / 1000,
        "number_of_errors": 0,
    }

    # Save results to a json file
    with open("bench_result.json", "w") as f:
        json.dump(bench_results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example TritonBLAS matrix multiplication with CLI parameters for m, n, k."
    )
    parser.add_argument(
        "--m",
        type=int,
        help="Number of rows in matrix A and C (default: 8192)",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of columns in matrix B (after transpose) and C (default: 8192)",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of columns in matrix A and rows in matrix B (default: 8192)",
    )
    parser.add_argument(
        "--ord0",
        type=int,
    )
    parser.add_argument(
        "--ord1",
        type=int,
    )
    parser.add_argument(
        "--wgm",
        type=int,
    )
    parser.add_argument(
        "--wgn",
        type=int,
    )
    args = parser.parse_args()
    main(args.m, args.n, args.k, args.ord0, args.ord1, args.wgm, args.wgn)