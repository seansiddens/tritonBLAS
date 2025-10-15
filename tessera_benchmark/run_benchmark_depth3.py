#!/usr/bin/env python3
"""
Benchmark script for tritonBLAS tessera depth-3 persistent matmul.
Measures performance of the persistent depth-3 tessera kernel and reports TFLOPS.
"""

import argparse
import json
import math

import torch
import triton
import tritonblas

from tritonblas.matmul import persistent_matmul_lt_tessera_depth3


def init_by_size_and_type(size, dtype, init_type):
    """
    Initialize a tensor of the given size and type using the specified initialization method.

    Args:
        size (tuple): The size of the tensor to be initialized.
        dtype (torch.dtype): The data type of the tensor.
        init_type (str): The initialization method ('hpl', 'trig_float', 'zeros', 'randn').

    Returns:
        torch.Tensor: The initialized tensor.
    """
    if init_type == "hpl":
        return torch.empty(size, device="cuda", dtype=dtype).uniform_(-0.5, 0.5)
    if init_type == "trig_float":
        m, n = size
        return (
            torch.reshape(
                torch.arange(0, m * n, device="cuda", dtype=torch.float32), (m, n)
            )
            .sin()
            .to(dtype=dtype)
        )
    if init_type == "zeros":
        return torch.zeros(size, dtype=dtype, device="cuda")
    if init_type == "randn":
        seeded = torch.randn(size, dtype=torch.float32, device="cuda")
        return seeded.to(dtype)
    raise ValueError(f"Unsupported init_type: {init_type}")


def benchmark_tessera_matmul_depth3(
    m,
    n,
    k,
    ordering0,
    ordering1,
    ordering2,
    L3Y,
    L3X,
    L2Y,
    L2X,
    dtype=torch.bfloat16,
    warmup=20,
    rep=20,
    chunk_size=-1,
    transA="T",
    transB="T",
):
    """
    Benchmark the depth-3 tessera persistent matmul implementation.
    """
    print("Benchmarking tessera depth-3 matmul:")
    print(f"  Dimensions: M={m}, N={n}, K={k}")
    print(f"  Orderings: ordering0={ordering0}, ordering1={ordering1}, ordering2={ordering2}")
    print(f"  L3 tile: {L3Y} x {L3X}")
    print(f"  L2 tile: {L2Y} x {L2X}")
    print(f"  Data type: {dtype}")
    print(f"  Chunk size: {chunk_size}")
    print()

    if transA == "T":
        a_shape = (m, k)
    else:
        a_shape = (k, m)

    if transB == "T":
        b_shape = (k, n)
    else:
        b_shape = (n, k)

    init_type = "randn"
    A = init_by_size_and_type(a_shape, dtype, init_type)
    B = init_by_size_and_type(b_shape, dtype, init_type)

    if transA == "N":
        A = A.T

    if transB == "N":
        B = B.T

    C = torch.zeros((m, n), device="cuda", dtype=dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    BLK_M, BLK_N, BLK_K, _ = selector.get_config()

    print(f"  Block sizes: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
    print(f"  Grid dimensions: {math.ceil(m / BLK_M)} x {math.ceil(n / BLK_N)}")
    print()

    def run_kernel():
        return persistent_matmul_lt_tessera_depth3(
            A,
            B,
            C,
            selector,
            ordering0,
            ordering1,
            ordering2,
            L3Y,
            L3X,
            L2Y,
            L2X,
            chunk_size=chunk_size,
        )

    elapsed_ms = triton.testing.do_bench(run_kernel, warmup=warmup, rep=rep)
    tflops = (2 * m * n * k * 1e-12) / (elapsed_ms * 1e-3)

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print("Tessera Depth-3 MatMul:")
    print(f"  Time:   {elapsed_ms:.3f} ms")
    print(f"  TFLOPS: {tflops:.3f}")

    results_json = {
        "ordering0": ordering0,
        "ordering1": ordering1,
        "ordering2": ordering2,
        "L3Y": L3Y,
        "L3X": L3X,
        "L2Y": L2Y,
        "L2X": L2X,
        "chunk_size": chunk_size,
        "dtype": str(dtype),
        "tflops": tflops,
        "ms": elapsed_ms,
        "transA": transA,
        "transB": transB,
        "init_type": init_type,
    }

    output_file = "benchmark_results_depth3.json"
    with open(output_file, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return {
        "elapsed_ms": elapsed_ms,
        "tflops": tflops,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tritonBLAS tessera depth-3 persistent matmul"
    )
    parser.add_argument("m", type=int, help="Matrix M dimension")
    parser.add_argument("n", type=int, help="Matrix N dimension")
    parser.add_argument("k", type=int, help="Matrix K dimension")
    parser.add_argument("ordering0", type=int, choices=[0, 1, 2, 3], help="Grid-level ordering")
    parser.add_argument("ordering1", type=int, choices=[0, 1, 2, 3], help="L3 ordering")
    parser.add_argument("ordering2", type=int, choices=[0, 1, 2, 3], help="L2 ordering")
    parser.add_argument("L3Y", type=int, help="L3 tile size along M")
    parser.add_argument("L3X", type=int, help="L3 tile size along N")
    parser.add_argument("L2Y", type=int, help="L2 tile size along M (workgroup)")
    parser.add_argument("L2X", type=int, help="L2 tile size along N (workgroup)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=-1,
        help="Chunk size for chiplet remapping (-1 disables chunking)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    benchmark_tessera_matmul_depth3(
        args.m,
        args.n,
        args.k,
        args.ordering0,
        args.ordering1,
        args.ordering2,
        args.L3Y,
        args.L3X,
        args.L2Y,
        args.L2X,
        dtype=dtype,
        warmup=args.warmup,
        rep=args.rep,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
