import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

import tritonblas


def parse_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "int8": torch.int8,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_str}'. Options: {', '.join(mapping.keys())}")
    return mapping[dtype_str]


def visualize_mapping(args):
    dtype = parse_dtype(args.dtype)
    workgroup_map, config = tritonblas.compute_persistent_workgroup_map(
        args.m,
        args.n,
        args.k,
        dtype,
        dtype,
        dtype,
        num_xcds=args.num_xcds,
        workgroup_schedule=args.workgroup_schedule,
        shuffle_seed=args.shuffle_seed,
    )
    grid = workgroup_map.cpu().numpy()
    color_data = grid % args.num_xcds

    num_pid_m = grid.shape[0]
    num_pid_n = grid.shape[1]
    total_tiles = num_pid_m * num_pid_n
    chunk_size = config["GROUP_SIZE_M"] * config["GROUP_SIZE_M"]
    message = (
        f"M={args.m}, N={args.n}, K={args.k} | "
        f"num_pid_m={num_pid_m}, num_pid_n={num_pid_n}, total_tiles={total_tiles} | "
        f"BLK_M={config['BLK_M']}, BLK_N={config['BLK_N']}, BLK_K={config['BLK_K']} | "
        f"GROUP_SIZE_M={config['GROUP_SIZE_M']}, NUM_XCDS={config['NUM_XCDS']}, CHUNK_SIZE={chunk_size} | "
        f"schedule={config['workgroup_schedule']}"
    )
    if config["workgroup_schedule"] == "random":
        message += f", LCG_A={config['LCG_A']}, LCG_C={config['LCG_C']}"
    print(message)
    print("\nWorkgroup remap grid (transformed_pid -> original_pid):")
    for row in grid:
        print(" ".join(f"{int(val):4d}" for val in row))

    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))
    cmap = plt.get_cmap("tab20", args.num_xcds)
    im = ax.imshow(color_data, cmap=cmap, interpolation="nearest")
    ax.set_title(
        f"Persistent Matmul Workgroup Mapping ({config['workgroup_schedule']})\n"
        f"M={args.m}, N={args.n}, K={args.k}, "
        f"BLK_M={config['BLK_M']}, BLK_N={config['BLK_N']}, GROUP={config['GROUP_SIZE_M']}"
    )
    ax.set_xlabel("pid_n (tile columns)")
    ax.set_ylabel("pid_m (tile rows)")
    cbar = fig.colorbar(im, ticks=list(range(args.num_xcds)))
    cbar.set_label("Original pid % NUM_XCDS")

    if args.annotate:
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, int(grid[r, c]), ha="center", va="center", fontsize=6, color="white")

    fig.tight_layout()
    fig.savefig(args.output, dpi=500, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the workgroup-to-grid mapping for persistent matmul kernels."
    )
    parser.add_argument("m", type=int, help="Rows of matrix A/C.")
    parser.add_argument("n", type=int, help="Columns of matrix B/C.")
    parser.add_argument("k", type=int, help="Shared dimension (A columns / B rows).")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32", "int8"],
        help="Data type used when running the heuristics.",
    )
    parser.add_argument(
        "--num-xcds",
        type=int,
        default=8,
        help="Number of XCDs to color by (must match the kernel launch).",
    )
    parser.add_argument(
        "--workgroup-schedule",
        type=str,
        default="default",
        choices=["default", "random"],
        help="Workgroup scheduling strategy to visualize.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional seed when using the random workgroup schedule.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        default=6.0,
        help="Figure size (width == height) for the matplotlib visualization.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate each cell with the original PID that produced it.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="workgroup_mapping.png",
        help="Path to save the rendered figure (PNG).",
    )
    args = parser.parse_args()
    visualize_mapping(args)


if __name__ == "__main__":
    main()
