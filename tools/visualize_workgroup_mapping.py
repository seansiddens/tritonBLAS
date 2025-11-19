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


def parse_hierarchical_config(arg: str | None):
    if arg is None:
        return None
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


def visualize_mapping(args):
    dtype = parse_dtype(args.dtype)
    hierarchical_config = parse_hierarchical_config(args.hierarchical_config)
    if args.workgroup_schedule == "hierarchical" and hierarchical_config is None:
        raise ValueError("hierarchical-config argument is required when using the hierarchical schedule.")
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
        hierarchical_config=hierarchical_config,
    )
    grid_tensor = workgroup_map.to(torch.int64)
    sorted_vals, _ = torch.sort(grid_tensor.flatten())
    expected_vals = torch.arange(grid_tensor.numel(), device=grid_tensor.device, dtype=grid_tensor.dtype)
    if not torch.equal(sorted_vals, expected_vals):
        raise ValueError("Workgroup mapping is not bijective.")
    else:
        print("Mapping is bijective.")
    grid = grid_tensor.cpu().numpy()
    color_data = grid % args.num_xcds

    num_pid_m = grid.shape[0]
    num_pid_n = grid.shape[1]
    total_tiles = num_pid_m * num_pid_n
    if config["workgroup_schedule"] == "hierarchical":
        chunk_size = config["chunk_size"]
    else:
        chunk_size = config["GROUP_SIZE_M"] * config["GROUP_SIZE_M"]
    message = (
        f"M={args.m}, N={args.n}, K={args.k} | "
        f"num_pid_m={num_pid_m}, num_pid_n={num_pid_n}, total_tiles={total_tiles} | "
        f"BLK_M={config['BLK_M']}, BLK_N={config['BLK_N']}, BLK_K={config['BLK_K']} | "
        f"GROUP_SIZE_M={config['GROUP_SIZE_M']}, NUM_XCDS={config['NUM_XCDS']}, CHUNK_SIZE={chunk_size}, NUM_L2_TILES={config['NUM_L2_TILES']} | "
        f"schedule={config['workgroup_schedule']}"
    )
    if config["workgroup_schedule"] in ("random", "workgroup_shuffle"):
        message += f", LCG_A={config['LCG_A']}, LCG_C={config['LCG_C']}"
    elif config["workgroup_schedule"] == "hierarchical":
        message += (
            f", ordering0={config['ordering0']}, ordering1={config['ordering1']}, ordering2={config['ordering2']}, "
            f"L3Y={config['L3Y']}, L3X={config['L3X']}, L2Y={config['L2Y']}, L2X={config['L2X']}, chunk_size={config['chunk_size']}"
        )
    print(message)
    # print("\nWorkgroup remap grid (transformed_pid -> original_pid):")
    # for row in grid:
    #     print(" ".join(f"{int(val):4d}" for val in row))

    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))
    cmap = plt.get_cmap("tab20", args.num_xcds)
    norm = plt.Normalize(vmin=0, vmax=max(args.num_xcds - 1, 1))
    colored = cmap(norm(color_data))
    if args.timestep is not None:
        timestep_mask = (grid // 256) != args.timestep
        colored[timestep_mask, :3] *= 0.2
    im = ax.imshow(colored, interpolation="nearest")
    ax.set_title(
        f"Persistent Matmul Workgroup Mapping ({config['workgroup_schedule']})\n"
        f"M={args.m}, N={args.n}, K={args.k}, "
        f"BLK_M={config['BLK_M']}, BLK_N={config['BLK_N']}, GROUP={config['GROUP_SIZE_M']}"
    )
    ax.set_xlabel("pid_n")
    ax.set_ylabel("pid_m")

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
        choices=["default", "random", "workgroup_shuffle", "hierarchical"],
        help="Workgroup scheduling strategy to visualize.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional seed when using the random workgroup schedule.",
    )
    parser.add_argument(
        "--hierarchical-config",
        type=str,
        default=None,
        help="Comma-separated ordering0,ordering1,ordering2,L3Y,L3X,L2Y,L2X when schedule=hierarchical.",
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
    parser.add_argument(
        "--timestep",
        type=int,
        default=None,
        help="Highlight only workgroups whose timestep (original_pid // 256) matches this value.",
    )
    args = parser.parse_args()
    visualize_mapping(args)


if __name__ == "__main__":
    main()
