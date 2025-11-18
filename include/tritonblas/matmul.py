import torch
import triton
import random
import functools
import time
import math
from dataclasses import dataclass
from .internal.persistent_matmul import (
    persistent_matmul,
    persistent_matmul_shuffled,
    persistent_matmul_debug_map,
    persistent_matmul_debug_map_shuffled,
    persistent_matmul_hierarchical,
    persistent_matmul_debug_map_hierarchical,
)
from .internal.streamk_matmul import streamk_matmul
from .origami import MatmulHeuristicResult
from typing import Dict, Tuple, Optional

_tensor_cache = {}
current_device_index = torch.cuda.current_device()
current_device = torch.cuda.get_device_properties(current_device_index)
MAX_SMS = current_device.multi_processor_count
# TODO: 256x256 for fp16/bf16, need adjust for fp8/fp4
MAX_BLOCK_SIZE = 65536

# Global pre-allocated buffers
_global_locks = torch.empty(MAX_SMS, device="cuda", dtype=torch.uint8)
_global_P = torch.empty(MAX_SMS, MAX_BLOCK_SIZE, device="cuda", dtype=torch.float32)


@dataclass(frozen=True)
class HierarchicalPersistentConfig:
    ordering0: int
    ordering1: int
    ordering2: int
    L3Y: int
    L3X: int
    L2Y: int
    L2X: int

    def __post_init__(self):
        dim_fields = ("L3Y", "L3X", "L2Y", "L2X")
        for name in dim_fields:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0 for hierarchical schedule (got {value}).")

    @property
    def chunk_size(self) -> int:
        return self.L2Y * self.L2X

    def to_kernel_kwargs(self) -> Dict[str, int]:
        return {
            "ordering0": self.ordering0,
            "ordering1": self.ordering1,
            "ordering2": self.ordering2,
            "L3Y": self.L3Y,
            "L3X": self.L3X,
            "L2Y": self.L2Y,
            "L2X": self.L2X,
            "chunk_size": self.chunk_size,
        }


# Function will behave like an LRU-Cache of heuristic results
# Saves several microseconds for previously seen problems by not rerunning the heuristic unnecessarily
@functools.lru_cache(maxsize=1024)
def _make_matmul_selector(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
):
    # Run Heuristic Results (Only if key has not been seen before)
    return MatmulHeuristicResult(M, N, K, a_dtype, b_dtype, c_dtype)


def _count_quantized_l2_tiles(num_pid_m: int, num_pid_n: int, tile_dim: int) -> int:
    if tile_dim <= 0:
        return 0
    quantized_m = (num_pid_m // tile_dim) * tile_dim
    quantized_n = (num_pid_n // tile_dim) * tile_dim
    tiles_per_row = quantized_n // tile_dim
    tiles_per_col = quantized_m // tile_dim
    return tiles_per_row * tiles_per_col


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _choose_lcg_params(
    n: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    if n <= 1:
        raise ValueError("shuffle requires at least two tiles in the grid")
    rng = rng or (random.Random(seed) if seed is not None else random.Random())
    if _is_power_of_two(n):
        valid_as = [a for a in range(5, n, 4)]
        if not valid_as:
            valid_as = [a for a in range(1, n, 2) if a != 1]
        valid_cs = list(range(n))
    else:
        valid_as = [a for a in range(1, n) if math.gcd(a, n) == 1 and a != 1]
        if not valid_as:
            valid_as = [a for a in range(1, n) if math.gcd(a, n) == 1]
        valid_cs = list(range(n))
    if not valid_as or not valid_cs:
        raise ValueError("No valid LCG parameters for given n")
    a = rng.choice(valid_as)
    c = rng.choice(valid_cs)
    return a, c


def choose_lcg_shuffle_params(
    num_tiles: int,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[int, int]:
    return _choose_lcg_params(num_tiles, seed=seed, rng=rng)


def persistent_matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    workgroup_schedule: str = "default",
    shuffle_seed: Optional[int] = None,
    hierarchical_config: Optional[HierarchicalPersistentConfig] = None,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    total_programs = total_tiles
    even_k = K % BLK_K == 0
    num_l2_tiles = _count_quantized_l2_tiles(total_blocks_M, total_blocks_N, gsize_m)

    # TODO: Separate these configs.
    # basica configs for most of compute bound sizes
    # TODO: set these values analytically?
    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1

    # Run in Data-parallel mode.
    grids = total_tiles

    # TODO: Support other matmul algs.
    kernel_kwargs = dict(
        A=a,
        B=b,
        C=c,
        bias_ptr=None,
        M=M,
        N=N,
        K=K,
        stride_am=a.stride(0),
        stride_bn=b.stride(1),
        stride_cm=c.stride(0),
        stride_cn=c.stride(1),
        stride_bias=0,
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        NUM_SMS=total_programs,
        NUM_XCDS=8,
        BIAS=False,
        EVEN_K=even_k,
    )

    if workgroup_schedule == "default":
        kernel = persistent_matmul
        kernel_kwargs.update({"GROUP_SIZE_M": gsize_m})
    elif workgroup_schedule == "random":
        if num_l2_tiles <= 1:
            raise ValueError(
                "Random workgroup schedule requires at least two full L2 tiles in the quantized region."
            )
        a_lcg, c_lcg = _choose_lcg_params(num_l2_tiles, seed=shuffle_seed)
        kernel = persistent_matmul_shuffled
        kernel_kwargs.update({"LCG_A": a_lcg, "LCG_C": c_lcg, "GROUP_SIZE_M": gsize_m})
    elif workgroup_schedule == "hierarchical":
        if hierarchical_config is None:
            raise ValueError("hierarchical_config is required when workgroup_schedule='hierarchical'.")
        kernel = persistent_matmul_hierarchical
        kernel_kwargs.update(hierarchical_config.to_kernel_kwargs())
    else:
        raise ValueError(
            f"Unknown workgroup_schedule '{workgroup_schedule}'. Expected 'default', 'random', or 'hierarchical'."
        )

    kernel[(grids,)](
        **kernel_kwargs,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def streamk_matmul_lt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, selector, sk_grid: Optional[int] = None
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()

    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    even_k = K % BLK_K == 0

    ##
    # Grid Size
    ##
    total_programs_streamk = selector.get_grid()

    if total_programs_streamk > 0:  # Stream-K
        total_tiles_streamk = total_tiles % total_programs_streamk
    else:  # all tiles are computed using classical blocking
        total_tiles_streamk = 0

    num_stages = 2
    num_warps = 8
    waves_per_eu = 0
    mfmaInstrSize = 16
    kpack = 1

    if sk_grid is not None:
        total_programs_streamk = sk_grid

    grids = total_programs_streamk
    block_size = BLK_M * BLK_N

    # Use global buffers with optimized zeroing
    if grids <= MAX_SMS and block_size <= MAX_BLOCK_SIZE:
        locks = _global_locks[:grids]
        P = _global_P[:grids, :block_size]
    else:
        locks = torch.empty(grids, device="cuda", dtype=torch.uint8)
        P = torch.empty(grids, block_size, device="cuda", dtype=torch.float32)

    kk = streamk_matmul[(grids,)](
        a,
        b,
        c,
        None,  # TODO: Enable bias.
        P,
        locks,
        M,
        N,
        K,
        a.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0,  # TODO: Enable bias stride.
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        BLOCK_SIZE_K=BLK_K,
        GROUP_SIZE_M=gsize_m,
        NUM_SMS=grids,
        NUM_XCDS=8,
        STREAMK_TILES=total_tiles_streamk,
        BIAS=False,
        EVEN_K=even_k,
        num_stages=num_stages,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=mfmaInstrSize,
        kpack=kpack,
    )

    return c


def matmul_lt(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    selector,
    enable_streamk=False,
    workgroup_schedule: str = "default",
    shuffle_seed: Optional[int] = None,
    hierarchical_config: Optional[HierarchicalPersistentConfig] = None,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"

    if enable_streamk:
        if workgroup_schedule != "default":
            raise ValueError("workgroup_schedule customization is only supported for persistent matmul.")
        return streamk_matmul_lt(a, b, c, selector)
    else:
        return persistent_matmul_lt(
            a,
            b,
            c,
            selector,
            workgroup_schedule=workgroup_schedule,
            shuffle_seed=shuffle_seed,
            hierarchical_config=hierarchical_config,
        )


def compute_persistent_workgroup_map(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    num_xcds: int = 8,
    workgroup_schedule: str = "default",
    shuffle_seed: Optional[int] = None,
    hierarchical_config: Optional[HierarchicalPersistentConfig] = None,
):
    selector = _make_matmul_selector(M, N, K, a_dtype, b_dtype, c_dtype)
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    total_blocks_M = triton.cdiv(M, BLK_M)
    total_blocks_N = triton.cdiv(N, BLK_N)
    total_tiles = total_blocks_M * total_blocks_N
    grids = total_tiles
    workgroup_map = torch.empty(total_tiles, device="cuda", dtype=torch.int32)
    num_l2_tiles = _count_quantized_l2_tiles(total_blocks_M, total_blocks_N, gsize_m)
    kernel_kwargs = dict(
        workgroup_map=workgroup_map,
        M=M,
        N=N,
        BLOCK_SIZE_M=BLK_M,
        BLOCK_SIZE_N=BLK_N,
        NUM_SMS=grids,
        NUM_XCDS=num_xcds,
    )
    config = {
        "BLK_M": BLK_M,
        "BLK_N": BLK_N,
        "BLK_K": BLK_K,
        "GROUP_SIZE_M": gsize_m,
        "NUM_XCDS": num_xcds,
        "workgroup_schedule": workgroup_schedule,
        "NUM_L2_TILES": num_l2_tiles,
    }
    if workgroup_schedule == "default":
        kernel = persistent_matmul_debug_map
        kernel_kwargs.update({"GROUP_SIZE_M": gsize_m})
    elif workgroup_schedule == "random":
        if num_l2_tiles <= 1:
            raise ValueError(
                "Random workgroup schedule requires at least two full L2 tiles in the quantized region."
            )
        a_lcg, c_lcg = _choose_lcg_params(num_l2_tiles, seed=shuffle_seed)
        kernel = persistent_matmul_debug_map_shuffled
        kernel_kwargs.update({"LCG_A": a_lcg, "LCG_C": c_lcg, "GROUP_SIZE_M": gsize_m})
        config["LCG_A"] = a_lcg
        config["LCG_C"] = c_lcg
    elif workgroup_schedule == "hierarchical":
        if hierarchical_config is None:
            raise ValueError("hierarchical_config is required when workgroup_schedule='hierarchical'.")
        kernel = persistent_matmul_debug_map_hierarchical
        kernel_kwargs.update(hierarchical_config.to_kernel_kwargs())
        config.update(hierarchical_config.to_kernel_kwargs())
    else:
        raise ValueError(
            f"Unknown workgroup_schedule '{workgroup_schedule}'. Expected 'default', 'random', or 'hierarchical'."
        )

    kernel[(grids,)](**kernel_kwargs)
    return (
        workgroup_map.view(total_blocks_M, total_blocks_N),
        config,
    )


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    enable_streamk=False,
    sk_grid=None,
    workgroup_schedule: str = "default",
    shuffle_seed: Optional[int] = None,
    hierarchical_config: Optional[HierarchicalPersistentConfig] = None,
):
    assert a.shape[1] == b.shape[0], "Incompatible Dimensions"
    M, K = a.shape
    _, N = b.shape

    selector = _make_matmul_selector(M, N, K, a.dtype, b.dtype, c.dtype)
    if enable_streamk:
        return streamk_matmul_lt(a, b, c, selector, sk_grid=sk_grid)
    else:
        return persistent_matmul_lt(
            a,
            b,
            c,
            selector,
            workgroup_schedule=workgroup_schedule,
            shuffle_seed=shuffle_seed,
            hierarchical_config=hierarchical_config,
        )
