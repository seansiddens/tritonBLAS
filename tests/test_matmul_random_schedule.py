import math

import pytest
import torch
import tritonblas


YAML_PROBLEMS = [
    (23904, 31744, 32000),
    (32560, 10496, 29744),
    (23760, 10688, 28528),
    (45552, 19536, 10208),
    (18480, 20960, 9200),
    (44464, 21072, 20848),
    (14656, 14960, 50448),
    (11024, 25536, 35184),
    (14384, 17264, 9408),
    (12160, 22528, 12160),
]


def _count_l2_tiles_for_problem(m, n, selector) -> tuple[int, int]:
    BLK_M, BLK_N, _, gsize_m = selector.get_config()
    total_blocks_M = math.ceil(m / BLK_M)
    total_blocks_N = math.ceil(n / BLK_N)
    tile = max(int(gsize_m), 1)
    if tile <= 0:
        return 0, total_blocks_M * total_blocks_N
    quantized_m = (total_blocks_M // tile) * tile
    quantized_n = (total_blocks_N // tile) * tile
    tiles_per_row = quantized_n // tile
    tiles_per_col = quantized_m // tile
    return tiles_per_row * tiles_per_col, total_blocks_M * total_blocks_N


def _find_problem_with_fewer_than_two_l2_tiles(dtype: torch.dtype):
    candidate_sizes = [256, 320, 384, 448, 512, 640, 768, 896, 1024]
    for m in candidate_sizes:
        for n in candidate_sizes:
            for k in candidate_sizes:
                selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
                num_l2_tiles, total_tiles = _count_l2_tiles_for_problem(m, n, selector)
                if num_l2_tiles <= 1 and total_tiles > 1:
                    return m, n, k
    return None


def make_hierarchical_config(selector):
    _, _, _, gsize_m = selector.get_config()
    tile = max(int(gsize_m), 1)
    return tritonblas.HierarchicalPersistentConfig(
        ordering0=0,
        ordering1=0,
        ordering2=0,
        L3Y=1,
        L3X=1,
        L2Y=tile,
        L2X=tile,
    )


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
        (6144, 4096, 4096),
        (4864, 8192, 4160),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("workgroup_schedule", ["random", "workgroup_shuffle"])
def test_persistent_random_schedule(m, n, k, dtype, workgroup_schedule):
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    C = torch.zeros((m, n), device="cuda", dtype=dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    tritonblas.matmul_lt(
        A,
        B,
        C,
        selector,
        enable_streamk=False,
        workgroup_schedule=workgroup_schedule,
        shuffle_seed=0,
    )

    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
        (6144, 4096, 4096),
        (4864, 8192, 4160),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
    ],
)
def test_persistent_hierarchical_schedule(m, n, k, dtype):
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    C = torch.zeros((m, n), device="cuda", dtype=dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    hier_config = make_hierarchical_config(selector)
    tritonblas.matmul_lt(
        A,
        B,
        C,
        selector,
        enable_streamk=False,
        workgroup_schedule="hierarchical",
        hierarchical_config=hier_config,
    )

    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)


@pytest.mark.parametrize("m, n, k", YAML_PROBLEMS)
@pytest.mark.parametrize("workgroup_schedule", ["random", "workgroup_shuffle"])
def test_persistent_random_schedule_yaml_cases(m, n, k, workgroup_schedule):
    dtype = torch.bfloat16
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    C = torch.zeros((m, n), device="cuda", dtype=dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    tritonblas.matmul_lt(
        A,
        B,
        C,
        selector,
        enable_streamk=False,
        workgroup_schedule=workgroup_schedule,
        shuffle_seed=0,
    )

    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)


@pytest.mark.parametrize("m, n, k", YAML_PROBLEMS)
def test_persistent_hierarchical_schedule_yaml_cases(m, n, k):
    dtype = torch.bfloat16
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    C = torch.zeros((m, n), device="cuda", dtype=dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    hier_config = make_hierarchical_config(selector)
    tritonblas.matmul_lt(
        A,
        B,
        C,
        selector,
        enable_streamk=False,
        workgroup_schedule="hierarchical",
        hierarchical_config=hier_config,
    )

    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)


def test_workgroup_shuffle_handles_small_grids():
    dtype = torch.float16
    maybe_problem = _find_problem_with_fewer_than_two_l2_tiles(dtype)
    if maybe_problem is None:
        pytest.skip("Could not find a problem with <= 1 L2 tiles for current heuristics.")
    m, n, k = maybe_problem
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    C = torch.zeros((m, n), device="cuda", dtype=dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    with pytest.raises(ValueError):
        tritonblas.matmul_lt(
            A,
            B,
            C.clone(),
            selector,
            enable_streamk=False,
            workgroup_schedule="random",
            shuffle_seed=0,
        )

    tritonblas.matmul_lt(
        A,
        B,
        C,
        selector,
        enable_streamk=False,
        workgroup_schedule="workgroup_shuffle",
        shuffle_seed=0,
    )
    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)
