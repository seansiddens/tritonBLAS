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

HIERARCHICAL_CONFIG = tritonblas.HierarchicalPersistentConfig(
    ordering0=1,
    ordering1=1,
    ordering2=1,
    L3Y=2,
    L3X=4,
    L2Y=8,
    L2X=4,
)


def _assert_large_grid(m, n, selector):
    blk_m, blk_n, _, _ = selector.get_config()
    num_pid_m = (m + blk_m - 1) // blk_m
    num_pid_n = (n + blk_n - 1) // blk_n
    assert num_pid_m >= 16 and num_pid_n >= 16


@pytest.mark.parametrize("m, n, k", YAML_PROBLEMS)
def test_hierarchical_depth3_matmul(m, n, k):
    dtype = torch.bfloat16
    A = torch.randn((m, k), device="cuda", dtype=dtype)
    B = torch.randn((k, n), device="cuda", dtype=dtype)
    C = torch.zeros((m, n), device="cuda", dtype=dtype)

    selector = tritonblas.MatmulHeuristicResult(m, n, k, dtype, dtype, dtype)
    _assert_large_grid(m, n, selector)

    tritonblas.matmul_lt(
        A,
        B,
        C,
        selector,
        enable_streamk=False,
        workgroup_schedule="hierarchical",
        hierarchical_config=HIERARCHICAL_CONFIG,
    )

    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)


@pytest.mark.parametrize("m, n, k", YAML_PROBLEMS[:3])
def test_hierarchical_depth3_workgroup_map(m, n, k):
    dtype = torch.bfloat16
    workgroup_map, _ = tritonblas.compute_persistent_workgroup_map(
        m,
        n,
        k,
        dtype,
        dtype,
        dtype,
        workgroup_schedule="hierarchical",
        hierarchical_config=HIERARCHICAL_CONFIG,
    )
    grid = workgroup_map.flatten().to(torch.int64)
    sorted_vals, _ = torch.sort(grid)
    expected = torch.arange(sorted_vals.numel(), device=grid.device, dtype=grid.dtype)
    assert torch.equal(sorted_vals, expected)
