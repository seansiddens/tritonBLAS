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
def test_persistent_random_schedule(m, n, k, dtype):
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
        workgroup_schedule="random",
        shuffle_seed=0,
    )

    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)


@pytest.mark.parametrize("m, n, k", YAML_PROBLEMS)
def test_persistent_random_schedule_yaml_cases(m, n, k):
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
        workgroup_schedule="random",
        shuffle_seed=0,
    )

    torch.testing.assert_close(C, torch.matmul(A, B), atol=1, rtol=1)
