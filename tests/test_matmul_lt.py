import pytest
import torch
import tritonblas


@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
        (4864, 8192, 4160),
        (4096, 4096, 4096),
        (2791, 9093, 1230),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "transA, transB",
    [
        ("T", "T"),
        ("N", "N"),
        ("T", "N"),
        ("N", "T"),
    ],
)
def test_matmul_default_schedule(m, n, k, in_dtype, out_dtype, transA, transB):
    A_size = (m, k) if transA == "T" else (k, m)
    B_size = (k, n) if transB == "T" else (n, k)
    A = torch.randn(A_size, device="cuda", dtype=in_dtype)
    B = torch.randn(B_size, device="cuda", dtype=in_dtype)
    if transA == "N":
        A = A.T
    if transB == "N":
        B = B.T

    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    tritonblas.matmul_lt(A, B, C, selector, enable_streamk=False)
    torch.testing.assert_close(C.to(out_dtype), torch.matmul(A, B), atol=1, rtol=1)
