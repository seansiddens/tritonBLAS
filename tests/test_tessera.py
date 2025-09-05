import pytest
import torch
import triton
import tritonblas

@pytest.mark.parametrize(
    "m, n, k",
    [
        (8192, 8192, 8192),
        (4864, 8192, 4160),
        (4096, 4096, 4096),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype", 
    [
        # (torch.float8_e4m3fn, torch.float8_e4m3fn),
        # (torch.float8_e5m2, torch.float8_e5m2),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "transA, transB", 
    [
        ("T", "T"),  # A^T @ B^T
        ("N", "N"),  # A @ B
        ("T", "N"),  # A^T @ B
        ("N", "T"),  # A @ B^T
    ],
)
def test_matmul_tessera(m, n, k, in_dtype, out_dtype, transA, transB):
    
    # Adjust dimensions for transposition and apply tensor.T if needed
    if transA == "T":
        A_size = (m, k)  # A is MxK
    else:
        A_size = (k, m)  # A is KxM (we will later transpose it with .T)

    if transB == "T":
        B_size = (k, n)  # B is KxN
    else:
        B_size = (n, k)  # B is NxK (we will later transpose it with .T)
    
    A = torch.randn(A_size, device="cuda", dtype=in_dtype)
    B = torch.randn(B_size, device="cuda", dtype=in_dtype)
    
    # Apply transpose on A or B if necessary (only needed for "N" case)
    if transA == "N":
        A = A.T  # Apply transpose to A if transA is "N"

    if transB == "N":
        B = B.T  # Apply transpose to B if transB is "N"
            
    # Allocate Tensors
    C = torch.zeros((m, n), device="cuda", dtype=out_dtype)
    bias = torch.zeros((m,), device="cuda", dtype=out_dtype)

    # Run TritonBLAS matmul tessera
    selector = tritonblas.MatmulHeuristicResult(m, n, k, A.dtype, B.dtype, C.dtype)
    
    # Set random values for tessera-specific parameters
    ordering0 = 1
    ordering1 = 2
    wgm = 4
    wgn = 8
    
    tritonblas.matmul_lt_tessera(A, B, C, selector, ordering0, ordering1, wgm, wgn)

    # Check correctness: Fix tolerance later
    torch_c = torch.matmul(A, B)
    torch.testing.assert_close(C.to(out_dtype), torch_c, atol=1, rtol=1)
