import pytest
import torch
import triton
import tritonblas
import math

def get_valid_wgm_wgn_combinations(m, n, blk_m, blk_n, max_wgm=16, max_wgn=16):
    """Generate valid WGM and WGN combinations that evenly divide the grid dimensions."""
    grid_m = math.ceil(m / blk_m)
    grid_n = math.ceil(n / blk_n)
    
    valid_wgm = [w for w in range(1, min(max_wgm + 1, grid_m)) if grid_m % w == 0]
    valid_wgn = [w for w in range(1, min(max_wgn + 1, grid_n)) if grid_n % w == 0]
    # valid_wgm = [w for w in range(1, min(max_wgm + 1, grid_m))]
    # valid_wgn = [w for w in range(1, min(max_wgn + 1, grid_n))]
    
    # Return all combinations
    combinations = []
    for wgm in valid_wgm:
        for wgn in valid_wgn:
            combinations.append((wgm, wgn))
    
    return combinations if combinations else [(1, 1)]  # fallback

@pytest.mark.parametrize(
    "m, n, k",
    [
        # (8192, 8192, 8192),
        # (4864, 8192, 4160),
        # (5184, 2768, 6000),
        (7744, 5616, 6080)
        # (4096, 4096, 4096),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype", 
    [
        # (torch.float8_e4m3fn, torch.float8_e4m3fn),
        # (torch.float8_e5m2, torch.float8_e5m2),
        # (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        # (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "transA, transB", 
    [
        # ("T", "T"),  # A^T @ B^T
        ("N", "N"),  # A @ B
        # ("T", "N"),  # A^T @ B
        # ("N", "T"),  # A @ B^T
    ],
)
@pytest.mark.parametrize(
    "ordering0, ordering1", 
    [
        (0, 0),  
        # (0, 1),  
        # (0, 2),  
        # (0, 3),  
        # (1, 0),  
        # (1, 1),  
        # (1, 2),  
        # (1, 3),  
        # (2, 0),  
        # (2, 1),  
        # (2, 2),  
        # (2, 3),  
        # (3, 0),  
        # (3, 1),  
        # (3, 2),  
        # (3, 3),  
    ],
)
def test_matmul_tessera(m, n, k, in_dtype, out_dtype, transA, transB, ordering0, ordering1):
    
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
    BLK_M, BLK_N, BLK_K, gsize_m = selector.get_config()
    print(f"Block sizes: BLK_M={BLK_M}, BLK_N={BLK_N}, BLK_K={BLK_K}")
    print(f"Grid dimensions: {math.ceil(m/BLK_M)} x {math.ceil(n/BLK_N)}")
    
    # Get valid WGM and WGN combinations that evenly divide the grid
    wgm_wgn_combinations = get_valid_wgm_wgn_combinations(m, n, BLK_M, BLK_N, max_wgm=8, max_wgn=8)
    print(f"Number of shape combinations to test: {len(wgm_wgn_combinations)}")
    for wgm_wgn in wgm_wgn_combinations:
        print(wgm_wgn)
    
    # Test each valid WGM/WGN combination
    for wgm, wgn in wgm_wgn_combinations:
        # Reset output tensor for each test
        C.fill_(0)
        
        tritonblas.matmul_lt_tessera(A, B, C, selector, ordering0, ordering1, wgm, wgn)
        
        # Check correctness for this WGM/WGN combination
        torch_c = torch.matmul(A, B)
        torch.testing.assert_close(C.to(out_dtype), torch_c)
