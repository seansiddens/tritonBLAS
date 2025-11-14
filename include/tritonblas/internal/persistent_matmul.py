import triton
import triton.language as tl
import torch


@triton.jit
def remap_xcd_chunked(
    pid,
    GRID_MN,
    NUM_XCDS: tl.constexpr = 8,
    CHUNK_SIZE: tl.constexpr = 2,
):
    xcd = pid % NUM_XCDS
    if pid > (GRID_MN // (NUM_XCDS * CHUNK_SIZE)) * (NUM_XCDS * CHUNK_SIZE):
        return pid
    local_pid = pid // NUM_XCDS
    chunk_idx = local_pid // CHUNK_SIZE
    pos_in_chunk = local_pid % CHUNK_SIZE
    return chunk_idx * NUM_XCDS * CHUNK_SIZE + xcd * CHUNK_SIZE + pos_in_chunk


@triton.jit
def swizzle_tile_l2(
    idx,
    grid_y,
    grid_x,
    TileDimY: tl.constexpr,
    TileDimX: tl.constexpr,
):
    quantized_x = (grid_x // TileDimX) * TileDimX
    quantized_y = (grid_y // TileDimY) * TileDimY
    total_quantized_size = quantized_x * quantized_y
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    y_region_start = (total_quantized_size - 1) + (non_quantized_x * grid_y)

    if idx > total_quantized_size - 1:
        if idx > y_region_start:
            new_grid_x = (
                (idx - total_quantized_size - (non_quantized_x * grid_y)) // non_quantized_y
            ) % grid_x
            new_grid_y = quantized_y + (idx % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            new_grid_x = quantized_x + (idx % non_quantized_x)
            new_grid_y = ((idx - total_quantized_size) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x

    tile_x_idx = idx % TileDimX
    tile_y_idx = (idx // TileDimX) % TileDimX
    tiles_per_row = grid_x // TileDimX
    tiles_per_col = grid_y // TileDimX
    temporal_x = (idx // (TileDimX * TileDimY)) % tiles_per_row
    temporal_y = (idx // (TileDimX * TileDimY * tiles_per_row)) % tiles_per_col
    new_grid_x = temporal_x * TileDimX + tile_x_idx
    new_grid_y = temporal_y * TileDimY + tile_y_idx
    return new_grid_y * grid_x + new_grid_x


@triton.jit
def swizzle_tile_l2_shuffled(
    idx,
    grid_y,
    grid_x,
    TileDimY: tl.constexpr,
    TileDimX: tl.constexpr,
    LCG_A,
    LCG_C,
):
    quantized_x = (grid_x // TileDimX) * TileDimX
    quantized_y = (grid_y // TileDimY) * TileDimY
    total_quantized_size = quantized_x * quantized_y
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    y_region_start = (total_quantized_size - 1) + (non_quantized_x * grid_y)

    if idx > total_quantized_size - 1:
        if idx > y_region_start:
            new_grid_x = (
                (idx - total_quantized_size - (non_quantized_x * grid_y)) // non_quantized_y
            ) % grid_x
            new_grid_y = quantized_y + (idx % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            new_grid_x = quantized_x + (idx % non_quantized_x)
            new_grid_y = ((idx - total_quantized_size) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x

    tile_x_idx = idx % TileDimX
    tile_y_idx = (idx // TileDimX) % TileDimX
    tiles_per_row = grid_x // TileDimX
    tiles_per_col = grid_y // TileDimX
    num_l2_tiles = tiles_per_row * tiles_per_col
    tile_linear_idx = idx // (TileDimX * TileDimY)
    safe_num_tiles = tl.where(num_l2_tiles > 0, num_l2_tiles, 1)
    shuffled_tile_idx = (LCG_A * tile_linear_idx + LCG_C) % safe_num_tiles
    shuffled_tile_idx = tl.where(num_l2_tiles > 1, shuffled_tile_idx, tile_linear_idx)
    temporal_x = shuffled_tile_idx % tiles_per_row
    temporal_y = shuffled_tile_idx // tiles_per_row
    new_grid_x = temporal_x * TileDimX + tile_x_idx
    new_grid_y = temporal_y * TileDimY + tile_y_idx
    return new_grid_y * grid_x + new_grid_x


@triton.jit()
def persistent_matmul(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(
            pid,
            NUM_SMS,
            NUM_XCDS=NUM_XCDS,
            CHUNK_SIZE=GROUP_SIZE_M * GROUP_SIZE_M,
        )
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        swizzled_pid = swizzle_tile_l2(
            tile_id,
            num_pid_m,
            num_pid_n,
            TileDimY=GROUP_SIZE_M,
            TileDimX=GROUP_SIZE_M,
        )
        pid_m = swizzled_pid // num_pid_n
        pid_n = swizzled_pid % num_pid_n
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
            else:
                A_BASE = tl.multiple_of(A_BASE, (16, 1))

            if stride_bk == 1:
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
            else:
                B_BASE = tl.multiple_of(B_BASE, (1, 16))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        if BIAS:
            c += bias[:, None]

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


@triton.jit()
def persistent_matmul_shuffled(
    A,
    B,
    C,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    LCG_A,
    LCG_C,
    ALLOW_TF32: tl.constexpr = torch.backends.cuda.matmul.allow_tf32,
):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(
            pid,
            NUM_SMS,
            NUM_XCDS=NUM_XCDS,
            CHUNK_SIZE=GROUP_SIZE_M * GROUP_SIZE_M,
        )
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(
            pid,
            NUM_SMS,
            NUM_XCDS=NUM_XCDS,
            CHUNK_SIZE=GROUP_SIZE_M * GROUP_SIZE_M,
        )
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    for tile_id in range(pid, total_tiles, NUM_SMS):
        swizzled_pid = swizzle_tile_l2_shuffled(
            tile_id,
            num_pid_m,
            num_pid_n,
            TileDimY=GROUP_SIZE_M,
            TileDimX=GROUP_SIZE_M,
            LCG_A=LCG_A,
            LCG_C=LCG_C,
        )
        pid_m = swizzled_pid // num_pid_n
        pid_n = swizzled_pid % num_pid_n
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        if BIAS:
            bias_ = bias_ptr + rm * stride_bias
            bias = tl.load(bias_, mask=rm < M, other=0.0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1
        tl.assume(loop_k > 1)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            if stride_ak == 1:
                a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            else:
                a = tl.load(tl.multiple_of(A_BASE, (16, 1)))

            if stride_bk == 1:
                b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            else:
                b = tl.load(tl.multiple_of(B_BASE, (1, 16)))

            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            if stride_ak == 1:
                A_BASE = tl.multiple_of(A_BASE, (1, 16))
            else:
                A_BASE = tl.multiple_of(A_BASE, (16, 1))

            if stride_bk == 1:
                B_BASE = tl.multiple_of(B_BASE, (16, 1))
            else:
                B_BASE = tl.multiple_of(B_BASE, (1, 16))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        if BIAS:
            c += bias[:, None]

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


@triton.jit()
def persistent_matmul_debug_map(
    workgroup_map,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    pid = tl.program_id(0)
    original_pid = pid
    if NUM_XCDS != 1:
        pid = remap_xcd_chunked(
            pid,
            NUM_SMS,
            NUM_XCDS=NUM_XCDS,
            CHUNK_SIZE=GROUP_SIZE_M * GROUP_SIZE_M,
        )

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        swizzled_pid = swizzle_tile_l2(
            tile_id,
            num_pid_m,
            num_pid_n,
            TileDimY=GROUP_SIZE_M,
            TileDimX=GROUP_SIZE_M,
        )
        tl.store(workgroup_map + swizzled_pid, original_pid)


@triton.jit()
def persistent_matmul_debug_map_shuffled(
    workgroup_map,
    M,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    LCG_A,
    LCG_C,
):
    pid = tl.program_id(0)
    original_pid = pid
    pid = remap_xcd_chunked(
        pid,
        NUM_SMS,
        NUM_XCDS=NUM_XCDS,
        CHUNK_SIZE=GROUP_SIZE_M * GROUP_SIZE_M,
    )

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        swizzled_pid = swizzle_tile_l2_shuffled(
            tile_id,
            num_pid_m,
            num_pid_n,
            TileDimY=GROUP_SIZE_M,
            TileDimX=GROUP_SIZE_M,
            LCG_A=LCG_A,
            LCG_C=LCG_C,
        )
        tl.store(workgroup_map + swizzled_pid, original_pid)
