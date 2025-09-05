import triton
import triton.language as tl

@triton.jit()
def row_major_index(x_idx, y_idx, width):
    """Convert 2D coordinates to 1D row-major index."""
    return y_idx * width + x_idx

@triton.jit()
def compute_level_index(
    index,
    level_x_radix,
    level_y_radix,
    order,
    cumulative_denominator
):
    """
    Compute level indices based on ordering type.
    
    Args:
        index: Input index
        level_x_radix: X dimension radix
        level_y_radix: Y dimension radix  
        order: Ordering type (ROW_MAJOR=0, COL_MAJOR=1, SNAKE=2)
        cumulative_denominator: Input/output cumulative denominator
    """
    if order == 0:  # ROW_MAJOR
        level_x_idx = (index // cumulative_denominator) % level_x_radix
        cumulative_denominator = cumulative_denominator * level_x_radix
        level_y_idx = (index // cumulative_denominator) % level_y_radix
        cumulative_denominator = cumulative_denominator * level_y_radix
    elif order == 1:  # COL_MAJOR
        level_y_idx = (index // cumulative_denominator) % level_y_radix
        cumulative_denominator = cumulative_denominator * level_y_radix
        level_x_idx = (index // cumulative_denominator) % level_x_radix
        cumulative_denominator = cumulative_denominator * level_x_radix
    elif order == 2:  # SNAKE
        level_x_idx = (index // cumulative_denominator) % level_x_radix
        cumulative_denominator = cumulative_denominator * level_x_radix
        level_y_idx = (index // cumulative_denominator) % level_y_radix
        cumulative_denominator = cumulative_denominator * level_y_radix
        # Apply snake transformation: reverse x for odd y rows
        if level_y_idx % 2 == 1:
            level_x_idx = level_x_radix - 1 - level_x_idx

    return level_x_idx, level_y_idx, cumulative_denominator

@triton.jit()
def transform_quantized(
    index,
    grid_y,
    grid_x,
    ordering0, 
    ordering1,
    wgm,
    wgn
):
    new_grid_x = 0
    new_grid_y = 0
    cumulative_denominator = 1
    cumulative_x = 1
    cumulative_y = 1

    # Compute L2 tile index (wgn x wgm tile within the grid)
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index(
        index, wgn, wgm, ordering1, cumulative_denominator
    )

    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    cumulative_x *= wgn
    cumulative_y *= wgm

    # Compute timestep index (which L2 tile in the grid)
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index(
        index, grid_x // wgn, grid_y // wgm, ordering0, cumulative_denominator
    )

    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    
    return new_grid_y * grid_x + new_grid_x
