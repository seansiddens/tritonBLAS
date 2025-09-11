import triton
import triton.language as tl
import torch
import enum

@triton.jit()
def chiplet_transform(
    pid,
    NUM_WORKGROUPS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    xcd = pid % NUM_XCDS 
    pos_in_xcd = pid // NUM_XCDS

    # Minimum number of workgroups each XCD gets.
    min_per_xcd = NUM_WORKGROUPS // NUM_XCDS
    extra_wgs = NUM_WORKGROUPS % NUM_XCDS

    offset = xcd * min_per_xcd + min(xcd, extra_wgs)

    return offset + pos_in_xcd

def chiplet_transform_cpu(
    pid,
    NUM_WORKGROUPS,
    NUM_XCDS,
):
    xcd = pid % NUM_XCDS 
    pos_in_xcd = pid // NUM_XCDS

    # Minimum number of workgroups each XCD gets.
    min_per_xcd = NUM_WORKGROUPS // NUM_XCDS
    extra_wgs = NUM_WORKGROUPS % NUM_XCDS

    offset = xcd * min_per_xcd + min(xcd, extra_wgs)

    return offset + pos_in_xcd

@triton.jit()
def row_major_index(x, y, width):
    return y * width + x

# @triton.jit()
# def spiral_hash(index, height, width):
#     """
#     Apply spiral hash transformation to an index.
#     Converts a linear index to spiral pattern coordinates.
#     """
    
#     # Calculate the number of layers
#     # Use conditional expression instead of tl.min for constexpr values
#     max_layers = ((height if height < width else width) + 1) // 2
    
#     result = -1
    
#     for layer in range(max_layers):
#         w = width - 2 * layer
#         h = height - 2 * layer
        
#         # Calculate perimeter of current layer
#         if w == 1:
#             continue
#             perimeter = h
#         elif h == 1:
#             perimeter = w
#         else:
#             perimeter = 2 * (w + h) - 4
        
#         if index < perimeter:
#             x = layer
#             y = layer
            
#             if index < w:
#                 # Right edge
#                 result = row_major_index(x + index, y, width)
#             else:
#                 index = index - w
#                 if index < h - 1:
#                     # Down edge
#                     result = row_major_index(x + w - 1, y + 1 + index, width)
#                 else:
#                     index = index - (h - 1)
#                     if index < w - 1:
#                         # Left edge
#                         result = row_major_index(x + w - 2 - index, y + h - 1, width)
#                     else:
#                         index = index - (w - 1)
#                         # Up edge
#                         result = row_major_index(x, y + h - 2 - index, width)
#         else:
#             index = index - perimeter
    
#     return result

@triton.jit()
def spiral_hash(index, grid_y, grid_x):
    """
    Map a 1D 'index' onto an outer-in spiral over a grid_y x grid_x grid,
    returning the row-major index of the resulting (x,y) location.

    No return/break/continue inside loops.
    """
    total = grid_x * grid_y
    # if index < 0 or index >= total:
    #     raise ValueError("index out of bounds for the given grid")

    # max_layers = (min(grid_y, grid_x) + 1) // 2
    max_layers = ((grid_y if grid_y < grid_x else grid_x) + 1) // 2

    # Phase 1: determine which layer the index falls into and offset within that layer
    selected_layer = -1
    idx_in_layer = -1
    remaining = index

    for layer in range(max_layers):
        w = grid_x - 2 * layer
        h = grid_y - 2 * layer

        # Skip degenerate layers by computing perimeter = 0
        if w <= 0 or h <= 0:
            perimeter = 0
        elif w == 1:
            perimeter = h
        elif h == 1:
            perimeter = w
        else:
            perimeter = 2 * (w + h) - 4

        if selected_layer < 0:
            if remaining < perimeter:
                selected_layer = layer
                idx_in_layer = remaining
            else:
                # Only subtract while we haven't selected a layer
                remaining = remaining - perimeter

    # if selected_layer < 0:
    #     # Should not happen for valid inputs
    #     raise RuntimeError("No spiral layer selected; invalid state.")

    # Phase 2: convert (layer, idx_in_layer) to (x, y)
    layer = selected_layer
    w = grid_x - 2 * layer
    h = grid_y - 2 * layer
    x0 = layer
    y0 = layer

    # Walk along the four sides of the perimeter (right, down, left, up)
    x = 0
    y = 0
    t = idx_in_layer

    if t < w:
        # top edge, moving right
        x = x0 + t
        y = y0
    else:
        t = t - w
        if t < h - 1:
            # right edge, moving down
            x = x0 + w - 1
            y = y0 + 1 + t
        else:
            t = t - (h - 1)
            if t < w - 1:
                # bottom edge, moving left
                x = x0 + w - 2 - t
                y = y0 + h - 1
            else:
                t = t - (w - 1)
                # left edge, moving up
                x = x0
                y = y0 + h - 2 - t

    result = row_major_index(x, y, grid_x)
    return result

@triton.jit()
def diagonal_hash(index, height, width):
    """
    Apply diagonal hash transformation to an index.
    Converts a linear index to diagonal pattern coordinates.
    """
    result = -1
    
    # Iterate through diagonals
    for d in range(width + height - 1):
        y_start = 0 if d - (width - 1) < 0 else d - (width - 1)
        y_end = d if d < height - 1 else height - 1
        count = y_end - y_start + 1
        
        if index < count:
            y = y_start + index
            x = d - y
            result = row_major_index(x, y, width)
        else:
            index = index - count
    
    return result

def row_major_index_cpu(x, y, width):
    return y * width + x

def spiral_hash_cpu(index, height, width):
    """
    Apply spiral hash transformation to an index.
    Converts a linear index to spiral pattern coordinates.
    """
    
    # Calculate the number of layers
    max_layers = (min(height, width) + 1) // 2
    
    for layer in range(max_layers):
        w = width - 2 * layer
        h = height - 2 * layer
        
        # Calculate perimeter of current layer
        if w == 1:
            perimeter = h
        elif h == 1:
            perimeter = w
        else:
            perimeter = 2 * (w + h) - 4
        
        if index < perimeter:
            x = layer
            y = layer
            
            if index < w:
                # Right edge
                return row_major_index_cpu(x + index, y, width)
            
            index = index - w
            if index < h - 1:
                # Down edge
                return row_major_index_cpu(x + w - 1, y + 1 + index, width)
            
            index = index - (h - 1)
            if index < w - 1:
                # Left edge
                return row_major_index_cpu(x + w - 2 - index, y + h - 1, width)
            
            index = index - (w - 1)
            # Up edge
            return row_major_index_cpu(x, y + h - 2 - index, width)
        
        index = index - perimeter
    
    # Shouldn't reach here, but return -1 if we do
    return -1


def diagonal_hash_cpu(index, height, width):
    """
    Apply diagonal hash transformation to an index.
    Converts a linear index to diagonal pattern coordinates.
    """
    # Convert to int32 for Triton compatibility
    
    # Iterate through diagonals
    for d in range(width + height - 1):
        y_start = max(0, d - (width - 1))
        y_end = min(d, height - 1)
        count = y_end - y_start + 1
        
        if index < count:
            y = y_start + index
            x = d - y
            return row_major_index_cpu(x, y, width)
        
        index = index - count
    
    # Shouldn't reach here, but return -1 if we do
    return -1

@triton.jit()
def compute_level_index(
    pid,
    level_x_radix,
    level_y_radix,
    order,
    cumulative_denominator,
):
    """
    Compute level indices based on ordering scheme.
    
    Args:
        workgroup_id: The workgroup index
        level_x_radix: X dimension radix for the level
        level_y_radix: Y dimension radix for the level
        order: Ordering scheme (ROW_MAJOR, COLUMN_MAJOR, SPIRAL, DIAGONAL, SNAKE)
    
    Returns:
        Tuple of (level_x_idx, level_y_idx, cumulative_denominator)
    """
    level_x_idx = 0
    level_y_idx = 0
    
    if order == 0:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
    elif order == 1:
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        
    elif order == 2:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
        # Apply spiral hash transformation
        level_idx = level_y_idx * level_x_radix + level_x_idx
        level_idx = spiral_hash(level_idx, level_y_radix, level_x_radix)
        level_x_idx = level_idx % level_x_radix
        level_y_idx = level_idx // level_x_radix
        
    elif order == 3:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
        # Apply diagonal hash transformation
        level_idx = level_y_idx * level_x_radix + level_x_idx
        level_idx = diagonal_hash(level_idx, level_y_radix, level_x_radix)
        level_x_idx = level_idx % level_x_radix
        level_y_idx = level_idx // level_x_radix
        
    elif order == 4:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
        # Apply snake transformation (reverse X direction for odd Y rows)
        if level_y_idx % 2 == 1:
            level_x_idx = level_x_radix - 1 - level_x_idx
    
    return level_x_idx, level_y_idx, cumulative_denominator

def compute_level_index_cpu(
    pid,
    level_x_radix,
    level_y_radix,
    order,
    cumulative_denominator,
):
    """
    Compute level indices based on ordering scheme.
    
    Args:
        workgroup_id: The workgroup index
        level_x_radix: X dimension radix for the level
        level_y_radix: Y dimension radix for the level
        order: Ordering scheme (ROW_MAJOR, COLUMN_MAJOR, SPIRAL, DIAGONAL, SNAKE)
    
    Returns:
        Tuple of (level_x_idx, level_y_idx, cumulative_denominator)
    """
    level_x_idx = 0
    level_y_idx = 0
    
    if order == 0:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
    elif order == 1:
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        
    elif order == 2:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
        # Apply spiral hash transformation
        level_idx = level_y_idx * level_x_radix + level_x_idx
        level_idx = spiral_hash_cpu(level_idx, level_y_radix, level_x_radix)
        level_x_idx = level_idx % level_x_radix
        level_y_idx = level_idx // level_x_radix
        
    elif order == 3:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
        # Apply diagonal hash transformation
        level_idx = level_y_idx * level_x_radix + level_x_idx
        level_idx = diagonal_hash_cpu(level_idx, level_y_radix, level_x_radix)
        level_x_idx = level_idx % level_x_radix
        level_y_idx = level_idx // level_x_radix
        
    elif order == 4:
        level_x_idx = (pid // cumulative_denominator) % level_x_radix
        cumulative_denominator *= level_x_radix
        level_y_idx = (pid // cumulative_denominator) % level_y_radix
        cumulative_denominator *= level_y_radix
        
        # Apply snake transformation (reverse X direction for odd Y rows)
        if level_y_idx % 2 == 1:
            level_x_idx = level_x_radix - 1 - level_x_idx
    
    return level_x_idx, level_y_idx, cumulative_denominator


@triton.jit()
def transform_quantized(
    pid,
    num_pid_m,
    num_pid_n,
    radix_level_0_order,
    radix_level_0_y,
    radix_level_0_x,
    radix_level_1_order,
    radix_level_1_y,
    radix_level_1_x,
):
    new_grid_x = 0;
    new_grid_y = 0;
    cumulative_denominator = 1;
    cumulative_x = 1;
    cumulative_y = 1;

    # Level 1
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index(
        pid,
        radix_level_1_x,
        radix_level_1_y,
        radix_level_1_order,
        cumulative_denominator,
    )

    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    cumulative_x *= radix_level_1_x
    cumulative_y *= radix_level_1_y

    # Level 0
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index(
        pid,
        radix_level_0_x,
        radix_level_0_y,
        radix_level_0_order,
        cumulative_denominator,
    )
    
    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y

    return new_grid_y * num_pid_n + new_grid_x

def transform_quantized_cpu(
    pid,
    num_pid_m,
    num_pid_n,
    radix_level_0_order,
    radix_level_0_y,
    radix_level_0_x,
    radix_level_1_order,
    radix_level_1_y,
    radix_level_1_x,
):
    new_grid_x = 0;
    new_grid_y = 0;
    cumulative_denominator = 1;
    cumulative_x = 1;
    cumulative_y = 1;

    # Level 1
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index_cpu(
        pid,
        radix_level_1_x,
        radix_level_1_y,
        radix_level_1_order,
        cumulative_denominator,
    )

    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    cumulative_x *= radix_level_1_x
    cumulative_y *= radix_level_1_y

    # Level 0
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index_cpu(
        pid,
        radix_level_0_x,
        radix_level_0_y,
        radix_level_0_order,
        cumulative_denominator,
    )
    
    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y

    return new_grid_y * num_pid_n + new_grid_x

@triton.jit()
def transform(
    pid,
    num_pid_m,
    num_pid_n,
    radix_level_0_order,
    radix_level_0_y,
    radix_level_0_x,
    radix_level_1_order,
    radix_level_1_y,
    radix_level_1_x,
):
    # Grid dimensions (num_pid_m = grid_y, num_pid_n = grid_x)
    grid_y = num_pid_m
    grid_x = num_pid_n
    
    # Timestep dimensions based on radix levels
    timestep_x_dim = radix_level_0_x * radix_level_1_x
    timestep_y_dim = radix_level_0_y * radix_level_1_y
    
    # Calculate temporal counts and quantized dimensions
    temporal_x_count = grid_x // timestep_x_dim
    temporal_y_count = grid_y // timestep_y_dim
    quantized_x = temporal_x_count * timestep_x_dim
    quantized_y = temporal_y_count * timestep_y_dim
    
    # Calculate non-quantized dimensions
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    
    # Total quantized size and region start
    total_quantized_size = quantized_x * quantized_y
    y_region_start = (total_quantized_size - 1) + (non_quantized_x * grid_y)
    
    # Compute non-quantized regions
    if pid > total_quantized_size - 1:
        if pid > y_region_start:
            # Un-quantized region Y
            new_grid_x = ((pid - total_quantized_size - (non_quantized_x * grid_y)) // non_quantized_y) % grid_x
            new_grid_y = quantized_y + (pid % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            # print("Unquantized region X: ", pid)
            # Un-quantized region X
            new_grid_x = quantized_x + (pid % non_quantized_x)
            new_grid_y = ((pid - total_quantized_size) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x
    
    return transform_quantized(
        pid,
        num_pid_m,
        num_pid_n,
        radix_level_0_order,
        radix_level_0_y,
        radix_level_0_x,
        radix_level_1_order,
        radix_level_1_y,
        radix_level_1_x,
    )


def transform_cpu(
    pid,
    num_pid_m,
    num_pid_n,
    radix_level_0_order,
    radix_level_0_y,
    radix_level_0_x,
    radix_level_1_order,
    radix_level_1_y,
    radix_level_1_x,
):
    # Grid dimensions (num_pid_m = grid_y, num_pid_n = grid_x)
    grid_y = num_pid_m
    grid_x = num_pid_n
    
    # Timestep dimensions based on radix levels
    timestep_x_dim = radix_level_0_x * radix_level_1_x
    timestep_y_dim = radix_level_0_y * radix_level_1_y
    
    # Calculate temporal counts and quantized dimensions
    temporal_x_count = grid_x // timestep_x_dim
    temporal_y_count = grid_y // timestep_y_dim
    quantized_x = temporal_x_count * timestep_x_dim
    quantized_y = temporal_y_count * timestep_y_dim
    
    # Calculate non-quantized dimensions
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    
    # Total quantized size and region start
    total_quantized_size = quantized_x * quantized_y
    y_region_start = (total_quantized_size - 1) + (non_quantized_x * grid_y)
    
    # Compute non-quantized regions
    if pid > total_quantized_size - 1:
        if pid > y_region_start:
            # Un-quantized region Y
            new_grid_x = ((pid - total_quantized_size - (non_quantized_x * grid_y)) // non_quantized_y) % grid_x
            new_grid_y = quantized_y + (pid % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            # print("Unquantized region X: ", pid)
            # Un-quantized region X
            new_grid_x = quantized_x + (pid % non_quantized_x)
            new_grid_y = ((pid - total_quantized_size) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x
    
    return transform_quantized_cpu(
        pid,
        num_pid_m,
        num_pid_n,
        radix_level_0_order,
        radix_level_0_y,
        radix_level_0_x,
        radix_level_1_order,
        radix_level_1_y,
        radix_level_1_x,
    )
