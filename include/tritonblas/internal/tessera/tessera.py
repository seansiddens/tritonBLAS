import triton
import triton.language as tl

@triton.jit()
def sgn(x):
    """Sign function: returns 1 if x > 0, -1 if x < 0, 0 if x == 0"""
    return tl.where(x > 0, 1, tl.where(x < 0, -1, 0))

def sgn_cpu(x):
    """Sign function: returns 1 if x > 0, -1 if x < 0, 0 if x == 0"""
    return tl.where(x > 0, 1, tl.where(x < 0, -1, 0))

@triton.jit()
def spiral_hash(index, grid_y, grid_x):
    """
    Spiral curve mapping from 1D index to 2D coordinates.
    Creates a spiral pattern starting from top-left corner.
    
    Args:
        index: 1D index to convert
        grid_y: Height of the grid
        grid_x: Width of the grid
    
    Returns:
        1D index in row-major 2D grid
    """
    # Calculate maximum number of layers
    max_layers = (tl.minimum(grid_y, grid_x) + 1) // 2
    
    # Initialize result variables
    result_x = 0
    result_y = 0
    found = False
    current_index = index
    
    # Iterate through layers
    for layer in range(max_layers):
        if not found:
            w = grid_x - 2 * layer
            h = grid_y - 2 * layer
            
            # Calculate perimeter of current layer
            if w == 1:
                perimeter = h
            elif h == 1:
                perimeter = w
            else:
                perimeter = 2 * (w + h) - 4
            
            # Check if index is in this layer
            if current_index < perimeter:
                x = layer
                y = layer
                temp_index = current_index
                
                if temp_index < w:
                    # Right edge
                    result_x = x + temp_index
                    result_y = y
                    found = True
                else:
                    temp_index = temp_index - w
                    if temp_index < h - 1:
                        # Down edge
                        result_x = x + w - 1
                        result_y = y + 1 + temp_index
                        found = True
                    else:
                        temp_index = temp_index - (h - 1)
                        if temp_index < w - 1:
                            # Left edge
                            result_x = x + w - 2 - temp_index
                            result_y = y + h - 1
                            found = True
                        else:
                            temp_index = temp_index - (w - 1)
                            # Up edge
                            result_x = x
                            result_y = y + h - 2 - temp_index
                            found = True
            else:
                # Move to next layer
                current_index = current_index - perimeter
    
    # Return the computed coordinates
    return row_major_index(result_x, result_y, grid_x)

def spiral_hash_cpu(index, grid_y, grid_x):
    """
    Spiral curve mapping from 1D index to 2D coordinates.
    Creates a spiral pattern starting from top-left corner.
    
    Args:
        index: 1D index to convert
        grid_y: Height of the grid
        grid_x: Width of the grid
    
    Returns:
        1D index in row-major 2D grid
    """
    # Calculate maximum number of layers
    max_layers = (min(grid_y, grid_x) + 1) // 2
    
    # Initialize result variables
    result_x = 0
    result_y = 0
    found = False
    current_index = index
    
    # Iterate through layers
    for layer in range(max_layers):
        if not found:
            w = grid_x - 2 * layer
            h = grid_y - 2 * layer
            
            # Calculate perimeter of current layer
            if w == 1:
                perimeter = h
            elif h == 1:
                perimeter = w
            else:
                perimeter = 2 * (w + h) - 4
            
            # Check if index is in this layer
            if current_index < perimeter:
                x = layer
                y = layer
                temp_index = current_index
                
                if temp_index < w:
                    # Right edge
                    result_x = x + temp_index
                    result_y = y
                    found = True
                else:
                    temp_index = temp_index - w
                    if temp_index < h - 1:
                        # Down edge
                        result_x = x + w - 1
                        result_y = y + 1 + temp_index
                        found = True
                    else:
                        temp_index = temp_index - (h - 1)
                        if temp_index < w - 1:
                            # Left edge
                            result_x = x + w - 2 - temp_index
                            result_y = y + h - 1
                            found = True
                        else:
                            temp_index = temp_index - (w - 1)
                            # Up edge
                            result_x = x
                            result_y = y + h - 2 - temp_index
                            found = True
            else:
                # Move to next layer
                current_index = current_index - perimeter
    
    # Return the computed coordinates
    return row_major_index_cpu(result_x, result_y, grid_x)

@triton.jit()
def gilbert(index, grid_y, grid_x, max_iterations=32):
    """
    Gilbert curve mapping from 1D index to 2D coordinates.
    Bounded version for Triton compatibility.
    
    Args:
        index: 1D index to convert
        grid_y: Height of the grid
        grid_x: Width of the grid
        max_iterations: Maximum number of iterations (default 32)
    
    Returns:
        1D index in row-major 2D grid
    """
    x = 0
    y = 0
    
    # Initialize ax, ay, bx, by based on grid dimensions
    if grid_x >= grid_y:
        ax = grid_x
        ay = 0
        bx = 0
        by = grid_y
    else:
        ax = 0
        ay = grid_y
        bx = grid_x
        by = 0
    
    cur_idx = 0
    
    # Bounded loop instead of infinite while
    for i in range(max_iterations):
        dax = sgn(ax)
        day = sgn(ay)
        dbx = sgn(bx)
        dby = sgn(by)
        
        w = tl.abs(ax + ay)
        h = tl.abs(bx + by)
        di = index - cur_idx
        
        # Base cases - check if we can return
        if h == 1:
            x = x + dax * di
            y = y + day * di
        elif w == 1:
            x = x + dbx * di
            y = y + dby * di
        else:
            # Calculate half dimensions
            ax2 = ax // 2
            ay2 = ay // 2
            bx2 = bx // 2
            by2 = by // 2
            w2 = tl.abs(ax2 + ay2)
            h2 = tl.abs(bx2 + by2)
            
            # Adjust ax2, ay2 if needed
            if (w2 % 2 == 1) and (w > 2):
                ax2 = ax2 + dax
                ay2 = ay2 + day
            
            # First recursive case
            if 2 * w > 3 * h:
                nxt_idx = cur_idx + tl.abs((ax2 + ay2) * (bx + by))
                if index < nxt_idx:
                    ax = ax2
                    ay = ay2
                else:
                    cur_idx = nxt_idx
                    x = x + ax2
                    y = y + ay2
                    ax = ax - ax2
                    ay = ay - ay2
            else:
                # Adjust bx2, by2 if needed
                if (h2 % 2 == 1) and (h > 2):
                    bx2 = bx2 + dbx
                    by2 = by2 + dby
                
                # Second recursive case
                nxt_idx = cur_idx + tl.abs((bx2 + by2) * (ax2 + ay2))
                if index < nxt_idx:
                    ax = bx2
                    ay = by2
                    bx = ax2
                    by = ay2
                else:
                    cur_idx = nxt_idx
                    nxt_idx = cur_idx + tl.abs((ax + ay) * ((bx - bx2) + (by - by2)))
                    if index < nxt_idx:
                        x = x + bx2
                        y = y + by2
                        bx = bx - bx2
                        by = by - by2
                    else:
                        # Third recursive case
                        cur_idx = nxt_idx
                        x = x + (ax - dax) + (bx2 - dbx)
                        y = y + (ay - day) + (by2 - dby)
                        tmp_ax = ax
                        tmp_ay = ay
                        ax = -bx2
                        ay = -by2
                        bx = -(tmp_ax - ax2)
                        by = -(tmp_ay - ay2)
    
    # Return the computed coordinates
    return y * grid_x + x

@triton.jit()
def row_major_index(x_idx, y_idx, width):
    """Convert 2D coordinates to 1D row-major index."""
    return y_idx * width + x_idx

def row_major_index_cpu(x_idx, y_idx, width):
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
        order: Ordering type (ROW_MAJOR=0, COL_MAJOR=1, SNAKE=2, SPIRAL=3, GILBERT=4)
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
    elif order == 3:  # SPIRAL
        # For spiral curve, we need to compute the 2D coordinates directly
        # and then convert back to our level indices
        level_idx = (index // cumulative_denominator) % (level_x_radix * level_y_radix)
        cumulative_denominator = cumulative_denominator * (level_x_radix * level_y_radix)
        
        # Convert 1D index to 2D coordinates using spiral curve
        spiral_idx = spiral_hash(level_idx, level_y_radix, level_x_radix)
        level_x_idx = spiral_idx % level_x_radix
        level_y_idx = spiral_idx // level_x_radix
    elif order == 4:  # GILBERT
        # For Gilbert curve, we need to compute the 2D coordinates directly
        # and then convert back to our level indices
        level_idx = (index // cumulative_denominator) % (level_x_radix * level_y_radix)
        cumulative_denominator = cumulative_denominator * (level_x_radix * level_y_radix)
        
        # Convert 1D index to 2D coordinates using Gilbert curve
        gilbert_idx = gilbert(level_idx, level_y_radix, level_x_radix)
        level_x_idx = gilbert_idx % level_x_radix
        level_y_idx = gilbert_idx // level_x_radix

    return level_x_idx, level_y_idx, cumulative_denominator

def compute_level_index_cpu(
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
        order: Ordering type (ROW_MAJOR=0, COL_MAJOR=1, SNAKE=2, SPIRAL=3, GILBERT=4)
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
    elif order == 3:  # SPIRAL
        # For spiral curve, we need to compute the 2D coordinates directly
        # and then convert back to our level indices
        level_idx = (index // cumulative_denominator) % (level_x_radix * level_y_radix)
        cumulative_denominator = cumulative_denominator * (level_x_radix * level_y_radix)
        
        # Convert 1D index to 2D coordinates using spiral curve
        spiral_idx = spiral_hash_cpu(level_idx, level_y_radix, level_x_radix)
        level_x_idx = spiral_idx % level_x_radix
        level_y_idx = spiral_idx // level_x_radix
    # elif order == 4:  # GILBERT
        # # For Gilbert curve, we need to compute the 2D coordinates directly
        # # and then convert back to our level indices
        # level_idx = (index // cumulative_denominator) % (level_x_radix * level_y_radix)
        # cumulative_denominator = cumulative_denominator * (level_x_radix * level_y_radix)
        
        # # Convert 1D index to 2D coordinates using Gilbert curve
        # gilbert_idx = gilbert(level_idx, level_y_radix, level_x_radix)
        # level_x_idx = gilbert_idx % level_x_radix
        # level_y_idx = gilbert_idx // level_x_radix

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

def transform_quantized_cpu(
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
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index_cpu(
        index, wgn, wgm, ordering1, cumulative_denominator
    )

    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    cumulative_x *= wgn
    cumulative_y *= wgm

    # Compute timestep index (which L2 tile in the grid)
    level_x_idx, level_y_idx, cumulative_denominator = compute_level_index_cpu(
        index, grid_x // wgn, grid_y // wgm, ordering0, cumulative_denominator
    )

    new_grid_x += level_x_idx * cumulative_x
    new_grid_y += level_y_idx * cumulative_y
    
    return new_grid_y * grid_x + new_grid_x

def transform(
    index,
    grid_y,
    grid_x,
    ordering0,
    ordering1,
    wgm,
    wgn
):
    """
    Transform index with quantized and non-quantized regions.
    Handles cases where WGM and WGN don't evenly divide the grid.
    
    Args:
        index: Input index
        grid_y: Grid height
        grid_x: Grid width
        ordering0: Ordering for timestep level
        ordering1: Ordering for L2 tile level
        wgm: Workgroup M dimension
        wgn: Workgroup N dimension
    
    Returns:
        Transformed index in row-major 2D grid
    """
    # Calculate quantized dimensions
    timestep_x_dim = wgn  # L2 tile width
    timestep_y_dim = wgm  # L2 tile height
    
    temporal_x_count = grid_x // timestep_x_dim
    temporal_y_count = grid_y // timestep_y_dim
    
    quantized_x = temporal_x_count * timestep_x_dim
    quantized_y = temporal_y_count * timestep_y_dim
    
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    
    total_quantized_size = quantized_x * quantized_y
    y_region_start = (total_quantized_size - 1) + (non_quantized_x * grid_y)
    
    # Check if index is in quantized region
    if index <= total_quantized_size - 1:
        # Quantized logic - use the existing transform_quantized function
        return transform_quantized(index, grid_y, grid_x, ordering0, ordering1, wgm, wgn)
    else:
        # Non-quantized regions
        if index > y_region_start:
            # Un-quantized region Y
            new_grid_x = ((index - total_quantized_size - (non_quantized_x * grid_y)) // non_quantized_y) % grid_x
            new_grid_y = quantized_y + (index % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            # Un-quantized region X
            new_grid_x = quantized_x + (index % non_quantized_x)
            new_grid_y = ((index - total_quantized_size) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x

@triton.jit()
def transform(
    index,
    grid_y,
    grid_x,
    ordering0,
    ordering1,
    wgm,
    wgn
):
    """
    Transform index with quantized and non-quantized regions.
    Handles cases where WGM and WGN don't evenly divide the grid.
    
    Args:
        index: Input index
        grid_y: Grid height
        grid_x: Grid width
        ordering0: Ordering for timestep level
        ordering1: Ordering for L2 tile level
        wgm: Workgroup M dimension
        wgn: Workgroup N dimension
    
    Returns:
        Transformed index in row-major 2D grid
    """
    # Calculate quantized dimensions
    timestep_x_dim = wgn  # L2 tile width
    timestep_y_dim = wgm  # L2 tile height
    
    temporal_x_count = grid_x // timestep_x_dim
    temporal_y_count = grid_y // timestep_y_dim
    
    quantized_x = temporal_x_count * timestep_x_dim
    quantized_y = temporal_y_count * timestep_y_dim
    
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    
    total_quantized_size = quantized_x * quantized_y
    y_region_start = (total_quantized_size - 1) + (non_quantized_x * grid_y)
    
    # Check if index is in quantized region
    if index <= total_quantized_size - 1:
        # Quantized logic - use the existing transform_quantized function
        return transform_quantized(index, grid_y, grid_x, ordering0, ordering1, wgm, wgn)
    else:
        # Non-quantized regions
        if index > y_region_start:
            # Un-quantized region Y
            new_grid_x = ((index - total_quantized_size - (non_quantized_x * grid_y)) // non_quantized_y) % grid_x
            new_grid_y = quantized_y + (index % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            # Un-quantized region X
            new_grid_x = quantized_x + (index % non_quantized_x)
            new_grid_y = ((index - total_quantized_size) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x

def transform_cpu(
    index,
    grid_y,
    grid_x,
    ordering0,
    ordering1,
    wgm,
    wgn
):
    """
    Transform index with quantized and non-quantized regions.
    Handles cases where WGM and WGN don't evenly divide the grid.
    
    Args:
        index: Input index
        grid_y: Grid height
        grid_x: Grid width
        ordering0: Ordering for timestep level
        ordering1: Ordering for L2 tile level
        wgm: Workgroup M dimension
        wgn: Workgroup N dimension
    
    Returns:
        Transformed index in row-major 2D grid
    """
    # Calculate quantized dimensions
    timestep_x_dim = wgn  # L2 tile width
    timestep_y_dim = wgm  # L2 tile height
    
    temporal_x_count = grid_x // timestep_x_dim
    temporal_y_count = grid_y // timestep_y_dim
    
    quantized_x = temporal_x_count * timestep_x_dim
    quantized_y = temporal_y_count * timestep_y_dim
    
    non_quantized_x = grid_x - quantized_x
    non_quantized_y = grid_y - quantized_y
    
    total_quantized_size = quantized_x * quantized_y
    y_region_start = (total_quantized_size - 1) + (non_quantized_x * grid_y)
    
    # Check if index is in quantized region
    if index <= total_quantized_size - 1:
        # Quantized logic - use the existing transform_quantized function
        return transform_quantized_cpu(index, grid_y, grid_x, ordering0, ordering1, wgm, wgn)
    else:
        # Non-quantized regions
        if index > y_region_start:
            # Un-quantized region Y
            new_grid_x = ((index - total_quantized_size - (non_quantized_x * grid_y)) // non_quantized_y) % grid_x
            new_grid_y = quantized_y + (index % non_quantized_y)
            return new_grid_y * grid_x + new_grid_x
        else:
            # Un-quantized region X
            new_grid_x = quantized_x + (index % non_quantized_x)
            new_grid_y = ((index - total_quantized_size) // non_quantized_x) % grid_y
            return (new_grid_y * grid_x) + new_grid_x

@triton.jit()
def chiplet_transform(index, num_workgroups, num_xcds):
    """
    Swizzle workgroup assignment across XCDs (eXecute Compute Dies).
    This is the Triton equivalent of the C++ swizzle_chiplet function.
    
    Args:
        index: Workgroup index to swizzle
        num_workgroups: Total number of workgroups
        num_xcds: Number of XCDs (default 8)
    
    Returns:
        Swizzled workgroup index
    """
    # Original round-robin assignment
    xcd = index % num_xcds
    pos_in_xcd = index // num_xcds
    
    # Minimum # of workgroups each XCD gets
    min_per_xcd = num_workgroups // num_xcds
    extra_wgs = num_workgroups % num_xcds  # Number of XCDs that get an extra WG
    
    # This is the total # of WGs assigned to all XCDs before this XCD.
    # Every XCD gets at least `min_per_xcd` WGs.
    if xcd < extra_wgs:
        offset = xcd * min_per_xcd + xcd
    else:
        offset = xcd * min_per_xcd + extra_wgs
    
    return offset + pos_in_xcd

def chiplet_transform_cpu(index, num_workgroups, num_xcds):
    """
    Swizzle workgroup assignment across XCDs (eXecute Compute Dies).
    This is the Triton equivalent of the C++ swizzle_chiplet function.
    
    Args:
        index: Workgroup index to swizzle
        num_workgroups: Total number of workgroups
        num_xcds: Number of XCDs (default 8)
    
    Returns:
        Swizzled workgroup index
    """
    # Original round-robin assignment
    xcd = index % num_xcds
    pos_in_xcd = index // num_xcds
    
    # Minimum # of workgroups each XCD gets
    min_per_xcd = num_workgroups // num_xcds
    extra_wgs = num_workgroups % num_xcds  # Number of XCDs that get an extra WG
    
    # This is the total # of WGs assigned to all XCDs before this XCD.
    # Every XCD gets at least `min_per_xcd` WGs.
    if xcd < extra_wgs:
        offset = xcd * min_per_xcd + xcd
    else:
        offset = xcd * min_per_xcd + extra_wgs
    
    return offset + pos_in_xcd
