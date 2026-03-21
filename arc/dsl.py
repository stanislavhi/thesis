"""
Grid DSL — Domain-Specific Language for ARC-AGI grid transformations.

Primitives operate on 2D numpy arrays of integers (0-9 = colors).
Each function takes a grid and returns a new grid (pure, no mutation).
"""
import numpy as np
from typing import Tuple, Optional


# ============================================================
# Geometric Transforms
# ============================================================

def rotate(grid: np.ndarray, n: int = 1) -> np.ndarray:
    """Rotate grid 90° clockwise, n times."""
    return np.rot90(grid, k=-n)


def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    """Flip grid left-right."""
    return np.fliplr(grid)


def flip_vertical(grid: np.ndarray) -> np.ndarray:
    """Flip grid top-bottom."""
    return np.flipud(grid)


def transpose(grid: np.ndarray) -> np.ndarray:
    """Transpose rows and columns."""
    return grid.T


# ============================================================
# Color Operations
# ============================================================

def replace_color(grid: np.ndarray, from_color: int, to_color: int) -> np.ndarray:
    """Replace all pixels of from_color with to_color."""
    result = grid.copy()
    result[result == from_color] = to_color
    return result


def swap_colors(grid: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    """Swap two colors throughout the grid."""
    result = grid.copy()
    mask_a = grid == color_a
    mask_b = grid == color_b
    result[mask_a] = color_b
    result[mask_b] = color_a
    return result


def fill_background(grid: np.ndarray, color: int) -> np.ndarray:
    """Fill all background (0) pixels with the given color."""
    result = grid.copy()
    result[result == 0] = color
    return result


def keep_only_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Keep only pixels of the given color, zero everything else."""
    result = np.zeros_like(grid)
    result[grid == color] = color
    return result


def most_common_color(grid: np.ndarray, exclude_zero: bool = True) -> int:
    """Return the most common color in the grid."""
    flat = grid.flatten()
    if exclude_zero:
        flat = flat[flat != 0]
    if len(flat) == 0:
        return 0
    values, counts = np.unique(flat, return_counts=True)
    return int(values[np.argmax(counts)])


# ============================================================
# Structural Operations
# ============================================================

def crop_to_content(grid: np.ndarray) -> np.ndarray:
    """Crop grid to the smallest bounding box containing non-zero pixels."""
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return grid
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return grid[rmin:rmax+1, cmin:cmax+1]


def tile(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Tile the grid n×m times."""
    return np.tile(grid, (rows, cols))


def scale_up(grid: np.ndarray, factor: int) -> np.ndarray:
    """Scale grid by repeating each pixel factor×factor times."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def pad(grid: np.ndarray, size: int, color: int = 0) -> np.ndarray:
    """Add a border of the given color and size around the grid."""
    return np.pad(grid, size, constant_values=color)


def overlay(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Overlay top grid on base — non-zero pixels from top replace base."""
    result = base.copy()
    # Crop to fit if sizes differ
    h = min(base.shape[0], top.shape[0])
    w = min(base.shape[1], top.shape[1])
    mask = top[:h, :w] != 0
    result[:h, :w][mask] = top[:h, :w][mask]
    return result


def mirror_horizontal(grid: np.ndarray) -> np.ndarray:
    """Double the grid by appending its horizontal mirror."""
    return np.hstack([grid, np.fliplr(grid)])


def mirror_vertical(grid: np.ndarray) -> np.ndarray:
    """Double the grid by appending its vertical mirror."""
    return np.vstack([grid, np.flipud(grid)])


# ============================================================
# Pattern Detection
# ============================================================

def extract_objects(grid: np.ndarray) -> list:
    """Extract connected non-zero regions as separate grids + positions."""
    from scipy import ndimage
    labeled, n_objects = ndimage.label(grid != 0)
    objects = []
    for i in range(1, n_objects + 1):
        mask = labeled == i
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        obj_grid = np.zeros_like(grid[rmin:rmax+1, cmin:cmax+1])
        obj_grid[mask[rmin:rmax+1, cmin:cmax+1]] = grid[rmin:rmax+1, cmin:cmax+1][mask[rmin:rmax+1, cmin:cmax+1]]
        objects.append({
            "grid": obj_grid,
            "position": (rmin, cmin),
            "size": (rmax - rmin + 1, cmax - cmin + 1),
        })
    return objects


def count_colors(grid: np.ndarray) -> dict:
    """Count occurrences of each color."""
    values, counts = np.unique(grid, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def get_grid_shape(grid: np.ndarray) -> Tuple[int, int]:
    """Return (rows, cols) of the grid."""
    return grid.shape


# ============================================================
# Spatial / Advanced Operations
# ============================================================

def flood_fill(grid: np.ndarray, row: int, col: int, color: int) -> np.ndarray:
    """Flood fill from (row, col) with the given color."""
    result = grid.copy()
    h, w = result.shape
    if row >= h or col >= w:
        return result
    target = result[row, col]
    if target == color:
        return result
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= h or c < 0 or c >= w:
            continue
        if result[r, c] != target:
            continue
        result[r, c] = color
        stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
    return result


def gravity(grid: np.ndarray, direction: int = 0) -> np.ndarray:
    """
    Apply gravity — non-zero pixels fall in the given direction.
    0=down, 1=up, 2=left, 3=right
    """
    result = grid.copy()
    h, w = result.shape
    if direction == 0:  # down
        for c in range(w):
            col = result[:, c]
            nonzero = col[col != 0]
            result[:, c] = 0
            result[h - len(nonzero):, c] = nonzero
    elif direction == 1:  # up
        for c in range(w):
            col = result[:, c]
            nonzero = col[col != 0]
            result[:, c] = 0
            result[:len(nonzero), c] = nonzero
    elif direction == 2:  # left
        for r in range(h):
            row = result[r, :]
            nonzero = row[row != 0]
            result[r, :] = 0
            result[r, :len(nonzero)] = nonzero
    elif direction == 3:  # right
        for r in range(h):
            row = result[r, :]
            nonzero = row[row != 0]
            result[r, :] = 0
            result[r, w - len(nonzero):] = nonzero
    return result


def draw_border(grid: np.ndarray, color: int) -> np.ndarray:
    """Draw a 1-pixel border around the grid."""
    result = grid.copy()
    result[0, :] = color
    result[-1, :] = color
    result[:, 0] = color
    result[:, -1] = color
    return result


def fill_rect(grid: np.ndarray, color: int) -> np.ndarray:
    """Fill the bounding box of all non-zero pixels with the given color."""
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        return grid
    result = grid.copy()
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    result[rmin:rmax+1, cmin:cmax+1] = color
    return result


def repeat_pattern(grid: np.ndarray, axis: int = 0) -> np.ndarray:
    """Repeat the grid along axis (0=vertical, 1=horizontal)."""
    if axis == 0:
        return np.vstack([grid, grid])
    else:
        return np.hstack([grid, grid])


def largest_object(grid: np.ndarray) -> np.ndarray:
    """Extract only the largest connected non-zero region."""
    from scipy import ndimage
    labeled, n = ndimage.label(grid != 0)
    if n == 0:
        return grid
    sizes = ndimage.sum(grid != 0, labeled, range(1, n + 1))
    largest_label = np.argmax(sizes) + 1
    result = np.zeros_like(grid)
    result[labeled == largest_label] = grid[labeled == largest_label]
    return result


def hollow(grid: np.ndarray) -> np.ndarray:
    """Remove interior pixels — keep only the border of each object."""
    from scipy import ndimage
    result = grid.copy()
    h, w = result.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if result[r, c] != 0:
                neighbors = [grid[r-1, c], grid[r+1, c], grid[r, c-1], grid[r, c+1]]
                if all(n != 0 for n in neighbors):
                    result[r, c] = 0
    return result


def invert_colors(grid: np.ndarray, max_color: int = 9) -> np.ndarray:
    """Invert all colors: c -> max_color - c (keeping 0 as 0)."""
    result = grid.copy()
    mask = result != 0
    result[mask] = max_color - result[mask]
    return result


# ============================================================
# DSL Registry — all operations with their signatures
# ============================================================

# (name, function, n_params, param_ranges)
# param_ranges: list of (min, max) for each int param
DSL_OPS = [
    # Geometric
    ("rotate", rotate, 1, [(1, 3)]),
    ("flip_h", flip_horizontal, 0, []),
    ("flip_v", flip_vertical, 0, []),
    ("transpose", transpose, 0, []),
    # Color
    ("replace_color", replace_color, 2, [(0, 9), (0, 9)]),
    ("swap_colors", swap_colors, 2, [(0, 9), (0, 9)]),
    ("fill_bg", fill_background, 1, [(1, 9)]),
    ("keep_color", keep_only_color, 1, [(1, 9)]),
    ("invert_colors", invert_colors, 0, []),
    # Structural
    ("crop", crop_to_content, 0, []),
    ("tile", tile, 2, [(1, 3), (1, 3)]),
    ("scale", scale_up, 1, [(2, 3)]),
    ("pad", pad, 2, [(1, 3), (0, 9)]),
    ("mirror_h", mirror_horizontal, 0, []),
    ("mirror_v", mirror_vertical, 0, []),
    # Spatial (new)
    ("flood_fill", flood_fill, 3, [(0, 5), (0, 5), (1, 9)]),
    ("gravity", gravity, 1, [(0, 3)]),
    ("draw_border", draw_border, 1, [(1, 9)]),
    ("fill_rect", fill_rect, 1, [(1, 9)]),
    ("repeat", repeat_pattern, 1, [(0, 1)]),
    ("largest_obj", largest_object, 0, []),
    ("hollow", hollow, 0, []),
]

# Lookup by name
DSL_REGISTRY = {name: (func, n_params, param_ranges) for name, func, n_params, param_ranges in DSL_OPS}


def apply_op(grid: np.ndarray, op_name: str, params: list) -> np.ndarray:
    """Apply a single DSL operation to a grid."""
    if op_name not in DSL_REGISTRY:
        return grid
    func, n_params, _ = DSL_REGISTRY[op_name]
    try:
        if n_params == 0:
            return func(grid)
        else:
            return func(grid, *params[:n_params])
    except Exception:
        return grid  # Gracefully handle invalid operations


def apply_program(grid: np.ndarray, program: list) -> np.ndarray:
    """Apply a sequence of (op_name, params) to a grid."""
    result = grid.copy()
    for op_name, params in program:
        result = apply_op(result, op_name, params)
        # Safety: don't let grids grow beyond 30x30
        if result.shape[0] > 30 or result.shape[1] > 30:
            result = result[:30, :30]
    return result
