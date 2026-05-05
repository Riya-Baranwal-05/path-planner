import numpy as np
"""
Map Generator

Creates 2D binary grids for path planning.
    0 = free space
    1 = obstacle

Usage:
    grid = random_blocks(seed=42)
    start, goal = get_start_goal(grid)
"""

def random_blocks(rows=40, cols=40, n_obstacles=15, min_size=2, max_size=6, seed=None):
    rng = np.random.default_rng(seed)
    grid = np.zeros((rows, cols), dtype=np.int8)
    for _ in range(n_obstacles):
        h = rng.integers(min_size, max_size + 1)
        w = rng.integers(min_size, max_size + 1)
        r = rng.integers(0, rows - h)
        c = rng.integers(0, cols - w)
        grid[r:r+h, c:c+w] = 1
    grid[0:2, 0:2] = 0 #top-left start
    grid[rows-2:rows, cols-2:cols] = 0 #bottom-right=goal
    return grid

def get_start_goal(grid):
    rows, cols = grid.shape
    return (0, 0), (rows-1, cols-1)