import numpy as np


def random_blocks(rows=40, cols=40, n_obstacles=15, min_size=2, max_size=6, seed=None):
    rng = np.random.default_rng(seed)
    grid = np.zeros((rows, cols), dtype=np.int8)
    for _ in range(n_obstacles):
        h = rng.integers(min_size, max_size + 1)
        w = rng.integers(min_size, max_size + 1)
        r = rng.integers(0, rows - h)
        c = rng.integers(0, cols - w)
        grid[r:r+h, c:c+w] = 1
    grid[0:2, 0:2] = 0
    grid[rows-2:rows, cols-2:cols] = 0
    return grid

def get_start_goal(grid):
    rows, cols = grid.shape
    return (0, 0), (rows-1, cols-1)