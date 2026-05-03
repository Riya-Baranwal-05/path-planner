import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

COLORS = {
    "obstacle":   "#2d2d2d",
    "free":       "#f5f5f0",
    "start":      "#2ecc71",
    "goal":       "#e74c3c",
    "explored_a": "#aed6f1",
    "explored_r": "#d5f5e3",
    "path_a":     "#2980b9",
    "path_r":     "#27ae60",
}

def plot_grid(ax, grid, title=""):
    cmap = ListedColormap([COLORS["free"], COLORS["obstacle"]])
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_astar_result(grid, path, came_from, start, goal, nodes_expanded, ax=None, title="A*"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    plot_grid(ax, grid, title)
    for (r, c) in came_from:
        ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                     color=COLORS["explored_a"], alpha=0.5, zorder=1))
    if path:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(path_cols, path_rows, color=COLORS["path_a"], linewidth=2.5, zorder=3)
    ax.plot(start[1], start[0], "o", color=COLORS["start"], markersize=10, zorder=5)
    ax.plot(goal[1],  goal[0],  "*", color=COLORS["goal"],  markersize=14, zorder=5)
    ax.set_xlabel(f"Nodes expanded: {nodes_expanded}  |  Path length: {len(path) if path else 0}", fontsize=9)
    return ax

def plot_rrt_result(grid, path, tree_nodes, tree_parents, start, goal, nodes_expanded, ax=None, title="RRT"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    plot_grid(ax, grid, title)
    for i, parent_idx in enumerate(tree_parents):
        if parent_idx == -1:
            continue
        child  = tree_nodes[i]
        parent = tree_nodes[parent_idx]
        ax.plot([parent[0], child[0]], [parent[1], child[1]],
                color=COLORS["explored_r"], linewidth=0.8, alpha=0.6, zorder=1)
    if path:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax.plot(path_cols, path_rows, color=COLORS["path_r"], linewidth=2.5, zorder=3)
    ax.plot(start[1], start[0], "o", color=COLORS["start"], markersize=10, zorder=5)
    ax.plot(goal[1],  goal[0],  "*", color=COLORS["goal"],  markersize=14, zorder=5)
    ax.set_xlabel(f"Nodes expanded: {nodes_expanded}  |  Path length: {len(path) if path else 0}", fontsize=9)
    return ax

def plot_comparison(grid, astar_result, rrt_result, start, goal, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.suptitle("A* vs RRT Path Planning", fontsize=14, fontweight="bold")
    plot_astar_result(grid, astar_result["path"], astar_result["came_from"],
                      start, goal, astar_result["nodes"], ax=axes[0], title="A*")
    plot_rrt_result(grid, rrt_result["path"], rrt_result["tree_nodes"],
                    rrt_result["tree_parents"], start, goal, rrt_result["nodes"],
                    ax=axes[1], title="RRT")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig