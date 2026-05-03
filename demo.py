import sys
import os
import time
import matplotlib.pyplot as plt
sys.path.insert(0, '.')

from planners.astar import AStarPlanner
from planners.rrt import RRTPlanner
from maps.map_generator import random_blocks, get_start_goal
from viz.animate import plot_comparison

grid = random_blocks(seed=42)
start, goal = get_start_goal(grid)

# --- A* ---
print("Running A*...")
planner_a = AStarPlanner()
t0 = time.perf_counter()
path_a, nodes_a, came_from_a = planner_a.plan(grid, start, goal)
time_a = (time.perf_counter() - t0) * 1000
print(f"  Path: {len(path_a)} steps | Nodes expanded: {nodes_a} | Time: {time_a:.1f}ms")

# --- RRT ---
print("Running RRT...")
planner_r = RRTPlanner(seed=42)
t0 = time.perf_counter()
path_r, nodes_r, tree_nodes_r, tree_parents_r = planner_r.plan(grid, start, goal)
time_r = (time.perf_counter() - t0) * 1000
print(f"  Path: {len(path_r)} steps | Nodes expanded: {nodes_r} | Time: {time_r:.1f}ms")

# --- Plot ---
os.makedirs("results", exist_ok=True)
plot_comparison(
    grid,
    astar_result={"path": path_a, "came_from": came_from_a, "nodes": nodes_a},
    rrt_result={"path": path_r, "tree_nodes": tree_nodes_r, "tree_parents": tree_parents_r, "nodes": nodes_r},
    start=start,
    goal=goal,
    save_path="results/comparison.png"
)
plt.show()