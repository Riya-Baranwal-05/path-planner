import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import sys
sys.path.insert(0, '.')
from maps.map_generator import random_blocks, get_start_goal

fig, ax = plt.subplots(figsize=(6, 6))
cmap = ListedColormap(["#f5f5f0", "#2d2d2d"])

def update(seed):
    ax.clear()
    grid = random_blocks(seed=seed)
    start, goal = get_start_goal(grid)
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=1)
    ax.plot(start[1], start[0], "o", color="#2ecc71", markersize=12)
    ax.plot(goal[1],  goal[0],  "*", color="#e74c3c", markersize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Random obstacle map (seed={seed})", fontsize=11)

anim = animation.FuncAnimation(fig, update, frames=range(10), interval=800)
anim.save("results/map.gif", writer="pillow", fps=1)
print("Saved results/map.gif")
plt.show()