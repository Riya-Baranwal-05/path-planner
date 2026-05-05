# A* / RRT Path Planner + Learned Heuristic



## What this does
A robot needs to get from A to B without hitting obstacles.
This project implements two classic path planning algorithms from scratch.

## Algorithms
- **A*** — searches a grid, guaranteed shortest path, uses a heuristic `f(n) = g(n) + h(n)`
- **RRT** — grows a random tree, works in continuous space, used in real robot arms

## Run it
```bash
pip install -r requirements.txt
python demo.py
```

## Project structure
planners/astar.py       — A* from scratch
planners/rrt.py         — RRT from scratch
maps/map_generator.py   — random obstacle maps
ml/heuristic_net.py     — neural heuristic (coming soon)

## Limitations
- Neural heuristic only sees position — not obstacles. 
  Needs CNN architecture to improve.
- RRT path is jagged — real robots would smooth it with RRT*
- Collision check uses discrete sampling — thin obstacles can be missed
- Grid size fixed at 40x40 — larger maps need retraining

## What I learned
- A* is Dijkstra + a heuristic — the heuristic is everything
- RRT's goal_sample_rate has huge effect on convergence speed
- Euclidean distance is admissible but optimistic with many obstacles

## References
- Russell & Norvig — *AI: A Modern Approach* (A*)
- LaValle — *Planning Algorithms* (RRT)