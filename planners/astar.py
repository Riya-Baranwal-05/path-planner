import heapq

"""
A* Path Planning — implemented from scratch.

f(n) = g(n) + h(n)
g(n) = exact cost from start to n
h(n) = heuristic estimate from n to goal

Guarantees shortest path when h(n) is admissible.
"""



class AStarPlanner:
    def __init__(self, heuristic=None):
        self.heuristic = heuristic or self._euclidean
        self.nodes_expanded = 0

    def plan(self, grid, start, goal):
        self.nodes_expanded = 0
        rows, cols = grid.shape
        g_cost = {start: 0.0}
        counter = 0
        open_set = [(self._euclidean(start, goal), counter, start)]
        came_from = {}
        closed_set = set()

        while open_set:
            f, _, current = heapq.heappop(open_set)
            if current in closed_set:
                continue
            closed_set.add(current)
            self.nodes_expanded += 1

            if current == goal:
                return self._reconstruct_path(came_from, start, goal), self.nodes_expanded, came_from

            for neighbor in self._get_neighbors(current, grid, rows, cols):
                if neighbor in closed_set:
                    continue
                tentative_g = g_cost[current] + self._move_cost(current, neighbor)
                if tentative_g < g_cost.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_cost[neighbor] = tentative_g
                    f_new = tentative_g + self.heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_new, counter, neighbor))

        return None, self.nodes_expanded, came_from

    def _get_neighbors(self, pos, grid, rows, cols):
        r, c = pos
        candidates = [
            (r-1,c),(r+1,c),(r,c-1),(r,c+1),
            (r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)
        ]
        return [(nr,nc) for nr,nc in candidates
                if 0<=nr<rows and 0<=nc<cols and grid[nr,nc]==0]

    def _move_cost(self, a, b):
        return 1.4142135623730951 if abs(a[0]-b[0])==1 and abs(a[1]-b[1])==1 else 1.0

    def _reconstruct_path(self, came_from, start, goal):
        path = [goal]
        current = goal
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _euclidean(pos, goal):
        return ((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2) ** 0.5