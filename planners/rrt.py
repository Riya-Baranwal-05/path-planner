import numpy as np


"""
RRT — Rapidly-exploring Random Tree

Five steps every iteration:
1. Sample  — random point in space
2. Nearest — closest tree node
3. Steer   — step toward sample
4. Check   — collision free?
5. Add     — grow the tree
"""

class RRTPlanner:
    def __init__(self, step_size=0.8, max_iterations=5000, goal_sample_rate=0.1, goal_tolerance=1.5, seed=None):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.rng = np.random.default_rng(seed)
        self.nodes_expanded = 0

    def plan(self, grid, start, goal):
        rows, cols = grid.shape #y ,x
        start_pt = np.array([start[1], start[0]], dtype=float) #x,y
        goal_pt  = np.array([goal[1],  goal[0]],  dtype=float) #x,y

        self.nodes  = [start_pt]
        self.parent = [-1]
        self.nodes_expanded = 0

        for _ in range(self.max_iterations):
            q_rand = self._sample(goal_pt, cols, rows)
            nearest_idx = self._nearest(q_rand)
            q_near = self.nodes[nearest_idx]
            q_new  = self._steer(q_near, q_rand)

            if self._collision_free(q_near, q_new, grid, rows, cols):
                self.nodes.append(q_new)
                self.parent.append(nearest_idx)
                self.nodes_expanded += 1

                if np.linalg.norm(q_new - goal_pt) <= self.goal_tolerance:
                    self.nodes.append(goal_pt)
                    self.parent.append(len(self.nodes) - 2)
                    path = self._extract_path(start, goal)
                    return path, self.nodes_expanded, self.nodes, self.parent

        return None, self.nodes_expanded, self.nodes, self.parent

    def _sample(self, goal_pt, max_x, max_y):
        if self.rng.random() < self.goal_sample_rate:
            return goal_pt.copy()
        return np.array([self.rng.uniform(0, max_x), self.rng.uniform(0, max_y)])

    def _nearest(self, q):
        nodes_array = np.array(self.nodes)
        dists = np.sum((nodes_array - q)**2, axis=1)
        return int(np.argmin(dists))

    def _steer(self, q_near, q_rand):
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist == 0:
            return q_near.copy()
        if dist <= self.step_size:
            return q_rand.copy()
        return q_near + self.step_size * (direction / dist)

    def _collision_free(self, q_from, q_to, grid, rows, cols, n_checks=10):
        for t in np.linspace(0, 1, n_checks):
            pt  = q_from + t * (q_to - q_from)
            col = int(round(np.clip(pt[0], 0, cols - 1)))
            row = int(round(np.clip(pt[1], 0, rows - 1)))
            if grid[row, col] == 1:
                return False
        return True

    def _extract_path(self, start, goal):
        path = []
        idx = len(self.nodes) - 1
        while idx != -1:
            node = self.nodes[idx]
            path.append((int(round(node[1])), int(round(node[0]))))
            idx = self.parent[idx]
        path.reverse()
        return path