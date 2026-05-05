import sys
import os
sys.path.insert(0,'.')
import torch
import torch.nn as nn

class HeuriticsAstar(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,128),# current row , current column, goal row, goal column
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.ReLU()
            
        )
    def forward(self,x):
        return self.net(x)
    


    def as_heuristic(self, grid_size, admissibility_factor=0.9):
        device = next(self.parameters()).device
        self.eval()

        def heuristic(pos, goal):
            curr_r = pos[0] / grid_size
            curr_c = pos[1] / grid_size
            goal_r = goal[0] / grid_size
            goal_c = goal[1] / grid_size

            x = torch.tensor(
                [[curr_r, curr_c, goal_r, goal_c]],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                pred = self(x).item()

            return pred * admissibility_factor

        return heuristic