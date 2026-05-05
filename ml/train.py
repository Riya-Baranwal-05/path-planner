import sys
import os
sys.path.insert(0,'.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from planners.astar import AStarPlanner
from maps.map_generator import random_blocks,get_start_goal
from ml.heuristics_net import HeuriticsAstar

def generate_training_data(n_maps,grid_size=40):
    planner = AStarPlanner()
    X_list, y_list =[],[]
    for i in range(n_maps):
        grid = random_blocks(seed=i)
        start,goal = get_start_goal(grid)
        path, node,came_from=planner.plan(grid,start,goal)
        if path is None:
            continue
        for step,pos in enumerate(path):
            remaining = len(path)-step-1
            curr_r = pos[0]/grid_size
            curr_c = pos[1]/grid_size
            goal_r = goal[0]/grid_size
            goal_c = goal[1]/grid_size
            X_list.append([curr_r,curr_c,goal_r,goal_c])
            y_list.append(float(remaining))
    return np.array(X_list,dtype=np.float32),np.array(y_list,dtype=np.float32)

    pass
def train():
    X,y = generate_training_data(n_maps=500)
    print(f"Data shape: {X.shape}, y shape: {y.shape}")
    print(f"y min: {y.min():.2f}, y max: {y.max():.2f}, y mean: {y.mean():.2f}")

    X = torch.tensor(X)
    y = torch.tensor(y).unsqueeze(1)
    model = HeuriticsAstar()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    print(f"LR: {optimizer.param_groups[0]['lr']}")
    print(f"X tensor shape: {X.shape}")
    print(f"y tensor shape: {y.shape}")
    for epoch in range(2000):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred,y)
        loss.backward()
        optimizer.step()

        if epoch%100 == 0:
            print(f"Epoch:{epoch} loss: {loss.item():.6f}")


    torch.save(model.state_dict(),"results/heuristics_net.pt")
    pass

def benchmark(model_path, n_maps =50,grid_size =40):
    model = HeuriticsAstar()
    model.load_state_dict(torch.load(model_path))
    planner_e = AStarPlanner()
    planner_h = AStarPlanner(heuristic=model.as_heuristic(grid_size=40))
    euclid_list,neural_list =[],[]
    for i in range(n_maps):
        grid = random_blocks(seed = i+500)
        start,goal = get_start_goal(grid)
        _,nodes_e,_=planner_e.plan(grid,start,goal)
        _,nodes_h,_=planner_h.plan(grid,start,goal)
        euclid_list.append(nodes_e)
        neural_list.append(nodes_h)
    print(f"Euclidean: {np.mean(euclid_list):.1f} nodes")
    print(f"Neural:    {np.mean(neural_list):.1f} nodes")
    print(f"Reduction: {(1 - np.mean(neural_list)/np.mean(euclid_list))*100:.1f}%")


if __name__ == "__main__":
    train()
    benchmark("results/heuristics_net.pt")