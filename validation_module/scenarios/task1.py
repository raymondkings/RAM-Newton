import torch
from interface import Task, ReachableRegion
from .task1_environment import make_task1_environment


# Single source of truth for room dimensions
ROOM_DEPTH = 1.0
ROOM_WIDTH = 1.0
ROOM_HEIGHT = 1.0
WALL_THICKNESS = 0.05


def make_task1() -> Task:
    env = make_task1_environment()
    
    # Reachable region = interior of the U-shaped room
    # Slightly inset from walls so it's clearly inside
    inset = WALL_THICKNESS + 0.05
    region = ReachableRegion(
        x_min=0.0 + inset,
        x_max=ROOM_DEPTH - inset,
        y_min=-ROOM_WIDTH/2 + inset,
        y_max=+ROOM_WIDTH/2 - inset,
        z_min=0.1,                          # off the floor
        z_max=ROOM_HEIGHT - inset,
    )
    
    # A few hand-picked goal poses inside the region (placeholder)
    # Eventually Jiyao samples these properly
    goals = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    goals[0, :3, 3] = torch.tensor([0.5, 0.0, 0.5])
    goals[1, :3, 3] = torch.tensor([0.7, 0.3, 0.5])
    goals[2, :3, 3] = torch.tensor([0.7, -0.3, 0.5])
    
    return Task(environment=env, goal_poses=goals, reachable_region=region)