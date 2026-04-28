import torch
from interface import Task, ReachableRegion
from .task1_environment import make_task1_environment

ROOM_WIDTH = 1.0
ROOM_DEPTH = 1.0
ROOM_HEIGHT = 1.0
WALL_THICKNESS = 0.05

def make_task1() -> Task:
    env = make_task1_environment()

    inset = WALL_THICKNESS + 0.05
    region = ReachableRegion(
        x_min=0.0 + inset,
        x_max=ROOM_WIDTH - inset,
        y_min=0.0 + inset,
        y_max=ROOM_DEPTH - inset,
        z_min=0.1,
        z_max=ROOM_HEIGHT - inset,
    )

    # Goals inside the room — robot must reach around the left wall to get here
    goals = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    goals[0, :3, 3] = torch.tensor([0.3, 0.5, 0.5])   # near left wall
    goals[1, :3, 3] = torch.tensor([0.5, 0.8, 0.5])   # back of room
    goals[2, :3, 3] = torch.tensor([0.8, 0.5, 0.5])   # near right wall

    return Task(environment=env, goal_poses=goals, reachable_region=region)