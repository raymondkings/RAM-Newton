import torch
from interface import Environment, Box

def make_task1_environment() -> Environment:
    """Three walls forming a U-shaped room, open on one side (Task 1)."""
    WALL_THICKNESS = 0.05
    ROOM_WIDTH = 1.0
    ROOM_DEPTH = 1.0
    ROOM_HEIGHT = 1.0
    
    obstacles = [
        # Back wall
        Box(
            center=torch.tensor([ROOM_DEPTH, 0.0, ROOM_HEIGHT / 2]),
            half_extents=torch.tensor([WALL_THICKNESS / 2, ROOM_WIDTH / 2, ROOM_HEIGHT / 2]),
        ),
        # Left wall
        Box(
            center=torch.tensor([ROOM_DEPTH / 2, +ROOM_WIDTH / 2, ROOM_HEIGHT / 2]),
            half_extents=torch.tensor([ROOM_DEPTH / 2, WALL_THICKNESS / 2, ROOM_HEIGHT / 2]),
        ),
        # Right wall
        Box(
            center=torch.tensor([ROOM_DEPTH / 2, -ROOM_WIDTH / 2, ROOM_HEIGHT / 2]),
            half_extents=torch.tensor([ROOM_DEPTH / 2, WALL_THICKNESS / 2, ROOM_HEIGHT / 2]),
        ),
    ]
    
    base_pose = torch.eye(4)   # robot mounted at origin
    return Environment(obstacles=obstacles, base_pose=base_pose)