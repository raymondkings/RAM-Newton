import torch
from interface import Environment, Box

def make_task1_environment() -> Environment:
    """U-shaped room. Robot is beside the left wall and must reach around it."""
    WALL_THICKNESS = 0.05
    ROOM_WIDTH = 1.0
    ROOM_DEPTH = 1.0
    ROOM_HEIGHT = 1.0

    obstacles = [
        # Back wall — far end at +y
        Box(
            center=torch.tensor([ROOM_WIDTH / 2, ROOM_DEPTH, ROOM_HEIGHT / 2]),
            half_extents=torch.tensor([ROOM_WIDTH / 2, WALL_THICKNESS / 2, ROOM_HEIGHT / 2]),
        ),
        # Left wall — the obstacle the robot must reach around
        Box(
            center=torch.tensor([0.0, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]),
            half_extents=torch.tensor([WALL_THICKNESS / 2, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]),
        ),
        # Right wall
        Box(
            center=torch.tensor([ROOM_WIDTH, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]),
            half_extents=torch.tensor([WALL_THICKNESS / 2, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]),
        ),
    ]

    # Robot base: LEFT of the room, beside the left wall midpoint
    base_pose = torch.eye(4)
    base_pose[0, 3] = -0.6    # x = -0.3, outside left wall
    base_pose[1, 3] = ROOM_DEPTH / 2   # y = 0.5, level with room midpoint
    return Environment(obstacles=obstacles, base_pose=base_pose)