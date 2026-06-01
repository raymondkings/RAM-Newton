import torch
from interface import Environment, Box


def l_environment() -> Environment:
    """L-shaped room with the robot base frame aligned to the world frame."""
    WALL_THICKNESS = 0.025
    ROOM_WIDTH = 0.5
    ROOM_DEPTH = 0.5
    ROOM_HEIGHT = 0.35

    # Room is shifted from the origin so the robot's base sits beside the left wall midpoint
    ROOM_OFFSET_X = 0.35

    obstacles = [
        # Back wall — far end at +y
        # Box(
        #     center=torch.tensor(
        #         [ROOM_OFFSET_X + ROOM_WIDTH / 2, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]
        #     ),
        #     half_extents=torch.tensor(
        #         [ROOM_WIDTH / 2, WALL_THICKNESS / 2, ROOM_HEIGHT / 2]
        #     ),
        # ),
        # Left wall — the obstacle the robot must reach around
        Box(
            center=torch.tensor([ROOM_OFFSET_X, 0.0, ROOM_HEIGHT / 2]),
            half_extents=torch.tensor(
                [WALL_THICKNESS / 2, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]
            ),
        ),
    ]

    return Environment(obstacles=obstacles)
