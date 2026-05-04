import torch
from interface import Environment, Box, Task, ReachableRegion

ROOM_WIDTH = 1.0
ROOM_DEPTH = 1.0
ROOM_HEIGHT = 0.5
WALL_THICKNESS = 0.05

def create_environment() -> Environment:
    """U-shaped room. Robot is beside the left wall and must reach around it."""

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
        # Box(
        #     center=torch.tensor([ROOM_WIDTH, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]),
        #     half_extents=torch.tensor([WALL_THICKNESS / 2, ROOM_DEPTH / 2, ROOM_HEIGHT / 2]),
        # ),
    ]

    # Robot base: LEFT of the room, beside the left wall midpoint
    base_pose = torch.eye(4)
    base_pose[0, 3] = -0.6              # x: outside left wall
    base_pose[1, 3] = ROOM_DEPTH / 2   # y: level with room midpoint
    base_pose[2, 3] = 0.3              # z: lifted clear of ground + Newton contact margin
    
    return Environment(obstacles=obstacles, base_pose=base_pose)

def create_task() -> Task:
    env = create_environment()

    # inset = WALL_THICKNESS + 0.05
    # region = ReachableRegion(
    #     x_min=0.0 + inset,
    #     x_max=ROOM_WIDTH - inset,
    #     y_min=0.0 + inset,
    #     y_max=ROOM_DEPTH - inset,
    #     z_min=0.1,
    #     z_max=ROOM_HEIGHT - inset,
    # )

    # Goals inside the room — robot must reach around the left wall to get here
    goals = torch.eye(4).unsqueeze(0).repeat(1, 1, 1)
    goals[0, :3, 3] = torch.tensor([0.3, 0.5, 0.5])   # near left wall
    # goals[1, :3, 3] = torch.tensor([0.5, 0.8, 0.5])   # back of room
    # goals[2, :3, 3] = torch.tensor([0.8, 0.5, 0.5])   # near right wall

    return Task(environment=env, goal_poses=goals)