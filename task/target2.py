import torch


def create_task():
    goals = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)

    # pos 0
    goals[0, :3, :3] = torch.tensor(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    goals[0, :3, 3] = torch.tensor([0.20, 0.0, 0.125])

    # pos 1 this goes around the wall
    goals[1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    goals[1, :3, 3] = torch.tensor([0.25, -0.40, 0.125])

    # pos 2
    goals[2, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    goals[2, :3, 3] = torch.tensor([0.30, 0.0, 0.125])

    return goals
