import torch


def create_task():
    goals = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)

    # Initial pose — in front of wall, pointing +x
    goals[0, :3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    goals[0, :3, 3] = torch.tensor([0.15, 0.0, 0.125])

    # Top — z_max center, pointing -z
    goals[1, :3, :3] = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    goals[1, :3, 3] = torch.tensor([0.25, 0.0, 0.40])

    # Back face — x_max center, pointing -x
    goals[2, :3, :3] = torch.tensor(
        [[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    goals[2, :3, 3] = torch.tensor([0.3175, 0.0, 0.125])

    return goals
