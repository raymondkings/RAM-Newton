import torch


def create_task():
    goals = torch.eye(4).unsqueeze(0).repeat(5, 1, 1)

    # Initial pose — in front of wall, pointing +x
    goals[0, :3, :3] = torch.tensor(
        [[0.0, -0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    goals[0, :3, 3] = torch.tensor([0.15, 0.0, 0.125])

    # Back face — x_max center, pointing -x
    goals[1, :3, :3] = torch.tensor(
        [[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    goals[1, :3, 3] = torch.tensor([0.3175, 0.0, 0.125])

    # Top — z_max center, pointing -z
    goals[2, :3, :3] = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    goals[2, :3, 3] = torch.tensor([0.25, 0.0, 0.40])

    # Above init — above pose 0, pointing -z
    goals[3, :3, :3] = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    goals[3, :3, 3] = torch.tensor([0.15, 0.0, 0.40])

    # Above back — above pose 1, pointing -z
    goals[4, :3, :3] = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    goals[4, :3, 3] = torch.tensor([0.3175, 0.0, 0.40])

    return goals
