import torch

def create_task():
    goals = torch.eye(4).unsqueeze(0).repeat(10, 1, 1)

    # Initial pose — EEF in front of wall, pointing +x
    goals[0, :3, :3] = torch.tensor([[0.0, -0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    goals[0, :3, 3]  = torch.tensor([0.15, 0.0, 0.125])

    # Target pose 1 — EEF perpendicular to wall (pointing -x)
    goals[1, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[1, :3, 3]  = torch.tensor([0.29, -0.158334, 0.00625])

    # Target pose 2 — EEF perpendicular to wall (pointing -x)
    goals[2, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[2, :3, 3]  = torch.tensor([0.29, -0.158334, 0.11875])

    # Target pose 3 — EEF perpendicular to wall (pointing -x)
    goals[3, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[3, :3, 3]  = torch.tensor([0.29, -0.158334, 0.23125])

    # Target pose 4 — EEF perpendicular to wall (pointing -x)
    goals[4, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[4, :3, 3]  = torch.tensor([0.29, 0.0, 0.00625])

    # Target pose 5 — EEF perpendicular to wall (pointing -x)
    goals[5, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[5, :3, 3]  = torch.tensor([0.29, 0.0, 0.11875])

    # Target pose 6 — EEF perpendicular to wall (pointing -x)
    goals[6, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[6, :3, 3]  = torch.tensor([0.29, 0.0, 0.23125])

    # Target pose 7 — EEF perpendicular to wall (pointing -x)
    goals[7, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[7, :3, 3]  = torch.tensor([0.29, 0.158334, 0.00625])

    # Target pose 8 — EEF perpendicular to wall (pointing -x)
    goals[8, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[8, :3, 3]  = torch.tensor([0.29, 0.158334, 0.11875])

    # Target pose 9 — EEF perpendicular to wall (pointing -x)
    goals[9, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [-0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[9, :3, 3]  = torch.tensor([0.29, 0.158334, 0.23125])

    return goals