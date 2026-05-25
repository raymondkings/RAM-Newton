import math
import torch

_s = math.sqrt(0.5)  # 1/√2, for 45-degree tilts in the xz-plane


def create_task():
    goals = torch.eye(4).unsqueeze(0).repeat(5, 1, 1)

    # Initial pose — in front of wall, pointing +x
    goals[0, :3, :3] = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[0, :3, 3]  = torch.tensor([0.19, 0.0, 0.125])

    # Above init — Z tilted 45° from +z toward -x  (Z col = [-s, 0, s])
    goals[1, :3, :3] = torch.tensor([[_s, 0.0, -_s], [0.0, 1.0, 0.0], [_s, 0.0, _s]])
    goals[1, :3, 3]  = torch.tensor([0.19, 0.0, 0.40])

    # Top — z_max center, pointing -z
    goals[2, :3, :3] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    goals[2, :3, 3]  = torch.tensor([0.25, 0.0, 0.40])

    # Above back — Z tilted 45° from +z toward +x  (Z col = [s, 0, s])
    goals[3, :3, :3] = torch.tensor([[_s, 0.0, _s], [0.0, 1.0, 0.0], [-_s, 0.0, _s]])
    goals[3, :3, 3]  = torch.tensor([0.31, 0.0, 0.40])
    
    # Back face — x_max center, pointing -x
    goals[4, :3, :3] = torch.tensor([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    goals[4, :3, 3]  = torch.tensor([0.31, 0.0, 0.125])

    return goals
