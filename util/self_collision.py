import torch
from typing import Optional


def get_capsules(
    mdh: torch.Tensor, poses: Optional[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a robot morphology, compute link-enclosing capsule start and endpoints.

    Args:
        mdh: Tensor [..., dofp1, 3], MDH morphology [alpha, a, d].
        poses: Tensor [..., dofp1, 4, 4], forward kinematics result.

    Returns:
        s_all, e_all: each Tensor [..., 2*dofp1, 3]
    """
    if poses is None:
        raise ValueError("poses must be provided for get_capsules().")

    *batch_shape, _, _ = mdh.shape
    device = mdh.device
    dtype = mdh.dtype

    # Prepend base frame T0. poses currently contains [T1, T2, ..., TN].
    identity = torch.eye(4, device=device, dtype=dtype).expand(*batch_shape, 1, 4, 4)
    poses = torch.cat([identity, poses], dim=-3)

    s_a = poses[..., :-1, :3, 3]
    e_d = poses[..., 1:, :3, 3]

    z_axis = poses[..., 1:, :3, 2]
    d = mdh[..., 2].unsqueeze(-1)
    e_a = s_d = e_d - d * z_axis

    s_all = torch.stack([s_a, s_d], dim=-2).flatten(-3, -2)
    e_all = torch.stack([e_a, e_d], dim=-2).flatten(-3, -2)
    return s_all, e_all
