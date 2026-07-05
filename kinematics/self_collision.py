import torch
from typing import Optional

LINK_RADIUS = 0.025
EPS = 1e-4


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


def get_joint_limits(morph: torch.Tensor) -> torch.Tensor:
    """
    Compute joint limits based on the morphology to avoid self-collisions.

    Args:
        morph: MDH parameters [..., dofp1, 3] encoding the robot geometry.

    Returns:
        Tensor [..., dofp1, 2], last dim is [range, offset]; the allowed joint
        interval is [offset, offset + range]. The final row corresponds to the
        fixed end-effector frame and is zero.
    """
    from kinematics.kinematics import (
        forward_kinematics,
        transformation_matrix,
    )

    joint_limits = torch.zeros(
        *morph.shape[:-1], 2, device=morph.device, dtype=morph.dtype
    )

    extended_morph = torch.cat([torch.zeros_like(morph[..., :1, :]), morph], dim=-2)
    alpha0, a0, d0 = extended_morph[..., :-2, :].split(1, dim=-1)
    alpha1, a1, d1 = extended_morph[..., 1:-1, :].split(1, dim=-1)

    coordinate_fix = torch.eye(4, device=morph.device, dtype=morph.dtype).repeat(
        *morph.shape[:-2], morph.shape[-2] - 1, 1, 1
    )
    wrist = (a1[..., 0] == 0) & (d1[..., 0] == 0)
    coordinate_fix[wrist] = transformation_matrix(alpha0, a0, d0, torch.zeros_like(d0))[
        wrist
    ]

    plane_normal = torch.stack(
        [
            torch.zeros_like(alpha1),
            -torch.sin(alpha1),
            torch.cos(alpha1),
            torch.zeros_like(alpha1),
        ],
        dim=-1,
    )
    plane_anchor = torch.stack(
        [a1, -d1 * torch.sin(alpha1), d1 * torch.cos(alpha1), torch.ones_like(alpha1)],
        dim=-1,
    )

    plane_normal = torch.sum(coordinate_fix * plane_normal, dim=-1)[..., :3]
    plane_anchor = torch.sum(coordinate_fix * plane_anchor, dim=-1)[..., :3]

    stacked_morph = torch.stack(
        [
            extended_morph[..., :-2, :],
            extended_morph[..., 1:-1, :],
            extended_morph[..., 2:, :],
        ],
        dim=-2,
    )
    stacked_morph[~wrist, 0, :] = 0.0
    stacked_poses = forward_kinematics(
        stacked_morph,
        torch.zeros(
            *stacked_morph.shape[:-1], 1, device=morph.device, dtype=morph.dtype
        ),
    )
    start, end = get_capsules(stacked_morph, stacked_poses)
    capsules = end - start

    # Closest non-zero capsule before joint
    pre_capsule = capsules[..., 3, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 2, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 1, :]

    # Closest non-zero capsule after joint
    post_capsule = capsules[..., -2, :]
    post_capsule[mask] = capsules[mask := post_capsule.norm(dim=-1) < 1e-6, -1, :]

    in_plane = ((pre_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6
    in_plane &= ((post_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6

    limited = (
        (pre_capsule.norm(dim=-1) > EPS) & (post_capsule.norm(dim=-1) > EPS) & in_plane
    )

    mask = post_capsule.norm(dim=-1) > pre_capsule.norm(dim=-1)
    arc = torch.arcsin(2 * LINK_RADIUS / post_capsule.norm(dim=-1))
    arc[mask] = torch.arcsin(2 * LINK_RADIUS / pre_capsule.norm(dim=-1))[mask]

    joint_limits[..., :-1, 0] = torch.where(
        limited, 2 * torch.pi - 2 * arc, 2 * torch.pi
    )
    angle = torch.atan2(
        torch.sum(
            torch.cross(pre_capsule, post_capsule, dim=-1) * plane_normal, dim=-1
        ),
        torch.sum(pre_capsule * post_capsule, dim=-1),
    )
    # If their angle becomes pi, they collide and are antiparallel.
    angle = torch.atan2(torch.sin(torch.pi - angle), torch.cos(torch.pi - angle))
    joint_limits[..., :-1, 1] = torch.where(limited, angle + arc, -torch.pi)

    return joint_limits
