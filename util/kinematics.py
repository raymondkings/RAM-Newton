import torch

def transformation_matrix(alpha: torch.Tensor, a: torch.Tensor, d: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute the modified Denavit-Hartenberg transformation matrix.

    Args:
        alpha: twist angle, shape [..., 1]
        a: link length, shape [..., 1]
        d: link offset, shape [..., 1]
        theta: joint angle, shape [..., 1]

    Returns:
        Homogeneous transform, shape [..., 4, 4]
    """
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    ct, st = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(alpha)
    one = torch.ones_like(alpha)

    return torch.stack(
        [
            torch.cat([ct, -st, zero, a], dim=-1),
            torch.cat([st * ca, ct * ca, -sa, -d * sa], dim=-1),
            torch.cat([st * sa, ct * sa, ca, d * ca], dim=-1),
            torch.cat([zero, zero, zero, one], dim=-1),
        ],
        dim=-2,
    )

def forward_kinematics(mdh: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Compute forward kinematics for a robot defined by MDH parameters.

    Args:
        mdh: Tensor [..., dofp1, 3], where each row is [alpha_i, a_i, d_i].
        theta: Tensor [..., dofp1, 1], joint angle for each MDH row.
               The last theta is usually zero for the end-effector transform.

    Returns:
        Tensor [..., dofp1, 4, 4], base-to-joint transforms.
    """
    transforms = transformation_matrix(mdh[..., 0:1], mdh[..., 1:2], mdh[..., 2:3], theta)

    poses = []
    pose = torch.eye(4, device=mdh.device, dtype=mdh.dtype).expand(*mdh.shape[:-2], 1, 4, 4)
    for i in range(mdh.shape[-2]):
        pose = pose @ transforms[..., i:i + 1, :, :]
        poses.append(pose)

    return torch.cat(poses, dim=-3)

def compute_link_world_poses(morph) -> torch.Tensor:
    """Forward kinematics at the rest pose (all joints zero).

    Returns:
        Tensor [n_links, 4, 4], base-to-link transforms.
    """
    mdh = morph.params
    theta = torch.zeros(mdh.shape[0], 1, device=mdh.device, dtype=mdh.dtype)
    return forward_kinematics(mdh, theta)
