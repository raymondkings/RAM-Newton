import torch
from jaxtyping import Float, Float64
from eaik.IK_Homogeneous import HomogeneousRobot

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


def morph_to_eaik(mdh: Float[torch.Tensor, "dofp1 3"]) -> HomogeneousRobot:
    """Wrap an MDH morphology in EAIK's HomogeneousRobot for analytical IK."""
    local_coord = transformation_matrix(
        mdh[:, 0:1], mdh[:, 1:2], mdh[:, 2:3], torch.zeros_like(mdh[:, 2:3])
    )
    global_coords = torch.empty_like(local_coord)
    global_coords[0] = local_coord[0]
    for i in range(1, len(local_coord)):
        global_coords[i] = global_coords[i - 1] @ local_coord[i]
    return HomogeneousRobot(global_coords.cpu().numpy())


def pure_analytical_inverse_kinematics(
    mdh: Float[torch.Tensor, "dofp1 3"],
    poses: Float[torch.Tensor, "batch 4 4"],
) -> list[Float[torch.Tensor, "n_solutions dofp1 1"]]:
    """All EAIK IK solutions per pose, unfiltered.
    Raises RuntimeError if the morphology has no known closed-form decomposition.
    """
    eaik_bot = morph_to_eaik(mdh)
    if not eaik_bot.hasKnownDecomposition():
        raise RuntimeError(f"Robot is not analytically solvable. {mdh}")
    solutions = eaik_bot.IK_batched(poses.cpu().numpy())
    if torch.tensor([sol.num_solutions() == 0 for sol in solutions]).all():
        raise RuntimeError("EAIK returned no solutions for any pose.")
    joints = [
        torch.cat(
            [
                torch.from_numpy(sol.Q.copy()).unsqueeze(-1),
                torch.zeros(sol.num_solutions(), 1, 1),
            ],
            dim=1,
        )
        if sol.num_solutions() != 0
        else torch.empty(0, mdh.shape[0], 1, dtype=torch.double)
        for sol in solutions
    ]
    return joints


def numerical_inverse_kinematics(
    mdh: Float[torch.Tensor, "dofp1 3"],
    target_pose: Float[torch.Tensor, "4 4"],
    n_seeds: int = 32,
    n_iter: int = 400,
    lr: float = 0.05,
    pos_weight: float = 1.0,
    rot_weight: float = 0.5,
) -> Float[torch.Tensor, "n_seeds dofp1 1"]:
    """Adam-based IK on Frobenius pose error. Returns all candidate joint configs.
    Caller filters by reach error and collisions externally.
    """
    n_joints = mdh.shape[0] - 1
    device = mdh.device
    dtype = mdh.dtype

    joints = (torch.rand(n_seeds, n_joints, 1, device=device, dtype=dtype) * 2 * torch.pi - torch.pi)
    joints = joints.requires_grad_(True)
    opt = torch.optim.Adam([joints], lr=lr)

    morph_b = mdh.unsqueeze(0).expand(n_seeds, -1, -1)
    target_b = target_pose.to(device=device, dtype=dtype).unsqueeze(0).expand(n_seeds, -1, -1)

    for _ in range(n_iter):
        full_joints = torch.cat([joints, torch.zeros(n_seeds, 1, 1, device=device, dtype=dtype)], dim=1)
        ee_pose = forward_kinematics(morph_b, full_joints)[:, -1]
        pos_err = ((ee_pose[:, :3, 3] - target_b[:, :3, 3]) ** 2).sum(dim=-1)
        rot_err = ((ee_pose[:, :3, :3] - target_b[:, :3, :3]) ** 2).sum(dim=(-1, -2))
        loss = (pos_weight * pos_err + rot_weight * rot_err).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    joints = torch.atan2(torch.sin(joints.detach()), torch.cos(joints.detach()))
    return torch.cat([joints, torch.zeros(n_seeds, 1, 1, device=device, dtype=dtype)], dim=1)