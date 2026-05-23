# -----------------------------------------------------------------------------
# Original work Copyright (c) 2023 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
#
# Standalone DOF-6 morphology sampler extracted from the NRM project.
#
# This file only depends on torch. It does NOT need:
#   - nrm package
#   - scipy / se3 / so3 lookup tables
#   - IK / EAIK
#   - zarr dataset files
#   - jaxtyping / beartype
#
# Main API:
#   sample_dof6_initial_morphologies(num_initial_samples, ...)
#
# Output:
#   Tensor [N, 7, 3]
#   Each morphology M is [dof + 1, 3], where each row is [alpha_i, a_i, d_i].
# -----------------------------------------------------------------------------

from __future__ import annotations


import torch

# Same as original _reject_morph: test each candidate morphology on 1000 random joint configurations.
REJECTION_TEST_CONFIGS = 1000
from torch import Tensor
from util.kinematics import forward_kinematics, transformation_matrix
from util.self_collision import get_capsules


# Same constants as the original self_collision.py
LINK_RADIUS = 0.025
EPS = 1e-4


# -----------------------------------------------------------------------------
# Minimal manipulability.py subset
# -----------------------------------------------------------------------------


def geometric_jacobian(poses: Tensor) -> Tensor:
    """
    Compute the geometric Jacobian for revolute joints in the base frame.

    Args:
        poses: Tensor [..., dofp1, 4, 4], including the end-effector as last pose.

    Returns:
        Tensor [..., 6, dof]
    """
    orientation = poses[..., :-1, :3, :3]
    positions = poses[..., :-1, :3, 3]
    eef_position = poses[..., -1:, :3, 3]
    joint_z_axes = orientation[..., :, 2]

    jacobian = torch.cat(
        [torch.cross(joint_z_axes, eef_position - positions, dim=-1), joint_z_axes],
        dim=-1,
    ).transpose(-1, -2)

    return jacobian


def yoshikawa_manipulability(jacobian: Tensor, soft: bool = False) -> Tensor:
    """
    Compute the Yoshikawa manipulability index.

    Args:
        jacobian: Tensor [..., 6, dof]
        soft: If True and dof < 6, use product of available singular values.

    Returns:
        Tensor [...]
    """
    if soft and jacobian.shape[-1] < 6:
        _, singular_values, _ = torch.linalg.svd(jacobian)
        return torch.prod(singular_values[..., : min(6, jacobian.shape[-1])], dim=-1)

    return torch.sqrt(torch.det(jacobian @ jacobian.transpose(-1, -2)).abs())


def signed_distance_capsule_capsule(
    s1: Tensor,
    e1: Tensor,
    r1: float,
    s2: Tensor,
    e2: Tensor,
    r2: float,
) -> Tensor:
    """Compute signed squared distance between two capsules."""
    l1 = e1 - s1
    l2 = e2 - s2
    ds = s1 - s2

    alpha = (l1 * l1).sum(dim=-1, keepdim=True)
    beta = (l2 * l2).sum(dim=-1, keepdim=True)
    gamma = (l1 * l2).sum(dim=-1, keepdim=True)
    delta = (l1 * ds).sum(dim=-1, keepdim=True)
    epsilon = (l2 * ds).sum(dim=-1, keepdim=True)

    det = alpha * beta - gamma**2

    t1 = torch.clamp((gamma * epsilon - beta * delta) / (det + 1e-10), 0.0, 1.0)
    t2 = torch.clamp((gamma * t1 + epsilon) / (beta + 1e-10), 0.0, 1.0)

    t1 = torch.where(
        (t2 == 0.0) | (t2 == 1.0),
        torch.clamp((t2 * gamma - delta) / (alpha + 1e-10), 0.0, 1.0),
        t1,
    )

    c1 = s1 + t1 * l1
    c2 = s2 + t2 * l2

    point_distance = ((c1 - c2) ** 2).sum(dim=-1)
    return point_distance - (r1 + r2) ** 2


def signed_distance_capsule_ball(
    s1: Tensor, e1: Tensor, r1: float, s2: Tensor, r2: float
) -> Tensor:
    """Compute signed squared distance between a capsule and a ball."""
    l1 = e1 - s1
    v = s2 - s1

    dot_v_l = (v * l1).sum(dim=-1, keepdim=True)
    len_sq_l = (l1 * l1).sum(dim=-1, keepdim=True)
    t = torch.clamp(dot_v_l / (len_sq_l + 1e-10), 0.0, 1.0)
    closest_point = s1 + t * l1

    point_distance = ((closest_point - s2) ** 2).sum(dim=-1)
    return point_distance - (r1 + r2) ** 2


PAIR_COMBINATIONS = [
    torch.triu_indices(2 * dof, 2 * dof, offset=2) for dof in range(1, 9)
]


def collision_check(
    mdh: Tensor, poses: Tensor, radius: float = LINK_RADIUS, debug: bool = False
) -> Tensor:
    """
    Compute whether the robot is in self-collision for each batch element.

    Args:
        mdh: Tensor [..., dofp1, 3]
        poses: Tensor [..., dofp1, 4, 4]
        radius: capsule radius
        debug: if True, return critical signed distances instead of bool collision mask.

    Returns:
        Bool Tensor [...] if debug=False.
    """
    global PAIR_COMBINATIONS
    if mdh.device != PAIR_COMBINATIONS[0].device:
        PAIR_COMBINATIONS = [pair.to(mdh.device) for pair in PAIR_COMBINATIONS]

    *batch_shape, dofp1, _ = mdh.shape

    s_all, e_all = get_capsules(mdh, poses)

    i_idx, j_idx = PAIR_COMBINATIONS[dofp1 - 1]
    num_pairs = i_idx.shape[0]

    s1 = s_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    e1 = e_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    s2 = s_all[..., j_idx, :].reshape(-1, num_pairs, 3)
    e2 = e_all[..., j_idx, :].reshape(-1, num_pairs, 3)

    # Ignore distances with missing capsules.
    collisions = (torch.norm(s1 - e1, dim=-1) > EPS) & (
        torch.norm(s2 - e2, dim=-1) > EPS
    )
    # Ignore adjacent capsules.
    collisions &= torch.norm(e1 - s2, dim=-1) > EPS

    distances = signed_distance_capsule_capsule(s1, e1, radius, s2, e2, radius)

    # Start/end kinematic chain ball checks.
    expansion_shape = s_all.shape
    s_end = torch.cat([s_all, s_all], dim=-2)
    e_end = torch.cat([e_all, e_all], dim=-2)
    c_end = torch.cat(
        [
            s_all[..., :1, :].expand(*expansion_shape),
            e_all[..., -1:, :].expand(*expansion_shape),
        ],
        dim=-2,
    )
    distance_end = signed_distance_capsule_ball(s_end, e_end, radius, c_end, radius)
    collisions_end = torch.norm(e_end - s_end, dim=-1) > EPS

    # Only interested in the second and second-to-last nonzero capsules.
    cum_sum = torch.cumsum(collisions_end, dim=-1)
    collisions_end &= (cum_sum == 2) | (cum_sum == (cum_sum[..., -1:] - 1))

    if not debug:
        collisions &= distances < 0.0
        collisions = collisions.any(dim=-1).reshape(batch_shape)
        collisions_end &= distance_end < 0.0
        collisions_end = collisions_end.any(dim=-1)
        return collisions | collisions_end

    critical_distance = (distances * collisions).min(dim=-1).values.reshape(batch_shape)
    critical_distance_end = (distance_end * collisions_end).min(dim=-1).values
    return (
        torch.stack([critical_distance, critical_distance_end], dim=-1)
        .min(dim=-1)
        .values
    )


def _sample_link_type(batch_size: int, dof: int, device: torch.device) -> Tensor:
    """
    Sample link types:
        0 <=> a != 0, d != 0  (general)
        1 <=> a == 0, d != 0  (only d / link offset)
        2 <=> a != 0, d == 0  (only a / link length)
        3 <=> a == 0, d == 0  (spherical wrist component)
    """
    return torch.randint(
        0, 4 if dof > 1 else 3, size=(batch_size, dof + 1), device=device
    )


def _reject_link_type(link_type: Tensor) -> Tensor:
    """Reject adjacent spherical-wrist-component link types."""
    return ((link_type[:, 1:] == 3) & (link_type[:, :-1] == 3)).any(dim=1)


def _sample_link_twist(link_type: Tensor) -> Tensor:
    """Sample alpha uniformly from {0, pi/2, -pi/2}."""
    link_twist_options = torch.tensor(
        [0.0, torch.pi / 2, -torch.pi / 2], device=link_type.device
    )
    link_twist_choice = torch.randint(
        0, 3, size=link_type.shape, device=link_type.device
    )
    return link_twist_options[link_twist_choice]


def _reject_link_twist(link_twist: Tensor, link_type: Tensor) -> Tensor:
    """Reject twist/type combinations that cause structural degeneracy or unmanufacturable wrists."""
    rejected = (
        (link_twist[:, :-2] == 0)
        & (link_twist[:, 1:-1] == 0)
        & (link_twist[:, 2:] == 0)
    ).any(dim=1)
    rejected |= (
        (link_twist[:, :-2] == 0) & (link_type[:, 1:-1] == 3) & (link_twist[:, 2:] == 0)
    ).any(dim=1)
    rejected |= (
        (link_type[:, :-1] == 1) & (link_type[:, 1:] == 1) & (link_twist[:, 1:] == 0)
    ).any(dim=1)
    rejected |= ((link_type == 3) & (link_twist == 0)).any(dim=1)
    return rejected


def _sample_analytically_solvable_link_types_and_twist(
    batch_size: int,
    dof: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Original project helper for optionally biasing samples toward analytically solvable structures.
    For main API this stays disabled by default.
    """
    link_type = _sample_link_type(batch_size, dof, device)
    link_twist = _sample_link_twist(link_type)

    if dof == 5:
        types = torch.randint(0, 3, size=(batch_size,), device=device)

        link_type[
            types == 0,
            torch.randint(0, 2, ((types == 0).sum().item(),), device=device) * 3 + 1,
        ] = 1

        axes_choice = torch.randint(0, 2, ((types == 1).sum().item(),), device=device)
        link_type[types == 1, axes_choice * 2 + 2] = 1
        link_twist[types == 1, (~axes_choice.bool()) * 2 + 2] = 0

        axes_choice = torch.randint(1, 4, ((types == 2).sum().item(),), device=device)
        link_twist[types == 2, axes_choice] = 0
        link_twist[types == 2, axes_choice + 1] = 0

    elif dof == 6:
        types = torch.randint(0, 3, size=(batch_size,), device=device)

        axes_choice = torch.randint(0, 2, ((types == 0).sum().item(),), device=device)
        link_type[types == 0, 1 + 3 * axes_choice] = 3
        link_type[types == 0, 2 + 3 * axes_choice] = 1

        axes_choice = torch.randint(0, 2, ((types == 1).sum().item(),), device=device)
        link_type[types == 1, 1 + 4 * axes_choice] = 1
        link_twist[types == 1, 1 + 3 * (~axes_choice.bool())] = 0
        link_twist[types == 1, 2 + 3 * (~axes_choice.bool())] = 0

        axes_choice = torch.randint(2, 4, ((types == 2).sum().item(),), device=device)
        link_twist[types == 2, axes_choice] = 0
        link_twist[types == 2, axes_choice + 1] = 0

    elif dof > 6:
        raise NotImplementedError(
            "Analytically solvable sampling is not implemented for DOF > 6."
        )

    return link_type, link_twist


def _sample_link_length(link_type: Tensor) -> Tensor:
    """
    Sample link lengths uniformly on a simplex, then split into a/d according to link type.

    Returns:
        Tensor [batch_size, dofp1, 2], where last dim is [a, d].
    """
    dist = torch.distributions.exponential.Exponential(
        torch.tensor([1.0], device=link_type.device)
    )
    link_lengths = dist.sample(torch.Size((*link_type.shape, 1)))[..., 0].repeat(
        1, 1, 2
    )

    # Normalize total non-wrist length budget.
    link_lengths /= (link_lengths * (link_type.unsqueeze(-1) != 3)).sum(
        dim=1, keepdim=True
    )
    link_lengths *= torch.sign(torch.rand_like(link_lengths) - 0.5)

    gamma = torch.rand((link_type == 0).sum(), device=link_type.device) * 2 * torch.pi
    link_lengths[..., 0][link_type == 0] *= torch.sin(gamma).abs()
    link_lengths[..., 1][link_type == 0] *= torch.cos(gamma).abs()

    link_lengths[..., 0][(link_type == 1) | (link_type == 3)] = 0
    link_lengths[..., 1][(link_type == 2) | (link_type == 3)] = 0
    return link_lengths


def _reject_link_length(link_length: Tensor) -> Tensor:
    """Reject nonzero link lengths shorter than 2*LINK_RADIUS and too-large first/last links."""
    rejected = ((link_length.abs() < 2 * LINK_RADIUS) & (link_length != 0)).any(
        dim=(1, 2)
    )
    if link_length.shape[1] > 2:
        rejected |= (
            torch.sqrt((link_length[:, 0, :] ** 2).sum(dim=-1))
            > 1 / link_length.shape[1]
        )
        rejected |= (
            torch.sqrt((link_length[:, -1, :] ** 2).sum(dim=-1))
            > 1 / link_length.shape[1]
        )
    return rejected


def _sample_morph(
    batch_size: int, dof: int, analytically_solvable: bool, device: torch.device
) -> Tensor:
    """
    Sample raw morphologies [batch_size, dof+1, 3] before final self-collision/manipulability rejection.
    """
    oversampling = 2 * batch_size

    if analytically_solvable:
        link_types, link_twists = _sample_analytically_solvable_link_types_and_twist(
            oversampling, dof, device
        )
        while (
            mask := _reject_link_type(link_types)
            | _reject_link_twist(link_twists, link_types)
        ).any():
            link_types[mask], link_twists[mask] = (
                _sample_analytically_solvable_link_types_and_twist(
                    mask.sum().item(),
                    dof,
                    device,
                )
            )
    else:
        link_types = _sample_link_type(oversampling, dof, device)
        while (mask := _reject_link_type(link_types)).any():
            link_types[mask] = _sample_link_type(mask.sum().item(), dof, device)

        link_twists = _sample_link_twist(link_types)
        while (mask := _reject_link_twist(link_twists, link_types)).any():
            link_twists[mask] = _sample_link_twist(link_types[mask])

    link_lengths = _sample_link_length(link_types)
    while oversampling - (mask := _reject_link_length(link_lengths)).sum() < batch_size:
        link_lengths[mask] = _sample_link_length(link_types[mask])

    link_lengths = link_lengths[~mask][:batch_size]
    link_twists = link_twists[~mask][:batch_size]

    return torch.cat([link_twists.unsqueeze(-1), link_lengths], dim=2)


def get_joint_limits(morph: Tensor) -> Tensor:
    """
    Compute joint limits based on morphology to avoid common self-collisions.

    Args:
        morph: Tensor [..., dofp1, 3]

    Returns:
        Tensor [..., dofp1, 2], where last dim is [range, offset].
    """
    joint_limits = torch.zeros(
        *morph.shape[:-1], 2, device=morph.device, dtype=morph.dtype
    )

    extended_morph = torch.cat([torch.zeros_like(morph[..., :1, :]), morph], dim=-2)
    alpha0, a0, d0 = extended_morph[..., :-2, :].split(1, dim=-1)
    alpha1, a1, d1 = extended_morph[..., 1:-1, :].split(1, dim=-1)

    coordinate_fix = torch.eye(4, device=morph.device, dtype=morph.dtype).repeat(
        *morph.shape[:-2],
        morph.shape[-2] - 1,
        1,
        1,
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
        [
            a1,
            -d1 * torch.sin(alpha1),
            d1 * torch.cos(alpha1),
            torch.ones_like(alpha1),
        ],
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

    pre_capsule = capsules[..., 3, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 2, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 1, :]

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


def _reject_morph(morph: Tensor) -> Tensor:
    """
    Reject morphologies that seem constantly self-colliding or Jacobian-degenerate.

    Args:
        morph: Tensor [batch_size, dofp1, 3]

    Returns:
        Bool Tensor [batch_size], True means rejected.
    """
    zero_joints = torch.zeros(
        *morph.shape[:-1], 1, device=morph.device, dtype=morph.dtype
    )
    zero_poses = forward_kinematics(morph, zero_joints)
    # Use cuRobo's effective radius (LINK_RADIUS + collision_sphere_buffer) so the
    # sampler's check matches cuRobo's stricter self-collision threshold.
    CUROBO_BUFFER = 0.002
    rejected = collision_check(morph, zero_poses, radius=LINK_RADIUS + CUROBO_BUFFER)

    joint_limits = get_joint_limits(morph)

    joint_limits = joint_limits.unsqueeze(1).expand(-1, REJECTION_TEST_CONFIGS, -1, -1)
    morph_expanded = morph.unsqueeze(1).expand(-1, REJECTION_TEST_CONFIGS, -1, -1)

    joints = (
        torch.rand(*joint_limits.shape[:-1], 1, device=morph.device, dtype=morph.dtype)
        * joint_limits[..., 0:1]
    )
    joints = joints + joint_limits[..., 1:2]

    poses = forward_kinematics(morph_expanded, joints)
    rejected |= collision_check(morph_expanded, poses).all(dim=1)

    jacobian = geometric_jacobian(poses)
    rejected |= (yoshikawa_manipulability(jacobian, soft=True) < 1e-4).all(dim=1)

    return rejected


def sample_morph(
    num_robots: int,
    dof: int,
    analytically_solvable: bool = False,
    device: torch.device | str = torch.device("cpu"),
) -> Tensor:
    """
    Sample valid morphologies encoded as modified DH parameters [alpha, a, d].

    Args:
        num_robots: number of morphologies to sample.
        dof: number of revolute DOFs. In our use case, dof=6.
        analytically_solvable: whether to bias toward analytical-IK-friendly structures.
        device: torch device.

    Returns:
        Tensor [num_robots, dof+1, 3].
    """
    if num_robots <= 0:
        raise ValueError("num_robots must be positive.")
    if dof <= 0:
        raise ValueError("dof must be positive.")
    if dof + 1 > len(PAIR_COMBINATIONS):
        raise ValueError(
            f"This standalone sampler supports up to dof={len(PAIR_COMBINATIONS) - 1}."
        )

    device = torch.device(device)

    num_samples = max(1, int(num_robots * 1.5))
    morph = _sample_morph(num_samples, dof, analytically_solvable, device)

    while num_samples - (mask := _reject_morph(morph)).sum() < num_robots:
        morph[mask] = _sample_morph(
            mask.sum().item(), dof, analytically_solvable, device
        )

    return morph[~mask][:num_robots]


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------


@torch.inference_mode()
def sample_dof6_initial_morphologies(
    num_initial_samples: int,
    seed: int | None = 0,
    device: str | torch.device | None = None,
    analytically_solvable: bool = False,
    as_list: bool = False,
) -> Tensor | list[Tensor]:
    """
    Sample valid DOF=6 morphologies as initial parameters for optimization.

    Args:
        num_initial_samples:
            Number of initial morphologies to sample.
        seed:
            Random seed. Use None if you do not want to reset randomness.
        device:
            "cpu", "cuda", or None. If None, uses cuda if available.
        analytically_solvable:
            Keep False for the same general morphology distribution used by mini inference config.
            True biases samples toward analytical-IK-friendly structures.
        cpu_output:
            If True, move output back to CPU before returning.
        as_list:
            If False, return Tensor [N, 7, 3].
            If True, return List[Tensor], each Tensor has shape [7, 3].

    Returns:
        Tensor [num_initial_samples, 7, 3], or list of [7, 3] tensors.
    """
    if num_initial_samples <= 0:
        raise ValueError("num_initial_samples must be positive.")

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    morphologies = sample_morph(
        num_robots=num_initial_samples,
        dof=6,
        analytically_solvable=analytically_solvable,
        device=device,
    ).float()

    if as_list:
        return [morphologies[i].clone() for i in range(morphologies.shape[0])]

    return morphologies


if __name__ == "__main__":
    morphs = sample_dof6_initial_morphologies(1, seed=0, device="cpu")
    print("morphs.shape =", morphs.shape)
    print("first morphology M [7, 3] =")
    print(morphs[0])
