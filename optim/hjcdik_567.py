"""
HJCD-IK morphology optimisation with selectable DOF (5, 6, or 7).
Original paper:
  https://arxiv.org/abs/2510.07514

Only change from hjcdik_baseline_6DOF.py: reads optimization_parameters["dof"]
and, if it differs from the input morphology's DOF, samples a fresh random
morphology of the requested DOF before running the same optimisation loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from interface import Morphology, Task
from util.kinematics import forward_kinematics
from util.self_collision import get_capsules
from task.morphology_sampler import get_joint_limits, sample_morph
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import run_optimization_validation

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_candidate_dofs(value) -> list[int]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        dofs = [int(p) for p in parts]
    else:
        dofs = [int(d) for d in value]
    if not dofs:
        raise ValueError("candidate_dofs must not be empty.")
    if any(d <= 0 for d in dofs):
        raise ValueError(f"candidate_dofs must be positive, got {dofs}.")
    return sorted(set(dofs))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPS = 1e-4

# ---------------------------------------------------------------------------
# Morphology preprocessing
# ---------------------------------------------------------------------------


class SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, param: Tensor, threshold: float) -> Tensor:
        mask = (param.abs() >= threshold).float()
        return param * mask

    @staticmethod
    def backward(_ctx, grad_output: Tensor):
        return grad_output, None


class BatchedNormaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param: Tensor) -> Tensor:
        l2_norm = torch.hypot(param[..., 0:1], param[..., 1:2])
        norm = l2_norm.sum(dim=-2, keepdim=True).clamp_min(1e-12)
        ctx.save_for_backward(param, l2_norm, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        param, l2_norm, norm = ctx.saved_tensors
        safe_l2 = l2_norm.clamp_min(1e-12)
        chain = torch.where(
            (param.abs() > EPS).any(dim=-1, keepdim=True),
            param / safe_l2,
            torch.zeros_like(param),
        )
        dot = (grad_output * param).sum(dim=(-2, -1), keepdim=True)
        return (grad_output * norm - chain * dot) / norm.pow(2)


def _preprocess_lengths(lengths: Tensor, link_radius: float) -> tuple[Tensor, Tensor]:
    threshold = 2.0 * link_radius
    norm_lengths = BatchedNormaliser.apply(lengths)
    squashed = SquasherSTE.apply(norm_lengths, threshold)
    return BatchedNormaliser.apply(squashed), norm_lengths


def _build_morphology_tensors(
    alpha: Tensor,
    lengths: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor]:
    processed_lengths, _ = _preprocess_lengths(lengths, link_radius)
    raw_morphologies = torch.cat([alpha, lengths], dim=-1)
    processed_morphologies = torch.cat([alpha, processed_lengths], dim=-1)
    return raw_morphologies, processed_morphologies


def _goal_poses_in_robot_frame(
    task: Task, device: torch.device, dtype: torch.dtype
) -> Tensor:
    # Robot base frame = world frame (identity base pose).
    return task.goal_poses.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# SE3 distance
# ---------------------------------------------------------------------------


def _rotation_error(reached: Tensor, target: Tensor) -> Tensor:
    r_rel = reached[..., :3, :3] @ target[..., :3, :3].transpose(-1, -2)
    trace = r_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    skew = torch.stack(
        [
            r_rel[..., 2, 1] - r_rel[..., 1, 2],
            r_rel[..., 0, 2] - r_rel[..., 2, 0],
            r_rel[..., 1, 0] - r_rel[..., 0, 1],
        ],
        dim=-1,
    )
    sin_angle = 0.5 * torch.linalg.norm(skew, dim=-1)
    cos_angle = (0.5 * (trace - 1.0)).clamp(-1.0, 1.0)
    return torch.atan2(sin_angle, cos_angle)


def _pose_se3_distance(reached: Tensor, target: Tensor) -> Tensor:
    dtype = reached.dtype
    target = target.to(dtype=dtype)
    pos_err_sq = ((reached[..., :3, 3] - target[..., :3, 3]) ** 2).sum(dim=-1)
    rot_err = _rotation_error(reached, target)
    return (pos_err_sq / 8.0 + rot_err**2 / (2.0 * torch.pi**2) + 1e-12).sqrt()


# ---------------------------------------------------------------------------
# Self-collision critical distance
# ---------------------------------------------------------------------------


def _signed_distance_capsule_capsule(
    s1: Tensor,
    e1: Tensor,
    r1: float,
    s2: Tensor,
    e2: Tensor,
    r2: float,
) -> Tensor:
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
    return ((c1 - c2) ** 2).sum(dim=-1) - (r1 + r2) ** 2


def _signed_distance_capsule_ball(
    s1: Tensor,
    e1: Tensor,
    r1: float,
    s2: Tensor,
    r2: float,
) -> Tensor:
    l1 = e1 - s1
    v = s2 - s1
    t = torch.clamp(
        (v * l1).sum(dim=-1, keepdim=True)
        / ((l1 * l1).sum(dim=-1, keepdim=True) + 1e-10),
        0.0,
        1.0,
    )
    closest = s1 + t * l1
    return ((closest - s2) ** 2).sum(dim=-1) - (r1 + r2) ** 2


def _collision_critical_distance(mdh: Tensor, poses: Tensor, radius: float) -> Tensor:
    *batch_shape, dofp1, _ = mdh.shape
    s_all, e_all = get_capsules(mdh, poses)
    i_idx, j_idx = torch.triu_indices(2 * dofp1, 2 * dofp1, offset=2, device=mdh.device)
    num_pairs = i_idx.shape[0]
    s1 = s_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    e1 = e_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    s2 = s_all[..., j_idx, :].reshape(-1, num_pairs, 3)
    e2 = e_all[..., j_idx, :].reshape(-1, num_pairs, 3)
    active = (torch.norm(s1 - e1, dim=-1) > EPS) & (torch.norm(s2 - e2, dim=-1) > EPS)
    active &= torch.norm(e1 - s2, dim=-1) > EPS
    distances = _signed_distance_capsule_capsule(s1, e1, radius, s2, e2, radius)
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
    dist_end = _signed_distance_capsule_ball(s_end, e_end, radius, c_end, radius)
    active_end = torch.norm(e_end - s_end, dim=-1) > EPS
    cum = torch.cumsum(active_end, dim=-1)
    active_end &= (cum == 2) | (cum == (cum[..., -1:] - 1))
    crit = (
        torch.where(active, distances, torch.ones_like(distances))
        .min(dim=-1)
        .values.reshape(batch_shape)
    )
    crit_end = (
        torch.where(active_end, dist_end, torch.ones_like(dist_end)).min(dim=-1).values
    )
    return torch.minimum(crit, crit_end)


# ---------------------------------------------------------------------------
# Validation wrapper
# ---------------------------------------------------------------------------


def _run_validation(
    *,
    processed_morphology: Tensor,
    morph: Morphology,
    task: Task,
    scene,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
) -> dict:
    gen = torch.Generator(device=device)
    gen.manual_seed(random_seed)
    result = run_optimization_validation(
        processed_morphology=processed_morphology.detach(),
        morph=morph,
        task=task,
        scene=scene,
        device=device,
        percentage_poses=percentage_poses,
        number_random_seed=number_random_seed,
        pose_sampling_generator=gen,
    )
    result["ik_success_pose_rate"] = (
        (result["best_se3_dist_per_pose"] <= EPS).float().mean()
    )
    return result


# ---------------------------------------------------------------------------
# Parameters


def _hjcdik_parameters(optimization_parameters: dict) -> dict[str, Any]:
    lr = float(optimization_parameters.get("learning_rate", 0.01))
    return {
        "num_iterations": int(optimization_parameters.get("num_iterations", 500)),
        "learning_rate_length": float(
            optimization_parameters.get("learning_rate_length", lr)
        ),
        "n_ccd_seeds": int(optimization_parameters.get("n_ccd_seeds", 32)),
        "n_ccd_iters": int(optimization_parameters.get("n_ccd_iters", 20)),
        "n_lm_top_k": int(optimization_parameters.get("n_lm_top_k", 8)),
        "n_lm_iters": int(optimization_parameters.get("n_lm_iters", 10)),
        "lambda_lm": float(optimization_parameters.get("lambda_lm", 0.01)),
        "trust_radius": float(optimization_parameters.get("trust_radius", 0.5)),
        "w_rot": float(optimization_parameters.get("w_rot", 0.5)),
        "collision_weight": float(optimization_parameters.get("collision_weight", 1.0)),
        "collision_margin": float(optimization_parameters.get("collision_margin", 0.0)),
        "zero_link_weight": float(optimization_parameters.get("zero_link_weight", 1.0)),
        "success_eps": float(optimization_parameters.get("success_eps", EPS)),
        "softmin_temperature": float(
            optimization_parameters.get("softmin_temperature", 0.01)
        ),
    }


# FK helpers


def _fk_batch(morphology_batch: Tensor, q: Tensor) -> Tensor:
    """Batched FK from joint angles (without the fixed last joint).

    morphology_batch : [B, seq_len, 3]
    q               : [B, n_dofs]        (n_dofs = seq_len - 1)
    Returns poses   : [B, seq_len, 4, 4]
    """
    B = q.shape[0]
    zeros = torch.zeros(B, 1, device=q.device, dtype=q.dtype)
    q_full = torch.cat([q, zeros], dim=-1).unsqueeze(-1)  # [B, seq_len, 1]
    return forward_kinematics(morphology_batch, q_full)  # [B, seq_len, 4, 4]


def _se3_residual(ee_poses: Tensor, target_poses: Tensor) -> Tensor:
    """SE3 residual vector [B, 6] = [pos_err(3), rot_err(3)].

    rot_err is the rotation vector extracted from R_target @ R_ee^T,
    approximated as (R_err − R_err^T) / 2 (skew-symmetric part / 2).
    """
    pos_err = target_poses[:, :3, 3] - ee_poses[:, :3, 3]  # [B, 3]
    R_err = target_poses[:, :3, :3] @ ee_poses[:, :3, :3].transpose(-1, -2)
    rot_err = (
        torch.stack(
            [
                R_err[:, 2, 1] - R_err[:, 1, 2],
                R_err[:, 0, 2] - R_err[:, 2, 0],
                R_err[:, 1, 0] - R_err[:, 0, 1],
            ],
            dim=-1,
        )
        / 2.0
    )  # [B, 3]
    return torch.cat([pos_err, rot_err], dim=-1)  # [B, 6]


def _analytical_jacobian(poses: Tensor, n_dofs: int) -> Tensor:
    """Analytical Jacobian from FK poses.

    For MDH revolute joint i:
      rotation axis  z_i = poses[:, i, :3, 2]   (z-col of frame i, invariant under Rot_z)
      axis origin    P_i = poses[:, i, :3, 3]   (frame i origin, lies on z_i axis)
      J_v_i = z_i × (P_ee − P_i)
      J_w_i = z_i

    poses : [B, seq_len, 4, 4]
    Returns J : [B, 6, n_dofs]
    """
    ee_pos = poses[:, -1, :3, 3]  # [B, 3]
    z = poses[:, :n_dofs, :3, 2]  # [B, n_dofs, 3]
    p = poses[:, :n_dofs, :3, 3]  # [B, n_dofs, 3]
    Jv = torch.linalg.cross(z, ee_pos[:, None] - p)  # [B, n_dofs, 3]
    J = torch.cat([Jv, z], dim=-1).transpose(1, 2)  # [B, 6, n_dofs]
    return J


# Joint limit helpers


def _compute_joint_bounds(morphology: Tensor, n_dofs: int) -> tuple[Tensor, Tensor]:
    """Compute per-joint lower/upper bounds from morphology self-collision limits.

    get_joint_limits returns [n_links, 2] where last dim = [range, offset].
    Allowed interval: [offset, offset + range].
    Only the first n_dofs rows are used (last row = fixed EE joint, range=0).
    """
    limits = get_joint_limits(morphology)  # [n_links, 2]
    offset = limits[:n_dofs, 1]  # [n_dofs]
    rng = limits[:n_dofs, 0]  # [n_dofs]
    return offset, offset + rng  # q_lo, q_hi


# Stage 1: PO-CCD


def _ccd_joint_update(
    poses: Tensor,
    q: Tensor,
    target_pos: Tensor,
    target_R: Tensor,
    j: int,
    delta: float,
    w_rot: float,
) -> Tensor:
    """Compute and apply CCD update for joint j across all seeds.

    Implements the orientation-aware CCD update from the paper:
      Δθ_j(pos) = sign(cross(û_proj, v̂_proj) · r̂_j) * arccos(û_proj · v̂_proj)
      Δθ_j(rot) = δ(k) * sign(â · r̂_j) * φ

    poses     : [B, seq_len, 4, 4]   current FK poses
    q         : [B, n_dofs]           current joint angles (modified in-place)
    target_pos: [B, 3]
    target_R  : [B, 3, 3]
    j         : joint index
    delta     : annealing factor for orientation component
    w_rot     : weight for orientation update relative to position update
    Returns updated q [B, n_dofs].
    """
    r_j = poses[:, j, :3, 2]  # [B, 3] rotation axis
    p_j = poses[:, j, :3, 3]  # [B, 3] joint origin
    ee_pos = poses[:, -1, :3, 3]  # [B, 3]

    # --- Position CCD (Eq. 1–3 in paper) ---
    u = ee_pos - p_j  # [B, 3]
    v = target_pos - p_j  # [B, 3]

    rr = (r_j * r_j).sum(-1, keepdim=True).clamp(min=1e-8)
    u_proj = u - (u * r_j).sum(-1, keepdim=True) * r_j / rr  # project ⊥ r_j
    v_proj = v - (v * r_j).sum(-1, keepdim=True) * r_j / rr

    u_n = F.normalize(u_proj, dim=-1, eps=1e-8)
    v_n = F.normalize(v_proj, dim=-1, eps=1e-8)

    cos_ang = (u_n * v_n).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    cross_uv = torch.cross(u_n, v_n, dim=-1)
    sign_pos = torch.sign((cross_uv * r_j).sum(-1))
    dtheta_pos = sign_pos * torch.acos(cos_ang)  # [B]

    # --- Orientation CCD (Eq. 4–6 in paper) ---
    ee_R = poses[:, -1, :3, :3]  # [B, 3, 3]
    R_err = target_R @ ee_R.transpose(-1, -2)  # [B, 3, 3]
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)  # [B]
    phi = torch.acos(((trace - 1.0) / 2.0).clamp(-1 + 1e-7, 1 - 1e-7))
    sin_phi = phi.sin().clamp(min=1e-8)
    rot_axis = torch.stack(
        [
            R_err[:, 2, 1] - R_err[:, 1, 2],
            R_err[:, 0, 2] - R_err[:, 2, 0],
            R_err[:, 1, 0] - R_err[:, 0, 1],
        ],
        dim=-1,
    ) / (2.0 * sin_phi.unsqueeze(-1))  # [B, 3]
    sign_rot = torch.sign((rot_axis * r_j).sum(-1))
    dtheta_rot = delta * sign_rot * phi  # [B]

    dtheta = dtheta_pos + w_rot * dtheta_rot
    q = q.clone()
    q[:, j] = q[:, j] + dtheta
    return q


@torch.no_grad()
def _po_ccd(
    morphology_batch: Tensor,
    target_poses_batch: Tensor,
    q: Tensor,
    n_iters: int,
    w_rot: float,
    q_lo: Tensor,
    q_hi: Tensor,
) -> Tensor:
    """PO-CCD: Stage 1 of HJCD-IK.

    morphology_batch : [B, seq_len, 3]   expanded morphology (same for all seeds)
    target_poses_batch: [B, 4, 4]        target per seed
    q               : [B, n_dofs]        initial joint angles
    q_lo / q_hi     : [n_dofs]           joint limits
    Returns refined q [B, n_dofs].
    """
    n_dofs = q.shape[-1]
    target_pos = target_poses_batch[:, :3, 3]
    target_R = target_poses_batch[:, :3, :3]

    for k in range(n_iters):
        delta = max(0.05, 1.0 * (0.9**k))  # annealing (paper §III-A)
        poses = _fk_batch(morphology_batch, q)  # [B, seq_len, 4, 4]
        for j in range(n_dofs):
            q = _ccd_joint_update(poses, q, target_pos, target_R, j, delta, w_rot)
            q[:, j] = q[:, j].clamp(q_lo[j].item(), q_hi[j].item())
            # Recompute FK so next joint sees the updated configuration (true CCD order).
            poses = _fk_batch(morphology_batch, q)

    return q


# Stage 2: PJ-IK (LM polishing)


@torch.no_grad()
def _lm_step(
    morphology_batch: Tensor,
    target_poses_batch: Tensor,
    q: Tensor,
    lambda_lm: float,
    trust_R: float,
    q_lo: Tensor,
    q_hi: Tensor,
) -> Tensor:
    """One Levenberg-Marquardt step with trust-region and line search.

    Three-tier fallback (paper §III-B):
      1. Primary  : LM normal equations → trust-region clip → line search
      2. Gradient : steepest descent step if LM direction has no improvement
      3. Identity : return q unchanged

    Returns updated q [B, n_dofs].
    """
    B, n_dofs = q.shape
    poses = _fk_batch(morphology_batch, q)  # [B, seq_len, 4, 4]
    r = _se3_residual(poses[:, -1], target_poses_batch)  # [B, 6]
    res0 = (r * r).sum(-1)  # [B]

    J = _analytical_jacobian(poses, n_dofs)  # [B, 6, n_dofs]
    JtJ = J.transpose(-1, -2) @ J  # [B, n_dofs, n_dofs]
    Jtr = (J.transpose(-1, -2) @ r.unsqueeze(-1)).squeeze(-1)  # [B, n_dofs]

    # Diagonal damping matrix D = diag(JtJ)
    D = torch.diag_embed(JtJ.diagonal(dim1=-2, dim2=-1))  # [B, n_dofs, n_dofs]
    A = JtJ + lambda_lm * D  # [B, n_dofs, n_dofs]

    # --- Tier 1: LM direction ---
    try:
        delta_q_lm = torch.linalg.solve(A, Jtr)  # [B, n_dofs]
        # Trust-region clip (paper §III-B)
        norms = delta_q_lm.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        delta_q_lm = torch.where(
            norms > trust_R, delta_q_lm * trust_R / norms, delta_q_lm
        )
    except Exception:
        delta_q_lm = None

    def _try_step(dq: Tensor) -> tuple[Tensor, bool]:
        """Line search: try [1, 0.5, 0.25, 0.125]; return best q."""
        best_q = q.clone()
        best_res = res0.clone()
        improved_any = False
        for alpha in [1.0, 0.5, 0.25, 0.125]:
            # q_try = q + alpha * dq
            q_try = (q + alpha * dq).clamp(q_lo, q_hi)
            poses_try = _fk_batch(morphology_batch, q_try)
            r_try = _se3_residual(poses_try[:, -1], target_poses_batch)
            res_try = (r_try * r_try).sum(-1)  # [B]
            better = res_try < best_res
            if better.any():
                improved_any = True
                best_q = torch.where(better.unsqueeze(-1), q_try, best_q)
                best_res = torch.where(better, res_try, best_res)
        return best_q, improved_any

    if delta_q_lm is not None:
        q_new, improved = _try_step(delta_q_lm)
        if improved:
            return q_new

    # --- Tier 2: gradient descent (simplified Dogleg) ---
    delta_q_gd = Jtr / (
        JtJ.diagonal(dim1=-2, dim2=-1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
    )
    norms = delta_q_gd.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    delta_q_gd = torch.where(norms > trust_R, delta_q_gd * trust_R / norms, delta_q_gd)
    q_new, improved = _try_step(delta_q_gd)
    if improved:
        return q_new

    # --- Tier 3: no improvement ---
    return q


@torch.no_grad()
def _pj_ik(
    morphology_batch: Tensor,
    target_poses_batch: Tensor,
    q: Tensor,
    n_iters: int,
    lambda_lm: float,
    trust_R: float,
    q_lo: Tensor,
    q_hi: Tensor,
) -> Tensor:
    """PJ-IK: LM polishing stage (Stage 2 of HJCD-IK).

    morphology_batch : [B, seq_len, 3]
    target_poses_batch: [B, 4, 4]
    q               : [B, n_dofs]    top-K seeds from CCD
    q_lo / q_hi     : [n_dofs]       joint limits
    Returns refined q [B, n_dofs].
    """
    for _ in range(n_iters):
        q = _lm_step(
            morphology_batch, target_poses_batch, q, lambda_lm, trust_R, q_lo, q_hi
        )
        q = q.clamp(q_lo, q_hi)
    return q


# Full HJCD-IK solve


@torch.no_grad()
def _hjcdik_solve(
    morphology: Tensor,
    target_poses: Tensor,
    *,
    n_seeds: int,
    n_ccd_iters: int,
    n_lm_top_k: int,
    n_lm_iters: int,
    lambda_lm: float,
    trust_R: float,
    w_rot: float,
    device: torch.device,
    dtype: torch.dtype,
    random_seed: int,
) -> Tensor:
    """Full HJCD-IK solver: CCD init → LM polish → best per pose.

    morphology  : [seq_len, 3]
    target_poses: [n_poses, 4, 4]
    Returns q_best : [n_poses, seq_len, 1]  (FK-ready, last joint = 0)
    """
    n_poses = target_poses.shape[0]
    seq_len = morphology.shape[0]
    n_dofs = seq_len - 1

    gen = torch.Generator(device=device)
    gen.manual_seed(random_seed)

    # Compute joint bounds from morphology self-collision limits.
    # Apply a small inward margin so solutions never land exactly on the
    # collision boundary (which would fail check_start_feasibility).
    _LIMIT_MARGIN = 0.05  # rad
    q_lo, q_hi = _compute_joint_bounds(morphology, n_dofs)  # [n_dofs] each
    q_lo = q_lo + _LIMIT_MARGIN
    q_hi = q_hi - _LIMIT_MARGIN

    # --- Stage 1: PO-CCD on M seeds per pose ---
    # Batch dim B = n_poses × n_seeds
    B = n_poses * n_seeds
    q = (
        torch.rand(n_poses, n_seeds, n_dofs, device=device, dtype=dtype, generator=gen)
        * (q_hi - q_lo)
        + q_lo  # seeds within joint limits
        # torch.rand(n_poses, n_seeds, n_dofs, device=device, dtype=dtype, generator=gen)
        # * 2 * math.pi - math.pi
    ).reshape(B, n_dofs)

    morph_b = morphology.unsqueeze(0).expand(B, -1, -1)  # [B, seq_len, 3]
    target_b = target_poses.unsqueeze(1).expand(-1, n_seeds, -1, -1).reshape(B, 4, 4)

    q = _po_ccd(
        morph_b, target_b, q, n_iters=n_ccd_iters, w_rot=w_rot, q_lo=q_lo, q_hi=q_hi
    )

    # Compute residuals and select top-K seeds per pose
    poses_ccd = _fk_batch(morph_b, q)  # [B, seq_len, 4, 4]
    res = (_se3_residual(poses_ccd[:, -1], target_b) ** 2).sum(-1)  # [B]
    res_pp = res.reshape(n_poses, n_seeds)  # [n_poses, n_seeds]
    K = min(n_lm_top_k, n_seeds)
    _, top_idx = res_pp.topk(K, dim=-1, largest=False)  # [n_poses, K]

    q_pp = q.reshape(n_poses, n_seeds, n_dofs)
    q_top = q_pp.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, n_dofs)).reshape(
        n_poses * K, n_dofs
    )  # [n_poses*K, n_dofs]

    # --- Stage 2: PJ-IK (LM) on top-K seeds ---
    B2 = n_poses * K
    morph_b2 = morphology.unsqueeze(0).expand(B2, -1, -1)
    target_b2 = target_poses.unsqueeze(1).expand(-1, K, -1, -1).reshape(B2, 4, 4)

    q_lm = _pj_ik(
        morph_b2,
        target_b2,
        q_top,
        n_iters=n_lm_iters,
        lambda_lm=lambda_lm,
        trust_R=trust_R,
        q_lo=q_lo,
        q_hi=q_hi,
    )

    # Select best per pose
    poses_lm = _fk_batch(morph_b2, q_lm)
    res_lm = (
        (_se3_residual(poses_lm[:, -1], target_b2) ** 2).sum(-1).reshape(n_poses, K)
    )
    best_idx = res_lm.argmin(dim=-1)  # [n_poses]
    q_best = q_lm.reshape(n_poses, K, n_dofs)[
        torch.arange(n_poses, device=device), best_idx
    ]  # [n_poses, n_dofs]

    # Pack to FK format [n_poses, seq_len, 1]
    zeros = torch.zeros(n_poses, 1, device=device, dtype=dtype)
    return torch.cat([q_best, zeros], dim=-1).unsqueeze(-1)  # [n_poses, seq_len, 1]


# Morphology loss (differentiable, uses detached IK solutions)


def _morphology_loss(
    alpha: Tensor,
    lengths: Tensor,
    q_solutions: Tensor,
    target_poses_local: Tensor,
    link_radius: float,
    collision_weight: float,
    collision_margin: float,
    success_eps: float,
    zero_link_weight: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """FK loss + collision penalty + alpha-aware zero-link penalty.

    q_solutions : [n_poses, seq_len, 1]  detached IK solutions
    Returns (loss, pose_success_rate, best_se3_mean, raw_morph, proc_morph)
    """
    raw_morph, proc_morph = _build_morphology_tensors(alpha, lengths, link_radius)

    n_poses = target_poses_local.shape[0]
    morph_exp = (
        proc_morph[0].unsqueeze(0).expand(n_poses, -1, -1)
    )  # [n_poses, seq_len, 3]

    reached = forward_kinematics(morph_exp, q_solutions)  # [n_poses, seq_len, 4, 4]
    ee_poses = reached[:, -1]  # [n_poses, 4, 4]

    se3 = _pose_se3_distance(ee_poses, target_poses_local)  # [n_poses]

    critical_dist = _collision_critical_distance(
        morph_exp, reached, radius=link_radius
    )  # [n_poses]
    collision_penalty = F.relu(collision_margin - critical_dist)

    # Alpha-aware zero-link penalty: stronger when alpha=0 (identity-transform risk).
    alpha_vals = alpha[0, :, 0]  # [seq_len]
    proc_len_sq = proc_morph[0, :, 1] ** 2 + proc_morph[0, :, 2] ** 2  # [seq_len]
    alpha_scale = 1.0 + torch.exp(-(alpha_vals**2) / 0.1)
    zero_link_penalty = (alpha_scale * torch.exp(-proc_len_sq / link_radius**2)).mean()

    loss = (
        se3 + collision_weight * collision_penalty
    ).mean() + zero_link_weight * zero_link_penalty
    best_se3_mean = se3.mean()

    pose_success_rate = (
        ((se3 <= success_eps) & (critical_dist >= collision_margin)).float().mean()
    )

    return loss, pose_success_rate, best_se3_mean, raw_morph[0], proc_morph[0]


# Main entry point


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """HJCD-IK morphology optimisation baseline for 6-DOF arms.

    Uses HJCD-IK (PO-CCD + LM) as the IK oracle at each morphology step.
    Joint solutions are detached; morphology gradients flow through a
    differentiable FK re-evaluation. Alpha is kept fixed; [a, d] are
    optimised with AdamW. Uses cuRobo for periodic and final validation.
    """
    params = _hjcdik_parameters(optimization_parameters)
    logging = bool(optimization_parameters.get("logging", True))
    eval_interval = int(optimization_parameters.get("eval_interval", 20))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1))

    device = morph.params.device
    dtype = morph.params.dtype

    # ── DOF selection ────────────────────────────────────────────────────────
    candidate_dofs_raw = optimization_parameters.get("candidate_dofs", None)
    morph_dof = morph.n_links - 1

    if candidate_dofs_raw is not None:
        dof = _resolve_candidate_dofs(candidate_dofs_raw)[0]
    else:
        dof = morph_dof

    if dof != morph_dof:
        if logging:
            print(
                f"[HJCD-IK] DOF mismatch: morph has {morph_dof}-DOF, "
                f"sampling fresh {dof}-DOF morphology."
            )
        torch.manual_seed(random_seed)
        sampled_params = sample_morph(
            1, dof, analytically_solvable=False, device=device
        )[0]
        working_morph = Morphology(params=sampled_params, link_radius=morph.link_radius)
    else:
        working_morph = morph

    if logging:
        print(f"[Info] Starting HJCD-IK {dof}-DOF baseline on device {device}.")
        print(
            f"[Info] n_ccd_seeds={params['n_ccd_seeds']}, n_ccd_iters={params['n_ccd_iters']}, "
            f"n_lm_top_k={params['n_lm_top_k']}, n_lm_iters={params['n_lm_iters']}"
        )

    target_poses_local = _goal_poses_in_robot_frame(task, device, dtype)
    scene = None
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    if logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    initial = working_morph.params.detach().clone()
    alpha = initial[:, 0:1].unsqueeze(0)  # [1, seq_len, 1]
    lengths = (
        initial[:, 1:].unsqueeze(0).clone().requires_grad_(True)
    )  # [1, seq_len, 2]

    morph_optimizer = torch.optim.AdamW(
        [lengths], lr=params["learning_rate_length"], weight_decay=0.0
    )

    loss = torch.tensor(0.0, device=device)
    pose_success_rate = torch.tensor(0.0, device=device)
    best_se3_mean = torch.tensor(0.0, device=device)

    try:
        progress = tqdm(
            range(params["num_iterations"]),
            desc="HJCD-IK baseline",
            disable=not logging,
            dynamic_ncols=True,
        )

        for update_idx in progress:
            morph_optimizer.zero_grad(set_to_none=True)

            # Current processed morphology (detached for IK solve)
            with torch.no_grad():
                _, proc_morph_ik = _build_morphology_tensors(
                    alpha, lengths.detach(), working_morph.link_radius
                )
                proc_morph_single = proc_morph_ik[0]  # [seq_len, 3]

            # --- HJCD-IK: solve IK for current morphology ---
            q_solutions = _hjcdik_solve(
                proc_morph_single,
                target_poses_local,
                n_seeds=params["n_ccd_seeds"],
                n_ccd_iters=params["n_ccd_iters"],
                n_lm_top_k=params["n_lm_top_k"],
                n_lm_iters=params["n_lm_iters"],
                lambda_lm=params["lambda_lm"],
                trust_R=params["trust_radius"],
                w_rot=params["w_rot"],
                device=device,
                dtype=dtype,
                random_seed=random_seed + update_idx,
            )  # [n_poses, seq_len, 1]

            # --- Morphology gradient via differentiable FK ---
            (
                loss,
                pose_success_rate,
                best_se3_mean,
                raw_morph_t,
                proc_morph_t,
            ) = _morphology_loss(
                alpha=alpha,
                lengths=lengths,
                q_solutions=q_solutions.detach(),
                target_poses_local=target_poses_local,
                link_radius=working_morph.link_radius,
                collision_weight=params["collision_weight"],
                collision_margin=params["collision_margin"],
                success_eps=params["success_eps"],
                zero_link_weight=params["zero_link_weight"],
            )

            # Periodic cuRobo validation for logging
            validation_data = None
            if eval_interval > 0 and update_idx % eval_interval == 0 and logging:
                validation_data = _run_validation(
                    processed_morphology=proc_morph_t.detach(),
                    morph=working_morph,
                    task=task,
                    scene=scene,
                    device=device,
                    percentage_poses=percentage_poses,
                    number_random_seed=number_random_seed,
                    random_seed=random_seed,
                )
                val_se3 = validation_data["best_se3_dist_mean"].detach().cpu().item()
                val_ik = validation_data["ik_success_pose_rate"].detach().cpu().item()
                tqdm.write(
                    f"[Iter {update_idx:>4}/{params['num_iterations']}] "
                    f"loss={loss.item():.6f}, "
                    f"hjcd_success={pose_success_rate.item() * 100:.2f}%, "
                    f"hjcd_se3={best_se3_mean.item():.6f}, "
                    f"val_se3={val_se3:.6f}, "
                    f"val_ik_success={val_ik * 100:.2f}%"
                )

            csv_logger.log_iteration(
                iteration=update_idx,
                loss=loss.detach(),
                reachability_probability=pose_success_rate.detach(),
                raw_morphology=raw_morph_t.detach(),
                processed_morphology=proc_morph_t.detach(),
                validation_data=validation_data,
            )

            loss.backward()
            morph_optimizer.step()
            with torch.no_grad():
                lengths[0, -1, 1].clamp_(min=0.0)

            if logging:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    success=f"{pose_success_rate.item():.3f}",
                    se3=f"{best_se3_mean.item():.4f}",
                )

        # --- Final validation ---
        with torch.no_grad():
            _, final_proc = _build_morphology_tensors(
                alpha, lengths.detach(), working_morph.link_radius
            )

        final_validation = _run_validation(
            processed_morphology=final_proc[0].detach(),
            morph=working_morph,
            task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            random_seed=random_seed,
        )

        with torch.no_grad():
            _, _, _, raw_f, proc_f = _morphology_loss(
                alpha=alpha,
                lengths=lengths.detach(),
                q_solutions=q_solutions.detach(),
                target_poses_local=target_poses_local,
                link_radius=working_morph.link_radius,
                collision_weight=params["collision_weight"],
                collision_margin=params["collision_margin"],
                success_eps=params["success_eps"],
                zero_link_weight=params["zero_link_weight"],
            )

        csv_logger.log_iteration(
            iteration=params["num_iterations"],
            loss=loss.detach(),
            reachability_probability=pose_success_rate.detach(),
            raw_morphology=raw_f.detach(),
            processed_morphology=proc_f.detach(),
            validation_data=final_validation,
        )

        final_se3 = final_validation["best_se3_dist_mean"].detach().cpu().item()
        final_ik = final_validation["ik_success_pose_rate"].detach().cpu().item()

        print(
            f"[Final HJCD-IK baseline] "
            f"loss={loss.item():.6f}, "
            f"hjcd_success={pose_success_rate.item() * 100:.2f}%, "
            f"hjcd_se3={best_se3_mean.item():.6f}, "
            f"final_val_se3={final_se3:.6f}, "
            f"ik_success_pose_rate={final_ik * 100:.2f}%"
        )

        return (
            Morphology(params=proc_f.detach(), link_radius=working_morph.link_radius),
            csv_logger.csv_path,
        )

    finally:
        csv_logger.close()
