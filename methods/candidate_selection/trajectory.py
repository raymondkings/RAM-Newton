# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
# Discrete-alpha candidate search with alternating morphology/trajectory
# optimization per candidate.
#
# This keeps the candidate-selection shell from candidate_selection/static.py:
#   - alpha candidate generation
#   - fixed-alpha initial morphology sampling and cache
#   - post-optimization distribution filter
#   - final-link d filter
#   - top-probability candidate validation
#   - final pick by validation success rate and the existing tie heuristic
#
# The per-candidate optimizer is replaced with the alternating NRM trajectory
# logic from methods/nrm_gradient/trajectory.py.  There is no per-candidate early stopping.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from core import Morphology, Task
from logutils.csv_logger import (
    InternalOptimizationCSVLogger,
)
from logutils.timing import OptimizationTimer
from methods._nrm_common import (
    _CHECKPOINT_PATH,
    EPS,
    _build_morphology_tensors,
    _rotation_angle_between,
    _se3_to_vector,
    _vector_to_se3,
)
from methods.candidate_selection._common import (
    TOP_PROBABILITY_FRACTION,
    _generate_initial_candidates,
    _parse_candidate_search_params,
    _postfilter_dof_group,
    _select_and_log_final_candidates,
    _setup_search_runtime,
)
from methods.nrm_model import MLP

num_steps = 20
num_iteratives = 50

MIN_WALL_CLEARANCE = 0.025
TRAJECTORY_REACHABILITY_WEIGHT = 1.0
TRAJECTORY_SMOOTHNESS_WEIGHT = 0.2
# adjacent poses heuristic
TRAJECTORY_DISTANCE_STEP_VARIANCE_WEIGHT = 6.0
TRAJECTORY_ROTATION_STEP_VARIANCE_WEIGHT = 2.0
POSITION_DEVIATION_WEIGHT = 0
ROTATION_DEVIATION_WEIGHT = 0
# Encourage the half of the points to be on the same side with start/goal.
TRAJECTORY_ENDPOINT_SIDE_WEIGHT = 0
WALL_CLEARANCE_WEIGHT = 500.0
# encourage to away from wall, bigger, more encourage
WALL_REPULSION_WEIGHT = 0.005
# decaying of the weight for the poses to be far from wall, bigger, slower decay
WALL_REPULSION_DISTANCE = 0.0005

# ----------------------------- hard-coded knobs -----------------------------
# Shared knobs (candidate DOFs, alpha-candidate count, batch sizes, zero-alpha
# run exclusion, top-probability fraction, checkpoint path) live in _common.

CANDIDATE_DOF: str | int | tuple[int, ...] = 7
MORPHOLOGY_FINAL_D_PENALTY_WEIGHT = 100.0

TRAJECTORY_INTERNAL_LOG_FIELDNAMES = [
    "dof",
    "iteration",
    "outer_iteration",
    "phase",
    "phase_iteration",
    "mean_loss",
    "mean_nrm_prob",
    "mean_morphology_loss",
    "mean_bce_loss",
    "mean_final_d_penalty",
    "mean_trajectory_loss",
    "mean_reachability_loss",
    "mean_smoothness_loss",
    "mean_distance_step_variance_loss",
    "mean_rotation_step_variance_loss",
    "mean_position_deviation_loss",
    "mean_rotation_deviation_loss",
    "mean_endpoint_side_loss",
    "mean_wall_loss",
    "mean_wall_repulsion_loss",
]

_MORPHOLOGY_LOSS_STAT_KEYS = [
    "bce_loss",
    "final_d_penalty",
]

_TRAJECTORY_LOSS_STAT_KEYS = [
    "reachability_loss",
    "smoothness_loss",
    "distance_step_variance_loss",
    "rotation_step_variance_loss",
    "position_deviation_loss",
    "rotation_deviation_loss",
    "endpoint_side_loss",
    "wall_loss",
    "wall_repulsion_loss",
]

# Per-candidate final loss log, written once for every candidate that survives
# post-optimization distribution + final-d filtering (not just the top-k).
ALL_CANDIDATES_LOG_FIELDNAMES = [
    "dof",
    "loss",
    "nrm_prob",
    "morphology_loss",
    *_MORPHOLOGY_LOSS_STAT_KEYS,
    *_TRAJECTORY_LOSS_STAT_KEYS,
]


# ------------------------------- SE(3) helpers ------------------------------


def _trajectory_from_intermediate(
    start_vec: Tensor,
    intermediate_vec: Tensor,
    goal_vec: Tensor,
) -> tuple[Tensor, Tensor]:
    num_candidates = intermediate_vec.shape[0]
    start = start_vec.view(1, 1, -1).expand(num_candidates, 1, -1)
    goal = goal_vec.view(1, 1, -1).expand(num_candidates, 1, -1)
    raw_vec = torch.cat([start, intermediate_vec, goal], dim=1)
    poses = _vector_to_se3(raw_vec)
    task_vec = _se3_to_vector(poses)
    return task_vec, poses


def _wrapped_angle_difference(angle: Tensor) -> Tensor:
    diff = angle[:, 1:] - angle[:, :-1]
    return torch.atan2(torch.sin(diff), torch.cos(diff))


# --------------------------- morphology/NRM loss ----------------------------


def _morphology_loss_and_prob_batched(
    *,
    model: MLP,
    alpha_candidates: Tensor,
    length_candidates: Tensor,
    task_vec: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Return morphology-phase loss/prob for each candidate, plus a breakdown
    of loss into its individual terms (see _MORPHOLOGY_LOSS_STAT_KEYS)."""
    num_candidates = alpha_candidates.shape[0]
    _, processed_morphologies = _build_morphology_tensors(
        alpha_candidates,
        length_candidates,
        link_radius,
    )

    if task_vec.ndim == 2:
        pose_vec = task_vec.unsqueeze(0).expand(num_candidates, -1, -1)
    else:
        pose_vec = task_vec

    num_poses = pose_vec.shape[1]
    bmorph = processed_morphologies.unsqueeze(1).expand(
        num_candidates,
        num_poses,
        -1,
        -1,
    )

    logit = model(
        bmorph.reshape(num_candidates * num_poses, processed_morphologies.shape[-2], 3),
        pose_vec.reshape(num_candidates * num_poses, 9),
    ).reshape(num_candidates, num_poses)

    target = torch.ones_like(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none").mean(
        dim=1
    )
    final_d_penalty = MORPHOLOGY_FINAL_D_PENALTY_WEIGHT * torch.relu(
        -processed_morphologies[:, -1, 2]
    )
    loss = bce + final_d_penalty
    prob = torch.sigmoid(logit).mean(dim=1)

    raw_morphologies = torch.cat([alpha_candidates, length_candidates], dim=-1)
    stats = {
        "bce_loss": bce.detach(),
        "final_d_penalty": final_d_penalty.detach(),
    }
    return loss, prob, raw_morphologies, processed_morphologies, stats


def _trajectory_reachability_loss_and_prob(
    *,
    model: MLP,
    processed_morphologies: Tensor,
    task_vec: Tensor,
) -> tuple[Tensor, Tensor]:
    num_candidates, num_poses = task_vec.shape[:2]
    bmorph = processed_morphologies.unsqueeze(1).expand(
        num_candidates,
        num_poses,
        -1,
        -1,
    )
    logit = model(
        bmorph.reshape(num_candidates * num_poses, processed_morphologies.shape[-2], 3),
        task_vec.reshape(num_candidates * num_poses, 9),
    ).reshape(num_candidates, num_poses)
    prob_per_pose = torch.sigmoid(logit)
    loss = F.mse_loss(torch.ones_like(prob_per_pose), prob_per_pose, reduction="none")
    return loss.mean(dim=1), prob_per_pose.mean(dim=1)


# ----------------------------- trajectory loss ------------------------------


def _trajectory_smoothness_loss_batched(poses: Tensor) -> Tensor:
    num_candidates, num_poses = poses.shape[:2]
    if num_poses <= 3:
        return poses.new_zeros(num_candidates)

    positions = poses[:, :, :3, 3]
    segments = positions[:, 1:] - positions[:, :-1]
    tangents = F.normalize(segments, dim=-1, eps=EPS)

    forward = F.normalize(positions[:, -1] - positions[:, 0], dim=-1, eps=EPS)
    world_z = torch.tensor([0.0, 0.0, 1.0], dtype=poses.dtype, device=poses.device)
    world_y = torch.tensor([0.0, 1.0, 0.0], dtype=poses.dtype, device=poses.device)
    use_z = torch.abs((forward.detach() * world_z).sum(dim=-1)) < 0.95
    seed_axis = torch.where(
        use_z.unsqueeze(-1),
        world_z.expand_as(forward),
        world_y.expand_as(forward),
    )

    lateral = seed_axis - (seed_axis * forward).sum(dim=-1, keepdim=True) * forward
    lateral = F.normalize(lateral, dim=-1, eps=EPS)
    binormal = torch.cross(forward, lateral, dim=-1)

    forward_component = (tangents * forward.unsqueeze(1)).sum(dim=-1)
    lateral_angle = torch.atan2(
        (tangents * lateral.unsqueeze(1)).sum(dim=-1),
        forward_component,
    )
    binormal_angle = torch.atan2(
        (tangents * binormal.unsqueeze(1)).sum(dim=-1),
        forward_component,
    )

    lateral_angle_diff = _wrapped_angle_difference(lateral_angle)
    binormal_angle_diff = _wrapped_angle_difference(binormal_angle)
    return torch.var(lateral_angle_diff, dim=1, unbiased=False) + torch.var(
        binormal_angle_diff,
        dim=1,
        unbiased=False,
    )


def _trajectory_step_variance_losses_batched(poses: Tensor) -> tuple[Tensor, Tensor]:
    num_candidates, num_poses = poses.shape[:2]
    if num_poses <= 2:
        zero = poses.new_zeros(num_candidates)
        return zero, zero

    position_steps = torch.linalg.vector_norm(
        poses[:, 1:, :3, 3] - poses[:, :-1, :3, 3],
        dim=-1,
    )
    rotation_steps = _rotation_angle_between(
        poses[:, :-1, :3, :3],
        poses[:, 1:, :3, :3],
    )
    return (
        torch.var(position_steps, dim=1, unbiased=False),
        torch.var(rotation_steps, dim=1, unbiased=False),
    )


def _trajectory_deviation_loss_batched(
    poses: Tensor,
    reference_poses: Tensor,
) -> tuple[Tensor, Tensor]:
    reference = reference_poses.unsqueeze(0)
    pos_loss = (poses[:, :, :3, 3] - reference[:, :, :3, 3]).pow(2).mean(dim=(1, 2))
    rot_angles = _rotation_angle_between(
        poses[:, :, :3, :3],
        reference[:, :, :3, :3],
    )
    rot_loss = rot_angles.pow(2).mean(dim=1)
    return pos_loss, rot_loss


def _trajectory_endpoint_side_loss_batched(poses: Tensor) -> Tensor:
    num_candidates, num_poses = poses.shape[:2]
    if num_poses <= 3:
        return poses.new_zeros(num_candidates)

    positions = poses[:, :, :3, 3]
    intermediate_positions = positions[:, 1:-1]
    pose_indices = torch.arange(
        1,
        num_poses - 1,
        dtype=poses.dtype,
        device=poses.device,
    )
    center_index = poses.new_tensor((num_poses - 1) / 2.0)
    left_span = torch.clamp(center_index - 1.0, min=EPS)
    right_span = torch.clamp((num_poses - 2.0) - center_index, min=EPS)

    start_weights = torch.where(
        pose_indices < center_index,
        (center_index - pose_indices) / left_span,
        torch.zeros_like(pose_indices),
    )
    goal_weights = torch.where(
        pose_indices > center_index,
        (pose_indices - center_index) / right_span,
        torch.zeros_like(pose_indices),
    )

    start_dist_sq = (intermediate_positions - positions[:, 0:1]).pow(2).sum(dim=-1)
    goal_dist_sq = (intermediate_positions - positions[:, -1:]).pow(2).sum(dim=-1)
    loss = start_weights.unsqueeze(0) * start_dist_sq
    loss = loss + goal_weights.unsqueeze(0) * goal_dist_sq
    return loss.sum(dim=1)


def _box_signed_line_clearance_xz_batched(
    points: Tensor,
    center: Tensor,
    half_extents: Tensor,
) -> Tensor:
    point_xz = torch.stack([points[..., 0], points[..., 2]], dim=-1)
    center_xz = torch.stack([center[0], center[2]], dim=0)
    half_xz = torch.stack([half_extents[0], half_extents[2]], dim=0)

    q = torch.abs(point_xz - center_xz) - half_xz
    outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.minimum(q.max(dim=-1).values, q.new_zeros(q.shape[:-1]))
    return outside + inside


def _nearest_wall_clearance_batched(task: Task, poses: Tensor) -> Tensor | None:
    clearances = []
    points = poses[:, :, :3, 3]

    for obstacle in task.environment.obstacles:
        if getattr(obstacle, "kind", None) != "box":
            continue

        center = obstacle.center.to(device=poses.device, dtype=poses.dtype)
        half_extents = obstacle.half_extents.to(device=poses.device, dtype=poses.dtype)
        clearances.append(
            _box_signed_line_clearance_xz_batched(points, center, half_extents)
        )

    if not clearances:
        return None

    return torch.stack(clearances, dim=0).min(dim=0).values


def _wall_clearance_loss_batched(task: Task, poses: Tensor) -> tuple[Tensor, Tensor]:
    num_candidates = poses.shape[0]
    nearest_clearance = _nearest_wall_clearance_batched(task, poses)
    if nearest_clearance is None:
        return (
            poses.new_zeros(num_candidates),
            poses.new_full((num_candidates,), float("inf")),
        )

    violation = torch.relu(MIN_WALL_CLEARANCE - nearest_clearance)
    return violation.pow(2).mean(dim=1), nearest_clearance.min(dim=1).values


def _wall_repulsion_loss_batched(task: Task, poses: Tensor) -> tuple[Tensor, Tensor]:
    num_candidates = poses.shape[0]
    nearest_clearance = _nearest_wall_clearance_batched(task, poses)
    if nearest_clearance is None:
        return (
            poses.new_zeros(num_candidates),
            poses.new_full((num_candidates,), float("inf")),
        )

    if WALL_REPULSION_WEIGHT <= 0.0 or WALL_REPULSION_DISTANCE <= 0.0:
        return poses.new_zeros(num_candidates), nearest_clearance.mean(dim=1)

    outside_clearance = torch.clamp(nearest_clearance, min=0.0)
    repulsion = torch.exp(-outside_clearance / WALL_REPULSION_DISTANCE)
    return repulsion.mean(dim=1), nearest_clearance.mean(dim=1)


def _trajectory_loss_and_stats_batched(
    *,
    model: MLP,
    processed_morphologies: Tensor,
    task: Task,
    task_vec: Tensor,
    poses: Tensor,
    reference_poses: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    reachability_loss, prob = _trajectory_reachability_loss_and_prob(
        model=model,
        processed_morphologies=processed_morphologies,
        task_vec=task_vec,
    )
    smoothness_loss = _trajectory_smoothness_loss_batched(poses)
    distance_step_variance_loss, rotation_step_variance_loss = (
        _trajectory_step_variance_losses_batched(poses)
    )
    position_deviation_loss, rotation_deviation_loss = (
        _trajectory_deviation_loss_batched(poses, reference_poses)
    )
    endpoint_side_loss = _trajectory_endpoint_side_loss_batched(poses)
    wall_loss, min_wall_clearance = _wall_clearance_loss_batched(task, poses)
    wall_repulsion_loss, mean_wall_clearance = _wall_repulsion_loss_batched(
        task,
        poses,
    )

    loss = (
        TRAJECTORY_REACHABILITY_WEIGHT * reachability_loss
        + TRAJECTORY_SMOOTHNESS_WEIGHT * smoothness_loss
        + TRAJECTORY_DISTANCE_STEP_VARIANCE_WEIGHT * distance_step_variance_loss
        + TRAJECTORY_ROTATION_STEP_VARIANCE_WEIGHT * rotation_step_variance_loss
        + POSITION_DEVIATION_WEIGHT * position_deviation_loss
        + ROTATION_DEVIATION_WEIGHT * rotation_deviation_loss
        + TRAJECTORY_ENDPOINT_SIDE_WEIGHT * endpoint_side_loss
        + WALL_CLEARANCE_WEIGHT * wall_loss
        + WALL_REPULSION_WEIGHT * wall_repulsion_loss
    )

    stats = {
        "reachability_loss": reachability_loss.detach(),
        "smoothness_loss": smoothness_loss.detach(),
        "distance_step_variance_loss": distance_step_variance_loss.detach(),
        "rotation_step_variance_loss": rotation_step_variance_loss.detach(),
        "position_deviation_loss": position_deviation_loss.detach(),
        "rotation_deviation_loss": rotation_deviation_loss.detach(),
        "endpoint_side_loss": endpoint_side_loss.detach(),
        "wall_loss": wall_loss.detach(),
        "wall_repulsion_loss": wall_repulsion_loss.detach(),
        "min_wall_clearance": min_wall_clearance.detach(),
        "mean_wall_clearance": mean_wall_clearance.detach(),
    }
    return loss, prob, stats


# -------------------------- alternating optimization -------------------------


def _trajectory_step_counts() -> tuple[int, int, int]:
    num_outer = int(num_iteratives)
    morphology_steps = int(num_steps)
    trajectory_steps = int(num_steps)
    if num_outer <= 0:
        raise ValueError("num_iteratives must be positive.")
    if morphology_steps < 0 or trajectory_steps < 0:
        raise ValueError("trajectory step counts must be non-negative.")
    if morphology_steps == 0 and trajectory_steps == 0:
        raise ValueError("At least one morphology or trajectory step is required.")
    return num_outer, morphology_steps, trajectory_steps


def _optimize_all_candidates_iterative(
    *,
    model: MLP,
    task: Task,
    alpha_candidates: Tensor,
    initial_length_candidates: Tensor,
    reference_poses: Tensor,
    link_radius: float,
    learning_rate: float,
    learning_rate_pose: float,
    candidate_batch_size: int,
    logging: bool,
    internal_logger: InternalOptimizationCSVLogger | None = None,
    dof: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Optimize all candidates with alternating morphology/trajectory steps.

    Returns:
        final_lengths:
            Optimized raw [a, d] length parameters, shape [N, seq_len, 2].
        final_trajectories:
            Optimized trajectory poses, shape [N, num_poses, 4, 4].
        final_losses:
            Final combined morphology + trajectory loss per candidate.
        final_probs:
            Final mean NRM probability per candidate on its optimized trajectory.
        final_raw_morphologies:
            Final raw morphology tensors.
        final_processed_morphologies:
            Final processed morphology tensors.
        final_loss_components:
            Per-candidate breakdown of the final loss, keyed by "morphology_loss"
            plus every key in _TRAJECTORY_LOSS_STAT_KEYS, each shape [N].
    """
    num_candidates = alpha_candidates.shape[0]
    if num_candidates == 0:
        raise ValueError("No candidates to optimize.")

    num_iteratives, morphology_steps, trajectory_steps = _trajectory_step_counts()
    reference_poses = reference_poses.detach().to(
        device=alpha_candidates.device,
        dtype=alpha_candidates.dtype,
    )
    initial_task_vec = _se3_to_vector(reference_poses)
    start_vec = initial_task_vec[0].detach()
    goal_vec = initial_task_vec[-1].detach()
    intermediate_template = initial_task_vec[1:-1].detach()

    length_candidates = initial_length_candidates.detach().clone().requires_grad_(True)
    intermediate_vec = (
        intermediate_template.unsqueeze(0)
        .expand(num_candidates, -1, -1)
        .clone()
        .requires_grad_(True)
    )

    length_optimizer = torch.optim.AdamW(
        [length_candidates],
        lr=learning_rate,
        weight_decay=0.0,
    )
    trajectory_optimizer = (
        torch.optim.AdamW(
            [intermediate_vec],
            lr=learning_rate_pose,
            weight_decay=0.0,
        )
        if intermediate_vec.numel() > 0 and trajectory_steps > 0
        else None
    )

    total_updates = num_iteratives * morphology_steps
    if trajectory_optimizer is not None:
        total_updates += num_iteratives * trajectory_steps

    if logging:
        pair_batch_size = (
            min(candidate_batch_size, num_candidates) * reference_poses.shape[0]
        )
        print(
            "[Info] Candidate trajectory optimization tensors: "
            f"num_candidates={num_candidates}, "
            f"num_poses={reference_poses.shape[0]}, "
            f"candidate_batch_size={candidate_batch_size}, "
            f"max_candidate_pose_pairs_per_batch={pair_batch_size}, "
            f"num_iteratives={num_iteratives}, "
            f"morphology_steps={morphology_steps}, "
            f"trajectory_steps={trajectory_steps}"
        )

    progress = tqdm(
        total=total_updates,
        desc="candidate trajectory optimization",
        disable=not logging,
        dynamic_ncols=True,
    )
    global_iteration = 0

    for outer_idx in range(num_iteratives):
        for morph_step_idx in range(morphology_steps):
            length_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                fixed_task_vec, _ = _trajectory_from_intermediate(
                    start_vec,
                    intermediate_vec.detach(),
                    goal_vec,
                )

            current_probs = torch.empty(num_candidates, device=alpha_candidates.device)
            current_loss_sum = 0.0
            current_stat_sums = {key: 0.0 for key in _MORPHOLOGY_LOSS_STAT_KEYS}

            for start in range(0, num_candidates, candidate_batch_size):
                end = min(start + candidate_batch_size, num_candidates)
                loss_per_candidate, prob_per_candidate, _, _, stats = (
                    _morphology_loss_and_prob_batched(
                        model=model,
                        alpha_candidates=alpha_candidates[start:end],
                        length_candidates=length_candidates[start:end],
                        task_vec=fixed_task_vec[start:end],
                        link_radius=link_radius,
                    )
                )
                (loss_per_candidate.sum() / num_candidates).backward()
                current_probs[start:end] = prob_per_candidate.detach()
                current_loss_sum += float(loss_per_candidate.detach().sum().item())
                for key in _MORPHOLOGY_LOSS_STAT_KEYS:
                    current_stat_sums[key] += float(stats[key].sum().item())

            length_optimizer.step()
            mean_loss = current_loss_sum / num_candidates
            mean_prob = current_probs.mean().item()
            mean_stats = {
                f"mean_{key}": value / num_candidates
                for key, value in current_stat_sums.items()
            }
            if internal_logger is not None:
                internal_logger.log_row(
                    dof=dof,
                    iteration=global_iteration,
                    outer_iteration=outer_idx,
                    phase="morph",
                    phase_iteration=morph_step_idx,
                    mean_loss=mean_loss,
                    mean_nrm_prob=mean_prob,
                    mean_morphology_loss=mean_loss,
                    **mean_stats,
                )
            if logging:
                progress.set_postfix(
                    phase="morph",
                    outer=f"{outer_idx + 1}/{num_iteratives}",
                    loss=f"{mean_loss:.4f}",
                    prob=f"{mean_prob:.3f}",
                )
            progress.update(1)
            global_iteration += 1

        if trajectory_optimizer is None:
            continue

        for trajectory_step_idx in range(trajectory_steps):
            trajectory_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                _, processed_morphologies = _build_morphology_tensors(
                    alpha_candidates,
                    length_candidates.detach(),
                    link_radius,
                )

            current_probs = torch.empty(num_candidates, device=alpha_candidates.device)
            current_loss_sum = 0.0
            current_stat_sums = {key: 0.0 for key in _TRAJECTORY_LOSS_STAT_KEYS}
            wall_values = []

            for start in range(0, num_candidates, candidate_batch_size):
                end = min(start + candidate_batch_size, num_candidates)
                task_vec, poses = _trajectory_from_intermediate(
                    start_vec,
                    intermediate_vec[start:end],
                    goal_vec,
                )
                loss_per_candidate, prob_per_candidate, stats = (
                    _trajectory_loss_and_stats_batched(
                        model=model,
                        processed_morphologies=processed_morphologies[start:end],
                        task=task,
                        task_vec=task_vec,
                        poses=poses,
                        reference_poses=reference_poses,
                    )
                )
                (loss_per_candidate.sum() / num_candidates).backward()
                current_probs[start:end] = prob_per_candidate.detach()
                current_loss_sum += float(loss_per_candidate.detach().sum().item())
                for key in _TRAJECTORY_LOSS_STAT_KEYS:
                    current_stat_sums[key] += float(stats[key].sum().item())
                wall_values.append(stats["min_wall_clearance"])

            trajectory_optimizer.step()
            mean_loss = current_loss_sum / num_candidates
            mean_prob = current_probs.mean().item()
            mean_stats = {
                f"mean_{key}": value / num_candidates
                for key, value in current_stat_sums.items()
            }
            if internal_logger is not None:
                internal_logger.log_row(
                    dof=dof,
                    iteration=global_iteration,
                    outer_iteration=outer_idx,
                    phase="trajectory",
                    phase_iteration=trajectory_step_idx,
                    mean_loss=mean_loss,
                    mean_nrm_prob=mean_prob,
                    mean_trajectory_loss=mean_loss,
                    **mean_stats,
                )
            if logging:
                wall_tensor = torch.cat(wall_values, dim=0)
                progress.set_postfix(
                    phase="trajectory",
                    outer=f"{outer_idx + 1}/{num_iteratives}",
                    loss=f"{mean_loss:.4f}",
                    prob=f"{mean_prob:.3f}",
                    wall=f"{wall_tensor.min().item():.3f}",
                )
            progress.update(1)
            global_iteration += 1

    progress.close()

    with torch.no_grad():
        final_task_vec, final_trajectories = _trajectory_from_intermediate(
            start_vec,
            intermediate_vec.detach(),
            goal_vec,
        )
        final_raw_morphologies, final_processed_morphologies = (
            _build_morphology_tensors(
                alpha_candidates,
                length_candidates.detach(),
                link_radius,
            )
        )

        final_losses = []
        final_probs = []
        final_morphology_losses = []
        final_stats: dict[str, list[Tensor]] = {
            key: []
            for key in (*_MORPHOLOGY_LOSS_STAT_KEYS, *_TRAJECTORY_LOSS_STAT_KEYS)
        }
        for start in range(0, num_candidates, candidate_batch_size):
            end = min(start + candidate_batch_size, num_candidates)
            morph_loss, _, _, _, morph_stats = _morphology_loss_and_prob_batched(
                model=model,
                alpha_candidates=alpha_candidates[start:end],
                length_candidates=length_candidates.detach()[start:end],
                task_vec=final_task_vec[start:end],
                link_radius=link_radius,
            )
            trajectory_loss, trajectory_prob, stats = (
                _trajectory_loss_and_stats_batched(
                    model=model,
                    processed_morphologies=final_processed_morphologies[start:end],
                    task=task,
                    task_vec=final_task_vec[start:end],
                    poses=final_trajectories[start:end],
                    reference_poses=reference_poses,
                )
            )
            final_losses.append((morph_loss + trajectory_loss).detach())
            final_probs.append(trajectory_prob.detach())
            final_morphology_losses.append(morph_loss.detach())
            for key in _MORPHOLOGY_LOSS_STAT_KEYS:
                final_stats[key].append(morph_stats[key])
            for key in _TRAJECTORY_LOSS_STAT_KEYS:
                final_stats[key].append(stats[key])

        final_losses = torch.cat(final_losses, dim=0)
        final_probs = torch.cat(final_probs, dim=0)
        final_loss_components: dict[str, Tensor] = {
            "morphology_loss": torch.cat(final_morphology_losses, dim=0),
        }
        final_loss_components.update(
            {key: torch.cat(values, dim=0) for key, values in final_stats.items()}
        )

    return (
        length_candidates.detach(),
        final_trajectories.detach(),
        final_losses,
        final_probs,
        final_raw_morphologies.detach(),
        final_processed_morphologies.detach(),
        final_loss_components,
    )


def _optimize_one_dof_group(
    *,
    dof: int,
    initial_candidate_morphologies: Tensor,
    model: MLP,
    task: Task,
    link_radius: float,
    learning_rate: float,
    learning_rate_pose: float,
    candidate_batch_size: int,
    distribution_batch_size: int,
    logging: bool,
    internal_logger: InternalOptimizationCSVLogger | None = None,
) -> list[dict[str, Any]]:
    """Optimize and post-filter one fixed-length DOF candidate group."""
    alpha_candidates = initial_candidate_morphologies[..., 0:1].detach()
    length_candidates = initial_candidate_morphologies[..., 1:].detach()
    initial_morphologies_for_log = initial_candidate_morphologies.detach()

    (
        _,
        trajectories,
        losses,
        probs,
        _,
        processed_morphologies,
        loss_components,
    ) = _optimize_all_candidates_iterative(
        model=model,
        task=task,
        alpha_candidates=alpha_candidates,
        initial_length_candidates=length_candidates,
        reference_poses=task.goal_poses,
        link_radius=link_radius,
        learning_rate=learning_rate,
        learning_rate_pose=learning_rate_pose,
        candidate_batch_size=candidate_batch_size,
        logging=logging,
        internal_logger=internal_logger,
        dof=dof,
    )

    return _postfilter_dof_group(
        dof=dof,
        losses=losses,
        probs=probs,
        initial_morphologies_for_log=initial_morphologies_for_log,
        processed_morphologies=processed_morphologies,
        distribution_batch_size=distribution_batch_size,
        logging=logging,
        extra_record_fields=lambda idx: {
            "trajectory": trajectories[idx],
            "loss_components": {
                key: values[idx] for key, values in loss_components.items()
            },
        },
    )


# --------------------------------- main API ---------------------------------


def optimize_morphology_and_trajectory(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[
    Morphology, Tensor, Path, list[float], list[tuple[Morphology, Tensor, float]]
]:
    """Discrete-alpha candidate search with per-candidate trajectory optimization."""
    dof_selector = CANDIDATE_DOF
    params = _parse_candidate_search_params(optimization_parameters, dof_selector)
    lr_pose = float(
        optimization_parameters.get(
            "learning_rate_pose",
            optimization_parameters.get("learning_rate_angle", params.learning_rate),
        )
    )

    if task.goal_poses.shape[0] < 2:
        raise ValueError("trajectory candidate selection requires at least 2 poses.")

    num_iteratives, morphology_steps, trajectory_steps = _trajectory_step_counts()
    device = morph.params.device
    timer = OptimizationTimer(device)
    timer.start()

    if params.logging:
        print(f"[Info] Starting DOF trajectory candidate optimization on {device}.")
        print(
            "[Info] "
            f"dof={dof_selector!r}, "
            f"candidate_dofs={params.candidate_dofs}, "
            f"num_iteratives={num_iteratives}, "
            f"morphology_steps={morphology_steps}, "
            f"trajectory_steps={trajectory_steps}, "
            f"learning_rate_length={params.learning_rate}, "
            f"learning_rate_pose={lr_pose}, "
            f"num_alpha_candidates={params.num_alpha_candidates}, "
            f"candidate_batch_size={params.candidate_batch_size}, "
            f"distribution_batch_size={params.distribution_batch_size}, "
            f"top_probability_fraction={TOP_PROBABILITY_FRACTION}, "
            f"random_seed={params.random_seed}, "
            f"number_random_seed={params.number_random_seed}, "
            f"percentage_poses={params.percentage_poses}"
        )
        print(
            "[Info] trajectory_weights="
            f"reachability:{TRAJECTORY_REACHABILITY_WEIGHT}, "
            f"smoothness:{TRAJECTORY_SMOOTHNESS_WEIGHT}, "
            f"distance_step_variance:"
            f"{TRAJECTORY_DISTANCE_STEP_VARIANCE_WEIGHT}, "
            f"rotation_step_variance:"
            f"{TRAJECTORY_ROTATION_STEP_VARIANCE_WEIGHT}, "
            f"position_deviation:{POSITION_DEVIATION_WEIGHT}, "
            f"rotation_deviation:{ROTATION_DEVIATION_WEIGHT}, "
            f"endpoint_side:{TRAJECTORY_ENDPOINT_SIDE_WEIGHT}, "
            f"wall_clearance:{WALL_CLEARANCE_WEIGHT}, "
            f"wall_repulsion:{WALL_REPULSION_WEIGHT}; "
            f"min_wall_clearance={MIN_WALL_CLEARANCE}, "
            f"wall_repulsion_distance={WALL_REPULSION_DISTANCE}"
        )
        print(f"[Info] Loading NRM checkpoint: {_CHECKPOINT_PATH}")

    scene, model, csv_logger, alpha_generator = _setup_search_runtime(
        task=task,
        device=device,
        optimization_parameters=optimization_parameters,
        params=params,
    )
    internal_logger: InternalOptimizationCSVLogger | None = None

    try:
        internal_logger = InternalOptimizationCSVLogger(
            csv_logger.csv_path,
            fieldnames=TRAJECTORY_INTERNAL_LOG_FIELDNAMES,
            enabled=params.csv_logging,
        )

        initial_candidate_morphologies_by_dof = _generate_initial_candidates(
            params=params,
            device=device,
            link_radius=morph.link_radius,
            alpha_generator=alpha_generator,
            csv_logger=csv_logger,
            internal_logger=internal_logger,
        )

        records: list[dict[str, Any]] = []
        for dof in params.candidate_dofs:
            records.extend(
                _optimize_one_dof_group(
                    dof=dof,
                    initial_candidate_morphologies=initial_candidate_morphologies_by_dof[
                        dof
                    ],
                    model=model,
                    task=task,
                    link_radius=morph.link_radius,
                    learning_rate=params.learning_rate,
                    learning_rate_pose=lr_pose,
                    candidate_batch_size=params.candidate_batch_size,
                    distribution_batch_size=params.distribution_batch_size,
                    logging=params.logging,
                    internal_logger=internal_logger,
                )
            )

        final_record, optimized_morph, candidates = _select_and_log_final_candidates(
            records=records,
            params=params,
            morph=morph,
            task=task,
            scene=scene,
            device=device,
            csv_logger=csv_logger,
            timer=timer,
            all_candidates_fieldnames=ALL_CANDIDATES_LOG_FIELDNAMES,
            validation_desc="validating top-probability trajectory candidates",
            final_print_label="Final trajectory candidate",
            final_print_extra=lambda record: (
                f", num_trajectory_poses={record['trajectory'].shape[0]}"
            ),
            build_candidate=lambda record, ik_success_pose_rate: (
                Morphology(
                    params=record["processed_morphology"].detach(),
                    link_radius=morph.link_radius,
                ),
                record["trajectory"].detach(),
                ik_success_pose_rate,
            ),
        )

        return (
            optimized_morph,
            final_record["trajectory"].detach(),
            csv_logger.csv_path,
            timer.result(),
            candidates,
        )

    finally:
        if internal_logger is not None:
            internal_logger.close()
        csv_logger.close()


optimize_morphology = optimize_morphology_and_trajectory
