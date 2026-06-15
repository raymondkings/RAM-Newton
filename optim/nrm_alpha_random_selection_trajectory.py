# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
# Discrete-alpha candidate search with alternating morphology/trajectory
# optimization per candidate.
#
# This keeps the candidate-selection shell from nrm_alpha_random_selection.py:
#   - alpha candidate generation
#   - fixed-alpha initial morphology sampling and cache
#   - post-optimization distribution filter
#   - final-link d filter
#   - top-probability candidate validation
#   - final pick by validation success rate and the existing tie heuristic
#
# The per-candidate optimizer is replaced with the alternating NRM trajectory
# logic from nrm_trajectory.py.  There is no per-candidate early stopping.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from optim import nrm_alpha_random_selection as candidate_base
from optim.model import MLP
from interface import Morphology, Task
from util.fixed_alpha_morphology_candidates import (
    DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
    generate_alpha_candidates,
    sample_fixed_alpha_morphology_candidates,
    sample_fixed_alpha_morphology_candidates_by_dof,
)
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)


EPS = 1e-4

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

_PROJECT_ROOT = Path(__file__).parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"
_CHECKPOINT_PATH = _WEIGHTS_DIR / "checkpoint_5-7.pth"

# ----------------------------- hard-coded knobs -----------------------------

DEFAULT_CANDIDATE_DOFS = (5, 6, 7)
CANDIDATE_DOF: str | int | tuple[int, ...] = 7

DEFAULT_NUM_ALPHA_CANDIDATES: int | str = "ALL"
DEFAULT_CANDIDATE_BATCH_SIZE = 64
DEFAULT_DISTRIBUTION_BATCH_SIZE = 128
ZERO_ALPHA_RUN_EXCLUSION_LENGTH = DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH
TOP_PROBABILITY_FRACTION = 0.025
MORPHOLOGY_FINAL_D_PENALTY_WEIGHT = 100.0


def _load_model(device: torch.device) -> MLP:
    metadata = json.loads((_WEIGHTS_DIR / "metadata.json").read_text())
    model = MLP(**metadata["hyperparameter"])
    model.load_state_dict(
        torch.load(
            _CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    )
    model = model.to(device)

    # cuDNN LSTM backward may require train mode.
    # The weights are frozen, but gradients still flow to optimized inputs.
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ------------------------------- candidates ---------------------------------


def _resolve_candidate_dofs(value: Any) -> list[int]:
    """Resolve the hard-coded DOF selector to a sorted non-empty subset of 5, 6, 7."""
    if value is None:
        return list(DEFAULT_CANDIDATE_DOFS)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "all":
            return list(DEFAULT_CANDIDATE_DOFS)
        parts = normalized.replace(",", " ").split()
        dofs = [int(part) for part in parts]
    elif isinstance(value, int):
        dofs = [int(value)]
    else:
        dofs = [int(dof) for dof in value]

    if not dofs:
        raise ValueError("dof must not be empty.")

    resolved = sorted(set(dofs))
    unsupported = [dof for dof in resolved if dof not in DEFAULT_CANDIDATE_DOFS]
    if unsupported:
        raise ValueError(
            f"dof supports only {list(DEFAULT_CANDIDATE_DOFS)} or 'all', "
            f"got {resolved}."
        )

    return resolved


def _generate_alpha_candidates_by_dof(
    *,
    dofs: list[int],
    requested_num_candidates: int | str | None,
    device: torch.device,
    generator: torch.Generator,
    logging: bool,
) -> dict[int, Tensor]:
    """Generate fixed alpha candidate tensors keyed by DOF."""
    alpha_candidates_by_dof: dict[int, Tensor] = {}

    for dof in dofs:
        seq_len = dof + 1
        alpha_candidates, max_alpha_candidates, using_all = generate_alpha_candidates(
            requested_num_candidates=requested_num_candidates,
            seq_len=seq_len,
            device=device,
            generator=generator,
            forbidden_zero_run_length=ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
        )
        alpha_candidates_by_dof[dof] = alpha_candidates

        if logging:
            print(
                f"[Info] DOF{dof} alpha candidates generated: "
                f"{alpha_candidates.shape[0]} / max_valid={max_alpha_candidates} "
                f"(using_all={using_all})."
            )

    return alpha_candidates_by_dof


def _sample_initial_candidate_morphologies_by_dof(
    *,
    alpha_candidates_by_dof: dict[int, Tensor],
    seed: int,
    link_radius: float,
    batch_size: int,
    logging: bool,
) -> dict[int, Tensor]:
    """Sample fixed-alpha initial morphologies while preserving cache schemas."""
    if len(alpha_candidates_by_dof) == 1:
        dof, alpha_candidates = next(iter(alpha_candidates_by_dof.items()))
        return {
            dof: sample_fixed_alpha_morphology_candidates(
                alpha_candidates=alpha_candidates,
                seed=seed,
                link_radius=link_radius,
                batch_size=batch_size,
                logging=logging,
            )
        }

    return sample_fixed_alpha_morphology_candidates_by_dof(
        alpha_candidates_by_dof=alpha_candidates_by_dof,
        seed=seed,
        link_radius=link_radius,
        batch_size=batch_size,
        logging=logging,
    )


# ------------------------------- SE(3) helpers ------------------------------


def _se3_to_vector(pose: Tensor) -> Tensor:
    """Convert SE(3) pose matrices [..., 4, 4] to 9D NRM pose vectors."""
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


def _rotation_6d_to_matrix(rot_6d: Tensor) -> Tensor:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1, eps=EPS)
    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_orthogonal, dim=-1, eps=EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _rotation_angle_between(rot_a: Tensor, rot_b: Tensor) -> Tensor:
    rel = rot_a.transpose(-1, -2) @ rot_b
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_angle = 0.5 * (trace - 1.0)
    skew = torch.stack(
        [
            rel[..., 2, 1] - rel[..., 1, 2],
            rel[..., 0, 2] - rel[..., 2, 0],
            rel[..., 1, 0] - rel[..., 0, 1],
        ],
        dim=-1,
    )
    sin_angle = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sin_angle, cos_angle)


def _vector_to_se3(vector: Tensor) -> Tensor:
    poses = torch.eye(4, dtype=vector.dtype, device=vector.device).expand(
        *vector.shape[:-1],
        4,
        4,
    )
    poses = poses.clone()
    poses[..., :3, :3] = _rotation_6d_to_matrix(vector[..., 3:9])
    poses[..., :3, 3] = vector[..., 0:3]
    return poses


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


def _build_morphology_tensors(
    alpha_candidates: Tensor,
    length_candidates: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor]:
    return candidate_base._build_morphology_tensors(
        alpha_candidates,
        length_candidates,
        link_radius,
    )


def _morphology_loss_and_prob_batched(
    *,
    model: MLP,
    alpha_candidates: Tensor,
    length_candidates: Tensor,
    task_vec: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return morphology-phase loss/prob for each candidate."""
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
    return loss, prob, raw_morphologies, processed_morphologies


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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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

    for outer_idx in range(num_iteratives):
        for _ in range(morphology_steps):
            length_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                fixed_task_vec, _ = _trajectory_from_intermediate(
                    start_vec,
                    intermediate_vec.detach(),
                    goal_vec,
                )

            current_probs = torch.empty(num_candidates, device=alpha_candidates.device)
            current_loss_sum = 0.0

            for start in range(0, num_candidates, candidate_batch_size):
                end = min(start + candidate_batch_size, num_candidates)
                loss_per_candidate, prob_per_candidate, _, _ = (
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

            length_optimizer.step()
            if logging:
                progress.set_postfix(
                    phase="morph",
                    outer=f"{outer_idx + 1}/{num_iteratives}",
                    loss=f"{current_loss_sum / num_candidates:.4f}",
                    prob=f"{current_probs.mean().item():.3f}",
                )
            progress.update(1)

        if trajectory_optimizer is None:
            continue

        for _ in range(trajectory_steps):
            trajectory_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                _, processed_morphologies = _build_morphology_tensors(
                    alpha_candidates,
                    length_candidates.detach(),
                    link_radius,
                )

            current_probs = torch.empty(num_candidates, device=alpha_candidates.device)
            current_loss_sum = 0.0
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
                wall_values.append(stats["min_wall_clearance"])

            trajectory_optimizer.step()
            if logging:
                wall_tensor = torch.cat(wall_values, dim=0)
                progress.set_postfix(
                    phase="trajectory",
                    outer=f"{outer_idx + 1}/{num_iteratives}",
                    loss=f"{current_loss_sum / num_candidates:.4f}",
                    prob=f"{current_probs.mean().item():.3f}",
                    wall=f"{wall_tensor.min().item():.3f}",
                )
            progress.update(1)

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
        for start in range(0, num_candidates, candidate_batch_size):
            end = min(start + candidate_batch_size, num_candidates)
            morph_loss, _, _, _ = _morphology_loss_and_prob_batched(
                model=model,
                alpha_candidates=alpha_candidates[start:end],
                length_candidates=length_candidates.detach()[start:end],
                task_vec=final_task_vec[start:end],
                link_radius=link_radius,
            )
            trajectory_loss, trajectory_prob, _ = _trajectory_loss_and_stats_batched(
                model=model,
                processed_morphologies=final_processed_morphologies[start:end],
                task=task,
                task_vec=final_task_vec[start:end],
                poses=final_trajectories[start:end],
                reference_poses=reference_poses,
            )
            final_losses.append((morph_loss + trajectory_loss).detach())
            final_probs.append(trajectory_prob.detach())

        final_losses = torch.cat(final_losses, dim=0)
        final_probs = torch.cat(final_probs, dim=0)

    return (
        length_candidates.detach(),
        final_trajectories.detach(),
        final_losses,
        final_probs,
        final_raw_morphologies.detach(),
        final_processed_morphologies.detach(),
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
    )

    post_valid_mask, _ = candidate_base._distribution_valid_mask(
        processed_morphologies,
        batch_size=distribution_batch_size,
        logging=logging,
        desc=f"post-checking DOF{dof} candidate distribution",
    )

    if logging:
        print(
            f"[Info] DOF{dof} post-optimization distribution filter: "
            f"kept {int(post_valid_mask.sum().item())}/"
            f"{processed_morphologies.shape[0]} candidates."
        )

    records: list[dict[str, Any]] = []
    valid_indices = torch.nonzero(post_valid_mask, as_tuple=False).squeeze(1)
    for idx in valid_indices.tolist():
        records.append(
            {
                "dof": dof,
                "loss": losses[idx],
                "prob": probs[idx],
                "raw_morphology": initial_morphologies_for_log[idx],
                "processed_morphology": processed_morphologies[idx],
                "trajectory": trajectories[idx],
            }
        )

    return records


# ------------------------------- validation ---------------------------------


def _validation_generator(device: torch.device, random_seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    return generator


def _validate_candidate(
    *,
    processed_morphology: Tensor,
    trajectory: Tensor,
    morph: Morphology,
    base_task: Task,
    scene,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
) -> dict:
    """Validate one candidate on its own optimized trajectory."""
    candidate_task = Task(
        environment=base_task.environment,
        goal_poses=trajectory.detach(),
        reachable_region=base_task.reachable_region,
        start_q=base_task.start_q,
    )
    return run_optimization_validation(
        processed_morphology=processed_morphology.detach(),
        morph=morph,
        task=candidate_task,
        scene=scene,
        device=device,
        percentage_poses=percentage_poses,
        number_random_seed=number_random_seed,
        pose_sampling_generator=_validation_generator(device, random_seed),
    )


def _validate_top_records(
    *,
    records: list[dict[str, Any]],
    morph: Morphology,
    task: Task,
    scene,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
    logging: bool,
) -> tuple[Tensor, Tensor, list[dict]]:
    """Run validation on selected candidate records and return scores plus data."""
    se3_scores = torch.empty(len(records), device=device)
    ik_success_rates = torch.empty(len(records), device=device)
    validation_data_list: list[dict] = []

    for idx in tqdm(
        range(len(records)),
        desc="validating top-probability trajectory candidates",
        disable=not logging,
        dynamic_ncols=True,
    ):
        validation_data = _validate_candidate(
            processed_morphology=records[idx]["processed_morphology"],
            trajectory=records[idx]["trajectory"],
            morph=morph,
            base_task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            random_seed=random_seed,
        )
        validation_data_list.append(validation_data)
        se3_scores[idx] = validation_data["best_se3_dist_mean"].detach().to(device)
        ik_success_rates[idx] = (
            validation_data["ik_success_pose_rate"].detach().to(device)
        )

    return se3_scores, ik_success_rates, validation_data_list


def _log_candidate(
    csv_logger: OptimizationCSVLogger,
    iteration_marker: int,
    loss: Tensor,
    prob: Tensor,
    raw_morphology: Tensor,
    processed_morphology: Tensor,
    validation_data: dict,
) -> None:
    candidate_base._log_candidate(
        csv_logger=csv_logger,
        iteration_marker=iteration_marker,
        loss=loss,
        prob=prob,
        raw_morphology=raw_morphology,
        processed_morphology=processed_morphology,
        validation_data=validation_data,
    )


# --------------------------------- main API ---------------------------------


def optimize_morphology_and_trajectory(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Tensor, Path]:
    """Discrete-alpha candidate search with per-candidate trajectory optimization."""
    lr = float(optimization_parameters.get("learning_rate", 0.01))
    lr_pose = float(
        optimization_parameters.get(
            "learning_rate_pose",
            optimization_parameters.get("learning_rate_angle", lr),
        )
    )
    logging = bool(optimization_parameters.get("logging", True))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1))
    ignore_ground = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles = bool(optimization_parameters.get("ignore_obstacles", False))
    dof_selector = CANDIDATE_DOF
    candidate_dofs = _resolve_candidate_dofs(dof_selector)

    num_alpha_candidates = optimization_parameters.get(
        "num_alpha_candidates",
        DEFAULT_NUM_ALPHA_CANDIDATES,
    )
    candidate_batch_size = int(
        optimization_parameters.get(
            "candidate_batch_size",
            DEFAULT_CANDIDATE_BATCH_SIZE,
        )
    )
    distribution_batch_size = int(
        optimization_parameters.get(
            "distribution_batch_size",
            DEFAULT_DISTRIBUTION_BATCH_SIZE,
        )
    )

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")
    if distribution_batch_size <= 0:
        raise ValueError("distribution_batch_size must be positive.")
    if task.goal_poses.shape[0] < 2:
        raise ValueError("trajectory candidate selection requires at least 2 poses.")

    num_iteratives, morphology_steps, trajectory_steps = _trajectory_step_counts()
    device = morph.params.device
    selected_label = ",".join(str(dof) for dof in candidate_dofs)

    if logging:
        print(f"[Info] Starting DOF trajectory candidate optimization on {device}.")
        print(
            "[Info] "
            f"dof={dof_selector!r}, "
            f"candidate_dofs={candidate_dofs}, "
            f"num_iteratives={num_iteratives}, "
            f"morphology_steps={morphology_steps}, "
            f"trajectory_steps={trajectory_steps}, "
            f"learning_rate_length={lr}, "
            f"learning_rate_pose={lr_pose}, "
            f"num_alpha_candidates={num_alpha_candidates}, "
            f"candidate_batch_size={candidate_batch_size}, "
            f"distribution_batch_size={distribution_batch_size}, "
            f"top_probability_fraction={TOP_PROBABILITY_FRACTION}, "
            f"random_seed={random_seed}, "
            f"number_random_seed={number_random_seed}, "
            f"percentage_poses={percentage_poses}"
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

    scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    model = _load_model(device)
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    alpha_generator = torch.Generator(device=device)
    alpha_generator.manual_seed(random_seed)

    try:
        alpha_candidates_by_dof = _generate_alpha_candidates_by_dof(
            dofs=candidate_dofs,
            requested_num_candidates=num_alpha_candidates,
            device=device,
            generator=alpha_generator,
            logging=logging,
        )

        initial_candidate_morphologies_by_dof = (
            _sample_initial_candidate_morphologies_by_dof(
                alpha_candidates_by_dof=alpha_candidates_by_dof,
                seed=random_seed,
                link_radius=morph.link_radius,
                batch_size=distribution_batch_size,
                logging=logging,
            )
        )

        if logging:
            print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")
            for dof in candidate_dofs:
                original_count = alpha_candidates_by_dof[dof].shape[0]
                accepted_count = initial_candidate_morphologies_by_dof[dof].shape[0]
                print(
                    f"[Info] DOF{dof} fixed-alpha sampler accepted "
                    f"{accepted_count}/{original_count} alpha candidates "
                    f"and dropped {original_count - accepted_count}."
                )

        records: list[dict[str, Any]] = []
        for dof in candidate_dofs:
            records.extend(
                _optimize_one_dof_group(
                    dof=dof,
                    initial_candidate_morphologies=initial_candidate_morphologies_by_dof[
                        dof
                    ],
                    model=model,
                    task=task,
                    link_radius=morph.link_radius,
                    learning_rate=lr,
                    learning_rate_pose=lr_pose,
                    candidate_batch_size=candidate_batch_size,
                    distribution_batch_size=distribution_batch_size,
                    logging=logging,
                )
            )

        if not records:
            raise RuntimeError(
                f"All optimized DOF {selected_label} candidates were rejected by "
                "the post-optimization distribution checker."
            )

        before_last_d_filter = len(records)
        records = [
            record
            for record in records
            if bool(
                candidate_base._last_d_nonnegative_mask(
                    record["processed_morphology"]
                ).item()
            )
        ]

        if not records:
            raise RuntimeError(
                f"All optimized DOF {selected_label} candidates were rejected by "
                "the final-link d filter (requires processed params[-1, 2] >= 0)."
            )

        if logging:
            print(
                "[Info] Final-link d filter: "
                f"kept {len(records)}/{before_last_d_filter} candidates "
                "with processed params[-1, 2] >= 0."
            )

        probs_valid = torch.stack(
            [record["prob"].detach().to(device) for record in records]
        )
        num_valid = len(records)
        top_k = max(1, int(math.ceil(num_valid * TOP_PROBABILITY_FRACTION)))
        top_indices = torch.argsort(probs_valid, descending=True)[:top_k]
        top_records = [records[int(idx.item())] for idx in top_indices]

        if logging:
            dof_counts = {
                dof: sum(1 for record in records if record["dof"] == dof)
                for dof in candidate_dofs
            }
            top_dof_counts = {
                dof: sum(1 for record in top_records if record["dof"] == dof)
                for dof in candidate_dofs
            }
            print(
                "[Info] Top-probability selection: "
                f"valid_candidates={num_valid}, top_k={top_k}, "
                f"valid_by_dof={dof_counts}, top_by_dof={top_dof_counts}, "
                f"best_prob={probs_valid[top_indices[0]].item():.6f}, "
                f"worst_top_prob={probs_valid[top_indices[-1]].item():.6f}"
            )

        _, ik_success_rates, validation_data_list = _validate_top_records(
            records=top_records,
            morph=morph,
            task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            random_seed=random_seed,
            logging=logging,
        )

        best_ik_success_rate = ik_success_rates.max()
        best_rate_mask = (ik_success_rates - best_ik_success_rate).abs() <= 1e-12
        final_tier_mask = best_rate_mask
        tied_indices = torch.nonzero(final_tier_mask, as_tuple=False).squeeze(1)

        tie_scores = []
        length_sums = []
        for record in top_records:
            tie_score, length_sum = candidate_base._tie_score(
                record["processed_morphology"],
                morph.link_radius,
            )
            tie_scores.append(tie_score)
            length_sums.append(length_sum)

        tie_scores_tensor = torch.stack(tie_scores).to(device)
        length_sums_tensor = torch.stack(length_sums).to(device)
        final_local_in_tied = torch.argmin(tie_scores_tensor[tied_indices])
        final_idx = int(tied_indices[final_local_in_tied].item())

        if logging:
            print(
                "[Info] Validation selection: "
                f"best_ik_success_pose_rate="
                f"{best_ik_success_rate.item() * 100.0:.2f}%, "
                f"num_best_rate_candidates={int(best_rate_mask.sum().item())}, "
                f"num_tie_break_candidates={int(final_tier_mask.sum().item())}, "
                f"final_idx={final_idx}, "
                f"final_dof={top_records[final_idx]['dof']}, "
                f"final_length_sum={length_sums_tensor[final_idx].item():.6f}"
            )

        for idx, record in enumerate(top_records):
            if idx == final_idx:
                marker = 2
            elif bool(final_tier_mask[idx]):
                marker = 1
            else:
                marker = 0

            _log_candidate(
                csv_logger=csv_logger,
                iteration_marker=marker,
                loss=record["loss"],
                prob=record["prob"],
                raw_morphology=record["raw_morphology"],
                processed_morphology=record["processed_morphology"],
                validation_data=validation_data_list[idx],
            )

        final_record = top_records[final_idx]
        final_processed_morphology = final_record["processed_morphology"]
        final_trajectory = final_record["trajectory"]
        final_validation_data = validation_data_list[final_idx]
        final_se3 = final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        final_ik_success_rate = (
            final_validation_data["ik_success_pose_rate"].detach().cpu().item()
        )

        print(
            "[Final trajectory candidate] "
            f"dof={final_record['dof']}, "
            f"loss={final_record['loss'].item():.6f}, "
            f"nrm_prob={final_record['prob'].item():.6f}, "
            f"final_se3_err={final_se3:.6f}, "
            f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%, "
            f"length_sum={length_sums_tensor[final_idx].item():.6f}, "
            f"num_trajectory_poses={final_trajectory.shape[0]}"
        )

        if logging:
            print("Final alpha [deg]:")
            print(final_processed_morphology[:, 0].detach().cpu() * 180.0 / math.pi)
            print("Final optimized morphology params:")
            print(final_processed_morphology.detach().cpu())

        optimized_morph = Morphology(
            params=final_processed_morphology.detach(),
            link_radius=morph.link_radius,
        )
        return optimized_morph, final_trajectory.detach(), csv_logger.csv_path

    finally:
        csv_logger.close()


optimize_morphology = optimize_morphology_and_trajectory
