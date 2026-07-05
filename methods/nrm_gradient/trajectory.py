# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Julian Arkenau / Shiyuan Zhang
# -----------------------------------------------------------------------------
# Alternating NRM optimization for morphology and intermediate trajectory poses.
# The morphology phase follows methods/legacy/nrm_gradient_static.py closely. The trajectory phase keeps
# morphology fixed, optimizes only intermediate poses, and leaves start/goal fixed.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from methods.nrm_model import MLP
from core import Morphology, Task
from logutils.csv_logger import OptimizationCSVLogger
from logutils.timing import OptimizationTimer
from validation.optimization_validation import run_optimization_validation


EPS = 1e-4

num_steps = 20
num_iteratives = 50

MIN_WALL_CLEARANCE = 0.025
TRAJECTORY_REACHABILITY_WEIGHT = 1.0
TRAJECTORY_SMOOTHNESS_WEIGHT = 0.1
# adjacent poses heuristic
TRAJECTORY_DISTANCE_STEP_VARIANCE_WEIGHT = 3.0
TRAJECTORY_ROTATION_STEP_VARIANCE_WEIGHT = 1.0
POSITION_DEVIATION_WEIGHT = 0
ROTATION_DEVIATION_WEIGHT = 0
# Encourage the half of the points to be on the same side with start/goal (linearly decaying)
TRAJECTORY_ENDPOINT_SIDE_WEIGHT = 0
WALL_CLEARANCE_WEIGHT = 500.0
# encourage to away from wall, bigger, more encourage
WALL_REPULSION_WEIGHT = 0.005
# decaying of the weight for the poses to be far from wall, bigger, slower decay
WALL_REPULSION_DISTANCE = 0.0005

from paths import PROJECT_ROOT as _PROJECT_ROOT, WEIGHTS_DIR as _WEIGHTS_DIR


def _se3_to_vector(pose: Tensor) -> Tensor:
    """Convert SE(3) matrices [..., 4, 4] to 9D NRM pose vectors."""
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


def _vector_to_se3(vector: Tensor) -> Tensor:
    """Convert 9D pose vectors [position, rotation_6d] to SE(3) matrices."""
    poses = torch.eye(4, dtype=vector.dtype, device=vector.device).repeat(
        vector.shape[0], 1, 1
    )
    poses[:, :3, :3] = _rotation_6d_to_matrix(vector[:, 3:9])
    poses[:, :3, 3] = vector[:, 0:3]
    return poses


def _load_model(device: torch.device) -> MLP:
    """Load pretrained NRM model on the same device as the morphology."""
    metadata = json.loads((_WEIGHTS_DIR / "metadata.json").read_text())

    model = MLP(**metadata["hyperparameter"])
    model.load_state_dict(
        torch.load(
            _WEIGHTS_DIR / "checkpoint_5-7.pth", map_location=device, weights_only=True
        )
    )
    model = model.to(device)

    # cuDNN LSTM backward may require train mode.
    # The weights are frozen, but gradients still flow to the optimized inputs.
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


class SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, param, threshold):
        mask = (param.abs() >= threshold).float()
        return param * mask

    @staticmethod
    def backward(_ctx, grad_output):
        return grad_output, None


class Normaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        l2_norm = torch.hypot(param[:, 0:1], param[:, 1:2])
        norm = l2_norm.sum(dim=0, keepdim=True)

        ctx.save_for_backward(param, l2_norm, norm)

        return param / norm

    @staticmethod
    def backward(ctx, grad_output):
        param, l2_norm, norm = ctx.saved_tensors

        chain = torch.where(
            (param.abs() > EPS).any(dim=1, keepdim=True),
            param / l2_norm,
            torch.zeros_like(param),
        )

        return (grad_output * norm - chain * (grad_output * param).sum()) / norm**2


def _preprocess(lengths: Tensor, link_radius: float) -> tuple[Tensor, Tensor]:
    """Apply normalize -> squash -> normalize to raw [a, d] lengths."""
    threshold = 2.0 * link_radius

    norm_lengths = Normaliser.apply(lengths)
    squashed = SquasherSTE.apply(norm_lengths, threshold)

    return Normaliser.apply(squashed), norm_lengths


def _build_morphology(
    alpha: Tensor,
    lengths: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor]:
    processed_lengths, _ = _preprocess(lengths, link_radius)
    raw_morphology = torch.cat([alpha, lengths], dim=1)
    processed_morphology = torch.cat([alpha, processed_lengths], dim=1)
    return raw_morphology, processed_morphology


def _morphology_loss_and_prob(
    *,
    model: MLP,
    alpha: Tensor,
    lengths: Tensor,
    task_vec: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raw_morphology, processed_morphology = _build_morphology(
        alpha=alpha,
        lengths=lengths,
        link_radius=link_radius,
    )

    bmorph = processed_morphology.unsqueeze(0).expand(task_vec.shape[0], -1, -1)
    logit = model(bmorph, task_vec)

    loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(logit, torch.ones_like(logit))
    loss = loss + 100.0 * torch.relu(-processed_morphology[-1, 2])
    prob = torch.sigmoid(logit).mean()

    return loss, prob, raw_morphology, processed_morphology


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


def _trajectory_from_intermediate(
    start_vec: Tensor,
    intermediate_vec: Tensor,
    goal_vec: Tensor,
) -> tuple[Tensor, Tensor]:
    raw_vec = torch.cat(
        [start_vec.unsqueeze(0), intermediate_vec, goal_vec.unsqueeze(0)],
        dim=0,
    )
    poses = _vector_to_se3(raw_vec)
    task_vec = _se3_to_vector(poses)
    return task_vec, poses


def _trajectory_smoothness_loss(poses: Tensor) -> Tensor:
    if poses.shape[0] <= 3:
        return poses.new_zeros(())

    positions = poses[:, :3, 3]
    segments = positions[1:] - positions[:-1]
    tangents = F.normalize(segments, dim=-1, eps=EPS)

    forward = F.normalize(
        (positions[-1] - positions[0]).unsqueeze(0),
        dim=-1,
        eps=EPS,
    )[0]

    world_z = torch.tensor([0.0, 0.0, 1.0], dtype=poses.dtype, device=poses.device)
    world_y = torch.tensor([0.0, 1.0, 0.0], dtype=poses.dtype, device=poses.device)
    seed_axis = torch.where(
        torch.abs(torch.dot(forward.detach(), world_z)) < 0.95,
        world_z,
        world_y,
    )
    lateral = seed_axis - torch.dot(seed_axis, forward) * forward
    lateral = F.normalize(lateral.unsqueeze(0), dim=-1, eps=EPS)[0]
    binormal = torch.cross(forward, lateral, dim=0)

    forward_component = tangents @ forward
    lateral_angle = torch.atan2(tangents @ lateral, forward_component)
    binormal_angle = torch.atan2(tangents @ binormal, forward_component)

    lateral_angle_diff = _wrapped_angle_difference(lateral_angle)
    binormal_angle_diff = _wrapped_angle_difference(binormal_angle)
    return torch.var(lateral_angle_diff, unbiased=False) + torch.var(
        binormal_angle_diff,
        unbiased=False,
    )


def _wrapped_angle_difference(angle: Tensor) -> Tensor:
    diff = angle[1:] - angle[:-1]
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def _trajectory_step_variance_losses(poses: Tensor) -> tuple[Tensor, Tensor]:
    if poses.shape[0] <= 2:
        zero = poses.new_zeros(())
        return zero, zero

    position_steps = torch.linalg.vector_norm(
        poses[1:, :3, 3] - poses[:-1, :3, 3],
        dim=-1,
    )
    rotation_steps = _rotation_angle_between(
        poses[:-1, :3, :3],
        poses[1:, :3, :3],
    )
    return (
        torch.var(position_steps, unbiased=False),
        torch.var(rotation_steps, unbiased=False),
    )


def _trajectory_deviation_loss(
    poses: Tensor,
    reference_poses: Tensor,
) -> tuple[Tensor, Tensor]:
    pos_loss = F.mse_loss(poses[:, :3, 3], reference_poses[:, :3, 3])
    rot_angles = _rotation_angle_between(
        poses[:, :3, :3],
        reference_poses[:, :3, :3],
    )
    rot_loss = rot_angles.pow(2).mean()
    return pos_loss, rot_loss


def _trajectory_endpoint_side_loss(poses: Tensor) -> Tensor:
    num_poses = poses.shape[0]
    if num_poses <= 3:
        return poses.new_zeros(())

    positions = poses[:, :3, 3]
    intermediate_positions = positions[1:-1]
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

    start_dist_sq = (intermediate_positions - positions[0]).pow(2).sum(dim=-1)
    goal_dist_sq = (intermediate_positions - positions[-1]).pow(2).sum(dim=-1)

    loss = start_weights * start_dist_sq + goal_weights * goal_dist_sq
    return loss.sum()


def _box_signed_line_clearance_xz(
    points: Tensor,
    center: Tensor,
    half_extents: Tensor,
) -> Tensor:
    point_xz = torch.stack([points[:, 0], points[:, 2]], dim=-1)
    center_xz = torch.stack([center[0], center[2]], dim=0)
    half_xz = torch.stack([half_extents[0], half_extents[2]], dim=0)

    q = torch.abs(point_xz - center_xz) - half_xz
    outside = torch.linalg.vector_norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.minimum(q.max(dim=-1).values, q.new_zeros(q.shape[0]))
    return outside + inside


def _nearest_wall_clearance(task: Task, poses: Tensor) -> Tensor | None:
    clearances = []
    points = poses[:, :3, 3]

    for obstacle in task.environment.obstacles:
        if getattr(obstacle, "kind", None) != "box":
            continue

        center = obstacle.center.to(device=poses.device, dtype=poses.dtype)
        half_extents = obstacle.half_extents.to(device=poses.device, dtype=poses.dtype)
        clearances.append(_box_signed_line_clearance_xz(points, center, half_extents))

    if not clearances:
        return None

    return torch.stack(clearances, dim=0).min(dim=0).values


def _wall_clearance_loss(task: Task, poses: Tensor) -> tuple[Tensor, Tensor]:
    nearest_clearance = _nearest_wall_clearance(task, poses)
    if nearest_clearance is None:
        return poses.new_zeros(()), poses.new_tensor(float("inf"))

    violation = torch.relu(MIN_WALL_CLEARANCE - nearest_clearance)
    return violation.pow(2).mean(), nearest_clearance.min()


def _wall_repulsion_loss(task: Task, poses: Tensor) -> tuple[Tensor, Tensor]:
    nearest_clearance = _nearest_wall_clearance(task, poses)
    if nearest_clearance is None:
        return poses.new_zeros(()), poses.new_tensor(float("inf"))

    if WALL_REPULSION_WEIGHT <= 0.0 or WALL_REPULSION_DISTANCE <= 0.0:
        return poses.new_zeros(()), nearest_clearance.mean()

    outside_clearance = torch.clamp(nearest_clearance, min=0.0)
    repulsion = torch.exp(-outside_clearance / WALL_REPULSION_DISTANCE)
    return repulsion.mean(), nearest_clearance.mean()


def _trajectory_loss_and_stats(
    *,
    model: MLP,
    processed_morphology: Tensor,
    task: Task,
    task_vec: Tensor,
    poses: Tensor,
    reference_poses: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    bmorph = processed_morphology.unsqueeze(0).expand(task_vec.shape[0], -1, -1)
    logit = model(bmorph, task_vec)
    prob = torch.sigmoid(logit).mean()

    reachability_loss = F.mse_loss(prob.new_ones(logit.shape), torch.sigmoid(logit))
    smoothness_loss = _trajectory_smoothness_loss(poses)
    distance_step_variance_loss, rotation_step_variance_loss = (
        _trajectory_step_variance_losses(poses)
    )
    position_deviation_loss, rotation_deviation_loss = _trajectory_deviation_loss(
        poses,
        reference_poses,
    )
    endpoint_side_loss = _trajectory_endpoint_side_loss(poses)
    wall_loss, min_wall_clearance = _wall_clearance_loss(task, poses)
    wall_repulsion_loss, mean_wall_clearance = _wall_repulsion_loss(task, poses)

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


def _validation_for_current_state(
    *,
    processed_morphology: Tensor,
    link_radius: float,
    base_task: Task,
    trajectory: Tensor,
    scene,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    pose_sampling_generator: torch.Generator,
    timer: OptimizationTimer | None = None,
) -> dict:
    current_task = Task(
        environment=base_task.environment,
        goal_poses=trajectory.detach(),
        reachable_region=base_task.reachable_region,
        start_q=base_task.start_q,
    )
    validation_morph = Morphology(
        params=processed_morphology.detach(),
        link_radius=link_radius,
    )
    if timer is None:
        return run_optimization_validation(
            processed_morphology=processed_morphology.detach(),
            morph=validation_morph,
            task=current_task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=pose_sampling_generator,
        )
    with timer.validation():
        return run_optimization_validation(
            processed_morphology=processed_morphology.detach(),
            morph=validation_morph,
            task=current_task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=pose_sampling_generator,
        )


def optimize_morphology_and_trajectory(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Tensor, Path, list[float]]:
    """Alternately optimize morphology lengths and intermediate trajectory poses.

    Returns:
        optimized_morphology:
            Final processed morphology.
        optimized_trajectory:
            Final trajectory, including fixed start and goal poses.
        csv_path:
            Path to output/<time>/morphology_history.csv.
        timing:
            [optimizer/backprop time, cuRobo validation time] in seconds.
    """
    lr = float(optimization_parameters.get("learning_rate", 0.01))
    lr_pose = float(
        optimization_parameters.get(
            "learning_rate_pose",
            optimization_parameters.get("learning_rate_angle", lr),
        )
    )
    logging = bool(optimization_parameters.get("logging", True))
    csv_logging = bool(optimization_parameters.get("csv_logging", True))
    eval_interval = int(optimization_parameters.get("eval_interval", 1))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1))

    device = morph.params.device
    timer = OptimizationTimer(device)
    timer.start()

    if task.goal_poses.shape[0] < 2:
        raise ValueError(
            "trajectory optimization requires at least start and goal poses."
        )

    if logging:
        print(
            f"[Info] Starting alternating NRM trajectory optimization on device {device}."
        )
        print(
            "[Info] "
            f"num_iteratives={num_iteratives}, "
            f"num_steps={num_steps}, "
            f"learning_rate_length={lr}, "
            f"learning_rate_pose={lr_pose}, "
            f"eval_interval={eval_interval}, "
            f"random_seed={random_seed}, "
            f"number_random_seed={number_random_seed}, "
            f"percentage_poses={percentage_poses}"
        )
        print(
            "[Info] "
            f"trajectory_weights="
            f"reachability:{TRAJECTORY_REACHABILITY_WEIGHT}, "
            f"smoothness:{TRAJECTORY_SMOOTHNESS_WEIGHT}, "
            f"distance_step_variance:{TRAJECTORY_DISTANCE_STEP_VARIANCE_WEIGHT}, "
            f"rotation_step_variance:{TRAJECTORY_ROTATION_STEP_VARIANCE_WEIGHT}, "
            f"position_deviation:{POSITION_DEVIATION_WEIGHT}, "
            f"rotation_deviation:{ROTATION_DEVIATION_WEIGHT}, "
            f"endpoint_side:{TRAJECTORY_ENDPOINT_SIDE_WEIGHT}, "
            f"wall_clearance:{WALL_CLEARANCE_WEIGHT}, "
            f"wall_repulsion:{WALL_REPULSION_WEIGHT}; "
            f"min_wall_clearance={MIN_WALL_CLEARANCE}, "
            f"wall_repulsion_distance={WALL_REPULSION_DISTANCE}"
        )

    scene = None

    reference_poses = task.goal_poses.to(device).detach()
    initial_task_vec = _se3_to_vector(reference_poses)
    start_vec = initial_task_vec[0].detach()
    goal_vec = initial_task_vec[-1].detach()
    intermediate_vec = initial_task_vec[1:-1].detach().clone()
    intermediate_vec.requires_grad_(True)

    alpha = morph.params[:, 0:1].clone().to(device)
    lengths = morph.params[:, 1:].clone().to(device)
    lengths.requires_grad_(True)

    length_optimizer = torch.optim.AdamW([lengths], lr=lr)
    trajectory_optimizer = (
        torch.optim.AdamW([intermediate_vec], lr=lr_pose)
        if intermediate_vec.numel() > 0
        else None
    )
    model = _load_model(device)

    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT, enabled=csv_logging)

    pose_sampling_generator = torch.Generator(device=device)
    pose_sampling_generator.manual_seed(random_seed)

    if logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    global_iteration = 0
    total_updates = num_iteratives * num_steps * (2 if trajectory_optimizer else 1)
    raw_morphology = torch.cat([alpha, lengths.detach()], dim=1)
    processed_morphology = raw_morphology.detach()

    try:
        progress_bar = tqdm(
            total=total_updates,
            desc="optimizing",
            dynamic_ncols=True,
        )
        for outer_idx in range(num_iteratives):
            for _ in range(num_steps):
                length_optimizer.zero_grad()

                with torch.no_grad():
                    fixed_task_vec, fixed_poses = _trajectory_from_intermediate(
                        start_vec,
                        intermediate_vec.detach(),
                        goal_vec,
                    )

                loss, prob, raw_morphology, processed_morphology = (
                    _morphology_loss_and_prob(
                        model=model,
                        alpha=alpha,
                        lengths=lengths,
                        task_vec=fixed_task_vec.detach(),
                        link_radius=morph.link_radius,
                    )
                )

                validation_data = None
                if (
                    eval_interval > 0
                    and logging
                    and global_iteration % eval_interval == 0
                ):
                    validation_data = _validation_for_current_state(
                        processed_morphology=processed_morphology,
                        link_radius=morph.link_radius,
                        base_task=task,
                        trajectory=fixed_poses,
                        scene=scene,
                        device=device,
                        percentage_poses=percentage_poses,
                        number_random_seed=number_random_seed,
                        pose_sampling_generator=pose_sampling_generator,
                        timer=timer,
                    )
                    best_se3 = (
                        validation_data["best_se3_dist_mean"].detach().cpu().item()
                    )
                    ik_success_rate = (
                        validation_data["ik_success_pose_rate"].detach().cpu().item()
                    )
                    tqdm.write(
                        f"[Iter {global_iteration:>4}/{total_updates}] "
                        f"phase=morph, outer={outer_idx + 1}/{num_iteratives}, "
                        f"loss={loss.item():.6f}, "
                        f"nrm_prob={prob.item():.6f}, "
                        f"best_se3={best_se3:.6f}, "
                        f"ik_success_pose_rate={ik_success_rate * 100.0:.2f}%"
                    )

                csv_logger.log_iteration(
                    iteration=global_iteration,
                    loss=loss.detach(),
                    reachability_probability=prob.detach(),
                    raw_morphology=raw_morphology.detach(),
                    processed_morphology=processed_morphology.detach(),
                    validation_data=validation_data,
                )

                loss.backward()
                length_optimizer.step()
                progress_bar.set_postfix(
                    phase="morph",
                    loss=f"{loss.item():.4f}",
                    prob=f"{prob.item():.3f}",
                )
                progress_bar.update(1)
                global_iteration += 1

            if trajectory_optimizer is None:
                continue

            for _ in range(num_steps):
                trajectory_optimizer.zero_grad()

                raw_morphology, processed_morphology = _build_morphology(
                    alpha=alpha,
                    lengths=lengths,
                    link_radius=morph.link_radius,
                )
                task_vec, poses = _trajectory_from_intermediate(
                    start_vec,
                    intermediate_vec,
                    goal_vec,
                )

                loss, prob, stats = _trajectory_loss_and_stats(
                    model=model,
                    processed_morphology=processed_morphology.detach(),
                    task=task,
                    task_vec=task_vec,
                    poses=poses,
                    reference_poses=reference_poses,
                )

                validation_data = None
                if (
                    eval_interval > 0
                    and logging
                    and global_iteration % eval_interval == 0
                ):
                    validation_data = _validation_for_current_state(
                        processed_morphology=processed_morphology,
                        link_radius=morph.link_radius,
                        base_task=task,
                        trajectory=poses,
                        scene=scene,
                        device=device,
                        percentage_poses=percentage_poses,
                        number_random_seed=number_random_seed,
                        pose_sampling_generator=pose_sampling_generator,
                        timer=timer,
                    )
                    best_se3 = (
                        validation_data["best_se3_dist_mean"].detach().cpu().item()
                    )
                    ik_success_rate = (
                        validation_data["ik_success_pose_rate"].detach().cpu().item()
                    )
                    tqdm.write(
                        f"[Iter {global_iteration:>4}/{total_updates}] "
                        f"phase=trajectory, outer={outer_idx + 1}/{num_iteratives}, "
                        f"loss={loss.item():.6f}, "
                        f"nrm_prob={prob.item():.6f}, "
                        f"endpoint_side={stats['endpoint_side_loss'].item():.6f}, "
                        f"min_wall_clearance={stats['min_wall_clearance'].item():.6f}, "
                        f"wall_repulsion={stats['wall_repulsion_loss'].item():.6f}, "
                        f"best_se3={best_se3:.6f}, "
                        f"ik_success_pose_rate={ik_success_rate * 100.0:.2f}%"
                    )

                csv_logger.log_iteration(
                    iteration=global_iteration,
                    loss=loss.detach(),
                    reachability_probability=prob.detach(),
                    raw_morphology=raw_morphology.detach(),
                    processed_morphology=processed_morphology.detach(),
                    validation_data=validation_data,
                )

                loss.backward()
                trajectory_optimizer.step()
                progress_bar.set_postfix(
                    phase="trajectory",
                    loss=f"{loss.item():.4f}",
                    prob=f"{prob.item():.3f}",
                    wall=f"{stats['min_wall_clearance'].item():.3f}",
                )
                progress_bar.update(1)
                global_iteration += 1

        progress_bar.close()

        with torch.no_grad():
            final_raw_morphology, final_processed_morphology = _build_morphology(
                alpha=alpha,
                lengths=lengths,
                link_radius=morph.link_radius,
            )
            final_task_vec, final_trajectory = _trajectory_from_intermediate(
                start_vec,
                intermediate_vec,
                goal_vec,
            )
            final_morph_loss, final_prob, _, _ = _morphology_loss_and_prob(
                model=model,
                alpha=alpha,
                lengths=lengths,
                task_vec=final_task_vec,
                link_radius=morph.link_radius,
            )
            final_trajectory_loss, _, final_stats = _trajectory_loss_and_stats(
                model=model,
                processed_morphology=final_processed_morphology,
                task=task,
                task_vec=final_task_vec,
                poses=final_trajectory,
                reference_poses=reference_poses,
            )
            final_loss = final_morph_loss + final_trajectory_loss

        final_validation_data = _validation_for_current_state(
            processed_morphology=final_processed_morphology,
            link_radius=morph.link_radius,
            base_task=task,
            trajectory=final_trajectory,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=pose_sampling_generator,
            timer=timer,
        )

        csv_logger.log_iteration(
            iteration=global_iteration,
            loss=final_loss,
            reachability_probability=final_prob,
            raw_morphology=final_raw_morphology,
            processed_morphology=final_processed_morphology,
            validation_data=final_validation_data,
        )

        final_se3_err = (
            final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        )
        final_ik_success_rate = (
            final_validation_data["ik_success_pose_rate"].detach().cpu().item()
        )

        print(
            f"[Iter {global_iteration:>4}/{total_updates}] "
            f"loss={final_loss.item():.6f}, "
            f"nrm_prob={final_prob.item():.6f}, "
            f"endpoint_side={final_stats['endpoint_side_loss'].item():.6f}, "
            f"min_wall_clearance={final_stats['min_wall_clearance'].item():.6f}, "
            f"wall_repulsion={final_stats['wall_repulsion_loss'].item():.6f}, "
            f"final_se3_err={final_se3_err:.6f}, "
            f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%"
        )

        optimized_morph = Morphology(
            params=final_processed_morphology.detach(),
            link_radius=morph.link_radius,
        )

        return (
            optimized_morph,
            final_trajectory.detach(),
            csv_logger.csv_path,
            timer.result(),
        )

    finally:
        csv_logger.close()


optimize_morphology = optimize_morphology_and_trajectory
