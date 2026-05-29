from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from interface import Morphology, Task
from util.kinematics import forward_kinematics, se3_distance
from util.optimization_csv_logger import OptimizationCSVLogger
from util.self_collision import get_capsules
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)


EPS = 1e-4

_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DirectIKMetrics:
    loss_per_candidate: Tensor
    pose_success_rate: Tensor
    best_se3_mean: Tensor
    self_collision_pose_rate: Tensor
    processed_morphologies: Tensor
    raw_morphologies: Tensor


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


def _goal_poses_in_robot_frame(task: Task, device: torch.device, dtype: torch.dtype) -> Tensor:
    base_pose_inv = torch.linalg.inv(
        task.environment.base_pose.to(device=device, dtype=dtype)
    )
    return base_pose_inv @ task.goal_poses.to(device=device, dtype=dtype)


def _wrap_joints(joint_raw: Tensor) -> Tensor:
    return torch.atan2(torch.sin(joint_raw), torch.cos(joint_raw))


def _append_fixed_ee_joint(joints: Tensor) -> Tensor:
    zeros = torch.zeros(
        *joints.shape[:-2],
        1,
        1,
        device=joints.device,
        dtype=joints.dtype,
    )
    return torch.cat([joints, zeros], dim=-2)


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
    pos_err = (reached[..., :3, 3] - target[..., :3, 3]).norm(dim=-1)
    rot_err = _rotation_error(reached, target)
    return se3_distance(pos_err, rot_err)


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
    point_distance = ((c1 - c2) ** 2).sum(dim=-1)
    return point_distance - (r1 + r2) ** 2


def _signed_distance_capsule_ball(
    s1: Tensor,
    e1: Tensor,
    r1: float,
    s2: Tensor,
    r2: float,
) -> Tensor:
    l1 = e1 - s1
    v = s2 - s1

    dot_v_l = (v * l1).sum(dim=-1, keepdim=True)
    len_sq_l = (l1 * l1).sum(dim=-1, keepdim=True)
    t = torch.clamp(dot_v_l / (len_sq_l + 1e-10), 0.0, 1.0)
    closest_point = s1 + t * l1

    point_distance = ((closest_point - s2) ** 2).sum(dim=-1)
    return point_distance - (r1 + r2) ** 2


def _collision_critical_distance(mdh: Tensor, poses: Tensor, radius: float) -> Tensor:
    *batch_shape, dofp1, _ = mdh.shape

    s_all, e_all = get_capsules(mdh, poses)

    i_idx, j_idx = torch.triu_indices(
        2 * dofp1,
        2 * dofp1,
        offset=2,
        device=mdh.device,
    )
    num_pairs = i_idx.shape[0]

    s1 = s_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    e1 = e_all[..., i_idx, :].reshape(-1, num_pairs, 3)
    s2 = s_all[..., j_idx, :].reshape(-1, num_pairs, 3)
    e2 = e_all[..., j_idx, :].reshape(-1, num_pairs, 3)

    active_pairs = (torch.norm(s1 - e1, dim=-1) > EPS) & (
        torch.norm(s2 - e2, dim=-1) > EPS
    )
    active_pairs &= torch.norm(e1 - s2, dim=-1) > EPS

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
    distance_end = _signed_distance_capsule_ball(s_end, e_end, radius, c_end, radius)
    active_end = torch.norm(e_end - s_end, dim=-1) > EPS

    cum_sum = torch.cumsum(active_end, dim=-1)
    active_end &= (cum_sum == 2) | (cum_sum == (cum_sum[..., -1:] - 1))

    critical_distance = torch.where(
        active_pairs,
        distances,
        torch.ones_like(distances),
    ).min(dim=-1).values.reshape(batch_shape)
    critical_distance_end = torch.where(
        active_end,
        distance_end,
        torch.ones_like(distance_end),
    ).min(dim=-1).values

    return torch.minimum(critical_distance, critical_distance_end)


def _make_joint_seeds(
    *,
    num_candidates: int,
    num_poses: int,
    num_joint_seeds: int,
    dof: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    joint_raw = (
        torch.rand(
            num_candidates,
            num_poses,
            num_joint_seeds,
            dof,
            1,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        * 2.0
        * math.pi
    ) - math.pi

    if num_joint_seeds > 0:
        joint_raw[:, :, 0] = 0.0

    return joint_raw


def _direct_ik_metrics(
    *,
    alpha: Tensor,
    lengths: Tensor,
    joint_raw: Tensor,
    target_poses_local: Tensor,
    link_radius: float,
    collision_weight: float,
    collision_margin: float,
    success_eps: float,
    softmin_temperature: float,
) -> DirectIKMetrics:
    raw_morphologies, processed_morphologies = _build_morphology_tensors(
        alpha,
        lengths,
        link_radius,
    )

    num_candidates, num_poses, num_joint_seeds = joint_raw.shape[:3]
    seq_len = processed_morphologies.shape[-2]

    full_joints = _append_fixed_ee_joint(_wrap_joints(joint_raw))
    morph_expanded = processed_morphologies[:, None, None].expand(
        num_candidates,
        num_poses,
        num_joint_seeds,
        seq_len,
        3,
    )

    flat_morph = morph_expanded.reshape(num_candidates * num_poses * num_joint_seeds, seq_len, 3)
    flat_joints = full_joints.reshape(num_candidates * num_poses * num_joint_seeds, seq_len, 1)

    reached_poses = forward_kinematics(flat_morph, flat_joints)
    ee_poses = reached_poses[:, -1].reshape(
        num_candidates,
        num_poses,
        num_joint_seeds,
        4,
        4,
    )

    target = target_poses_local[None, :, None].expand_as(ee_poses)
    se3 = _pose_se3_distance(ee_poses, target)

    critical_distance = _collision_critical_distance(
        flat_morph,
        reached_poses,
        radius=link_radius,
    ).reshape(num_candidates, num_poses, num_joint_seeds)

    collision_penalty = F.relu(collision_margin - critical_distance)
    seed_score = se3 + collision_weight * collision_penalty

    if softmin_temperature > 0.0:
        weights = torch.softmax(-seed_score / softmin_temperature, dim=-1)
        best_score = (weights * seed_score).sum(dim=-1)
        best_se3 = (weights * se3).sum(dim=-1)
    else:
        best_score, best_idx = seed_score.min(dim=-1)
        best_se3 = se3.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

    loss_per_candidate = best_score.mean(dim=-1)
    best_se3_mean = best_se3.mean(dim=-1)

    success_per_seed = (se3 <= success_eps) & (critical_distance >= collision_margin)
    pose_success_rate = success_per_seed.any(dim=-1).float().mean(dim=-1)
    self_collision_pose_rate = (critical_distance < 0.0).any(dim=-1).float().mean(dim=-1)

    return DirectIKMetrics(
        loss_per_candidate=loss_per_candidate,
        pose_success_rate=pose_success_rate,
        best_se3_mean=best_se3_mean,
        self_collision_pose_rate=self_collision_pose_rate,
        processed_morphologies=processed_morphologies,
        raw_morphologies=raw_morphologies,
    )


def _run_direct_ik_length_optimization(
    *,
    initial_morphologies: Tensor,
    target_poses_local: Tensor,
    link_radius: float,
    num_iterations: int,
    learning_rate_length: float,
    learning_rate_joint: float,
    num_joint_seeds: int,
    joint_refinement_steps: int,
    collision_weight: float,
    collision_margin: float,
    success_eps: float,
    softmin_temperature: float,
    random_seed: int,
    logging: bool,
    desc: str,
) -> tuple[Tensor, DirectIKMetrics]:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")
    if num_joint_seeds <= 0:
        raise ValueError("num_joint_seeds must be positive.")
    if joint_refinement_steps <= 0:
        raise ValueError("joint_refinement_steps must be positive.")

    device = initial_morphologies.device
    dtype = initial_morphologies.dtype
    num_candidates, seq_len, _ = initial_morphologies.shape
    dof = seq_len - 1

    alpha = initial_morphologies[..., 0:1].detach().clone()
    lengths = initial_morphologies[..., 1:].detach().clone().requires_grad_(True)

    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    joint_raw = _make_joint_seeds(
        num_candidates=num_candidates,
        num_poses=target_poses_local.shape[0],
        num_joint_seeds=num_joint_seeds,
        dof=dof,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(True)

    length_optimizer = torch.optim.AdamW([lengths], lr=learning_rate_length, weight_decay=0.0)
    joint_optimizer = torch.optim.AdamW([joint_raw], lr=learning_rate_joint, weight_decay=0.0)

    iterator = tqdm(
        range(num_iterations),
        desc=desc,
        disable=not logging,
        dynamic_ncols=True,
    )

    final_metrics: DirectIKMetrics | None = None

    for _update_idx in iterator:
        for _ in range(joint_refinement_steps):
            joint_optimizer.zero_grad(set_to_none=True)
            joint_metrics = _direct_ik_metrics(
                alpha=alpha,
                lengths=lengths.detach(),
                joint_raw=joint_raw,
                target_poses_local=target_poses_local,
                link_radius=link_radius,
                collision_weight=collision_weight,
                collision_margin=collision_margin,
                success_eps=success_eps,
                softmin_temperature=softmin_temperature,
            )
            joint_loss = joint_metrics.loss_per_candidate.mean()
            joint_loss.backward()
            joint_optimizer.step()

        length_optimizer.zero_grad(set_to_none=True)
        final_metrics = _direct_ik_metrics(
            alpha=alpha,
            lengths=lengths,
            joint_raw=joint_raw.detach(),
            target_poses_local=target_poses_local,
            link_radius=link_radius,
            collision_weight=collision_weight,
            collision_margin=collision_margin,
            success_eps=success_eps,
            softmin_temperature=softmin_temperature,
        )
        length_loss = final_metrics.loss_per_candidate.mean()
        length_loss.backward()
        length_optimizer.step()

        if logging:
            iterator.set_postfix(
                loss=f"{length_loss.item():.4f}",
                success=f"{final_metrics.pose_success_rate.mean().item():.3f}",
                se3=f"{final_metrics.best_se3_mean.mean().item():.4f}",
            )

    with torch.no_grad():
        final_metrics = _direct_ik_metrics(
            alpha=alpha,
            lengths=lengths.detach(),
            joint_raw=joint_raw.detach(),
            target_poses_local=target_poses_local,
            link_radius=link_radius,
            collision_weight=collision_weight,
            collision_margin=collision_margin,
            success_eps=success_eps,
            softmin_temperature=softmin_temperature,
        )

    return lengths.detach(), final_metrics


def _validation_generator(device: torch.device, random_seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    return generator


def _run_validation(
    *,
    processed_morphology: Tensor,
    morph: Morphology,
    task: Task,
    scene,
    base_pose_inv: Tensor,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
) -> dict:
    return run_optimization_validation(
        processed_morphology=processed_morphology.detach(),
        morph=morph,
        task=task,
        scene=scene,
        base_pose_inv=base_pose_inv,
        device=device,
        percentage_poses=percentage_poses,
        number_random_seed=number_random_seed,
        pose_sampling_generator=_validation_generator(device, random_seed),
    )


def _direct_ik_parameters(optimization_parameters: dict) -> dict[str, Any]:
    lr_fallback = float(optimization_parameters.get("learning_rate", 0.01))
    return {
        "num_iterations": int(optimization_parameters.get("num_iterations", 100)),
        "learning_rate_length": float(
            optimization_parameters.get("learning_rate_length", lr_fallback)
        ),
        "learning_rate_joint": float(
            optimization_parameters.get("learning_rate_joint", 0.05)
        ),
        "num_joint_seeds": int(optimization_parameters.get("num_joint_seeds", 16)),
        "joint_refinement_steps": int(
            optimization_parameters.get("joint_refinement_steps", 5)
        ),
        "collision_weight": float(optimization_parameters.get("collision_weight", 10.0)),
        "collision_margin": float(optimization_parameters.get("collision_margin", 0.0)),
        "success_eps": float(optimization_parameters.get("success_eps", EPS)),
        "softmin_temperature": float(
            optimization_parameters.get("softmin_temperature", 0.01)
        ),
    }


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Optimize one initial morphology's continuous MDH lengths with a direct IK baseline.

    This is a PyTorch-native baseline inspired by the RAM implicit-differentiation IK
    experiment. It does not use NRM. For each morphology step it first refines
    per-goal joint seeds against the current morphology, then updates only the
    continuous morphology parameters [a, d] using the FK pose error and a
    differentiable self-collision penalty.
    """
    params = _direct_ik_parameters(optimization_parameters)
    logging = bool(optimization_parameters.get("logging", True))
    eval_interval = int(optimization_parameters.get("eval_interval", 1))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1))
    ignore_ground = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles = bool(optimization_parameters.get("ignore_obstacles", False))

    device = morph.params.device
    dtype = morph.params.dtype

    if logging:
        print(f"[Info] Starting direct IK baseline on device {device}.")
        print(
            "[Info] "
            f"num_iterations={params['num_iterations']}, "
            f"learning_rate_length={params['learning_rate_length']}, "
            f"learning_rate_joint={params['learning_rate_joint']}, "
            f"num_joint_seeds={params['num_joint_seeds']}, "
            f"joint_refinement_steps={params['joint_refinement_steps']}, "
            f"success_eps={params['success_eps']}, "
            f"random_seed={random_seed}"
        )

    base_pose_inv, scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )
    target_poses_local = _goal_poses_in_robot_frame(task, device, dtype)
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    initial_params = morph.params.detach().clone()
    alpha = initial_params[:, 0:1].unsqueeze(0)
    lengths = initial_params[:, 1:].unsqueeze(0).clone().requires_grad_(True)

    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    joint_raw = _make_joint_seeds(
        num_candidates=1,
        num_poses=target_poses_local.shape[0],
        num_joint_seeds=params["num_joint_seeds"],
        dof=morph.params.shape[0] - 1,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(True)

    length_optimizer = torch.optim.AdamW(
        [lengths],
        lr=params["learning_rate_length"],
        weight_decay=0.0,
    )
    joint_optimizer = torch.optim.AdamW(
        [joint_raw],
        lr=params["learning_rate_joint"],
        weight_decay=0.0,
    )

    if logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    try:
        progress_bar = tqdm(
            range(params["num_iterations"]),
            desc="direct IK baseline",
            disable=not logging,
            dynamic_ncols=True,
        )

        for update_idx in progress_bar:
            for _ in range(params["joint_refinement_steps"]):
                joint_optimizer.zero_grad(set_to_none=True)
                joint_metrics = _direct_ik_metrics(
                    alpha=alpha,
                    lengths=lengths.detach(),
                    joint_raw=joint_raw,
                    target_poses_local=target_poses_local,
                    link_radius=morph.link_radius,
                    collision_weight=params["collision_weight"],
                    collision_margin=params["collision_margin"],
                    success_eps=params["success_eps"],
                    softmin_temperature=params["softmin_temperature"],
                )
                joint_loss = joint_metrics.loss_per_candidate.mean()
                joint_loss.backward()
                joint_optimizer.step()

            length_optimizer.zero_grad(set_to_none=True)
            metrics = _direct_ik_metrics(
                alpha=alpha,
                lengths=lengths,
                joint_raw=joint_raw.detach(),
                target_poses_local=target_poses_local,
                link_radius=morph.link_radius,
                collision_weight=params["collision_weight"],
                collision_margin=params["collision_margin"],
                success_eps=params["success_eps"],
                softmin_temperature=params["softmin_temperature"],
            )
            loss = metrics.loss_per_candidate.mean()

            validation_data = None
            if eval_interval > 0 and update_idx % eval_interval == 0 and logging:
                validation_data = _run_validation(
                    processed_morphology=metrics.processed_morphologies[0],
                    morph=morph,
                    task=task,
                    scene=scene,
                    base_pose_inv=base_pose_inv,
                    device=device,
                    percentage_poses=percentage_poses,
                    number_random_seed=number_random_seed,
                    random_seed=random_seed,
                )
                best_se3 = validation_data["best_se3_dist_mean"].detach().cpu().item()
                ik_success_rate = (
                    validation_data["ik_success_pose_rate"].detach().cpu().item()
                )
                tqdm.write(
                    f"[Iter {update_idx:>4}/{params['num_iterations']}] "
                    f"loss={loss.item():.6f}, "
                    f"direct_ik_success={metrics.pose_success_rate.mean().item() * 100.0:.2f}%, "
                    f"direct_se3={metrics.best_se3_mean.mean().item():.6f}, "
                    f"validation_se3={best_se3:.6f}, "
                    f"validation_ik_success={ik_success_rate * 100.0:.2f}%"
                )

            csv_logger.log_iteration(
                iteration=update_idx,
                loss=loss.detach(),
                reachability_probability=metrics.pose_success_rate.mean().detach(),
                raw_morphology=metrics.raw_morphologies[0].detach(),
                processed_morphology=metrics.processed_morphologies[0].detach(),
                validation_data=validation_data,
            )

            loss.backward()
            length_optimizer.step()

            if logging:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    success=f"{metrics.pose_success_rate.mean().item():.3f}",
                    se3=f"{metrics.best_se3_mean.mean().item():.4f}",
                )

        with torch.no_grad():
            final_metrics = _direct_ik_metrics(
                alpha=alpha,
                lengths=lengths.detach(),
                joint_raw=joint_raw.detach(),
                target_poses_local=target_poses_local,
                link_radius=morph.link_radius,
                collision_weight=params["collision_weight"],
                collision_margin=params["collision_margin"],
                success_eps=params["success_eps"],
                softmin_temperature=params["softmin_temperature"],
            )

        final_validation_data = _run_validation(
            processed_morphology=final_metrics.processed_morphologies[0],
            morph=morph,
            task=task,
            scene=scene,
            base_pose_inv=base_pose_inv,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            random_seed=random_seed,
        )

        csv_logger.log_iteration(
            iteration=params["num_iterations"],
            loss=final_metrics.loss_per_candidate.mean(),
            reachability_probability=final_metrics.pose_success_rate.mean(),
            raw_morphology=final_metrics.raw_morphologies[0],
            processed_morphology=final_metrics.processed_morphologies[0],
            validation_data=final_validation_data,
        )

        final_se3_err = (
            final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        )
        final_ik_success_rate = (
            final_validation_data["ik_success_pose_rate"].detach().cpu().item()
        )

        print(
            f"[Final direct IK baseline] "
            f"loss={final_metrics.loss_per_candidate.mean().item():.6f}, "
            f"direct_ik_success={final_metrics.pose_success_rate.mean().item() * 100.0:.2f}%, "
            f"direct_se3={final_metrics.best_se3_mean.mean().item():.6f}, "
            f"final_se3_err={final_se3_err:.6f}, "
            f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%"
        )

        optimized_morph = Morphology(
            params=final_metrics.processed_morphologies[0].detach(),
            link_radius=morph.link_radius,
        )
        return optimized_morph, csv_logger.csv_path

    finally:
        csv_logger.close()

