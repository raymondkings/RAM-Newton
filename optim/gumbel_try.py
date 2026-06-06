"""Gumbel-Softmax morphology optimisation — 5/6/7-DOF.

Jointly optimises:
  * discrete  alpha ∈ {0, π/2, −π/2}  via Gumbel-Softmax annealing
  * continuous lengths [a, d]           via AdamW + Normaliser/SquasherSTE

A per-joint scaling vector  s_i = 1 / dist(joint_i_pos, mean_target_pos)
is computed at each step and applied to the alpha-logit gradients after
backward(), equalising how much each joint's alpha contributes to EEF
displacement.  Set `scale_alpha_grad=False` to disable.

Config keys (via optimization_parameters):
    num_iterations      int    Gradient steps. Default: 100.
    learning_rate       float  AdamW LR for lengths. Default: 0.01.
    learning_rate_alpha float  AdamW LR for alpha logits. Default: 0.05.
    gumbel_tau_start    float  Initial Gumbel temperature. Default: 1.0.
    gumbel_tau_min      float  Minimum temperature. Default: 0.05.
    gumbel_tau_decay    float  Multiplicative decay per step. Default: 0.95.
    scale_alpha_grad    bool   Apply per-joint distance scaling to alpha
                               gradients. Default: True.
    collision_weight    float  Default: 10.0.
    collision_margin    float  Default: 0.0.
    eval_interval       int    cuRobo validation every N steps. Default: 1.
    random_seed         int    Default: 42.
    number_random_seed  int    Default: 32.
    percentage_poses    float  Default: 1.0.
    ignore_ground       bool   Default: False.
    ignore_obstacles    bool   Default: False.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor

from interface import Morphology, Task
from util.direct_ik_common import _collision_critical_distance
from util.kinematics import forward_kinematics
from util.nrm_model import MLP
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)

EPS = 1e-4
_PROJECT_ROOT = Path(__file__).parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"

# Discrete alpha choices: 0, +π/2, −π/2  (index 0, 1, 2)
_ALPHA_VALUES = [0.0, torch.pi / 2, -torch.pi / 2]


def _load_model(device: torch.device) -> MLP:
    metadata = json.loads((_WEIGHTS_DIR / "metadata.json").read_text())
    model = MLP(**metadata["hyperparameter"])
    model.load_state_dict(
        torch.load(_WEIGHTS_DIR / "checkpoint.pth", map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _se3_to_vector(pose: Tensor) -> Tensor:
    """SE(3) [4,4] → 9-D NRM pose vector (position + first two cols of R)."""
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


class _SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, param, threshold):
        return param * (param.abs() >= threshold).float()

    @staticmethod
    def backward(_ctx, grad_output):
        return grad_output, None


class _Normaliser(torch.autograd.Function):
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
        return (grad_output * norm - chain * (grad_output * param).sum()) / norm ** 2


def _preprocess(lengths: Tensor, link_radius: float) -> tuple[Tensor, Tensor]:
    threshold = 2.0 * link_radius
    norm = _Normaliser.apply(lengths)
    squashed = _SquasherSTE.apply(norm, threshold)
    return _Normaliser.apply(squashed), norm


def _alpha_logits_from_morph(morph_params: Tensor, device, dtype) -> Tensor:
    """Initialise alpha logits to match current discrete alpha values."""
    n_links = morph_params.shape[0]
    alpha_val = morph_params[:, 0]
    logits = torch.zeros(n_links, 3, device=device, dtype=dtype)
    logits[alpha_val.abs() < EPS, 0] = 3.0
    logits[(alpha_val - torch.pi / 2).abs() < EPS, 1] = 3.0
    logits[(alpha_val + torch.pi / 2).abs() < EPS, 2] = 3.0
    return logits


def _joint_scaling(
    processed_morphology: Tensor,
    target_poses: Tensor,
) -> Tensor:
    """Per-joint scaling vector  s_i = 1 / dist(joint_i_origin, mean_target_pos).

    Returns [n_links] tensor.  Used to rescale alpha-logit gradients so that
    each joint's alpha update has equal leverage on EEF displacement.

    Computed at zero configuration (theta=0) to avoid IK dependency.
    """
    n_links = processed_morphology.shape[0]
    device = processed_morphology.device
    dtype = processed_morphology.dtype

    zero_theta = torch.zeros(n_links, 1, device=device, dtype=dtype)
    with torch.no_grad():
        # poses: [n_links, 4, 4]; poses[i, :3, 3] = origin of joint-i frame
        poses = forward_kinematics(processed_morphology.detach(), zero_theta)
        joint_origins = poses[:, :3, 3]  # [n_links, 3]

    target_mean = target_poses[:, :3, 3].mean(dim=0)  # [3]
    dists = torch.norm(joint_origins - target_mean.unsqueeze(0), dim=-1)  # [n_links]
    return 1.0 / (dists + 1e-6)  # [n_links]


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Jointly optimise alpha (Gumbel-Softmax) and lengths (AdamW).

    Returns:
        optimized_morphology: final morphology with hard (discrete) alpha.
        csv_path:             path to output/log_<time>.csv.
    """
    n_iter          = int(optimization_parameters.get("num_iterations", 100))
    lr_lengths      = float(optimization_parameters.get("learning_rate", 0.01))
    lr_alpha        = float(optimization_parameters.get("learning_rate_alpha", 0.05))
    logging         = bool(optimization_parameters.get("logging", True))
    eval_interval   = int(optimization_parameters.get("eval_interval", 1))
    random_seed     = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses   = float(optimization_parameters.get("percentage_poses", 1.0))
    ignore_ground      = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles   = bool(optimization_parameters.get("ignore_obstacles", False))
    collision_weight   = float(optimization_parameters.get("collision_weight", 10.0))
    collision_margin   = float(optimization_parameters.get("collision_margin", 0.0))
    tau_start          = float(optimization_parameters.get("gumbel_tau_start", 1.0))
    tau_min            = float(optimization_parameters.get("gumbel_tau_min", 0.05))
    tau_decay          = float(optimization_parameters.get("gumbel_tau_decay", 0.95))
    scale_alpha_grad   = bool(optimization_parameters.get("scale_alpha_grad", True))

    device = morph.params.device
    dtype  = morph.params.dtype
    n_links = morph.params.shape[0]

    # Alpha discrete choices as a [3] tensor for soft weighted sum
    alpha_choices = torch.tensor(
        _ALPHA_VALUES, device=device, dtype=dtype
    )  # [3]

    # ── initialise optimisable parameters ────────────────────────────────────
    alpha_logits = _alpha_logits_from_morph(morph.params, device, dtype)
    alpha_logits.requires_grad_(True)

    lengths = morph.params[:, 1:].clone().to(device).requires_grad_(True)

    # Two separate optimisers: different LRs for discrete vs continuous params
    opt_lengths = torch.optim.AdamW([lengths],      lr=lr_lengths, weight_decay=0.0)
    opt_alpha   = torch.optim.AdamW([alpha_logits], lr=lr_alpha,   weight_decay=0.0)

    model    = _load_model(device)
    task_vec = _se3_to_vector(task.goal_poses.to(device))  # [N_poses, 9]

    scene = build_optimization_validation_context(
        task=task, device=device,
        ignore_ground=ignore_ground, ignore_obstacles=ignore_obstacles,
    )

    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)
    pose_sampling_generator = torch.Generator(device=device)
    pose_sampling_generator.manual_seed(random_seed)

    tau = tau_start

    if logging:
        print(
            f"[Gumbel] {n_links}-link robot, {n_iter} iters on device={device}\n"
            f"[Gumbel] tau: {tau_start}→{tau_min} (decay={tau_decay:.3f}), "
            f"lr_lengths={lr_lengths}, lr_alpha={lr_alpha}, "
            f"scale_alpha_grad={scale_alpha_grad}\n"
            f"[Gumbel] Writing CSV to: {csv_logger.csv_path}"
        )

    try:
        progress_bar = tqdm(range(n_iter), desc="Gumbel optimising", dynamic_ncols=True)

        for update_idx in progress_bar:
            opt_lengths.zero_grad()
            opt_alpha.zero_grad()

            # ── soft alpha via Gumbel-Softmax ─────────────────────────────────
            # alpha_soft: [n_links, 3]  (soft one-hot, differentiable)
            alpha_soft = F.gumbel_softmax(alpha_logits, tau=tau, hard=False, dim=-1)
            alpha = (alpha_soft * alpha_choices).sum(dim=-1, keepdim=True)  # [n_links, 1]

            # ── preprocess lengths ────────────────────────────────────────────
            processed_lengths, _ = _preprocess(lengths, morph.link_radius)
            raw_morphology       = torch.cat([alpha.detach(), lengths.detach()], dim=1)
            processed_morphology = torch.cat([alpha, processed_lengths], dim=1)

            # ── NRM BCE loss ──────────────────────────────────────────────────
            bmorph = processed_morphology.unsqueeze(0).expand(task_vec.shape[0], -1, -1)
            logit  = model(bmorph, task_vec)  # [N_poses]
            loss   = F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
            prob   = torch.sigmoid(logit).mean()

            # ── differentiable self-collision penalty ─────────────────────────
            zero_theta = torch.zeros(n_links, 1, device=device, dtype=dtype)
            zero_poses = forward_kinematics(processed_morphology, zero_theta)
            critical_dist = _collision_critical_distance(
                processed_morphology.unsqueeze(0),
                zero_poses.unsqueeze(0),
                morph.link_radius,
            )
            col_penalty = F.relu(collision_margin - critical_dist).mean()
            loss = loss + collision_weight * col_penalty

            # ── backward ─────────────────────────────────────────────────────
            loss.backward()

            # ── per-joint alpha gradient scaling ─────────────────────────────
            # scaling[i] = 1 / dist(joint_i_origin, mean_target_pos)
            # Joints far from the target have large lever-arm → naturally larger
            # gradient magnitude.  Scaling by 1/dist equalises the per-joint
            # alpha update magnitude.
            # If NRM gradients already encode this (i.e. the gradient norms
            # already correlate with distance), setting scale_alpha_grad=False
            # disables this pre-conditioning step.
            if scale_alpha_grad and alpha_logits.grad is not None:
                scaling = _joint_scaling(
                    processed_morphology.detach(),
                    task.goal_poses.to(device),
                )  # [n_links]
                # Normalise so the mean scale factor = 1 (preserves effective LR)
                scaling = scaling / (scaling.mean() + 1e-8)
                alpha_logits.grad.mul_(scaling.unsqueeze(-1))

            opt_lengths.step()
            opt_alpha.step()
            tau = max(tau_min, tau * tau_decay)

            with torch.no_grad():
                lengths[-1, 1].clamp_(min=0.0)

            # ── validation ────────────────────────────────────────────────────
            validation_data = None
            if eval_interval > 0 and update_idx % eval_interval == 0 and logging:
                validation_data = run_optimization_validation(
                    processed_morphology=processed_morphology.detach(),
                    morph=morph, task=task, scene=scene, device=device,
                    percentage_poses=percentage_poses,
                    number_random_seed=number_random_seed,
                    pose_sampling_generator=pose_sampling_generator,
                )
                tqdm.write(
                    f"[Iter {update_idx:>4}/{n_iter}] "
                    f"loss={loss.item():.6f}, prob={prob.item():.4f}, "
                    f"tau={tau:.4f}, "
                    f"val_se3={validation_data['best_se3_dist_mean'].item():.6f}"
                )

            csv_logger.log_iteration(
                iteration=update_idx,
                loss=loss.detach(),
                reachability_probability=prob.detach(),
                raw_morphology=raw_morphology.detach(),
                processed_morphology=processed_morphology.detach(),
                validation_data=validation_data,
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                prob=f"{prob.item():.3f}",
                tau=f"{tau:.3f}",
            )

        # ── final: snap alpha to argmax (hard discrete) ───────────────────────
        with torch.no_grad():
            best_cls = alpha_logits.argmax(dim=-1)          # [n_links]
            alpha_hard = alpha_choices[best_cls].unsqueeze(-1)  # [n_links, 1]

            final_processed_lengths, _ = _preprocess(lengths.detach(), morph.link_radius)
            final_processed_morphology = torch.cat([alpha_hard, final_processed_lengths], dim=1)
            final_raw_morphology       = torch.cat([alpha_hard, lengths.detach()], dim=1)

            final_bmorph = final_processed_morphology.unsqueeze(0).expand(
                task_vec.shape[0], -1, -1
            )
            final_logit = model(final_bmorph, task_vec)
            final_loss  = F.binary_cross_entropy_with_logits(
                final_logit, torch.ones_like(final_logit)
            )
            final_prob  = torch.sigmoid(final_logit).mean()

        final_validation_data = run_optimization_validation(
            processed_morphology=final_processed_morphology,
            morph=morph, task=task, scene=scene, device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=pose_sampling_generator,
        )

        csv_logger.log_iteration(
            iteration=n_iter,
            loss=final_loss,
            reachability_probability=final_prob,
            raw_morphology=final_raw_morphology,
            processed_morphology=final_processed_morphology,
            validation_data=final_validation_data,
        )

        final_se3_err = final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        final_ik_rate = final_validation_data["ik_success_pose_rate"].detach().cpu().item()

        print(
            f"[Iter {n_iter:>4}/{n_iter}] "
            f"loss={final_loss.item():.6f}, prob={final_prob.item():.4f}, "
            f"final_val_se3={final_se3_err:.6f}, "
            f"ik_success_rate={final_ik_rate * 100:.2f}%"
        )
        print(
            f"[Gumbel] Final alpha (discrete): "
            f"{alpha_hard.squeeze(-1).tolist()}"
        )

        return (
            Morphology(
                params=final_processed_morphology.detach(),
                link_radius=morph.link_radius,
            ),
            csv_logger.csv_path,
        )

    finally:
        csv_logger.close()
