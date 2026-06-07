# -----------------------------------------------------------------------------
#   - final validation always runs once after the last optimizer step.
#   - CSV logging uses raw_morphology / processed_morphology only.
#
# Alpha behaviour:
#   - during optimisation, alpha is parameterised as logits over {0, π/2, -π/2}
#     and relaxed with Gumbel-Softmax (temperature anneals from tau_start → tau_min).
#   - the final returned morphology snaps alpha to the argmax (hard discrete).
# -----------------------------------------------------------------------------

import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor

from util.nrm_model import MLP
from util.direct_ik_common import (
    _collision_critical_distance,
    _make_joint_seeds,
    _wrap_joints,
    _append_fixed_ee_joint,
    _pose_se3_distance,
)
from util.kinematics import forward_kinematics
from interface import Morphology, Task
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import run_optimization_validation


EPS = 1e-4

_PROJECT_ROOT = Path(__file__).parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"

# Discrete alpha choices: index 0 → 0, index 1 → +π/2, index 2 → -π/2
_ALPHA_CHOICES = torch.tensor([0.0, math.pi / 2, -math.pi / 2])


def _se3_to_vector(pose: Tensor) -> Tensor:
    """Convert SE(3) matrices [..., 4, 4] to 9D NRM pose vectors."""
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


def _load_model(device: torch.device) -> MLP:
    """Load pretrained NRM model on the same device as the morphology."""
    metadata = json.loads((_WEIGHTS_DIR / "metadata.json").read_text())

    model = MLP(**metadata["hyperparameter"])
    model.load_state_dict(
        torch.load(
            _WEIGHTS_DIR / "checkpoint.pth", map_location=device, weights_only=True
        )
    )
    model = model.to(device)

    # cuDNN LSTM backward may require train mode.
    # The weights are frozen, but gradients still flow to the morphology input.
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


class SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, param: Tensor, threshold: float) -> Tensor:
        mask = (param.abs() >= threshold).float()
        return param * mask

    @staticmethod
    def backward(_ctx, grad_output: Tensor):
        return grad_output, None


class Normaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param: Tensor) -> Tensor:
        l2_norm = torch.hypot(param[:, 0:1], param[:, 1:2])
        norm = l2_norm.sum(dim=0, keepdim=True)
        ctx.save_for_backward(param, l2_norm, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
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


def _init_alpha_logits(morph_params: Tensor, device, dtype) -> Tensor:
    """Initialise alpha logits to strongly favour the current discrete alpha value."""
    n_links = morph_params.shape[0]
    alpha_val = morph_params[:, 0]
    logits = torch.zeros(n_links, 3, device=device, dtype=dtype)
    logits[alpha_val.abs() < EPS, 0] = 3.0  # → 0
    logits[(alpha_val - math.pi / 2).abs() < EPS, 1] = 3.0  # → +π/2
    logits[(alpha_val + math.pi / 2).abs() < EPS, 2] = 3.0  # → -π/2
    return logits


def _hard_alpha(alpha_logits: Tensor, choices: Tensor) -> Tensor:
    """Argmax snap: return [n_links, 1] hard discrete alpha from logits."""
    return choices[alpha_logits.argmax(dim=-1)].unsqueeze(-1)


def _compute_loss_and_prob(
    model: MLP,
    processed_morphology: Tensor,
    task_vec: Tensor,
) -> tuple[Tensor, Tensor]:
    bmorph = processed_morphology.unsqueeze(0).expand(task_vec.shape[0], -1, -1)
    logit = model(bmorph, task_vec)
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(logit, torch.ones_like(logit))
    prob = torch.sigmoid(logit).mean()
    return loss, prob


def _format_alpha_degrees(alpha: Tensor) -> str:
    vals = alpha.detach().cpu().squeeze(-1) * 180.0 / math.pi
    return "[" + ", ".join(f"{v:.1f}°" for v in vals.tolist()) + "]"


def _optimize_morphology_impl(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Core Gumbel-Softmax morphology optimisation — called by DOF-specific wrappers.

    Alpha is parameterised as logits over {0, π/2, -π/2} and soft-sampled
    via Gumbel-Softmax.  Temperature decays from `gumbel_tau_start` to
    `gumbel_tau_min` over the course of training.  The final morphology
    uses the hard argmax over the learned logits.
    """
    n_iter = optimization_parameters.get("num_iterations", 100)
    lr_fallback = optimization_parameters.get("learning_rate", 0.01)
    lr_angle = optimization_parameters.get("learning_rate_angle", lr_fallback)
    lr_length = optimization_parameters.get("learning_rate_length", lr_fallback)
    logging = optimization_parameters.get("logging", True)
    eval_interval = int(optimization_parameters.get("eval_interval", 1))
    random_seed = optimization_parameters.get("random_seed", 42)
    number_random_seed = optimization_parameters.get("number_random_seed", 32)
    percentage_poses = optimization_parameters.get("percentage_poses", 1)
    collision_weight = float(optimization_parameters.get("collision_weight", 10.0))
    collision_margin = float(optimization_parameters.get("collision_margin", 0.0))
    tau_start = float(optimization_parameters.get("gumbel_tau_start", 1.0))
    tau_min = float(optimization_parameters.get("gumbel_tau_min", 0.05))
    tau_decay = float(optimization_parameters.get("gumbel_tau_decay", 0.95))
    collision_seeds = int(optimization_parameters.get("collision_seeds", 8))
    joint_refinement_steps = int(
        optimization_parameters.get("joint_refinement_steps", 3)
    )
    lr_joint = float(optimization_parameters.get("learning_rate_joint", 0.05))
    # Phase 1: optimize lengths with alpha fixed before Gumbel phase begins.
    # This ensures a/d get clean gradient signal before Gumbel noise is introduced.
    warmup_iters = int(optimization_parameters.get("warmup_iters", 30))

    device = morph.params.device
    dtype = morph.params.dtype

    if logging:
        print(
            f"[Info] Starting alpha-Gumbel optimization with {n_iter} iterations on device {device}."
        )
        print(
            "[Info] "
            f"lr_angle={lr_angle}, "
            f"lr_length={lr_length}, "
            f"tau: {tau_start}→{tau_min} (decay={tau_decay}), "
            f"eval_interval={eval_interval}, "
            f"random_seed={random_seed}, "
            f"number_random_seed={number_random_seed}, "
            f"percentage_poses={percentage_poses}"
        )

    scene = None

    task_vec = _se3_to_vector(task.goal_poses.to(device))

    # Alpha choices on the correct device/dtype
    choices = _ALPHA_CHOICES.to(device=device, dtype=dtype)  # [3]

    # Discrete alpha → logits initialised to favour the current morph's alpha
    alpha_logits = _init_alpha_logits(morph.params, device, dtype)
    alpha_logits.requires_grad_(True)

    lengths = morph.params[:, 1:].clone().to(device)
    lengths.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [
            {"params": [alpha_logits], "lr": lr_angle},
            {"params": [lengths], "lr": lr_length},
        ]
    )
    model = _load_model(device)
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    pose_sampling_generator = torch.Generator(device=device)
    pose_sampling_generator.manual_seed(random_seed)

    n_links = morph.params.shape[0]
    dof = n_links - 1
    n_poses = task_vec.shape[0]
    goal_poses_local = task.goal_poses.to(device=device, dtype=dtype)

    _joint_gen = torch.Generator(device=device)
    _joint_gen.manual_seed(random_seed)
    joint_raw = (
        _make_joint_seeds(
            num_candidates=1,
            num_poses=n_poses,
            num_joint_seeds=collision_seeds,
            dof=dof,
            device=device,
            dtype=dtype,
            generator=_joint_gen,
        )
        .squeeze(0)
        .requires_grad_(True)
    )  # [n_poses, collision_seeds, dof, 1]
    joint_optimizer = torch.optim.AdamW([joint_raw], lr=lr_joint)

    tau = tau_start

    if logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    try:
        progress_bar = tqdm(
            range(n_iter),
            desc="optimizing alpha-gumbel",
            dynamic_ncols=True,
            disable=not logging,
        )

        for update_idx in progress_bar:
            optimizer.zero_grad()

            in_warmup = update_idx < warmup_iters

            if in_warmup:
                # Phase 1: alpha fixed at hard discrete, only lengths get gradients.
                alpha = _hard_alpha(alpha_logits.detach(), choices)
            else:
                # Phase 2: Gumbel-Softmax relaxation for joint alpha + length co-optimisation.
                alpha_soft = F.gumbel_softmax(alpha_logits, tau=tau, hard=False, dim=-1)
                alpha = (alpha_soft * choices).sum(dim=-1, keepdim=True)

            processed_lengths, _ = _preprocess(lengths, morph.link_radius)
            raw_morphology = torch.cat([alpha.detach(), lengths.detach()], dim=1)
            processed_morphology = torch.cat([alpha, processed_lengths], dim=1)

            loss, prob = _compute_loss_and_prob(model, processed_morphology, task_vec)

            # Self-collision penalty: inner IK drives joint_raw toward goal poses,
            # then morphology gradient flows through collision at those configs.
            inner_morph = (
                processed_morphology.detach()
                .unsqueeze(0)
                .expand(n_poses * collision_seeds, -1, -1)
            )
            for _ in range(joint_refinement_steps):
                joint_optimizer.zero_grad()
                full_j = _append_fixed_ee_joint(
                    _wrap_joints(joint_raw.reshape(n_poses * collision_seeds, dof, 1))
                )
                reached = forward_kinematics(inner_morph, full_j)
                ee = reached[:, -1].reshape(n_poses, collision_seeds, 4, 4)
                target = goal_poses_local[:, None].expand_as(ee)
                se3_inner = _pose_se3_distance(ee, target)
                se3_inner.mean().backward()
                joint_optimizer.step()

            flat_morph = processed_morphology.unsqueeze(0).expand(
                n_poses * collision_seeds, -1, -1
            )
            full_j = _append_fixed_ee_joint(
                _wrap_joints(
                    joint_raw.detach().reshape(n_poses * collision_seeds, dof, 1)
                )
            )
            reached = forward_kinematics(flat_morph, full_j)
            critical_dist = _collision_critical_distance(
                flat_morph, reached, morph.link_radius
            )
            col_penalty = F.relu(collision_margin - critical_dist).mean()
            loss = loss + collision_weight * col_penalty

            validation_data = None
            if logging and eval_interval > 0 and update_idx % eval_interval == 0:
                validation_data = run_optimization_validation(
                    processed_morphology=processed_morphology.detach(),
                    morph=morph,
                    task=task,
                    scene=scene,
                    device=device,
                    percentage_poses=percentage_poses,
                    number_random_seed=number_random_seed,
                    pose_sampling_generator=pose_sampling_generator,
                )

                best_se3 = validation_data["best_se3_dist_mean"].detach().cpu().item()
                ik_success_rate = (
                    validation_data["ik_success_pose_rate"].detach().cpu().item()
                )
                tqdm.write(
                    f"[Iter {update_idx:>4}/{n_iter}] "
                    f"loss={loss.item():.6f}, "
                    f"nrm_prob={prob.item():.6f}, "
                    f"tau={tau:.4f}, "
                    f"best_se3={best_se3:.6f}, "
                    f"ik_success_pose_rate={ik_success_rate * 100.0:.2f}%"
                )

            csv_logger.log_iteration(
                iteration=update_idx,
                loss=loss.detach(),
                reachability_probability=prob.detach(),
                raw_morphology=raw_morphology.detach(),
                processed_morphology=processed_morphology.detach(),
                validation_data=validation_data,
            )

            loss.backward()
            if in_warmup:
                # Don't let AdamW accumulate stale momentum on alpha_logits during warmup.
                if alpha_logits.grad is not None:
                    alpha_logits.grad.zero_()
            optimizer.step()
            if not in_warmup:
                tau = max(tau_min, tau * tau_decay)

            with torch.no_grad():
                lengths[-1, 1].clamp_(min=0.0)

            if logging:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    prob=f"{prob.item():.3f}",
                    tau=f"{tau:.3f}",
                )

        # Final state: snap alpha logits → argmax (hard discrete)
        with torch.no_grad():
            final_alpha = _hard_alpha(alpha_logits.detach(), choices)  # [n_links, 1]
            final_processed_lengths, _ = _preprocess(
                lengths.detach(), morph.link_radius
            )

            final_raw_morphology = torch.cat([final_alpha, lengths.detach()], dim=1)
            final_processed_morphology = torch.cat(
                [final_alpha, final_processed_lengths], dim=1
            )

            final_loss, final_prob = _compute_loss_and_prob(
                model, final_processed_morphology, task_vec
            )

        # ALWAYS do a validation for the final result!
        final_validation_data = run_optimization_validation(
            processed_morphology=final_processed_morphology.detach(),
            morph=morph,
            task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=pose_sampling_generator,
        )

        csv_logger.log_iteration(
            iteration=n_iter,
            loss=final_loss.detach(),
            reachability_probability=final_prob.detach(),
            raw_morphology=final_raw_morphology.detach(),
            processed_morphology=final_processed_morphology.detach(),
            validation_data=final_validation_data,
        )

        final_se3_err = (
            final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        )
        final_ik_success_rate = (
            final_validation_data["ik_success_pose_rate"].detach().cpu().item()
        )
        print(
            f"[Iter {n_iter:>4}/{n_iter}] "
            f"loss={final_loss.item():.6f}, "
            f"nrm_prob={final_prob.item():.6f}, "
            f"final_se3_err={final_se3_err:.6f}, "
            f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%"
        )

        if logging:
            print(f"Final alpha (hard discrete): {_format_alpha_degrees(final_alpha)}")
            print("Final optimized morphology params:")
            print(final_processed_morphology.detach().cpu())

        optimized_morph = Morphology(
            params=final_processed_morphology.detach(),
            link_radius=morph.link_radius,
        )
        return optimized_morph, csv_logger.csv_path

    finally:
        csv_logger.close()


# ---------------------------------------------------------------------------
# Per-DOF defaults (override any key in optimization_parameters)
# ---------------------------------------------------------------------------

_DOF_DEFAULTS: dict[int, dict] = {
    5: {
        "gumbel_tau_start": 1.0,
        "gumbel_tau_min": 0.05,
        "gumbel_tau_decay": 0.97,
        "learning_rate_angle": 0.008,
        "learning_rate_length": 0.008,
        "collision_weight": 10.0,
        "collision_seeds": 8,
        "joint_refinement_steps": 3,
    },
    6: {
        "gumbel_tau_start": 1.0,
        "gumbel_tau_min": 0.05,
        "gumbel_tau_decay": 0.95,
        "learning_rate_angle": 0.005,
        "learning_rate_length": 0.005,
        "collision_weight": 10.0,
        "collision_seeds": 8,
        "joint_refinement_steps": 3,
    },
    7: {
        "gumbel_tau_start": 1.0,
        "gumbel_tau_min": 0.05,
        "gumbel_tau_decay": 0.93,
        "learning_rate_angle": 0.003,
        "learning_rate_length": 0.003,
        "collision_weight": 15.0,
        "collision_seeds": 12,
        "joint_refinement_steps": 4,
    },
}


def _merge_params(dof: int, user_params: dict) -> dict:
    """Merge DOF defaults with user-supplied params (user always wins)."""
    merged = dict(_DOF_DEFAULTS.get(dof, {}))
    merged.update(user_params)
    return merged


def optimize_morphology_5dof(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Gumbel-Softmax optimisation for 5-DOF morphologies."""
    return _optimize_morphology_impl(
        morph, task, _merge_params(5, optimization_parameters)
    )


def optimize_morphology_6dof(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Gumbel-Softmax optimisation for 6-DOF morphologies."""
    return _optimize_morphology_impl(
        morph, task, _merge_params(6, optimization_parameters)
    )


def optimize_morphology_7dof(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Gumbel-Softmax optimisation for 7-DOF morphologies."""
    return _optimize_morphology_impl(
        morph, task, _merge_params(7, optimization_parameters)
    )


_DOF_DISPATCH: dict[int, Callable] = {
    5: optimize_morphology_5dof,
    6: optimize_morphology_6dof,
    7: optimize_morphology_7dof,
}


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Dispatcher: reads DOF from morph and calls the matching DOF-specific optimiser."""
    dof = morph.params.shape[0] - 1
    fn = _DOF_DISPATCH.get(dof)
    if fn is None:
        raise ValueError(
            f"nrm_alpha_gumbel: no optimiser registered for DOF={dof}. "
            f"Supported: {sorted(_DOF_DISPATCH)}."
        )
    return fn(morph, task, optimization_parameters)
