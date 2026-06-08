# -----------------------------------------------------------------------------
#   - final validation always runs once after the last optimizer step.
#   - CSV logging uses raw_morphology / processed_morphology only.
#
# Alpha behaviour:
#   - during optimisation, alpha is parameterised as logits over {0, π/2, -π/2}
#     and relaxed with Gumbel-Softmax (temperature anneals from tau_start → tau_min).
#   - the final returned morphology snaps alpha to the argmax (hard discrete).
# -----------------------------------------------------------------------------
"""
LEGACY:
    If you want to run this code, simply put in inside the optim folder.
    This method simply doesn't work, alpha will almost not change, and even if
    deliberately set very strange hyperparameter to get a different alpha, the
    result (se3 error/ ik successrate/ loss) is also not better.
"""

import json
import math
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor

from optim.model import MLP
from interface import Morphology, Task
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import run_optimization_validation


EPS = 1e-4
DELTA_EARLY_STOPPING = 1e-5
EARLY_STOPPING_PATIENCE = 50

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
    loss = loss + 100.0 * torch.relu(-processed_morphology[-1, 2])
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
    tau_start = float(optimization_parameters.get("gumbel_tau_start", 1.0))
    tau_min = float(optimization_parameters.get("gumbel_tau_min", 0.05))
    tau_decay = float(optimization_parameters.get("gumbel_tau_decay", 0.95))
    # Phase 1: optimize lengths with alpha fixed before Gumbel phase begins.
    # This ensures a/d get clean gradient signal before Gumbel noise is introduced.
    warmup_iters = int(optimization_parameters.get("warmup_iters", 30))
    early_stopping_patience = int(
        optimization_parameters.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
    )
    delta_early_stopping = float(
        optimization_parameters.get("delta_early_stopping", DELTA_EARLY_STOPPING)
    )

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
            f"percentage_poses={percentage_poses}, "
            f"early_stopping_patience={early_stopping_patience}, "
            f"delta_early_stopping={delta_early_stopping}"
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

    tau = tau_start

    if logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    best_loss = float("inf")
    best_alpha_logits = alpha_logits.detach().clone()
    best_lengths = lengths.detach().clone()
    best_iteration = 0
    stale_count = 0
    final_iteration = n_iter

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
            loss_value = float(loss.detach().cpu().item())

            improved = loss_value < best_loss - delta_early_stopping
            if improved:
                best_loss = loss_value
                best_alpha_logits = alpha_logits.detach().clone()
                best_lengths = lengths.detach().clone()
                best_iteration = update_idx
                stale_count = 0
            elif not in_warmup:
                stale_count += 1

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

            should_stop = (
                early_stopping_patience > 0
                and not in_warmup
                and stale_count >= early_stopping_patience
                and (update_idx + 1) < n_iter
            )
            if should_stop:
                final_iteration = update_idx + 1
                if logging:
                    tqdm.write(
                        "[Early stopping] "
                        f"loss did not improve for {stale_count} updates; "
                        f"stopping at iteration {final_iteration}. "
                        f"best_iteration={best_iteration}, best_loss={best_loss:.6f}"
                    )
                break

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
                    stale=stale_count,
                )

        # Final state: snap best alpha logits → argmax (hard discrete)
        with torch.no_grad():
            final_alpha = _hard_alpha(best_alpha_logits, choices)  # [n_links, 1]
            final_processed_lengths, _ = _preprocess(best_lengths, morph.link_radius)

            final_raw_morphology = torch.cat([final_alpha, best_lengths], dim=1)
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
            iteration=final_iteration,
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
            f"[Iter {final_iteration:>4}/{n_iter}] "
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
        "gumbel_tau_decay": 0.999,
        "learning_rate_angle": 0.01,
        "learning_rate_length": 0.1,
    },
    6: {
        "gumbel_tau_start": 1.0,
        "gumbel_tau_min": 0.05,
        "gumbel_tau_decay": 0.999,
        "learning_rate_angle": 0.01,
        "learning_rate_length": 0.1,
    },
    7: {
        "gumbel_tau_start": 1.0,
        "gumbel_tau_min": 0.05,
        "gumbel_tau_decay": 0.999,
        "learning_rate_angle": 0.01,
        "learning_rate_length": 0.1,
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
