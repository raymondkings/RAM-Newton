# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
#   - final validation always runs once after the last optimizer step.
#   - CSV logging uses raw_morphology / processed_morphology only.
#
# Alpha behaviour:
#   - during optimization, NRM sees continuous alpha directly.
#   - the final returned morphology maps alpha to {-pi/2, 0, pi/2}.
# -----------------------------------------------------------------------------

import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from torch import Tensor

from optim.model import MLP
from interface import Morphology, Task
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)


EPS = 1e-4

_PROJECT_ROOT = Path(__file__).parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"


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


def _map_alpha_nearest(alpha: Tensor) -> Tensor:
    """Nearest-map alpha to {-pi/2, 0, pi/2}. No STE is needed for final output."""
    levels = alpha.new_tensor([-math.pi / 2.0, 0.0, math.pi / 2.0])
    distances = (alpha.unsqueeze(-1) - levels.view(*([1] * alpha.ndim), 3)).abs()
    indices = distances.argmin(dim=-1)
    return levels[indices]


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


def _format_alpha_degrees(alpha: Tensor) -> Tensor:
    return alpha.detach().cpu().squeeze(-1) * 180.0 / math.pi


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Optimize morphology with continuous alpha during NRM forward."""
    n_iter = optimization_parameters.get("num_iterations", 100)
    lr_fallback = optimization_parameters.get("learning_rate", 0.01)
    lr_angle = optimization_parameters.get("learning_rate_angle", lr_fallback)
    lr_length = optimization_parameters.get("learning_rate_length", lr_fallback)
    logging = optimization_parameters.get("logging", True)
    eval_interval = int(optimization_parameters.get("eval_interval", 1))
    random_seed = optimization_parameters.get("random_seed", 42)
    number_random_seed = optimization_parameters.get("number_random_seed", 32)
    percentage_poses = optimization_parameters.get("percentage_poses", 1)
    ignore_ground = optimization_parameters.get("ignore_ground", False)
    ignore_obstacles = optimization_parameters.get("ignore_obstacles", False)

    device = morph.params.device

    if logging:
        print(
            f"[Info] Starting alpha-continuous optimization with {n_iter} iterations on device {device}."
        )
        print(
            "[Info] "
            f"lr_angle={lr_angle}, "
            f"lr_length={lr_length}, "
            f"eval_interval={eval_interval}, "
            f"random_seed={random_seed}, "
            f"number_random_seed={number_random_seed}, "
            f"percentage_poses={percentage_poses}"
        )

    scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    task_vec = _se3_to_vector(task.goal_poses.to(device))

    alpha = morph.params[:, 0:1].clone().to(device)
    lengths = morph.params[:, 1:].clone().to(device)
    alpha.requires_grad_(True)
    lengths.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [
            {"params": [alpha], "lr": lr_angle},
            {"params": [lengths], "lr": lr_length},
        ]
    )
    model = _load_model(device)
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    pose_sampling_generator = torch.Generator(device=device)
    pose_sampling_generator.manual_seed(random_seed)

    if logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    try:
        progress_bar = tqdm(
            range(n_iter),
            desc="optimizing alpha-continuous",
            dynamic_ncols=True,
            disable=not logging,
        )

        for update_idx in progress_bar:
            # Current state is after update_idx optimizer steps.
            optimizer.zero_grad()

            processed_lengths, _ = _preprocess(lengths, morph.link_radius)
            raw_morphology = torch.cat([alpha, lengths], dim=1)
            processed_morphology = torch.cat([alpha, processed_lengths], dim=1)

            loss, prob = _compute_loss_and_prob(model, processed_morphology, task_vec)

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
            optimizer.step()

            if logging:
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}", prob=f"{prob.item():.3f}"
                )

        # Final state after n_iter optimizer steps.
        with torch.no_grad():
            final_alpha_raw = alpha.detach().clone()
            final_alpha_mapped = _map_alpha_nearest(final_alpha_raw)
            final_processed_lengths, _ = _preprocess(
                lengths.detach(), morph.link_radius
            )

            final_raw_morphology = torch.cat([final_alpha_raw, lengths.detach()], dim=1)
            final_processed_morphology = torch.cat(
                [final_alpha_mapped, final_processed_lengths], dim=1
            )

            final_loss, final_prob = _compute_loss_and_prob(
                model, final_processed_morphology, task_vec
            )

        # ALWAYS do a validation for the final data!
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
            print("\nFinal alpha raw before nearest-map [deg]:")
            print(_format_alpha_degrees(final_alpha_raw))
            print("Final alpha after nearest-map [deg]:")
            print(_format_alpha_degrees(final_alpha_mapped))
            print("Final optimized morphology params:")
            print(final_processed_morphology.detach().cpu())

        optimized_morph = Morphology(
            params=final_processed_morphology.detach(),
            link_radius=morph.link_radius,
        )
        return optimized_morph, csv_logger.csv_path

    finally:
        csv_logger.close()
