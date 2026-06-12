# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Julian Arkenau / Shiyuan Zhang
# -----------------------------------------------------------------------------
# This version keeps the original optimization structure, but removes plotting
# and recorder/timelapse creation. All data needed for later plotting/video
# reconstruction is written to output/log_<time>.csv.
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import torch
from tqdm import tqdm
from torch import Tensor

from optim.model import MLP
from interface import Morphology, Task
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import run_optimization_validation


EPS = 1e-4
DELTA_EARLY_STOPPING = 1e-5
EARLY_STOPPING_PATIENCE = 10

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
            _WEIGHTS_DIR / "checkpoint_5-7.pth", map_location=device, weights_only=True
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
    """Apply normalize -> squash -> normalize to raw [a, d] lengths.

    Returns:
        processed_lengths:
            normalize -> squash -> normalize result.
        norm_lengths:
            normalize-only result, useful for CSV/debugging.
    """
    threshold = 2.0 * link_radius

    norm_lengths = Normaliser.apply(lengths)
    squashed = SquasherSTE.apply(norm_lengths, threshold)

    return Normaliser.apply(squashed), norm_lengths


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Optimize morphology link lengths (a, d) for the given task.

    Returns:
        optimized_morphology:
            Final processed morphology.
        csv_path:
            Path to output/log_<time>.csv.
    """
    n_iter = optimization_parameters.get("num_iterations", 100)
    lr = optimization_parameters.get("learning_rate", 0.01)
    logging = optimization_parameters.get("logging", True)
    eval_interval = optimization_parameters.get("eval_interval", 1)
    eval_interval = int(eval_interval)
    random_seed = optimization_parameters.get("random_seed", 42)
    number_random_seed = optimization_parameters.get("number_random_seed", 32)
    percentage_poses = optimization_parameters.get("percentage_poses", 1)
    early_stopping_patience = int(
        optimization_parameters.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
    )
    delta_early_stopping = float(
        optimization_parameters.get("delta_early_stopping", DELTA_EARLY_STOPPING)
    )

    device = morph.params.device

    if logging:
        print(
            f"[Info] Starting morphology optimization with {n_iter} iterations on device {device}."
        )
        print(
            "[Info] "
            f"eval_interval={eval_interval}, "
            f"random_seed={random_seed}, "
            f"number_random_seed={number_random_seed}, "
            f"percentage_poses={percentage_poses}, "
            f"early_stopping_patience={early_stopping_patience}, "
            f"delta_early_stopping={delta_early_stopping}"
        )

    scene = None

    task_vec = _se3_to_vector(task.goal_poses.to(device))

    alpha = morph.params[:, 0:1].clone().to(device)
    lengths = morph.params[:, 1:].clone().to(device)
    lengths.requires_grad_(True)

    optimizer = torch.optim.AdamW([lengths], lr=lr)
    model = _load_model(device)

    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    # for pose sampling (each validation sample different poses if not full pose validation)
    pose_sampling_generator = torch.Generator(device=device)
    pose_sampling_generator.manual_seed(random_seed)

    if logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    best_loss = float("inf")
    best_lengths = lengths.detach().clone()
    best_iteration = 0
    stale_count = 0
    final_iteration = n_iter

    try:
        progress_bar = tqdm(range(n_iter), desc="optimizing", dynamic_ncols=True)
        for update_idx in progress_bar:
            # Current state is after update_idx optimizer steps.
            optimizer.zero_grad()

            processed_lengths, _ = _preprocess(lengths, morph.link_radius)
            raw_morphology = torch.cat([alpha, lengths], dim=1)
            processed_morphology = torch.cat([alpha, processed_lengths], dim=1)

            bmorph = processed_morphology.unsqueeze(0).expand(task_vec.shape[0], -1, -1)
            logit = model(bmorph, task_vec)

            loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(
                logit, torch.ones_like(logit)
            )
            loss = loss + 100.0 * torch.relu(-processed_morphology[-1, 2])
            prob = torch.sigmoid(logit).mean()
            loss_value = float(loss.detach().cpu().item())

            improved = loss_value < best_loss - delta_early_stopping
            if improved:
                best_loss = loss_value
                best_lengths = lengths.detach().clone()
                best_iteration = update_idx
                stale_count = 0
            else:
                stale_count += 1

            validation_data = None

            if eval_interval > 0 and update_idx % eval_interval == 0 and logging:
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
                msg = (
                    f"[Iter {update_idx:>4}/{n_iter}] "
                    f"loss={loss.item():.6f}, "
                    f"val_se3={best_se3:.6f}, "
                    f"ik_success_pose_rate={ik_success_rate * 100.0:.2f}%, "
                    f"nrm_prob={prob.item():.6f}"
                )

                tqdm.write(msg)

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
            optimizer.step()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                prob=f"{prob.item():.3f}",
                stale=stale_count,
            )

        # Final state after all optimizer steps, or the best state before early stopping.
        with torch.no_grad():
            final_processed_lengths, _ = _preprocess(
                best_lengths,
                morph.link_radius,
            )

            final_raw_morphology = torch.cat([alpha, best_lengths], dim=1)
            final_processed_morphology = torch.cat(
                [alpha, final_processed_lengths], dim=1
            )

            final_bmorph = final_processed_morphology.unsqueeze(0).expand(
                task_vec.shape[0], -1, -1
            )
            final_logit = model(final_bmorph, task_vec)
            final_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(
                final_logit,
                torch.ones_like(final_logit),
            )
            final_loss = final_loss + 100.0 * torch.relu(
                -final_processed_morphology[-1, 2]
            )
            final_prob = torch.sigmoid(final_logit).mean()

        # you will always do a validation for the final result even if not in logging mode
        final_validation_data = run_optimization_validation(
            processed_morphology=final_processed_morphology,
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

        msg = (
            f"[Final NRM] "
            f"loss={final_loss.item():.6f}, "
            f"final_se3_err={final_se3_err:.6f}, "
            f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%, "
            f"nrm_prob={final_prob.item():.6f}"
        )
        print(msg)

        optimized_morph = Morphology(
            params=final_processed_morphology.detach(),
            link_radius=morph.link_radius,
        )

        return optimized_morph, csv_logger.csv_path

    finally:
        csv_logger.close()
