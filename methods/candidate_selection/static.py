# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
# Single-round discrete-alpha candidate search across selectable DOFs.
#
# Logic:
#   1. Resolve candidate DOFs from the CANDIDATE_DOF constant below.
#      - "all" means DOF 5, 6, and 7.
#      - Strings like "5,6" or "7" select specific supported DOFs.
#   2. Generate alpha candidates from {-pi/2, 0, pi/2} for each DOF.
#      - num_alpha_candidates can be "ALL" or an integer.
#      - candidates with 3 or more consecutive zero-alpha entries are excluded
#        already during alpha generation.
#   3. For each fixed alpha candidate, sample a valid initial morphology using
#      the original sampler's link_type / length / rejection logic. This avoids
#      blindly reusing one initial morphology's [a, d] for all alpha candidates.
#   4. Optimize each DOF group in its own fixed-length batched optimization run.
#      - no multi-stage filtering
#      - no validation during optimization
#      - per-candidate early stopping based on stable NRM probability
#   5. Post-check optimized morphologies with the distribution checker.
#   6. Merge selected DOF groups, then keep the top TOP_PROBABILITY_FRACTION by
#      NRM probability.
#   7. Run IK/FK validation only on those top-probability candidates.
#   8. Select final candidate by highest IK pose success rate, then the existing
#      morphology heuristic tiebreaker.
#
# CSV convention:
#   iteration = 0  -> validated top-probability candidate, not final-tier
#   iteration = 1  -> final-tier candidate but not selected by tiebreak
#   iteration = 2  -> final selected candidate
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

from core import Morphology, Task
from logutils.csv_logger import (
    InternalOptimizationCSVLogger,
    OptimizationCSVLogger,
)
from logutils.timing import OptimizationTimer
from methods.nrm_model import MLP
from tasks.sampling.fixed_alpha_candidates import (
    DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
    generate_alpha_candidates,
    sample_fixed_alpha_morphology_candidates,
    sample_fixed_alpha_morphology_candidates_by_dof,
)
from validation.distribution_checker import check_morphology_distribution
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)

EPS = 1e-4
from paths import PROJECT_ROOT as _PROJECT_ROOT
from paths import WEIGHTS_DIR as _WEIGHTS_DIR

_CHECKPOINT_PATH = _WEIGHTS_DIR / "checkpoint_5-7.pth"

# ----------------------------- hard-coded knobs -----------------------------

# control for the input TODO: change this into main file
DEFAULT_CANDIDATE_DOFS = (5, 6, 7)
CANDIDATE_DOF: str | int | tuple[int, ...] = "all"

# number of candidates for alpha to be picked as initial candidates
DEFAULT_NUM_ALPHA_CANDIDATES: int | str = "ALL"
# number of candidate to be updated at once
# glance for estimated VRAM: 300 --> 25GB, 128 --> 10GB
DEFAULT_CANDIDATE_BATCH_SIZE = 64
# number of candidates to be checked for distribution at once
DEFAULT_DISTRIBUTION_BATCH_SIZE = 128

# Exclude alpha candidates with this many consecutive zero-alpha entries.
# The actual alpha-generation/filtering logic lives in
# util.fixed_alpha_morphology_candidates.generate_alpha_candidates.
# For DOF=6 / seq_len=7, run_length=3 changes the alpha maximum
# from 3^7=2187 to 1892.
# DEFAULT = 3 for non-degenerate robot, you can also change this
# to your preferred number to avoid consecutive zero-alpha entries.
ZERO_ALPHA_RUN_EXCLUSION_LENGTH = DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH

# Per-candidate early stopping:
# stop a candidate when its NRM probability changes by less than this threshold
# for EARLY_STOPPING_PATIENCE consecutive optimization updates.
DELTA_EARLY_STOPPING = 1e-4
EARLY_STOPPING_PATIENCE = 5

# After post-optimization distribution filtering, only validate the top 10% by
# final NRM probability.
TOP_PROBABILITY_FRACTION = 0.025

INTERNAL_LOG_FIELDNAMES = [
    "dof",
    "iteration",
    "mean_loss",
    "mean_nrm_prob",
]

# Per-candidate final loss log, written once for every candidate that survives
# post-optimization distribution + final-d filtering (not just the top-k).
ALL_CANDIDATES_LOG_FIELDNAMES = [
    "dof",
    "loss",
    "nrm_prob",
]

# ------------------------------- model helpers ------------------------------


def _se3_to_vector(pose: Tensor) -> Tensor:
    """Convert SE(3) pose matrices [..., 4, 4] to 9D NRM pose vectors."""
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


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
    # The weights are frozen, but gradients still flow to morphology inputs.
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


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
            f"dof supports only {list(DEFAULT_CANDIDATE_DOFS)} or 'all', got {resolved}."
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


# --------------------------- morphology preprocessing -------------------------


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
        norm = l2_norm.sum(dim=-2, keepdim=True)
        ctx.save_for_backward(param, l2_norm, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        param, l2_norm, norm = ctx.saved_tensors
        chain = torch.where(
            (param.abs() > EPS).any(dim=-1, keepdim=True),
            param / l2_norm,
            torch.zeros_like(param),
        )
        dot = (grad_output * param).sum(dim=(-2, -1), keepdim=True)
        return (grad_output * norm - chain * dot) / norm.pow(2)


def _preprocess_lengths(lengths: Tensor, link_radius: float) -> tuple[Tensor, Tensor]:
    """Apply normalize -> squash -> normalize to [a, d] lengths."""
    threshold = 2.0 * link_radius
    norm_lengths = BatchedNormaliser.apply(lengths)
    squashed = SquasherSTE.apply(norm_lengths, threshold)
    return BatchedNormaliser.apply(squashed), norm_lengths


def _build_morphology_tensors(
    alpha_candidates: Tensor,
    length_candidates: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor]:
    """Build raw and processed morphology tensors for batched candidates."""
    processed_lengths, _ = _preprocess_lengths(length_candidates, link_radius)
    raw_morphologies = torch.cat([alpha_candidates, length_candidates], dim=-1)
    processed_morphologies = torch.cat([alpha_candidates, processed_lengths], dim=-1)
    return raw_morphologies, processed_morphologies


def _last_d_nonnegative_mask(processed_morphologies: Tensor) -> Tensor:
    """Return candidates whose final processed d is positive or zero."""
    if processed_morphologies.ndim < 2 or processed_morphologies.shape[-1] < 3:
        raise ValueError(
            "Expected processed_morphologies with shape [..., n_links, 3]."
        )
    return processed_morphologies[..., -1, 2] >= 0.0


# -------------------------- NRM score / optimization -------------------------


def _candidate_loss_and_prob(
    model: MLP,
    alpha_candidates: Tensor,
    length_candidates: Tensor,
    task_vec: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor]:
    """Compute one BCE loss and one mean reachability probability per candidate."""
    num_candidates = alpha_candidates.shape[0]
    num_poses = task_vec.shape[0]

    _, morph_params = _build_morphology_tensors(
        alpha_candidates,
        length_candidates,
        link_radius,
    )

    bmorph = morph_params.unsqueeze(1).expand(num_candidates, num_poses, -1, -1)
    bpose = task_vec.unsqueeze(0).expand(num_candidates, num_poses, -1)

    bmorph = bmorph.reshape(num_candidates * num_poses, morph_params.shape[-2], 3)
    bpose = bpose.reshape(num_candidates * num_poses, 9)

    logit = model(bmorph, bpose).reshape(num_candidates, num_poses)
    target = torch.ones_like(logit)
    loss_per_pose = F.binary_cross_entropy_with_logits(logit, target, reduction="none")

    loss = loss_per_pose.mean(dim=1)
    prob = torch.sigmoid(logit).mean(dim=1)

    return loss, prob


@torch.no_grad()
def _evaluate_scores_batched(
    model: MLP,
    alpha_candidates: Tensor,
    length_candidates: Tensor,
    task_vec: Tensor,
    link_radius: float,
    candidate_batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Return current NRM loss and mean reachability probability for all candidates."""
    losses = []
    probs = []

    for start in range(0, alpha_candidates.shape[0], candidate_batch_size):
        end = min(start + candidate_batch_size, alpha_candidates.shape[0])
        loss, prob = _candidate_loss_and_prob(
            model=model,
            alpha_candidates=alpha_candidates[start:end],
            length_candidates=length_candidates[start:end],
            task_vec=task_vec,
            link_radius=link_radius,
        )
        losses.append(loss.detach())
        probs.append(prob.detach())

    return torch.cat(losses, dim=0), torch.cat(probs, dim=0)


def _optimize_all_candidates_single_round(
    model: MLP,
    alpha_candidates: Tensor,
    initial_length_candidates: Tensor,
    task_vec: Tensor,
    link_radius: float,
    num_iterations: int,
    learning_rate: float,
    candidate_batch_size: int,
    logging: bool,
    internal_logger: InternalOptimizationCSVLogger | None = None,
    dof: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Optimize all candidates in one batched round with per-candidate early stopping.

    Returns:
        final_lengths:
            Optimized raw [a, d] parameters, shape [N, seq_len, 2].
        final_losses:
            Final NRM BCE loss per candidate.
        final_probs:
            Final mean reachability probability per candidate.
        stop_iterations:
            Iteration index at which each candidate stopped. Candidates that did
            not early-stop have value num_iterations.
    """
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive.")

    num_candidates = alpha_candidates.shape[0]
    if logging:
        pair_batch_size = min(candidate_batch_size, num_candidates) * task_vec.shape[0]
        print(
            "[Info] NRM optimization tensors: "
            f"model_device={next(model.parameters()).device}, "
            f"alpha_device={alpha_candidates.device}, "
            f"length_device={initial_length_candidates.device}, "
            f"task_vec_device={task_vec.device}, "
            f"num_candidates={num_candidates}, "
            f"num_poses={task_vec.shape[0]}, "
            f"candidate_batch_size={candidate_batch_size}, "
            f"max_candidate_pose_pairs_per_batch={pair_batch_size}"
        )
    length_candidates = initial_length_candidates.detach().clone().requires_grad_(True)

    # weight_decay=0 prevents stopped/inactive rows from drifting due to AdamW's
    # decoupled weight decay. We also restore inactive rows after each step to
    # prevent momentum state from moving already-stopped candidates.
    optimizer = torch.optim.AdamW(
        [length_candidates], lr=learning_rate, weight_decay=0.0
    )

    active_mask = torch.ones(
        num_candidates, dtype=torch.bool, device=alpha_candidates.device
    )
    frozen_lengths = initial_length_candidates.detach().clone()
    stable_counts = torch.zeros(
        num_candidates, dtype=torch.long, device=alpha_candidates.device
    )
    previous_probs = torch.full(
        (num_candidates,),
        float("nan"),
        dtype=torch.float32,
        device=alpha_candidates.device,
    )
    stop_iterations = torch.full(
        (num_candidates,),
        num_iterations,
        dtype=torch.long,
        device=alpha_candidates.device,
    )

    iterator = tqdm(
        range(num_iterations),
        desc="single-round NRM optimization",
        disable=not logging,
        dynamic_ncols=True,
    )

    for update_idx in iterator:
        if not active_mask.any():
            if logging:
                tqdm.write(
                    f"[Early stopping] all candidates stopped at iteration {update_idx}."
                )
            break

        optimizer.zero_grad(set_to_none=True)

        active_indices = torch.nonzero(active_mask, as_tuple=False).squeeze(1)
        num_active = int(active_indices.numel())
        current_probs = torch.full_like(previous_probs, float("nan"))
        current_loss_sum = 0.0

        for start in range(0, num_active, candidate_batch_size):
            local = active_indices[start : start + candidate_batch_size]
            loss_per_candidate, prob_per_candidate = _candidate_loss_and_prob(
                model=model,
                alpha_candidates=alpha_candidates[local],
                length_candidates=length_candidates[local],
                task_vec=task_vec,
                link_radius=link_radius,
            )

            # Mean over active candidates, accumulated over chunks.
            (loss_per_candidate.sum() / num_active).backward()
            current_probs[local] = prob_per_candidate.detach()
            current_loss_sum += float(loss_per_candidate.detach().sum().item())

        optimizer.step()

        # Restore candidates that had already stopped before this optimizer step.
        # This protects them from any optimizer state/momentum side effects.
        with torch.no_grad():
            inactive_mask = ~active_mask
            if inactive_mask.any():
                length_candidates[inactive_mask] = frozen_lengths[inactive_mask]

        with torch.no_grad():
            has_previous = torch.isfinite(previous_probs)
            delta = (current_probs - previous_probs).abs()
            stable_now = active_mask & has_previous & (delta < DELTA_EARLY_STOPPING)

            stable_counts[stable_now] += 1
            stable_counts[active_mask & ~stable_now] = 0

            # Do not mark candidates as "early stopped" on the final
            # optimizer step; at that point they simply reached max iteration.
            can_stop_early = (update_idx + 1) < num_iterations
            newly_stopped = active_mask & (stable_counts >= EARLY_STOPPING_PATIENCE)
            newly_stopped &= can_stop_early
            if newly_stopped.any():
                frozen_lengths[newly_stopped] = length_candidates.detach()[
                    newly_stopped
                ]
                active_mask[newly_stopped] = False
                stop_iterations[newly_stopped] = update_idx + 1

            previous_probs[torch.isfinite(current_probs)] = current_probs[
                torch.isfinite(current_probs)
            ]

        mean_loss = current_loss_sum / max(num_active, 1)
        mean_prob = torch.nanmean(current_probs).item()
        if internal_logger is not None:
            internal_logger.log_row(
                dof=dof,
                iteration=update_idx,
                mean_loss=mean_loss,
                mean_nrm_prob=mean_prob,
            )

        if logging:
            num_stopped = int((~active_mask).sum().item())
            num_active_after = int(active_mask.sum().item())
            num_batches = int(math.ceil(max(num_active, 1) / candidate_batch_size))
            iterator.set_postfix(
                active=num_active_after,
                stopped=num_stopped,
                total=num_candidates,
                batches=num_batches,
                mean_prob=f"{mean_prob:.4f}",
            )

    final_lengths = length_candidates.detach()
    final_losses, final_probs = _evaluate_scores_batched(
        model=model,
        alpha_candidates=alpha_candidates,
        length_candidates=final_lengths,
        task_vec=task_vec,
        link_radius=link_radius,
        candidate_batch_size=candidate_batch_size,
    )

    return final_lengths, final_losses, final_probs, stop_iterations


# --------------------------- distribution filtering --------------------------


def _distribution_valid_mask(
    processed_morphologies: Tensor,
    batch_size: int,
    logging: bool,
    desc: str,
) -> tuple[Tensor, list[Any]]:
    """Run distribution checker in chunks and return a bool mask plus reports."""
    reports: list[Any] = []

    for start in tqdm(
        range(0, processed_morphologies.shape[0], batch_size),
        desc=desc,
        disable=not logging,
        dynamic_ncols=True,
    ):
        end = min(start + batch_size, processed_morphologies.shape[0])
        chunk_reports = check_morphology_distribution(
            processed_morphologies[start:end].detach(),
            num_joint_samples=1000,
            seed=0,
        )
        reports.extend(chunk_reports)

    valid_mask = torch.tensor(
        [report.valid for report in reports],
        dtype=torch.bool,
        device=processed_morphologies.device,
    )

    return valid_mask, reports


# TIE BREAK HEURISTIC
def _tie_score(
    processed_morphology: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor]:
    """Return the existing morphology heuristic score and length sum."""
    ad_abs = processed_morphology[..., 1:].abs()
    link_mag = torch.linalg.norm(processed_morphology[..., 1:], dim=-1)

    eps_zero = 1e-6
    min_good_nonzero = 4.0 * link_radius

    length_sum = ad_abs.sum()
    zero_reward = (ad_abs <= eps_zero).float().sum()

    is_tiny_nonzero = (ad_abs > eps_zero) & (ad_abs < min_good_nonzero)
    tiny_penalty = (min_good_nonzero - ad_abs).clamp_min(0.0)
    tiny_penalty = (tiny_penalty * is_tiny_nonzero.float()).sum()

    active_link = link_mag > eps_zero
    mean_link = (
        link_mag * active_link.float()
    ).sum() / active_link.float().sum().clamp_min(1.0)
    balance_penalty = ((link_mag - mean_link) ** 2 * active_link.float()).sum()

    score = (
        10.0 * tiny_penalty
        + 2.0 * balance_penalty
        + 0.02 * length_sum
        - 0.003 * zero_reward
    )
    return score, length_sum


def _optimize_one_dof_group(
    *,
    dof: int,
    initial_candidate_morphologies: Tensor,
    model: MLP,
    task_vec: Tensor,
    link_radius: float,
    total_iterations: int,
    learning_rate: float,
    candidate_batch_size: int,
    distribution_batch_size: int,
    logging: bool,
    internal_logger: InternalOptimizationCSVLogger | None = None,
) -> list[dict[str, Any]]:
    """Optimize and post-filter one fixed-length DOF candidate group."""
    alpha_candidates = initial_candidate_morphologies[..., 0:1].detach()
    length_candidates = initial_candidate_morphologies[..., 1:].detach()
    initial_morphologies_for_log = initial_candidate_morphologies.detach()

    length_candidates, losses, probs, stop_iterations = (
        _optimize_all_candidates_single_round(
            model=model,
            alpha_candidates=alpha_candidates,
            initial_length_candidates=length_candidates,
            task_vec=task_vec,
            link_radius=link_radius,
            num_iterations=total_iterations,
            learning_rate=learning_rate,
            candidate_batch_size=candidate_batch_size,
            logging=logging,
            internal_logger=internal_logger,
            dof=dof,
        )
    )

    if logging:
        n_stopped = int((stop_iterations < total_iterations).sum().item())
        print(
            f"[Info] DOF{dof} early stopping summary: "
            f"{n_stopped}/{alpha_candidates.shape[0]} candidates stopped before max iteration."
        )

    _, processed_morphologies = _build_morphology_tensors(
        alpha_candidates,
        length_candidates,
        link_radius,
    )
    processed_morphologies = processed_morphologies.detach()

    post_valid_mask, _ = _distribution_valid_mask(
        processed_morphologies,
        batch_size=distribution_batch_size,
        logging=logging,
        desc=f"post-checking DOF{dof} candidate distribution",
    )

    if logging:
        print(
            f"[Info] DOF{dof} post-optimization distribution filter: "
            f"kept {int(post_valid_mask.sum().item())}/{processed_morphologies.shape[0]} candidates."
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
            }
        )

    return records


# ------------------------------- validation ----------------------------------


def _validation_generator(device: torch.device, random_seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    return generator


def _validate_candidate(
    processed_morphology: Tensor,
    morph: Morphology,
    task: Task,
    scene,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
    timer: OptimizationTimer | None = None,
) -> dict:
    """Validate one candidate using IK/FK with a deterministic pose subset."""
    pose_sampling_generator = _validation_generator(device, random_seed)
    if timer is None:
        return run_optimization_validation(
            processed_morphology=processed_morphology.detach(),
            morph=morph,
            task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=pose_sampling_generator,
        )
    with timer.validation():
        return run_optimization_validation(
            processed_morphology=processed_morphology.detach(),
            morph=morph,
            task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=pose_sampling_generator,
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
    timer: OptimizationTimer | None = None,
) -> tuple[Tensor, Tensor, list[dict]]:
    """Run validation on selected candidate records and return scores plus data."""
    se3_scores = torch.empty(len(records), device=device)
    ik_success_rates = torch.empty(len(records), device=device)
    validation_data_list: list[dict] = []

    for idx in tqdm(
        range(len(records)),
        desc="validating top-probability candidates",
        disable=not logging,
        dynamic_ncols=True,
    ):
        validation_data = _validate_candidate(
            processed_morphology=records[idx]["processed_morphology"],
            morph=morph,
            task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            random_seed=random_seed,
            timer=timer,
        )
        validation_data_list.append(validation_data)
        se3_scores[idx] = validation_data["best_se3_dist_mean"].detach().to(device)
        ik_success_rates[idx] = (
            validation_data["ik_success_pose_rate"].detach().to(device)
        )

    return se3_scores, ik_success_rates, validation_data_list


def _sample_initial_candidate_morphologies_by_dof(
    *,
    alpha_candidates_by_dof: dict[int, Tensor],
    seed: int,
    link_radius: float,
    batch_size: int,
    logging: bool,
) -> dict[int, Tensor]:
    """Sample fixed-alpha initial morphologies while preserving cache schemas.

    A single requested DOF uses the historical single-DOF cache payload under
    initial_candidates/DOF{dof}_seed{seed}/candidates.json.  Multiple requested
    DOFs use the combined cache payload, e.g. DOF5-7_seed0.
    """
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


# ---------------------------------- logging ----------------------------------


def _log_candidate(
    csv_logger: OptimizationCSVLogger,
    iteration_marker: int,
    loss: Tensor,
    prob: Tensor,
    raw_morphology: Tensor,
    processed_morphology: Tensor,
    validation_data: dict,
) -> None:
    csv_logger.log_iteration(
        iteration=iteration_marker,
        loss=loss.detach(),
        reachability_probability=prob.detach(),
        raw_morphology=raw_morphology.detach(),
        processed_morphology=processed_morphology.detach(),
        validation_data=validation_data,
    )


# --------------------------------- main API ----------------------------------


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path, list[float], list[tuple[Morphology, float]]]:
    """Single-round discrete-alpha candidate search over selected DOF groups.

    Set CANDIDATE_DOF at the top of this file to "all", "5,6", "5,7",
    "6,7", "5", "6", or "7".
    """
    num_plan_candidates = max(
        1, int(optimization_parameters.get("num_plan_candidates", 1))
    )
    total_iterations = int(optimization_parameters.get("num_iterations", 100))
    lr = float(optimization_parameters.get("learning_rate", 0.01))
    logging = bool(optimization_parameters.get("logging", True))
    csv_logging = bool(optimization_parameters.get("csv_logging", True))
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
            "candidate_batch_size", DEFAULT_CANDIDATE_BATCH_SIZE
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

    device = morph.params.device
    timer = OptimizationTimer(device)
    timer.start()
    selected_label = ",".join(str(dof) for dof in candidate_dofs)

    if logging:
        print(f"[Info] Starting DOF candidate optimization on device {device}.")
        print(
            "[Info] "
            f"dof={dof_selector!r}, "
            f"candidate_dofs={candidate_dofs}, "
            f"num_iterations={total_iterations}, "
            f"learning_rate={lr}, "
            f"num_alpha_candidates={num_alpha_candidates}, "
            f"candidate_batch_size={candidate_batch_size}, "
            f"distribution_batch_size={distribution_batch_size}, "
            f"top_probability_fraction={TOP_PROBABILITY_FRACTION}, "
            f"early_stopping_patience={EARLY_STOPPING_PATIENCE}, "
            f"delta_early_stopping={DELTA_EARLY_STOPPING}, "
            f"random_seed={random_seed}, "
            f"number_random_seed={number_random_seed}, "
            f"percentage_poses={percentage_poses}"
        )
        print(f"[Info] Loading NRM checkpoint: {_CHECKPOINT_PATH}")

    scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    task_vec = _se3_to_vector(task.goal_poses.to(device))
    model = _load_model(device)
    log_root_dir = optimization_parameters.get("log_root_dir")
    csv_logger = (
        OptimizationCSVLogger(
            root_dir=Path(log_root_dir), output_subdir=None, enabled=csv_logging
        )
        if log_root_dir
        else OptimizationCSVLogger(root_dir=_PROJECT_ROOT, enabled=csv_logging)
    )
    internal_logger: InternalOptimizationCSVLogger | None = None

    alpha_generator = torch.Generator(device=device)
    alpha_generator.manual_seed(random_seed)

    try:
        internal_logger = InternalOptimizationCSVLogger(
            csv_logger.csv_path,
            fieldnames=INTERNAL_LOG_FIELDNAMES,
            enabled=csv_logging,
        )

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
            print(f"[Info] Writing internal CSV log to: {internal_logger.csv_path}")
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
                    task_vec=task_vec,
                    link_radius=morph.link_radius,
                    total_iterations=total_iterations,
                    learning_rate=lr,
                    candidate_batch_size=candidate_batch_size,
                    distribution_batch_size=distribution_batch_size,
                    logging=logging,
                    internal_logger=internal_logger,
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
            if bool(_last_d_nonnegative_mask(record["processed_morphology"]).item())
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

        all_candidates_logger = InternalOptimizationCSVLogger(
            csv_logger.csv_path,
            fieldnames=ALL_CANDIDATES_LOG_FIELDNAMES,
            suffix="final_candidates",
            enabled=csv_logging,
        )
        if logging:
            print(
                "[Info] Writing all-candidates CSV log to: "
                f"{all_candidates_logger.csv_path}"
            )
        try:
            for record in records:
                all_candidates_logger.log_row(
                    dof=record["dof"],
                    loss=record["loss"],
                    nrm_prob=record["prob"],
                )
        finally:
            all_candidates_logger.close()

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
            timer=timer,
        )

        # change here for the final selection before tie break score
        best_ik_success_rate = ik_success_rates.max()
        best_rate_mask = (ik_success_rates - best_ik_success_rate).abs() <= 1e-12
        final_tier_mask = best_rate_mask
        tied_indices = torch.nonzero(final_tier_mask, as_tuple=False).squeeze(1)

        tie_scores = []
        length_sums = []
        for record in top_records:
            tie_score, length_sum = _tie_score(
                record["processed_morphology"],
                morph.link_radius,
            )
            tie_scores.append(tie_score)
            length_sums.append(length_sum)

        tie_scores_tensor = torch.stack(tie_scores).to(device)
        length_sums_tensor = torch.stack(length_sums).to(device)
        final_local_in_tied = torch.argmin(tie_scores_tensor[tied_indices])
        final_idx = int(tied_indices[final_local_in_tied].item())

        # Full ranking by the same key as the winner above: best ik_success_rate
        # first, ties broken by min tie_score. Rank 0 is exactly `final_idx`.
        ranking = sorted(
            range(len(top_records)),
            key=lambda idx: (
                -ik_success_rates[idx].item(),
                tie_scores_tensor[idx].item(),
            ),
        )
        selected_indices = ranking[: min(num_plan_candidates, len(ranking))]

        if logging:
            print(
                "[Info] Validation selection: "
                f"best_ik_success_pose_rate={best_ik_success_rate.item() * 100.0:.2f}%, "
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
        final_validation_data = validation_data_list[final_idx]
        final_se3 = final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        final_ik_success_rate = (
            final_validation_data["ik_success_pose_rate"].detach().cpu().item()
        )

        print(
            "[Final candidate] "
            f"dof={final_record['dof']}, "
            f"loss={final_record['loss'].item():.6f}, "
            f"nrm_prob={final_record['prob'].item():.6f}, "
            f"final_se3_err={final_se3:.6f}, "
            f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%, "
            f"length_sum={length_sums_tensor[final_idx].item():.6f}"
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

        candidates: list[tuple[Morphology, float]] = []
        for idx in selected_indices:
            record = top_records[idx]
            candidates.append(
                (
                    Morphology(
                        params=record["processed_morphology"].detach(),
                        link_radius=morph.link_radius,
                    ),
                    validation_data_list[idx]["ik_success_pose_rate"]
                    .detach()
                    .cpu()
                    .item(),
                )
            )

        return optimized_morph, csv_logger.csv_path, timer.result(), candidates

    finally:
        if internal_logger is not None:
            internal_logger.close()
        csv_logger.close()
