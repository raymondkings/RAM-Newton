# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
# Single-round discrete-alpha candidate search + batched gradient optimization.
#
# New logic:
#   1. Generate alpha candidates from {-pi/2, 0, pi/2}.
#      - num_alpha_candidates can be "ALL" or an integer.
#      - candidates with 3 or more consecutive zero-alpha entries are excluded
#        already during alpha generation.
#   2. For each fixed alpha candidate, sample a valid initial morphology using
#      the original sampler's link_type / length / rejection logic. This avoids
#      blindly reusing one initial morphology's [a, d] for all alpha candidates.
#   3. Optimize all sampled candidates in one batched optimization run.
#      - no multi-stage filtering
#      - no validation during optimization
#      - per-candidate early stopping based on stable NRM probability
#   4. Post-check optimized morphologies with the distribution checker.
#   5. Keep the top TOP_PROBABILITY_FRACTION by NRM probability.
#   6. Run IK/FK validation only on those top-probability candidates.
#   7. Select final candidate by lowest validation SE3 error; if tied, use
#      smallest total |a|+|d| as the tiebreaker.
#
# CSV convention:
#   iteration = 0  -> validated top-probability candidate, not SE3-best
#   iteration = 1  -> SE3-best candidate but not finally selected by tiebreak
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

from optim.model import MLP
from interface import Morphology, Task
from util.distribution_checker import check_morphology_distribution
from util.fixed_alpha_morphology_candidates import (
    DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
    generate_alpha_candidates,
    sample_fixed_alpha_morphology_candidates,
)
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)


EPS = 1e-4
_PROJECT_ROOT = Path(__file__).parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"

# ----------------------------- hard-coded knobs -----------------------------

DEFAULT_NUM_ALPHA_CANDIDATES: int | str = "ALL"
# number of candidate to be updated at once
DEFAULT_CANDIDATE_BATCH_SIZE = 64
# number of candidates to be checked for distribution at once
DEFAULT_DISTRIBUTION_BATCH_SIZE = 128

# Exclude alpha candidates with this many consecutive zero-alpha entries.
# The actual alpha-generation/filtering logic lives in
# util.fixed_alpha_morphology_candidates.generate_alpha_candidates.
# For DOF=6 / seq_len=7, run_length=3 changes the alpha maximum
# from 3^7=2187 to 1892.
ZERO_ALPHA_RUN_EXCLUSION_LENGTH = DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH

# Per-candidate early stopping:
# stop a candidate when its NRM probability changes by less than this threshold
# for EARLY_STOPPING_PATIENCE consecutive optimization updates.
DELTA_EARLY_STOPPING = 1e-4
EARLY_STOPPING_PATIENCE = 5

# After post-optimization distribution filtering, only validate the top 10% by
# final NRM probability.
TOP_PROBABILITY_FRACTION = 0.10

# Tie tolerance for validation SE3 errors.
SE3_TIE_EPS = 1e-12


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
            _WEIGHTS_DIR / "checkpoint.pth",
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

        if logging:
            mean_prob = torch.nanmean(current_probs).item()
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
    base_pose_inv: Tensor,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
) -> dict:
    """Validate one candidate using IK/FK with a deterministic pose subset."""
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


def _validate_top_candidates(
    processed_morphologies: Tensor,
    morph: Morphology,
    task: Task,
    scene,
    base_pose_inv: Tensor,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
    logging: bool,
) -> tuple[Tensor, list[dict]]:
    """Run validation on selected candidates and return SE3 scores plus data."""
    se3_scores = torch.empty(processed_morphologies.shape[0], device=device)
    validation_data_list: list[dict] = []

    for idx in tqdm(
        range(processed_morphologies.shape[0]),
        desc="validating top-probability candidates",
        disable=not logging,
        dynamic_ncols=True,
    ):
        validation_data = _validate_candidate(
            processed_morphology=processed_morphologies[idx],
            morph=morph,
            task=task,
            scene=scene,
            base_pose_inv=base_pose_inv,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            random_seed=random_seed,
        )
        validation_data_list.append(validation_data)
        se3_scores[idx] = validation_data["best_se3_dist_mean"].detach().to(device)

    return se3_scores, validation_data_list


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
) -> tuple[Morphology, Path]:
    """Single-round random/all discrete-alpha search plus length optimization."""
    total_iterations = int(optimization_parameters.get("num_iterations", 100))
    lr = float(optimization_parameters.get("learning_rate", 0.01))
    logging = bool(optimization_parameters.get("logging", True))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1))
    ignore_ground = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles = bool(optimization_parameters.get("ignore_obstacles", False))

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
    seq_len = morph.params.shape[0]

    if logging:
        print(
            f"[Info] Starting single-round alpha candidate optimization on device {device}."
        )
        print(
            "[Info] "
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

    base_pose_inv, scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    task_vec = _se3_to_vector(task.goal_poses.to(device))
    model = _load_model(device)
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    alpha_generator = torch.Generator(device=device)
    alpha_generator.manual_seed(random_seed)

    try:
        # ------------------------------------------------------------------
        # 1. Generate alpha candidates.
        # ------------------------------------------------------------------
        alpha_candidates, max_alpha_candidates, using_all = generate_alpha_candidates(
            requested_num_candidates=num_alpha_candidates,
            seq_len=seq_len,
            device=device,
            generator=alpha_generator,
            forbidden_zero_run_length=ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
        )

        if logging:
            print(
                "[Info] Alpha candidates generated: "
                f"{alpha_candidates.shape[0]} / max_valid={max_alpha_candidates} "
                f"(using_all={using_all})."
            )
            print(
                f"[Info] For seq_len={seq_len}, excluding "
                f"{ZERO_ALPHA_RUN_EXCLUSION_LENGTH}+ consecutive zeros gives "
                f"max_valid={max_alpha_candidates} instead of 3^{seq_len}={3**seq_len}."
            )

        # ------------------------------------------------------------------
        # 2. For every fixed alpha, sample a valid initial morphology from the
        #    original sampler distribution.  This replaces the old approach of
        #    reusing one initial morphology's [a, d] lengths for every alpha.
        # ------------------------------------------------------------------
        initial_candidate_morphologies = sample_fixed_alpha_morphology_candidates(
            alpha_candidates=alpha_candidates,
            seed=random_seed,
            link_radius=morph.link_radius,
            batch_size=distribution_batch_size,
            logging=logging,
        )
        original_alpha_count = alpha_candidates.shape[0]
        accepted_alpha_count = initial_candidate_morphologies.shape[0]
        dropped_alpha_count = original_alpha_count - accepted_alpha_count

        alpha_candidates = initial_candidate_morphologies[..., 0:1].detach()
        length_candidates = initial_candidate_morphologies[..., 1:].detach()

        if alpha_candidates.shape[0] == 0:
            raise RuntimeError(
                "Fixed-alpha sampler generated zero valid initial morphologies."
            )

        if logging:
            print(
                "[Info] Fixed-alpha sampler accepted "
                f"{accepted_alpha_count}/{original_alpha_count} alpha candidates "
                f"and dropped {dropped_alpha_count}."
            )
            print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

        # ------------------------------------------------------------------
        # 3. One single batched NRM optimization round with early stopping.
        # ------------------------------------------------------------------
        length_candidates, losses, probs, stop_iterations = (
            _optimize_all_candidates_single_round(
                model=model,
                alpha_candidates=alpha_candidates,
                initial_length_candidates=length_candidates,
                task_vec=task_vec,
                link_radius=morph.link_radius,
                num_iterations=total_iterations,
                learning_rate=lr,
                candidate_batch_size=candidate_batch_size,
                logging=logging,
            )
        )

        if logging:
            n_stopped = int((stop_iterations < total_iterations).sum().item())
            print(
                "[Info] Early stopping summary: "
                f"{n_stopped}/{alpha_candidates.shape[0]} candidates stopped before max iteration."
            )

        raw_morphologies, processed_morphologies = _build_morphology_tensors(
            alpha_candidates,
            length_candidates,
            morph.link_radius,
        )
        raw_morphologies = raw_morphologies.detach()
        processed_morphologies = processed_morphologies.detach()

        # ------------------------------------------------------------------
        # 4. Post-optimization distribution filtering.
        # ------------------------------------------------------------------
        post_valid_mask, _ = _distribution_valid_mask(
            processed_morphologies,
            batch_size=distribution_batch_size,
            logging=logging,
            desc="post-checking candidate distribution",
        )

        if not post_valid_mask.any():
            raise RuntimeError(
                "All optimized candidates were rejected by the post-optimization "
                "distribution checker."
            )

        raw_valid = raw_morphologies[post_valid_mask]
        processed_valid = processed_morphologies[post_valid_mask]
        losses_valid = losses[post_valid_mask]
        probs_valid = probs[post_valid_mask]

        if logging:
            print(
                "[Info] Post-optimization distribution filter: "
                f"kept {processed_valid.shape[0]}/{processed_morphologies.shape[0]} candidates."
            )

        # ------------------------------------------------------------------
        # 5. Keep only the top 10% by final NRM probability.
        # ------------------------------------------------------------------
        num_valid = processed_valid.shape[0]
        top_k = max(1, int(math.ceil(num_valid * TOP_PROBABILITY_FRACTION)))
        top_indices = torch.argsort(probs_valid, descending=True)[:top_k]

        raw_top = raw_valid[top_indices]
        processed_top = processed_valid[top_indices]
        losses_top = losses_valid[top_indices]
        probs_top = probs_valid[top_indices]

        if logging:
            print(
                "[Info] Top-probability selection: "
                f"valid_candidates={num_valid}, top_k={top_k}, "
                f"best_prob={probs_top[0].item():.6f}, "
                f"worst_top_prob={probs_top[-1].item():.6f}"
            )

        # ------------------------------------------------------------------
        # 6. Validate only these top-probability candidates.
        # ------------------------------------------------------------------
        se3_scores, validation_data_list = _validate_top_candidates(
            processed_morphologies=processed_top,
            morph=morph,
            task=task,
            scene=scene,
            base_pose_inv=base_pose_inv,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            random_seed=random_seed,
            logging=logging,
        )

        # ------------------------------------------------------------------
        # 7. Select final candidate: lowest SE3; tie -> smallest total |a|+|d|.
        # ------------------------------------------------------------------
        best_se3 = se3_scores.min()
        best_se3_mask = (se3_scores - best_se3).abs() <= SE3_TIE_EPS

        length_sums = processed_top[..., 1:].abs().sum(dim=(-2, -1))
        tied_indices = torch.nonzero(best_se3_mask, as_tuple=False).squeeze(1)
        final_local_in_tied = torch.argmin(length_sums[tied_indices])
        final_idx = int(tied_indices[final_local_in_tied].item())

        if logging:
            print(
                "[Info] Validation selection: "
                f"best_se3={best_se3.item():.12f}, "
                f"num_se3_ties={int(best_se3_mask.sum().item())}, "
                f"final_idx={final_idx}, "
                f"final_length_sum={length_sums[final_idx].item():.6f}"
            )

        # Log all validated top-probability candidates.
        # iteration marker: 0 = ordinary, 1 = SE3-best tie, 2 = final selected.
        for idx in range(processed_top.shape[0]):
            if idx == final_idx:
                marker = 2
            elif bool(best_se3_mask[idx]):
                marker = 1
            else:
                marker = 0

            _log_candidate(
                csv_logger=csv_logger,
                iteration_marker=marker,
                loss=losses_top[idx],
                prob=probs_top[idx],
                raw_morphology=raw_top[idx],
                processed_morphology=processed_top[idx],
                validation_data=validation_data_list[idx],
            )

        final_processed_morphology = processed_top[final_idx]
        final_validation_data = validation_data_list[final_idx]
        final_se3 = final_validation_data["best_se3_dist_mean"].detach().cpu().item()

        print(
            "[Final candidate] "
            f"loss={losses_top[final_idx].item():.6f}, "
            f"nrm_prob={probs_top[final_idx].item():.6f}, "
            f"final_se3_err={final_se3:.6f}, "
            f"length_sum={length_sums[final_idx].item():.6f}"
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
        return optimized_morph, csv_logger.csv_path

    finally:
        csv_logger.close()
