# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
# Shared helpers for the discrete-alpha candidate-selection modules
# (static.py and trajectory.py).  Both drivers reuse the same model loading,
# alpha-candidate generation, morphology preprocessing, distribution filtering,
# validation, tie-break heuristic, and CSV logging; this module is their single
# source of truth so the two cannot silently drift.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from core import Morphology, Task
from logutils.csv_logger import (
    InternalOptimizationCSVLogger,
    OptimizationCSVLogger,
)
from logutils.timing import OptimizationTimer
from methods.nrm_model import MLP
from paths import PROJECT_ROOT, WEIGHTS_DIR
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

_CHECKPOINT_PATH = WEIGHTS_DIR / "checkpoint_5-7.pth"

# ----------------------------- shared knobs ---------------------------------

DEFAULT_CANDIDATE_DOFS = (5, 6, 7)
DEFAULT_NUM_ALPHA_CANDIDATES: int | str = "ALL"
DEFAULT_CANDIDATE_BATCH_SIZE = 64
DEFAULT_DISTRIBUTION_BATCH_SIZE = 128
ZERO_ALPHA_RUN_EXCLUSION_LENGTH = DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH

# After post-optimization distribution filtering, only validate the top fraction
# by final NRM probability.
TOP_PROBABILITY_FRACTION = 0.025


# ------------------------------- model helpers ------------------------------


def _se3_to_vector(pose: Tensor) -> Tensor:
    """Convert SE(3) pose matrices [..., 4, 4] to 9D NRM pose vectors."""
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


def _load_model(device: torch.device) -> MLP:
    metadata = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
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


# ---------------------------- distribution filter ----------------------------


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


# ----------------------------- tie-break heuristic ---------------------------


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


# ------------------------------- validation ----------------------------------


def _validation_generator(device: torch.device, random_seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    return generator


def _validate_candidate(
    *,
    processed_morphology: Tensor,
    morph: Morphology,
    task: Task,
    scene,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
    trajectory: Tensor | None = None,
    timer: OptimizationTimer | None = None,
) -> dict:
    """Validate one candidate using IK/FK with a deterministic pose subset.

    When ``trajectory`` is given, validation runs on a per-candidate task built
    from that trajectory (used by the trajectory pipeline); otherwise it runs on
    ``task`` directly.
    """
    if trajectory is not None:
        task = Task(
            environment=task.environment,
            goal_poses=trajectory.detach(),
            start_q=task.start_q,
        )
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
    desc: str = "validating top-probability candidates",
    timer: OptimizationTimer | None = None,
) -> tuple[Tensor, Tensor, list[dict]]:
    """Run validation on selected candidate records and return scores plus data.

    Records may carry a ``trajectory`` entry; when present each candidate is
    validated on its own optimized trajectory.
    """
    se3_scores = torch.empty(len(records), device=device)
    ik_success_rates = torch.empty(len(records), device=device)
    validation_data_list: list[dict] = []

    for idx in tqdm(
        range(len(records)),
        desc=desc,
        disable=not logging,
        dynamic_ncols=True,
    ):
        validation_data = _validate_candidate(
            processed_morphology=records[idx]["processed_morphology"],
            trajectory=records[idx].get("trajectory"),
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


# ----------------------------- driver skeleton --------------------------------
# The static and trajectory drivers share their whole shell: parameter parsing,
# runtime setup, initial-candidate generation, per-DOF post-filtering, and the
# final top-probability selection/validation/logging tail.  Only the inner
# optimizer and the trajectory-specific record fields differ.


@dataclass
class CandidateSearchParams:
    """optimization_parameters fields parsed identically by both drivers."""

    num_plan_candidates: int
    learning_rate: float
    logging: bool
    csv_logging: bool
    random_seed: int
    number_random_seed: int
    percentage_poses: float
    ignore_ground: bool
    ignore_obstacles: bool
    candidate_dofs: list[int]
    selected_label: str
    num_alpha_candidates: int | str | None
    candidate_batch_size: int
    distribution_batch_size: int


def _parse_candidate_search_params(
    optimization_parameters: dict,
    dof_selector: Any,
) -> CandidateSearchParams:
    candidate_dofs = _resolve_candidate_dofs(dof_selector)

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

    return CandidateSearchParams(
        num_plan_candidates=max(
            1, int(optimization_parameters.get("num_plan_candidates", 1))
        ),
        learning_rate=float(optimization_parameters.get("learning_rate", 0.01)),
        logging=bool(optimization_parameters.get("logging", True)),
        csv_logging=bool(optimization_parameters.get("csv_logging", True)),
        random_seed=int(optimization_parameters.get("random_seed", 42)),
        number_random_seed=int(optimization_parameters.get("number_random_seed", 32)),
        percentage_poses=float(optimization_parameters.get("percentage_poses", 1)),
        ignore_ground=bool(optimization_parameters.get("ignore_ground", False)),
        ignore_obstacles=bool(optimization_parameters.get("ignore_obstacles", False)),
        candidate_dofs=candidate_dofs,
        selected_label=",".join(str(dof) for dof in candidate_dofs),
        num_alpha_candidates=optimization_parameters.get(
            "num_alpha_candidates",
            DEFAULT_NUM_ALPHA_CANDIDATES,
        ),
        candidate_batch_size=candidate_batch_size,
        distribution_batch_size=distribution_batch_size,
    )


def _setup_search_runtime(
    *,
    task: Task,
    device: torch.device,
    optimization_parameters: dict,
    params: CandidateSearchParams,
) -> tuple[Any, MLP, OptimizationCSVLogger, torch.Generator]:
    """Build the validation scene, NRM model, CSV logger, and alpha generator."""
    scene = build_optimization_validation_context(
        task=task,
        ignore_ground=params.ignore_ground,
        ignore_obstacles=params.ignore_obstacles,
    )

    model = _load_model(device)
    log_root_dir = optimization_parameters.get("log_root_dir")
    csv_logger = (
        OptimizationCSVLogger(
            root_dir=Path(log_root_dir), output_subdir=None, enabled=params.csv_logging
        )
        if log_root_dir
        else OptimizationCSVLogger(root_dir=PROJECT_ROOT, enabled=params.csv_logging)
    )

    alpha_generator = torch.Generator(device=device)
    alpha_generator.manual_seed(params.random_seed)

    return scene, model, csv_logger, alpha_generator


def _generate_initial_candidates(
    *,
    params: CandidateSearchParams,
    device: torch.device,
    link_radius: float,
    alpha_generator: torch.Generator,
    csv_logger: OptimizationCSVLogger,
    internal_logger: InternalOptimizationCSVLogger,
) -> dict[int, Tensor]:
    """Generate alpha candidates and sample initial morphologies per DOF."""
    alpha_candidates_by_dof = _generate_alpha_candidates_by_dof(
        dofs=params.candidate_dofs,
        requested_num_candidates=params.num_alpha_candidates,
        device=device,
        generator=alpha_generator,
        logging=params.logging,
    )

    initial_candidate_morphologies_by_dof = (
        _sample_initial_candidate_morphologies_by_dof(
            alpha_candidates_by_dof=alpha_candidates_by_dof,
            seed=params.random_seed,
            link_radius=link_radius,
            batch_size=params.distribution_batch_size,
            logging=params.logging,
        )
    )

    if params.logging:
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")
        print(f"[Info] Writing internal CSV log to: {internal_logger.csv_path}")
        for dof in params.candidate_dofs:
            original_count = alpha_candidates_by_dof[dof].shape[0]
            accepted_count = initial_candidate_morphologies_by_dof[dof].shape[0]
            print(
                f"[Info] DOF{dof} fixed-alpha sampler accepted "
                f"{accepted_count}/{original_count} alpha candidates "
                f"and dropped {original_count - accepted_count}."
            )

    return initial_candidate_morphologies_by_dof


def _postfilter_dof_group(
    *,
    dof: int,
    losses: Tensor,
    probs: Tensor,
    initial_morphologies_for_log: Tensor,
    processed_morphologies: Tensor,
    distribution_batch_size: int,
    logging: bool,
    extra_record_fields: Callable[[int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Distribution-filter one optimized DOF group and pack candidate records."""
    post_valid_mask, _ = _distribution_valid_mask(
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
        record = {
            "dof": dof,
            "loss": losses[idx],
            "prob": probs[idx],
            "raw_morphology": initial_morphologies_for_log[idx],
            "processed_morphology": processed_morphologies[idx],
        }
        if extra_record_fields is not None:
            record.update(extra_record_fields(idx))
        records.append(record)

    return records


def _select_and_log_final_candidates(
    *,
    records: list[dict[str, Any]],
    params: CandidateSearchParams,
    morph: Morphology,
    task: Task,
    scene,
    device: torch.device,
    csv_logger: OptimizationCSVLogger,
    timer: OptimizationTimer,
    all_candidates_fieldnames: list[str],
    build_candidate: Callable[[dict[str, Any], float], Any],
    validation_desc: str = "validating top-probability candidates",
    final_print_label: str = "Final candidate",
    final_print_extra: Callable[[dict[str, Any]], str] | None = None,
) -> tuple[dict[str, Any], Morphology, list[Any]]:
    """Shared post-optimization tail: filter, validate, select, and log.

    Keeps the top TOP_PROBABILITY_FRACTION records by NRM probability, validates
    them, picks the final candidate by IK success rate with the tie heuristic,
    and writes the candidate CSV rows.  Returns the final record, its
    Morphology, and the ``num_plan_candidates`` best candidates built via
    ``build_candidate(record, ik_success_pose_rate)``.
    """
    if not records:
        raise RuntimeError(
            f"All optimized DOF {params.selected_label} candidates were rejected by "
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
            f"All optimized DOF {params.selected_label} candidates were rejected by "
            "the final-link d filter (requires processed params[-1, 2] >= 0)."
        )

    if params.logging:
        print(
            "[Info] Final-link d filter: "
            f"kept {len(records)}/{before_last_d_filter} candidates "
            "with processed params[-1, 2] >= 0."
        )

    all_candidates_logger = InternalOptimizationCSVLogger(
        csv_logger.csv_path,
        fieldnames=all_candidates_fieldnames,
        suffix="final_candidates",
        enabled=params.csv_logging,
    )
    if params.logging:
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
                **record.get("loss_components", {}),
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

    if params.logging:
        dof_counts = {
            dof: sum(1 for record in records if record["dof"] == dof)
            for dof in params.candidate_dofs
        }
        top_dof_counts = {
            dof: sum(1 for record in top_records if record["dof"] == dof)
            for dof in params.candidate_dofs
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
        percentage_poses=params.percentage_poses,
        number_random_seed=params.number_random_seed,
        random_seed=params.random_seed,
        logging=params.logging,
        desc=validation_desc,
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
    selected_indices = ranking[: min(params.num_plan_candidates, len(ranking))]

    if params.logging:
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
        f"[{final_print_label}] "
        f"dof={final_record['dof']}, "
        f"loss={final_record['loss'].item():.6f}, "
        f"nrm_prob={final_record['prob'].item():.6f}, "
        f"final_se3_err={final_se3:.6f}, "
        f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%, "
        f"length_sum={length_sums_tensor[final_idx].item():.6f}"
        + (final_print_extra(final_record) if final_print_extra is not None else "")
    )

    if params.logging:
        print("Final alpha [deg]:")
        print(final_processed_morphology[:, 0].detach().cpu() * 180.0 / math.pi)
        print("Final optimized morphology params:")
        print(final_processed_morphology.detach().cpu())

    optimized_morph = Morphology(
        params=final_processed_morphology.detach(),
        link_radius=morph.link_radius,
    )

    candidates: list[Any] = []
    for idx in selected_indices:
        record = top_records[idx]
        candidates.append(
            build_candidate(
                record,
                validation_data_list[idx]["ik_success_pose_rate"].detach().cpu().item(),
            )
        )

    return final_record, optimized_morph, candidates
