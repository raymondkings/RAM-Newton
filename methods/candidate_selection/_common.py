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
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from core import Morphology, Task
from logutils.csv_logger import OptimizationCSVLogger
from logutils.timing import OptimizationTimer
from methods.nrm_model import MLP
from paths import WEIGHTS_DIR
from tasks.sampling.fixed_alpha_candidates import (
    DEFAULT_ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
    generate_alpha_candidates,
    sample_fixed_alpha_morphology_candidates,
    sample_fixed_alpha_morphology_candidates_by_dof,
)
from validation.distribution_checker import check_morphology_distribution
from validation.optimization_validation import run_optimization_validation

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
