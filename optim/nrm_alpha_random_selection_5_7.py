# -----------------------------------------------------------------------------
# Multi-DOF variant of nrm_alpha_random_selection.
#
# This keeps the candidate-selection logic from nrm_alpha_random_selection, but
# generates and optimizes initial candidates for DOF 5, 6, and 7. Each DOF is
# sampled and optimized as its own fixed-length batch, then candidates are merged
# for the final top-probability validation/selection stage.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from interface import Morphology, Task
from optim.model import MLP
from optim.nrm_alpha_random_selection import (
    DEFAULT_CANDIDATE_BATCH_SIZE,
    DEFAULT_DISTRIBUTION_BATCH_SIZE,
    DEFAULT_NUM_ALPHA_CANDIDATES,
    DELTA_EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    SE3_TIE_EPS,
    TOP_PROBABILITY_FRACTION,
    ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
    _build_morphology_tensors,
    _distribution_valid_mask,
    _log_candidate,
    _optimize_all_candidates_single_round,
    _se3_to_vector,
    _validate_candidate,
)
from util.fixed_alpha_morphology_candidates import (
    generate_alpha_candidates,
    sample_fixed_alpha_morphology_candidates_by_dof,
)
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import build_optimization_validation_context


_PROJECT_ROOT = Path(__file__).parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"
_CHECKPOINT_PATH = _WEIGHTS_DIR / "checkpoint_5-7.pth"

DEFAULT_CANDIDATE_DOFS = (5, 6, 7)


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
    if value is None:
        return list(DEFAULT_CANDIDATE_DOFS)

    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        dofs = [int(part) for part in parts]
    else:
        dofs = [int(dof) for dof in value]

    if not dofs:
        raise ValueError("candidate_dofs must not be empty.")
    if any(dof <= 0 for dof in dofs):
        raise ValueError(f"candidate_dofs must be positive, got {dofs}.")

    return sorted(set(dofs))


def _generate_alpha_candidates_by_dof(
    *,
    dofs: list[int],
    requested_num_candidates: int | str | None,
    device: torch.device,
    generator: torch.Generator,
    logging: bool,
) -> dict[int, Tensor]:
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


def _tie_score(
    processed_morphology: Tensor, link_radius: float
) -> tuple[Tensor, Tensor]:
    """Return morphology heuristic score and length sum for one candidate."""
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
) -> list[dict[str, Any]]:
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


def _validate_top_records(
    *,
    records: list[dict[str, Any]],
    morph: Morphology,
    task: Task,
    scene,
    base_pose_inv: Tensor,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    random_seed: int,
    logging: bool,
) -> tuple[Tensor, Tensor, list[dict]]:
    se3_scores = torch.empty(len(records), device=device)
    ik_success_rates = torch.empty(len(records), device=device)
    validation_data_list: list[dict] = []

    for idx in tqdm(
        range(len(records)),
        desc="validating top-probability DOF5-7 candidates",
        disable=not logging,
        dynamic_ncols=True,
    ):
        validation_data = _validate_candidate(
            processed_morphology=records[idx]["processed_morphology"],
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
        ik_success_rates[idx] = (
            validation_data["ik_success_pose_rate"].detach().to(device)
        )

    return se3_scores, ik_success_rates, validation_data_list


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Single-round random/all discrete-alpha search across DOF 5, 6, and 7."""
    total_iterations = int(optimization_parameters.get("num_iterations", 100))
    lr = float(optimization_parameters.get("learning_rate", 0.01))
    logging = bool(optimization_parameters.get("logging", True))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1))
    ignore_ground = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles = bool(optimization_parameters.get("ignore_obstacles", False))
    candidate_dofs = _resolve_candidate_dofs(
        optimization_parameters.get("candidate_dofs", DEFAULT_CANDIDATE_DOFS)
    )

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

    if logging:
        print(
            f"[Info] Starting DOF5-7 alpha candidate optimization on device {device}."
        )
        print(
            "[Info] "
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
        alpha_candidates_by_dof = _generate_alpha_candidates_by_dof(
            dofs=candidate_dofs,
            requested_num_candidates=num_alpha_candidates,
            device=device,
            generator=alpha_generator,
            logging=logging,
        )

        initial_candidate_morphologies_by_dof = (
            sample_fixed_alpha_morphology_candidates_by_dof(
                alpha_candidates_by_dof=alpha_candidates_by_dof,
                seed=random_seed,
                link_radius=morph.link_radius,
                batch_size=distribution_batch_size,
                logging=logging,
            )
        )

        if logging:
            print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")
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
                )
            )

        if not records:
            raise RuntimeError(
                "All optimized DOF5-7 candidates were rejected by the "
                "post-optimization distribution checker."
            )

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
                "[Info] DOF5-7 top-probability selection: "
                f"valid_candidates={num_valid}, top_k={top_k}, "
                f"valid_by_dof={dof_counts}, top_by_dof={top_dof_counts}, "
                f"best_prob={probs_valid[top_indices[0]].item():.6f}, "
                f"worst_top_prob={probs_valid[top_indices[-1]].item():.6f}"
            )

        se3_scores, ik_success_rates, validation_data_list = _validate_top_records(
            records=top_records,
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

        best_ik_success_rate = ik_success_rates.max()
        best_rate_mask = (ik_success_rates - best_ik_success_rate).abs() <= 1e-12
        best_rate_indices = torch.nonzero(best_rate_mask, as_tuple=False).squeeze(1)

        best_se3 = se3_scores[best_rate_indices].min()
        final_tier_mask = best_rate_mask & (
            (se3_scores - best_se3).abs() <= SE3_TIE_EPS
        )
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

        if logging:
            print(
                "[Info] DOF5-7 validation selection: "
                f"best_ik_success_pose_rate={best_ik_success_rate.item() * 100.0:.2f}%, "
                f"num_best_rate_candidates={int(best_rate_mask.sum().item())}, "
                f"best_se3_within_best_rate={best_se3.item():.12f}, "
                f"num_final_ties={int(final_tier_mask.sum().item())}, "
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
            "[Final DOF5-7 candidate] "
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
        return optimized_morph, csv_logger.csv_path

    finally:
        csv_logger.close()
