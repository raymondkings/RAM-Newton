from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from interface import Morphology, Task
from optim.direct_ik_baseline import (
    _direct_ik_parameters,
    _run_direct_ik_length_optimization,
    _run_validation,
)
from util.fixed_alpha_morphology_candidates import (
    DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
    DEFAULT_FIXED_ALPHA_RANDOM_TRIES,
    generate_alpha_candidates,
    _sample_one_chunk_with_fixed_alpha,
    _set_sampler_seed,
)
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import build_optimization_validation_context


_PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_CANDIDATE_DOFS = (5, 6, 7)
DEFAULT_NUM_ALPHA_CANDIDATES: int | str = 64
DEFAULT_CANDIDATE_BATCH_SIZE = 16
DEFAULT_DISTRIBUTION_BATCH_SIZE = 64
DEFAULT_TOP_FRACTION = 0.10
ZERO_ALPHA_RUN_EXCLUSION_LENGTH = 3
SE3_TIE_EPS = 1e-12


def _resolve_candidate_dofs(value: Any) -> list[int]:
    if value is None:
        return list(DEFAULT_CANDIDATE_DOFS)
    if isinstance(value, str):
        dofs = [int(part) for part in value.replace(",", " ").split()]
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
        alpha_candidates, max_alpha_candidates, using_all = generate_alpha_candidates(
            requested_num_candidates=requested_num_candidates,
            seq_len=dof + 1,
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


def _sample_fixed_alpha_morphology_candidates_by_dof_no_cache(
    *,
    alpha_candidates_by_dof: dict[int, Tensor],
    seed: int | None,
    link_radius: float,
    batch_size: int,
    max_attempts_per_alpha: int,
    dynamic_rejection_batch_size: int,
    logging: bool,
) -> dict[int, Tensor]:
    if not alpha_candidates_by_dof:
        raise ValueError("alpha_candidates_by_dof must not be empty.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_attempts_per_alpha <= 0:
        raise ValueError("max_attempts_per_alpha must be positive.")
    if dynamic_rejection_batch_size <= 0:
        raise ValueError("dynamic_rejection_batch_size must be positive.")

    first_alpha = next(iter(alpha_candidates_by_dof.values()))
    _set_sampler_seed(seed, first_alpha.device)

    sampled_by_dof: dict[int, Tensor] = {}
    for dof, alpha_candidates in sorted(alpha_candidates_by_dof.items()):
        if alpha_candidates.ndim != 3 or alpha_candidates.shape[-1] != 1:
            raise ValueError(
                f"Expected alpha_candidates for DOF{dof} to have shape "
                f"[N, dof+1, 1], got {tuple(alpha_candidates.shape)}."
            )
        if alpha_candidates.shape[1] - 1 != int(dof):
            raise ValueError(
                f"DOF key {dof} does not match alpha candidate shape "
                f"{tuple(alpha_candidates.shape)}."
            )

        chunks: list[Tensor] = []
        total_failed = 0
        iterator = tqdm(
            range(0, alpha_candidates.shape[0], batch_size),
            desc=f"sampling uncached DOF{dof} fixed-alpha morphologies",
            disable=not logging,
            dynamic_ncols=True,
        )
        for start in iterator:
            end = min(start + batch_size, alpha_candidates.shape[0])
            sampled, failed_count = _sample_one_chunk_with_fixed_alpha(
                alpha_candidates[start:end].detach(),
                link_radius=link_radius,
                max_attempts_per_alpha=max_attempts_per_alpha,
                dynamic_rejection_batch_size=dynamic_rejection_batch_size,
            )
            if sampled.numel() > 0:
                chunks.append(sampled)
            total_failed += failed_count

            if logging:
                kept = sum(chunk.shape[0] for chunk in chunks)
                iterator.set_postfix(
                    kept=kept,
                    failed=total_failed,
                    tries=max_attempts_per_alpha,
                )

        if not chunks:
            raise RuntimeError(
                f"Fixed-alpha sampler failed to generate any DOF{dof} morphology."
            )

        sampled_by_dof[int(dof)] = torch.cat(chunks, dim=0)
        if logging:
            print(
                f"[Info] Uncached DOF{dof} fixed-alpha sampling: "
                f"sampled {sampled_by_dof[int(dof)].shape[0]}/"
                f"{alpha_candidates.shape[0]} candidates, failed={total_failed}."
            )

    return sampled_by_dof


def _distribution_valid_mask(
    processed_morphologies: Tensor,
    batch_size: int,
    logging: bool,
    desc: str,
) -> Tensor:
    from util.distribution_checker import check_morphology_distribution

    reports = []
    for start in tqdm(
        range(0, processed_morphologies.shape[0], batch_size),
        desc=desc,
        disable=not logging,
        dynamic_ncols=True,
    ):
        end = min(start + batch_size, processed_morphologies.shape[0])
        reports.extend(
            check_morphology_distribution(
                processed_morphologies[start:end],
                num_joint_samples=1000,
                seed=0,
            )
        )

    return torch.tensor(
        [report.valid for report in reports],
        dtype=torch.bool,
        device=processed_morphologies.device,
    )


def _tie_score(
    processed_morphology: Tensor, link_radius: float
) -> tuple[Tensor, Tensor]:
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


def _candidate_sort_key(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        -float(record["direct_success"].detach().cpu().item()),
        float(record["loss"].detach().cpu().item()),
        float(record["direct_se3"].detach().cpu().item()),
    )


def _optimize_one_dof_group(
    *,
    dof: int,
    initial_candidate_morphologies: Tensor,
    target_poses_local: Tensor,
    link_radius: float,
    direct_params: dict[str, Any],
    candidate_batch_size: int,
    distribution_batch_size: int,
    random_seed: int,
    logging: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for start in range(
        0, initial_candidate_morphologies.shape[0], candidate_batch_size
    ):
        end = min(start + candidate_batch_size, initial_candidate_morphologies.shape[0])
        batch_initial = initial_candidate_morphologies[start:end]

        _, metrics = _run_direct_ik_length_optimization(
            initial_morphologies=batch_initial,
            target_poses_local=target_poses_local,
            link_radius=link_radius,
            num_iterations=direct_params["num_iterations"],
            learning_rate_length=direct_params["learning_rate_length"],
            learning_rate_joint=direct_params["learning_rate_joint"],
            num_joint_seeds=direct_params["num_joint_seeds"],
            joint_refinement_steps=direct_params["joint_refinement_steps"],
            collision_weight=direct_params["collision_weight"],
            collision_margin=direct_params["collision_margin"],
            success_eps=direct_params["success_eps"],
            softmin_temperature=direct_params["softmin_temperature"],
            random_seed=random_seed + start,
            logging=logging,
            desc=f"direct IK DOF{dof} candidates {start}-{end}",
        )

        valid_mask = _distribution_valid_mask(
            metrics.processed_morphologies.detach(),
            batch_size=distribution_batch_size,
            logging=logging,
            desc=f"post-checking DOF{dof} direct IK candidates {start}-{end}",
        )

        if logging:
            print(
                f"[Info] DOF{dof} direct IK batch {start}-{end}: "
                f"kept {int(valid_mask.sum().item())}/{valid_mask.numel()} after distribution check."
            )

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        for local_idx in valid_indices.tolist():
            records.append(
                {
                    "dof": dof,
                    "loss": metrics.loss_per_candidate[local_idx].detach(),
                    "direct_success": metrics.pose_success_rate[local_idx].detach(),
                    "direct_se3": metrics.best_se3_mean[local_idx].detach(),
                    "raw_morphology": metrics.raw_morphologies[local_idx].detach(),
                    "processed_morphology": metrics.processed_morphologies[
                        local_idx
                    ].detach(),
                }
            )

    return records


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Run a direct IK baseline with discrete-alpha candidate selection.

    This mirrors the high-level flow of nrm_alpha_random_selection_5_7:
    generate fixed alpha candidates, sample one valid initial morphology per alpha,
    optimize [a, d] with the direct IK objective, filter invalid morphologies, then
    validate the best candidates with the shared cuRobo IK/FK validator.
    """
    direct_params = _direct_ik_parameters(optimization_parameters)
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
    top_fraction = float(
        optimization_parameters.get("top_fraction", DEFAULT_TOP_FRACTION)
    )
    max_attempts_per_alpha = int(
        optimization_parameters.get(
            "max_attempts_per_alpha",
            DEFAULT_FIXED_ALPHA_RANDOM_TRIES,
        )
    )
    dynamic_rejection_batch_size = int(
        optimization_parameters.get(
            "dynamic_rejection_batch_size",
            DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
        )
    )

    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")
    if distribution_batch_size <= 0:
        raise ValueError("distribution_batch_size must be positive.")
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")

    device = morph.params.device
    dtype = morph.params.dtype

    if logging:
        print(f"[Info] Starting direct IK candidate baseline on device {device}.")
        print(
            "[Info] "
            f"candidate_dofs={candidate_dofs}, "
            f"num_alpha_candidates={num_alpha_candidates}, "
            f"candidate_batch_size={candidate_batch_size}, "
            f"top_fraction={top_fraction}, "
            f"direct_params={direct_params}"
        )

    scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )
    target_poses_local = task.goal_poses.to(device=device, dtype=dtype)
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
            _sample_fixed_alpha_morphology_candidates_by_dof_no_cache(
                alpha_candidates_by_dof=alpha_candidates_by_dof,
                seed=random_seed,
                link_radius=morph.link_radius,
                batch_size=distribution_batch_size,
                max_attempts_per_alpha=max_attempts_per_alpha,
                dynamic_rejection_batch_size=dynamic_rejection_batch_size,
                logging=logging,
            )
        )

        if logging:
            print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

        records: list[dict[str, Any]] = []
        for dof in candidate_dofs:
            records.extend(
                _optimize_one_dof_group(
                    dof=dof,
                    initial_candidate_morphologies=initial_candidate_morphologies_by_dof[
                        dof
                    ],
                    target_poses_local=target_poses_local,
                    link_radius=morph.link_radius,
                    direct_params=direct_params,
                    candidate_batch_size=candidate_batch_size,
                    distribution_batch_size=distribution_batch_size,
                    random_seed=random_seed,
                    logging=logging,
                )
            )

        if not records:
            raise RuntimeError(
                "All direct IK candidate morphologies were rejected by the "
                "post-optimization distribution checker."
            )

        records = sorted(records, key=_candidate_sort_key)
        top_k = max(1, int(math.ceil(len(records) * top_fraction)))
        top_records = records[:top_k]

        if logging:
            print(
                "[Info] Direct IK top-candidate selection: "
                f"valid_candidates={len(records)}, top_k={top_k}, "
                f"best_direct_success={top_records[0]['direct_success'].item() * 100.0:.2f}%, "
                f"best_direct_loss={top_records[0]['loss'].item():.6f}"
            )

        se3_scores = torch.empty(len(top_records), device=device)
        ik_success_rates = torch.empty(len(top_records), device=device)
        validation_data_list: list[dict] = []

        for idx in tqdm(
            range(len(top_records)),
            desc="validating top direct IK candidates",
            disable=not logging,
            dynamic_ncols=True,
        ):
            validation_data = _run_validation(
                processed_morphology=top_records[idx]["processed_morphology"],
                morph=morph,
                task=task,
                scene=scene,
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

        for idx, record in enumerate(top_records):
            if idx == final_idx:
                marker = 2
            elif bool(final_tier_mask[idx]):
                marker = 1
            else:
                marker = 0

            csv_logger.log_iteration(
                iteration=marker,
                loss=record["loss"],
                reachability_probability=record["direct_success"],
                raw_morphology=record["raw_morphology"],
                processed_morphology=record["processed_morphology"],
                validation_data=validation_data_list[idx],
            )

        final_record = top_records[final_idx]
        final_validation_data = validation_data_list[final_idx]
        final_se3 = final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        final_ik_success_rate = (
            final_validation_data["ik_success_pose_rate"].detach().cpu().item()
        )

        print(
            "[Final direct IK candidate] "
            f"dof={final_record['dof']}, "
            f"loss={final_record['loss'].item():.6f}, "
            f"direct_ik_success={final_record['direct_success'].item() * 100.0:.2f}%, "
            f"direct_se3={final_record['direct_se3'].item():.6f}, "
            f"final_se3_err={final_se3:.6f}, "
            f"ik_success_pose_rate={final_ik_success_rate * 100.0:.2f}%, "
            f"length_sum={length_sums_tensor[final_idx].item():.6f}"
        )

        optimized_morph = Morphology(
            params=final_record["processed_morphology"].detach(),
            link_radius=morph.link_radius,
        )
        return optimized_morph, csv_logger.csv_path

    finally:
        csv_logger.close()
