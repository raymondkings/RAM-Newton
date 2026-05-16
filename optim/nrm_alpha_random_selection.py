# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Shiyuan Zhang
# -----------------------------------------------------------------------------
# Random discrete-alpha candidate selection + gradient-based length optimization.
# -----------------------------------------------------------------------------

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
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

DEFAULT_NUM_ALPHA_CANDIDATES = 1000
DEFAULT_NUM_SELECTION_STAGES = 8
DEFAULT_CANDIDATE_BATCH_SIZE = 32
ALPHA_VALUES = (-math.pi / 2.0, 0.0, math.pi / 2.0)


def _se3_to_vector(pose: Tensor) -> Tensor:
    """Convert SE(3) pose matrices [..., 4, 4] to 9D NRM pose vectors."""
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


def _load_model(device: torch.device) -> MLP:
    metadata = json.loads((_WEIGHTS_DIR / "metadata.json").read_text())
    model = MLP(**metadata["hyperparameter"])
    model.load_state_dict(
        torch.load(_WEIGHTS_DIR / "checkpoint.pth", map_location=device, weights_only=True)
    )
    model = model.to(device)

    # cuDNN LSTM backward may require train mode.
    # The weights are frozen, but gradients still flow to morphology inputs.
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


def _round_iterations(total_iterations: int, round_idx: int, num_selection_stages: int) -> int:
    """Number of length-optimization steps for one selection/final round.

    For num_selection_stages=3, round_idx 0..3 gives denominators 8, 8, 4, 2.
    The final round is round_idx == num_selection_stages.
    """
    if num_selection_stages <= 0:
        return max(1, total_iterations)
    if round_idx == 0:
        denom = 2 ** num_selection_stages
    else:
        denom = 2 ** (num_selection_stages - round_idx + 1)
    return max(1, total_iterations // denom)


def _keep_count(current_count: int) -> int:
    """Keep the best half of the currently active candidates."""
    return max(1, current_count // 2)


def _generate_random_alpha_candidates(
    num_candidates: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tensor:
    """Sample unique alpha combinations from {-pi/2, 0, pi/2}^seq_len."""
    total_combinations = 3 ** seq_len
    if num_candidates > total_combinations:
        raise ValueError(
            f"Requested {num_candidates} alpha candidates, but only "
            f"{total_combinations} unique combinations exist for seq_len={seq_len}."
        )

    flat_ids = torch.randperm(total_combinations, generator=generator, device=device)[:num_candidates]

    digits = []
    x = flat_ids.clone()
    for _ in range(seq_len):
        digits.append(x % 3)
        x = x // 3
    alpha_indices = torch.stack(digits, dim=1)

    alpha_values = torch.tensor(ALPHA_VALUES, device=device, dtype=torch.float32)
    return alpha_values[alpha_indices].unsqueeze(-1)


def _candidate_loss(
    model: MLP,
    alpha_candidates: Tensor,
    length_candidates: Tensor,
    task_vec: Tensor,
    link_radius: float,
) -> Tensor:
    """Compute one mean BCE loss per candidate."""
    num_candidates = alpha_candidates.shape[0]
    num_poses = task_vec.shape[0]

    processed_lengths, _ = _preprocess_lengths(length_candidates, link_radius)
    morph_params = torch.cat([alpha_candidates, processed_lengths], dim=-1)

    bmorph = morph_params.unsqueeze(1).expand(num_candidates, num_poses, -1, -1)
    bpose = task_vec.unsqueeze(0).expand(num_candidates, num_poses, -1)

    bmorph = bmorph.reshape(num_candidates * num_poses, morph_params.shape[-2], 3)
    bpose = bpose.reshape(num_candidates * num_poses, 9)

    logit = model(bmorph, bpose).reshape(num_candidates, num_poses)
    target = torch.ones_like(logit)
    loss_per_pose = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return loss_per_pose.mean(dim=1)


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

    with torch.no_grad():
        for start in range(0, alpha_candidates.shape[0], candidate_batch_size):
            end = min(start + candidate_batch_size, alpha_candidates.shape[0])
            alpha_batch = alpha_candidates[start:end]
            length_batch = length_candidates[start:end]
            num_candidates = alpha_batch.shape[0]
            num_poses = task_vec.shape[0]

            processed_lengths, _ = _preprocess_lengths(length_batch, link_radius)
            morph_params = torch.cat([alpha_batch, processed_lengths], dim=-1)

            bmorph = morph_params.unsqueeze(1).expand(num_candidates, num_poses, -1, -1)
            bpose = task_vec.unsqueeze(0).expand(num_candidates, num_poses, -1)

            bmorph = bmorph.reshape(num_candidates * num_poses, morph_params.shape[-2], 3)
            bpose = bpose.reshape(num_candidates * num_poses, 9)

            logit = model(bmorph, bpose).reshape(num_candidates, num_poses)
            target = torch.ones_like(logit)
            loss_per_pose = F.binary_cross_entropy_with_logits(logit, target, reduction="none")

            losses.append(loss_per_pose.mean(dim=1).detach())
            probs.append(torch.sigmoid(logit).mean(dim=1).detach())

    return torch.cat(losses, dim=0), torch.cat(probs, dim=0)


def _candidate_morphologies(
    alpha_candidate: Tensor,
    length_candidate: Tensor,
    link_radius: float,
) -> tuple[Tensor, Tensor]:
    """Build raw and processed morphology tensors for one candidate."""
    processed_lengths, _ = _preprocess_lengths(length_candidate, link_radius)
    raw_morphology = torch.cat([alpha_candidate, length_candidate], dim=-1)
    processed_morphology = torch.cat([alpha_candidate, processed_lengths], dim=-1)
    return raw_morphology, processed_morphology


def _compute_single_candidate_loss_prob(
    model: MLP,
    processed_morphology: Tensor,
    task_vec: Tensor,
) -> tuple[Tensor, Tensor]:
    bmorph = processed_morphology.unsqueeze(0).expand(task_vec.shape[0], -1, -1)
    logit = model(bmorph, task_vec)
    loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(logit, torch.ones_like(logit))
    prob = torch.sigmoid(logit).mean()
    return loss, prob


def _optimize_active_candidates(
    model: MLP,
    active_alpha: Tensor,
    active_lengths: Tensor,
    task_vec: Tensor,
    link_radius: float,
    num_iterations: int,
    learning_rate: float,
    candidate_batch_size: int,
    logging: bool,
) -> Tensor:
    """Optimize only a,d lengths for currently active fixed-alpha candidates."""
    active_lengths = active_lengths.detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([active_lengths], lr=learning_rate)
    num_active = active_alpha.shape[0]

    iterator = tqdm(
        range(num_iterations),
        desc="optimizing candidate lengths",
        disable=not logging,
        dynamic_ncols=True,
    )

    for _ in iterator:
        optimizer.zero_grad()

        # Accumulate gradients over chunks without changing the objective:
        # total loss = mean over all active candidates.
        for start in range(0, num_active, candidate_batch_size):
            end = min(start + candidate_batch_size, num_active)
            loss_per_candidate = _candidate_loss(
                model=model,
                alpha_candidates=active_alpha[start:end],
                length_candidates=active_lengths[start:end],
                task_vec=task_vec,
                link_radius=link_radius,
            )
            (loss_per_candidate.sum() / num_active).backward()

        optimizer.step()

    return active_lengths.detach()


def _stage_generator(device: torch.device, random_seed: int, stage_idx: int) -> torch.Generator:
    """Create a deterministic validation generator for one selection stage."""
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed + 42 * stage_idx)
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
    stage_idx: int,
) -> dict:
    """Validate one candidate using the current optimization_validation API."""
    return run_optimization_validation(
        processed_morphology=processed_morphology.detach(),
        morph=morph,
        task=task,
        scene=scene,
        base_pose_inv=base_pose_inv,
        device=device,
        percentage_poses=percentage_poses,
        number_random_seed=number_random_seed,
        pose_sampling_generator=_stage_generator(device, random_seed, stage_idx),
    )


def _log_candidate(
    csv_logger: OptimizationCSVLogger,
    iteration: int,
    loss: Tensor,
    prob: Tensor,
    raw_morphology: Tensor,
    processed_morphology: Tensor,
    validation_data: dict | None,
) -> None:
    csv_logger.log_iteration(
        iteration=iteration,
        loss=loss.detach(),
        reachability_probability=prob.detach(),
        raw_morphology=raw_morphology.detach(),
        processed_morphology=processed_morphology.detach(),
        validation_data=validation_data,
    )


def _log_kept_candidates(
    csv_logger: OptimizationCSVLogger,
    iteration: int,
    kept_indices: Tensor,
    active_alpha: Tensor,
    active_lengths: Tensor,
    losses: Tensor,
    probs: Tensor,
    link_radius: float,
    validation_cache: dict[int, dict] | None = None,
) -> None:
    for idx_tensor in kept_indices:
        idx = int(idx_tensor.item())
        raw_morphology, processed_morphology = _candidate_morphologies(
            active_alpha[idx],
            active_lengths[idx],
            link_radius,
        )
        validation_data = None if validation_cache is None else validation_cache.get(idx)
        _log_candidate(
            csv_logger=csv_logger,
            iteration=iteration,
            loss=losses[idx],
            prob=probs[idx],
            raw_morphology=raw_morphology,
            processed_morphology=processed_morphology,
            validation_data=validation_data,
        )


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Random discrete alpha search plus gradient optimization of a,d lengths."""
    total_iterations = optimization_parameters.get("num_iterations", 100)
    lr = optimization_parameters.get("learning_rate", 0.01)
    logging = optimization_parameters.get("logging", True)
    random_seed = optimization_parameters.get("random_seed", 42)
    number_random_seed = optimization_parameters.get("number_random_seed", 32)
    percentage_poses = optimization_parameters.get("percentage_poses", 1)
    ignore_ground = optimization_parameters.get("ignore_ground", False)
    ignore_obstacles = optimization_parameters.get("ignore_obstacles", False)

    num_alpha_candidates = int(
        optimization_parameters.get("num_alpha_candidates", DEFAULT_NUM_ALPHA_CANDIDATES)
    )
    num_selection_stages = int(
        optimization_parameters.get("num_selection_stages", DEFAULT_NUM_SELECTION_STAGES)
    )
    candidate_batch_size = int(
        optimization_parameters.get("candidate_batch_size", DEFAULT_CANDIDATE_BATCH_SIZE)
    )

    if num_alpha_candidates <= 0:
        raise ValueError("num_alpha_candidates must be positive.")
    if num_selection_stages < 0:
        raise ValueError("num_selection_stages must be non-negative.")
    if candidate_batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive.")

    device = morph.params.device

    if logging:
        print(f"[Info] Starting random-alpha candidate optimization on device {device}.")
        print(
            "[Info] "
            f"total_iterations={total_iterations}, "
            f"learning_rate={lr}, "
            f"num_alpha_candidates={num_alpha_candidates}, "
            f"num_selection_stages={num_selection_stages}, "
            f"candidate_batch_size={candidate_batch_size}, "
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

    goal_poses = task.goal_poses.to(device)
    task_vec = _se3_to_vector(goal_poses)
    model = _load_model(device)
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    alpha_generator = torch.Generator(device=device)
    alpha_generator.manual_seed(random_seed)

    try:
        seq_len = morph.params.shape[0]
        alpha_candidates = _generate_random_alpha_candidates(
            num_candidates=num_alpha_candidates,
            seq_len=seq_len,
            device=device,
            generator=alpha_generator,
        )

        initial_lengths = morph.params[:, 1:].clone().to(device)
        length_candidates = initial_lengths.unsqueeze(0).expand(num_alpha_candidates, -1, -1).clone()

        if logging:
            print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

        # One row for the very initial morphology from the morphology sampler.
        with torch.no_grad():
            #initial morphology need no normalization and squash
            initial_raw_morphology = morph.params.detach().clone().to(device)
            initial_loss, initial_prob = _compute_single_candidate_loss_prob(
                model=model,
                processed_morphology=initial_raw_morphology,
                task_vec=task_vec,
            )

        csv_logger.log_iteration(
            iteration=0,
            loss=initial_loss.detach(),
            reachability_probability=initial_prob.detach(),
            raw_morphology=initial_raw_morphology.detach(),
            processed_morphology=initial_raw_morphology.detach(),
            validation_data=None,
        )

        active_alpha = alpha_candidates
        active_lengths = length_candidates

        # num_selection_stages selection rounds plus one final optimization round.
        for round_idx in range(num_selection_stages + 1):
            stage_idx = round_idx + 1
            active_count = active_alpha.shape[0]
            round_iterations = _round_iterations(total_iterations, round_idx, num_selection_stages)

            if logging:
                print(
                    f"\n[Stage {stage_idx}/{num_selection_stages + 1}] "
                    f"active_candidates={active_count}, "
                    f"length_iterations={round_iterations}"
                )

            active_lengths = _optimize_active_candidates(
                model=model,
                active_alpha=active_alpha,
                active_lengths=active_lengths,
                task_vec=task_vec,
                link_radius=morph.link_radius,
                num_iterations=round_iterations,
                learning_rate=lr,
                candidate_batch_size=candidate_batch_size,
                logging=logging,
            )

            losses, probs = _evaluate_scores_batched(
                model=model,
                alpha_candidates=active_alpha,
                length_candidates=active_lengths,
                task_vec=task_vec,
                link_radius=morph.link_radius,
                candidate_batch_size=candidate_batch_size,
            )

            is_final_round = round_idx == num_selection_stages
            if is_final_round:
                best_idx = int(torch.argmin(losses).item())
                raw_morphology, final_processed_morphology = _candidate_morphologies(
                    active_alpha[best_idx],
                    active_lengths[best_idx],
                    morph.link_radius,
                )
                final_validation_data = _validate_candidate(
                    processed_morphology=final_processed_morphology,
                    morph=morph,
                    task=task,
                    scene=scene,
                    base_pose_inv=base_pose_inv,
                    device=device,
                    percentage_poses=percentage_poses,
                    number_random_seed=number_random_seed,
                    random_seed=random_seed,
                    stage_idx=stage_idx,
                )

                _log_candidate(
                    csv_logger=csv_logger,
                    iteration=stage_idx,
                    loss=losses[best_idx],
                    prob=probs[best_idx],
                    raw_morphology=raw_morphology,
                    processed_morphology=final_processed_morphology,
                    validation_data=final_validation_data,
                )

                final_se3 = final_validation_data["best_se3_dist_mean"].detach().cpu().item()
                print(
                    f"[Final candidate] "
                    f"iteration={stage_idx}, "
                    f"loss={losses[best_idx].item():.6f}, "
                    f"nrm_prob={probs[best_idx].item():.6f}, "
                    f"final_se3_err={final_se3:.6f}"
                )
                if logging:
                    print("Final alpha [deg]:")
                    print(active_alpha[best_idx].detach().cpu().squeeze(-1) * 180.0 / math.pi)
                    print("Final optimized morphology params:")
                    print(final_processed_morphology.detach().cpu())

                optimized_morph = Morphology(
                    params=final_processed_morphology.detach(),
                    link_radius=morph.link_radius,
                )
                return optimized_morph, csv_logger.csv_path

            keep = _keep_count(active_count)
            use_validation_selection = round_idx == num_selection_stages

            if use_validation_selection:
                if logging:
                    print(
                        "Validation-based selection stage: validating all active candidates "
                        f"({active_count}) and keeping {keep}."
                    )

                se3_scores = torch.empty(active_count, device=device)
                validation_cache: dict[int, dict] = {}

                for local_idx in tqdm(
                    range(active_count),
                    desc="validating candidates",
                    disable=not logging,
                    dynamic_ncols=True,
                ):
                    _, processed_morphology = _candidate_morphologies(
                        active_alpha[local_idx],
                        active_lengths[local_idx],
                        morph.link_radius,
                    )
                    validation_data = _validate_candidate(
                        processed_morphology=processed_morphology,
                        morph=morph,
                        task=task,
                        scene=scene,
                        base_pose_inv=base_pose_inv,
                        device=device,
                        percentage_poses=percentage_poses,
                        number_random_seed=number_random_seed,
                        random_seed=random_seed,
                        stage_idx=stage_idx,
                    )
                    validation_cache[local_idx] = validation_data
                    se3_scores[local_idx] = validation_data["best_se3_dist_mean"].detach().to(device)

                kept_indices = torch.argsort(se3_scores)[:keep]
                if logging:
                    print(
                        f"Validation selection: keep {keep}/{active_count}. "
                        f"best_se3={se3_scores[kept_indices[0]].item():.6f}, "
                        f"worst_kept_se3={se3_scores[kept_indices[-1]].item():.6f}"
                    )

                _log_kept_candidates(
                    csv_logger=csv_logger,
                    iteration=stage_idx,
                    kept_indices=kept_indices,
                    active_alpha=active_alpha,
                    active_lengths=active_lengths,
                    losses=losses,
                    probs=probs,
                    link_radius=morph.link_radius,
                    validation_cache=validation_cache,
                )

            else:
                kept_indices = torch.argsort(losses)[:keep]
                if logging:
                    print(
                        f"NRM selection: keep {keep}/{active_count}. "
                        f"best_loss={losses[kept_indices[0]].item():.6f}, "
                        f"worst_kept_loss={losses[kept_indices[-1]].item():.6f}"
                    )

                _log_kept_candidates(
                    csv_logger=csv_logger,
                    iteration=stage_idx,
                    kept_indices=kept_indices,
                    active_alpha=active_alpha,
                    active_lengths=active_lengths,
                    losses=losses,
                    probs=probs,
                    link_radius=morph.link_radius,
                    validation_cache=None,
                )

            active_alpha = active_alpha[kept_indices].detach()
            active_lengths = active_lengths[kept_indices].detach()

        raise RuntimeError("Optimization failed to return a final morphology.")

    finally:
        csv_logger.close()
