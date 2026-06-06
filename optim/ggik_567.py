"""GGIK morphology optimisation — mirrors nrm.py with GGIK as the IK oracle.

Drop-in replacement for nrm.py.  NRM's BCE loss is replaced by GGIK FK error
plus a differentiable self-collision penalty on the capsule geometry.

Config keys (all via optimization_parameters):
    dof                     int    DOF to optimise (5, 6, or 7). If different
                                   from morph.params, a fresh random morphology
                                   of that DOF is sampled. Default: morph DOF.
    num_iterations          int    Gradient-descent steps. Default: 100.
    learning_rate           float  AdamW LR for [a, d]. Default: 0.01.
    eval_interval           int    cuRobo validation every N steps. Default: 1.
    collision_weight        float  Weight for self-collision penalty. Default: 10.
    collision_margin        float  Minimum safe capsule distance. Default: 0.0.
    success_eps             float  SE3 threshold for IK success. Default: 1e-4.
    softmin_temperature     float  Soft-min over GGIK samples. Default: 0.01.
    random_seed             int    Default: 42.
    number_random_seed      int    IK validation seeds. Default: 32.
    percentage_poses        float  Fraction of poses validated. Default: 1.
    ignore_ground           bool   Default: False.
    ignore_obstacles        bool   Default: False.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from interface import Morphology, Task

EPS = 1e-4

from task.morphology_sampler import sample_morph
from util.ggik_baseline_common import (
    _build_ggik_solver,
    _ggik_metrics,
    _ggik_parameters,
)
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)

_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_candidate_dofs(value) -> list[int]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        dofs = [int(p) for p in parts]
    else:
        dofs = [int(d) for d in value]
    if not dofs:
        raise ValueError("candidate_dofs must not be empty.")
    if any(d <= 0 for d in dofs):
        raise ValueError(f"candidate_dofs must be positive, got {dofs}.")
    return sorted(set(dofs))


# ── preprocessing (same as nrm.py) ───────────────────────────────────────────


class _SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, param, threshold):
        return param * (param.abs() >= threshold).float()

    @staticmethod
    def backward(_ctx, grad_output):
        return grad_output, None


class _Normaliser(torch.autograd.Function):
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
        return (grad_output * norm - chain * (grad_output * param).sum()) / norm ** 2


def _preprocess(lengths, link_radius):
    threshold = 2.0 * link_radius
    norm = _Normaliser.apply(lengths)
    squashed = _SquasherSTE.apply(norm, threshold)
    return _Normaliser.apply(squashed), norm


# ── goal poses in robot-local frame ──────────────────────────────────────────


def _goal_poses_local(task: Task, device, dtype):
    """Return goal poses in robot-local frame (world frame = robot base frame)."""
    return task.goal_poses.to(device=device, dtype=dtype)


# ── main entry point ─────────────────────────────────────────────────────────


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """Optimise morphology [a, d] using GGIK FK error + self-collision penalty.

    Mirrors nrm.py: same loop structure, same CSV logging, same cuRobo
    validation.  NRM's BCE loss is replaced by the GGIK-based SE3 loss.

    Returns:
        optimized_morphology: final processed morphology.
        csv_path:             path to output/log_<time>.csv.
    """
    n_iter         = int(optimization_parameters.get("num_iterations", 100))
    lr             = float(optimization_parameters.get("learning_rate", 0.01))
    logging        = bool(optimization_parameters.get("logging", True))
    eval_interval  = int(optimization_parameters.get("eval_interval", 1))
    random_seed    = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses   = float(optimization_parameters.get("percentage_poses", 1))
    ignore_ground      = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles   = bool(optimization_parameters.get("ignore_obstacles", False))

    ggik_params = _ggik_parameters(optimization_parameters)

    device = morph.params.device
    dtype  = morph.params.dtype

    # ── DOF selection ────────────────────────────────────────────────────────
    candidate_dofs_raw = optimization_parameters.get("candidate_dofs", None)
    morph_dof          = morph.n_links - 1

    if candidate_dofs_raw is not None:
        dof = _resolve_candidate_dofs(candidate_dofs_raw)[0]
    else:
        dof = morph_dof

    link_radius = morph.link_radius
    if dof != morph_dof:
        if logging:
            print(f"[GGIK] DOF mismatch: morph has {morph_dof}-DOF, "
                  f"sampling fresh {dof}-DOF morphology.")
        torch.manual_seed(random_seed)
        sampled = sample_morph(1, dof, analytically_solvable=False, device=device)[0]
        alpha   = sampled[:, 0:1].clone()
        lengths = sampled[:, 1:].clone().requires_grad_(True)
    else:
        alpha   = morph.params[:, 0:1].clone().to(device)
        lengths = morph.params[:, 1:].clone().to(device).requires_grad_(True)

    if logging:
        print(f"[GGIK] Starting {dof}-DOF optimisation, "
              f"{n_iter} iterations on device {device}.")

    # ── validation context (cuRobo) ──────────────────────────────────────────
    scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    target_poses_local = _goal_poses_local(task, device, dtype)

    # ── GGIK solver ──────────────────────────────────────────────────────────
    solver = _build_ggik_solver(optimization_parameters, device=device, dtype=dtype)

    optimizer = torch.optim.AdamW([lengths], lr=lr, weight_decay=0.0)
    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)

    pose_sampling_generator = torch.Generator(device=device)
    pose_sampling_generator.manual_seed(random_seed)

    if logging:
        print(f"[GGIK] Writing CSV log to: {csv_logger.csv_path}")

    try:
        progress_bar = tqdm(range(n_iter), desc="GGIK optimising", dynamic_ncols=True)

        for update_idx in progress_bar:
            optimizer.zero_grad()

            processed_lengths, _ = _preprocess(lengths, link_radius)
            raw_morphology       = torch.cat([alpha, lengths.detach()], dim=1)
            processed_morphology = torch.cat([alpha, processed_lengths], dim=1)

            # GGIK metrics: [1-candidate batch]
            metrics = _ggik_metrics(
                alpha=alpha.unsqueeze(0),
                lengths=processed_lengths.unsqueeze(0),
                solver=solver,
                target_poses_local=target_poses_local,
                link_radius=link_radius,
                collision_weight=ggik_params["collision_weight"],
                collision_margin=ggik_params["collision_margin"],
                success_eps=ggik_params["success_eps"],
                softmin_temperature=ggik_params["softmin_temperature"],
            )

            loss    = metrics.loss_per_candidate[0]
            success = metrics.pose_success_rate[0]
            se3     = metrics.best_se3_mean[0]
            col_rate = metrics.self_collision_pose_rate[0]

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
                best_se3_val = validation_data["best_se3_dist_mean"].detach().cpu().item()
                tqdm.write(
                    f"[Iter {update_idx:>4}/{n_iter}] "
                    f"loss={loss.item():.6f}, "
                    f"ggik_success={success.item():.3f}, "
                    f"ggik_se3={se3.item():.6f}, "
                    f"self_col={col_rate.item():.3f}, "
                    f"val_se3={best_se3_val:.6f}"
                )

            csv_logger.log_iteration(
                iteration=update_idx,
                loss=loss.detach(),
                reachability_probability=success.detach(),
                raw_morphology=raw_morphology.detach(),
                processed_morphology=processed_morphology.detach(),
                validation_data=validation_data,
            )

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                success=f"{success.item():.3f}",
                self_col=f"{col_rate.item():.3f}",
            )

        # ── final evaluation ─────────────────────────────────────────────────
        with torch.no_grad():
            final_processed_lengths, _ = _preprocess(lengths.detach(), link_radius)
            final_raw_morphology       = torch.cat([alpha, lengths.detach()], dim=1)
            final_processed_morphology = torch.cat([alpha, final_processed_lengths], dim=1)

            final_metrics = _ggik_metrics(
                alpha=alpha.unsqueeze(0),
                lengths=final_processed_lengths.unsqueeze(0),
                solver=solver,
                target_poses_local=target_poses_local,
                link_radius=link_radius,
                collision_weight=ggik_params["collision_weight"],
                collision_margin=ggik_params["collision_margin"],
                success_eps=ggik_params["success_eps"],
                softmin_temperature=ggik_params["softmin_temperature"],
            )
            final_loss     = final_metrics.loss_per_candidate[0]
            final_success  = final_metrics.pose_success_rate[0]
            final_se3      = final_metrics.best_se3_mean[0]
            final_col_rate = final_metrics.self_collision_pose_rate[0]

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
            iteration=n_iter,
            loss=final_loss,
            reachability_probability=final_success,
            raw_morphology=final_raw_morphology,
            processed_morphology=final_processed_morphology,
            validation_data=final_validation_data,
        )

        final_se3_err = final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        print(
            f"[Iter {n_iter:>4}/{n_iter}] "
            f"loss={final_loss.item():.6f}, "
            f"ggik_success={final_success.item():.3f}, "
            f"ggik_se3={final_se3.item():.6f}, "
            f"self_col={final_col_rate.item():.3f}, "
            f"final_val_se3={final_se3_err:.6f}"
        )

        return (
            Morphology(
                params=final_processed_morphology.detach(),
                link_radius=link_radius,
            ),
            csv_logger.csv_path,
        )

    finally:
        csv_logger.close()
