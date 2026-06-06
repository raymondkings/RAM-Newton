# nrm.py adapted to support selectable DOF (5, 6, 7).
#
# Only change from nrm.py: reads optimization_parameters["dof"] and, if it
# differs from the input morphology's DOF, samples a fresh random morphology
# of the requested DOF before running the same NRM gradient-descent loop.

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor

from util.nrm_model import MLP
from util.direct_ik_common import _collision_critical_distance
from util.kinematics import forward_kinematics
from interface import Morphology, Task
from task.morphology_sampler import sample_morph
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import (
    build_optimization_validation_context,
    run_optimization_validation,
)


EPS = 1e-4

_PROJECT_ROOT = Path(__file__).parent.parent
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"


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


def _se3_to_vector(pose: Tensor) -> Tensor:
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


def _load_model(device: torch.device) -> MLP:
    metadata = json.loads((_WEIGHTS_DIR / "metadata.json").read_text())
    model = MLP(**metadata["hyperparameter"])
    model.load_state_dict(
        torch.load(
            _WEIGHTS_DIR / "checkpoint.pth", map_location=device, weights_only=True
        )
    )
    model = model.to(device)
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
        return (grad_output * norm - chain * (grad_output * param).sum()) / norm ** 2


def _preprocess(lengths: Tensor, link_radius: float) -> tuple[Tensor, Tensor]:
    threshold = 2.0 * link_radius
    norm_lengths = Normaliser.apply(lengths)
    squashed = SquasherSTE.apply(norm_lengths, threshold)
    return Normaliser.apply(squashed), norm_lengths


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """NRM morphology optimisation with selectable DOF (5, 6, or 7).

    Identical to nrm.py except for the optional 'dof' parameter:
    if optimization_parameters['dof'] differs from morph's DOF, a fresh
    random morphology of the requested DOF is sampled before optimisation.

    Returns:
        optimized_morphology: final processed morphology.
        csv_path:             path to output/log_<time>.csv.
    """
    n_iter             = int(optimization_parameters.get("num_iterations", 100))
    lr                 = float(optimization_parameters.get("learning_rate", 0.01))
    logging            = bool(optimization_parameters.get("logging", True))
    eval_interval      = int(optimization_parameters.get("eval_interval", 1))
    random_seed        = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses   = float(optimization_parameters.get("percentage_poses", 1))
    ignore_ground      = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles   = bool(optimization_parameters.get("ignore_obstacles", False))
    collision_weight   = float(optimization_parameters.get("collision_weight", 10.0))
    collision_margin   = float(optimization_parameters.get("collision_margin", 0.0))

    device = morph.params.device

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
            print(f"[NRM] DOF mismatch: morph has {morph_dof}-DOF, "
                  f"sampling fresh {dof}-DOF morphology.")
        torch.manual_seed(random_seed)
        sampled = sample_morph(1, dof, analytically_solvable=False, device=device)[0]
        alpha   = sampled[:, 0:1].clone()
        lengths = sampled[:, 1:].clone().requires_grad_(True)
    else:
        alpha   = morph.params[:, 0:1].clone().to(device)
        lengths = morph.params[:, 1:].clone().to(device).requires_grad_(True)

    if logging:
        print(f"[NRM] Starting {dof}-DOF optimisation, "
              f"{n_iter} iterations on device {device}.")
        print(f"[NRM] eval_interval={eval_interval}, "
              f"random_seed={random_seed}, "
              f"number_random_seed={number_random_seed}, "
              f"percentage_poses={percentage_poses}")

    # ── validation context + NRM model ───────────────────────────────────────
    scene = build_optimization_validation_context(
        task=task,
        device=device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    task_vec  = _se3_to_vector(task.goal_poses.to(device))
    model     = _load_model(device)
    optimizer = torch.optim.AdamW([lengths], lr=lr)

    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)
    pose_sampling_generator = torch.Generator(device=device)
    pose_sampling_generator.manual_seed(random_seed)

    if logging:
        print(f"[NRM] Writing CSV log to: {csv_logger.csv_path}")

    try:
        progress_bar = tqdm(range(n_iter), desc=f"NRM {dof}-DOF optimising",
                            dynamic_ncols=True)

        for update_idx in progress_bar:
            optimizer.zero_grad()

            processed_lengths, _ = _preprocess(lengths, link_radius)
            raw_morphology        = torch.cat([alpha, lengths.detach()], dim=1)
            processed_morphology  = torch.cat([alpha, processed_lengths], dim=1)

            bmorph = processed_morphology.unsqueeze(0).expand(task_vec.shape[0], -1, -1)
            logit  = model(bmorph, task_vec)

            loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(
                logit, torch.ones_like(logit)
            )
            prob = torch.sigmoid(logit).mean()

            # Differentiable self-collision penalty using zero-config FK
            n_links = processed_morphology.shape[0]
            zero_theta = torch.zeros(n_links, 1, device=device, dtype=processed_morphology.dtype)
            zero_poses = forward_kinematics(processed_morphology, zero_theta)  # [n_links, 4, 4]
            critical_dist = _collision_critical_distance(
                processed_morphology.unsqueeze(0), zero_poses.unsqueeze(0), link_radius
            )
            col_penalty = F.relu(collision_margin - critical_dist).mean()
            loss = loss + collision_weight * col_penalty

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
                tqdm.write(
                    f"[Iter {update_idx:>4}/{n_iter}] "
                    f"loss={loss.item():.6f}, "
                    f"nrm_prob={prob.item():.6f}, "
                    f"best_se3={best_se3:.6f}"
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

            # Clamp EEF d (last row, column 1) to ≥ 0.
            with torch.no_grad():
                lengths[-1, 1].clamp_(min=0.0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                prob=f"{prob.item():.3f}",
            )

        # ── final evaluation ─────────────────────────────────────────────────
        with torch.no_grad():
            final_processed_lengths, _ = _preprocess(lengths.detach(), link_radius)
            final_raw_morphology       = torch.cat([alpha, lengths.detach()], dim=1)
            final_processed_morphology = torch.cat([alpha, final_processed_lengths], dim=1)

            final_bmorph = final_processed_morphology.unsqueeze(0).expand(
                task_vec.shape[0], -1, -1
            )
            final_logit = model(final_bmorph, task_vec)
            final_loss  = torch.nn.BCEWithLogitsLoss(reduction="mean")(
                final_logit, torch.ones_like(final_logit)
            )
            final_prob = torch.sigmoid(final_logit).mean()

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
            reachability_probability=final_prob,
            raw_morphology=final_raw_morphology,
            processed_morphology=final_processed_morphology,
            validation_data=final_validation_data,
        )

        final_se3_err = final_validation_data["best_se3_dist_mean"].detach().cpu().item()
        print(
            f"[Iter {n_iter:>4}/{n_iter}] "
            f"loss={final_loss.item():.6f}, "
            f"nrm_prob={final_prob.item():.6f}, "
            f"final_se3_err={final_se3_err:.6f}"
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
