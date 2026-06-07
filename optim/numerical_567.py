# numerical_567.py  (GPU-batch, optimised backward)
#
# Three-stage pipeline:
#   forward : _batched_lm_ik  — GPU LM, all N poses in parallel, FD Jacobian
#   IFT     : _IKSolveBatchIFT backward — O(n_joints+2) batched FK calls total
#             (scalar trick: one batched FK + one backward replaces N×6 grads)
#   outer   : torch.optim.LBFGS (strong-Wolfe)

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from interface import Morphology, Task
from task.morphology_sampler import sample_morph
from util.kinematics import forward_kinematics
from util.direct_ik_common import _collision_critical_distance
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import run_optimization_validation

EPS = 1e-4
_PROJECT_ROOT = Path(__file__).parent.parent


# ── Morphology preprocessing ─────────────────────────────────────────────────


class _SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, param, threshold):
        return param * (param.abs() >= threshold).to(param.dtype)

    @staticmethod
    def backward(_ctx, grad):
        return grad, None


class _Normaliser(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param):
        l2 = torch.hypot(param[:, 0:1], param[:, 1:2])
        norm = l2.sum(dim=0, keepdim=True)
        ctx.save_for_backward(param, l2, norm)
        return param / norm

    @staticmethod
    def backward(ctx, grad):
        param, l2, norm = ctx.saved_tensors
        chain = torch.where(
            (param.abs() > EPS).any(dim=1, keepdim=True),
            param / l2,
            torch.zeros_like(param),
        )
        return (grad * norm - chain * (grad * param).sum()) / norm**2


def _preprocess(lengths: Tensor, link_radius: float):
    threshold = 2.0 * link_radius
    nl = _Normaliser.apply(lengths)
    sq = _SquasherSTE.apply(nl, threshold)
    return _Normaliser.apply(sq), nl


# ── DOF selection ─────────────────────────────────────────────────────────────


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


# ── Batched SE3 residual (uses forward_kinematics batch dims, no vmap) ────────

_TIKHONOV = 1e-4


def _res_batch(theta: Tensor, mdh: Tensor, goal_poses: Tensor) -> Tensor:
    """
    theta:      [N, n_joints]
    mdh:        [n_links, 3]   (may require_grad for the IFT scalar trick)
    goal_poses: [N, 4, 4]
    Returns:    [N, 12 + n_joints]  — pos_err(3) + rot_mat_diff(9) + tikhonov(n_joints)
    """
    N = theta.shape[0]
    theta_full = torch.cat([theta, theta.new_zeros(N, 1)], dim=1).unsqueeze(
        -1
    )  # [N, n_links, 1]
    mdh_batch = mdh.unsqueeze(0).expand(N, -1, -1)  # [N, n_links, 3]
    ee = forward_kinematics(mdh_batch, theta_full)[:, -1]  # [N, 4, 4]

    pos_err = ee[:, :3, 3] - goal_poses[:, :3, 3]  # [N, 3]
    rot_err = (ee[:, :3, :3] - goal_poses[:, :3, :3]).reshape(N, 9)  # [N, 9]
    reg = _TIKHONOV * theta  # [N, n_joints]
    return torch.cat([pos_err, rot_err, reg], dim=1)  # [N, 12+n_joints]


# ── GPU-batched Levenberg-Marquardt IK ────────────────────────────────────────


def _batched_lm_ik(
    mdh: Tensor,
    goal_poses: Tensor,
    n_joints: int,
    max_iter: int = 30,
    tol: float = 1e-4,
    lam: float = 1e-2,
    fd_eps: float = 1e-5,
) -> Tensor:
    """GPU LM IK for N poses in parallel. FD Jacobian avoids torch.func overhead."""
    device = mdh.device
    dtype = mdh.dtype
    N = goal_poses.shape[0]
    theta = torch.zeros(N, n_joints, device=device, dtype=dtype)
    I_n = torch.eye(n_joints, device=device, dtype=dtype)

    res_dim = 12 + n_joints
    for _ in range(max_iter):
        f = _res_batch(theta, mdh, goal_poses)  # [N, 12+n_joints]
        if f[:, :12].norm(dim=-1).max() < tol:  # convergence on pos+rot only
            break
        # FD Jacobian: n_joints batched FK calls
        J = torch.empty(N, res_dim, n_joints, device=device, dtype=dtype)
        for j in range(n_joints):
            tp = theta.clone()
            tp[:, j] += fd_eps
            J[:, :, j] = (_res_batch(tp, mdh, goal_poses) - f) / fd_eps
        JtJ = J.mT @ J
        Jtf = (J.mT @ f.unsqueeze(-1)).squeeze(-1)
        delta = torch.linalg.solve(JtJ + lam * I_n, -Jtf)
        theta = theta + delta

    return theta


# ── Custom autograd: batch IK with efficient IFT backward ────────────────────


class _IKSolveBatchIFT(torch.autograd.Function):
    """
    Forward  : GPU-batch LM → θ* [N, n_joints].
    Backward : IFT via scalar trick — total cost ≈ (n_joints + 2) batched FK calls.

    Scalar trick:
        dL/dmdh = -∂/∂mdh [Σ_i  lam_i · F_i(θ*_i, mdh, goal_i)]
        where lam_i = dL/dθ*_i @ pinv(∂F_i/∂θ)

    The weighted sum is differentiated through ONE batched FK (expand trick),
    so autograd accumulates Σ_i lam_i · ∂F_i/∂mdh in a single backward pass.
    """

    @staticmethod
    def forward(ctx, mdh_flat, goal_poses, n_joints, n_links, ik_tol, ik_max_iter):
        mdh = mdh_flat.detach().reshape(n_links, 3)
        with torch.no_grad():
            theta_star = _batched_lm_ik(
                mdh,
                goal_poses.detach(),
                n_joints,
                max_iter=ik_max_iter,
                tol=ik_tol,
            )
        ctx.save_for_backward(mdh_flat.detach(), theta_star, goal_poses.detach())
        ctx.n_joints = n_joints
        ctx.n_links = n_links
        return theta_star  # [N, n_joints]

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: [N, n_joints]
        mdh_flat, theta_star, goal_poses = ctx.saved_tensors
        n_joints = ctx.n_joints
        n_links = ctx.n_links
        N = theta_star.shape[0]
        device = mdh_flat.device

        res_dim = 12 + n_joints

        # ── Step 1: J_theta via FD [N, 12+n_joints, n_joints] ───────────────
        with torch.no_grad():
            mdh_det = mdh_flat.detach().reshape(n_links, 3)
            f_base = _res_batch(theta_star, mdh_det, goal_poses)  # [N, 12+n_joints]

            FD_EPS = 1e-5
            J_theta = torch.empty(
                N, res_dim, n_joints, device=device, dtype=theta_star.dtype
            )
            for j in range(n_joints):
                tp = theta_star.clone()
                tp[:, j] += FD_EPS
                J_theta[:, :, j] = (
                    _res_batch(tp, mdh_det, goal_poses) - f_base
                ) / FD_EPS

        # ── Step 2: lam_i = dL/dθ*_i @ pinv(J_theta_i) ─────────────────────
        J_pinv = torch.linalg.pinv(J_theta.double())  # [N, n_joints, 12+n_joints]
        lam = (grad_output.double().unsqueeze(1) @ J_pinv).squeeze(
            1
        )  # [N, 12+n_joints]

        # ── Step 3: grad_mdh via scalar trick + single batched backward ──────
        # Tikhonov term has zero ∂/∂mdh, so only pos+rot contribute to grad_mdh.
        with torch.enable_grad():
            mdh_d = mdh_flat.detach().requires_grad_(True)
            theta_full = torch.cat(
                [theta_star.detach(), theta_star.detach().new_zeros(N, 1)], dim=1
            ).unsqueeze(-1)  # [N, n_links, 1]
            mdh_batch = mdh_d.reshape(n_links, 3).unsqueeze(0).expand(N, -1, -1)
            ee_batch = forward_kinematics(mdh_batch, theta_full)[:, -1]  # [N, 4, 4]

            pos_errs = ee_batch[:, :3, 3] - goal_poses[:, :3, 3]  # [N, 3]
            rot_errs = (ee_batch[:, :3, :3] - goal_poses[:, :3, :3]).reshape(
                N, 9
            )  # [N, 9]
            reg = _TIKHONOV * theta_star.detach()  # [N, n_joints]
            f_all = torch.cat([pos_errs, rot_errs, reg], dim=1)  # [N, 12+n_joints]

            weighted_sum = (lam.to(f_all.dtype) * f_all).sum()
            grad_mdh_raw = torch.autograd.grad(weighted_sum, mdh_d, create_graph=False)[
                0
            ]

        grad_mdh = (-grad_mdh_raw).to(mdh_flat.dtype)
        return grad_mdh, None, None, None, None, None


# ── Main entry point ──────────────────────────────────────────────────────────


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """
    GPU-batch NumericalIK + IFT + L-BFGS morphology optimisation.

    Per-iteration cost ≈ (n_joints + 2) × 2 batched FK calls per closure call
    (forward LM + backward scalar trick), vs original scipy serial version.

    Key parameters:
        percentage_poses   fraction of task poses used per step (default 0.2)
        ik_max_iter        LM iterations inside IK solver         (default 30)
        lbfgs_max_iter     LBFGS inner line-search steps         (default 10)
    """
    n_iter = int(optimization_parameters.get("num_iterations", 100))
    lr = float(optimization_parameters.get("learning_rate", 0.05))
    logging_ = bool(optimization_parameters.get("logging", True))
    eval_interval = int(optimization_parameters.get("eval_interval", 1))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 0.2))
    ik_tol = float(optimization_parameters.get("ik_tol", 1e-4))
    ik_max_iter = int(optimization_parameters.get("ik_max_iter", 30))
    lbfgs_max_iter = int(optimization_parameters.get("lbfgs_max_iter", 10))
    history_size = int(optimization_parameters.get("lbfgs_history_size", 10))
    collision_weight = float(optimization_parameters.get("collision_weight", 10.0))
    collision_margin = float(optimization_parameters.get("collision_margin", 0.0))

    device = morph.params.device

    # ── DOF selection ────────────────────────────────────────────────────────
    candidate_dofs_raw = optimization_parameters.get("candidate_dofs", None)
    morph_dof = morph.n_links - 1
    dof = (
        _resolve_candidate_dofs(candidate_dofs_raw)[0]
        if candidate_dofs_raw is not None
        else morph_dof
    )

    n_joints = dof
    link_radius = morph.link_radius

    if dof != morph_dof:
        if logging_:
            print(
                f"[NumericalIK] DOF mismatch: morph={morph_dof}, sampling fresh {dof}-DOF."
            )
        torch.manual_seed(random_seed)
        sampled = sample_morph(1, dof, analytically_solvable=False, device=device)[0]
        alpha = sampled[:, 0:1].clone()
        lengths = sampled[:, 1:].clone().requires_grad_(True)
    else:
        alpha = morph.params[:, 0:1].clone().to(device)
        lengths = morph.params[:, 1:].clone().to(device).requires_grad_(True)

    if logging_:
        print(f"[NumericalIK] {dof}-DOF, {n_iter} iters, device={device}")
        print(
            f"[NumericalIK] lr={lr}, lbfgs_max_iter={lbfgs_max_iter}, "
            f"ik_max_iter={ik_max_iter}, ik_tol={ik_tol}, "
            f"percentage_poses={percentage_poses}"
        )

    scene = None

    goal_poses = task.goal_poses.to(device)
    total_poses = goal_poses.shape[0]
    n_sample = (
        max(1, int(round(total_poses * percentage_poses)))
        if percentage_poses <= 1.0
        else min(total_poses, int(round(percentage_poses)))
    )

    optimizer = torch.optim.LBFGS(
        [lengths],
        lr=lr,
        max_iter=lbfgs_max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(random_seed)
    val_generator = torch.Generator(device=device)
    val_generator.manual_seed(random_seed + 1)

    if logging_:
        print(
            f"[NumericalIK] n_sample={n_sample}/{total_poses}, CSV: {csv_logger.csv_path}"
        )

    try:
        progress_bar = tqdm(
            range(n_iter), desc=f"NumericalIK {dof}-DOF", dynamic_ncols=True
        )
        _loss_store = [None]

        for update_idx in progress_bar:
            perm = torch.randperm(
                total_poses, device=device, generator=train_generator
            )[:n_sample]
            sampled_goals = goal_poses[perm].detach()  # [N, 4, 4]
            N_cur = sampled_goals.shape[0]

            def closure():
                optimizer.zero_grad()
                processed_lengths, _ = _preprocess(lengths, link_radius)
                mdh = torch.cat([alpha, processed_lengths], dim=1)
                n_links = mdh.shape[0]

                # Batch IK on GPU
                theta_star = _IKSolveBatchIFT.apply(
                    mdh.reshape(-1),
                    sampled_goals,
                    n_joints,
                    n_links,
                    ik_tol,
                    ik_max_iter,
                )  # [N, n_joints]

                # Batched FK — gradient through both mdh (direct) and theta_star (IFT)
                theta_full = torch.cat(
                    [theta_star, theta_star.new_zeros(N_cur, 1)], dim=1
                ).unsqueeze(-1)
                all_poses = forward_kinematics(
                    mdh.unsqueeze(0).expand(N_cur, -1, -1), theta_full
                )  # [N, n_links, 4, 4]
                ee_poses = all_poses[:, -1]  # [N, 4, 4]

                p_err = (ee_poses[:, :3, 3] - sampled_goals[:, :3, 3]).norm(dim=-1)
                R_rel = ee_poses[:, :3, :3] @ sampled_goals[:, :3, :3].mT
                trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(-3.0, 3.0)
                r_err = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
                se3_loss = (
                    (p_err**2 / 8.0 + r_err**2 / (2.0 * torch.pi**2) + 1e-12)
                    .sqrt()
                    .mean()
                )

                # Differentiable self-collision penalty
                critical_dist = _collision_critical_distance(
                    mdh.unsqueeze(0).expand(N_cur, -1, -1), all_poses, link_radius
                )  # [N, ...]
                col_penalty = F.relu(collision_margin - critical_dist).mean()
                loss = se3_loss + collision_weight * col_penalty

                if not loss.isfinite():
                    loss = torch.tensor(1.0, device=device, requires_grad=True)

                loss.backward()
                _loss_store[0] = loss.detach()
                return loss

            optimizer.step(closure)

            # Clamp EEF d (last row, column 1) to ≥ 0 after each optimizer step.
            # Negative EEF d inverts the capsule structure (barrel extends into workpiece).
            with torch.no_grad():
                lengths[-1, 1].clamp_(min=0.0)
            loss_val = _loss_store[0]

            validation_data = None
            if eval_interval > 0 and update_idx % eval_interval == 0 and logging_:
                with torch.no_grad():
                    pl, _ = _preprocess(lengths.detach(), link_radius)
                    proc = torch.cat([alpha, pl], dim=1)
                validation_data = run_optimization_validation(
                    processed_morphology=proc.detach(),
                    morph=morph,
                    task=task,
                    scene=scene,
                    device=device,
                    percentage_poses=percentage_poses,
                    number_random_seed=number_random_seed,
                    pose_sampling_generator=val_generator,
                )
                best_se3 = validation_data["best_se3_dist_mean"].detach().cpu().item()
                tqdm.write(
                    f"[Iter {update_idx:>4}/{n_iter}] "
                    f"loss={loss_val.item():.6f}, best_se3={best_se3:.6f}"
                )

            with torch.no_grad():
                pl_log, _ = _preprocess(lengths.detach(), link_radius)
                raw_morph = torch.cat([alpha, lengths.detach()], dim=1)
                proc_morph = torch.cat([alpha, pl_log], dim=1)

            csv_logger.log_iteration(
                iteration=update_idx,
                loss=loss_val if loss_val is not None else torch.zeros(1),
                reachability_probability=torch.zeros(1),
                raw_morphology=raw_morph.detach(),
                processed_morphology=proc_morph.detach(),
                validation_data=validation_data,
            )
            if loss_val is not None:
                progress_bar.set_postfix(loss=f"{loss_val.item():.4f}")  # type: ignore[union-attr]

        # ── Final evaluation ─────────────────────────────────────────────────
        with torch.no_grad():
            final_pl, _ = _preprocess(lengths.detach(), link_radius)
            final_raw = torch.cat([alpha, lengths.detach()], dim=1)
            final_proc = torch.cat([alpha, final_pl], dim=1)

        final_val = run_optimization_validation(
            processed_morphology=final_proc,
            morph=morph,
            task=task,
            scene=scene,
            device=device,
            percentage_poses=percentage_poses,
            number_random_seed=number_random_seed,
            pose_sampling_generator=val_generator,
        )
        csv_logger.log_iteration(
            iteration=n_iter,
            loss=torch.zeros(1),
            reachability_probability=torch.zeros(1),
            raw_morphology=final_raw,
            processed_morphology=final_proc,
            validation_data=final_val,
        )
        final_se3 = final_val["best_se3_dist_mean"].detach().cpu().item()
        print(f"[Iter {n_iter:>4}/{n_iter}] final_se3={final_se3:.6f}")

        return (
            Morphology(params=final_proc.detach(), link_radius=link_radius),
            csv_logger.csv_path,
        )

    finally:
        csv_logger.close()
