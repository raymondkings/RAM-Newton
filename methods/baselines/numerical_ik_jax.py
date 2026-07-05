# numerical_ik_jax.py
#
# Morphology optimiser following the NAGE baseline
# (paper_archive/rq3_design_optimisation/baseline.py in nrm-master).
#
# Architecture:
#   inner : optx.LevenbergMarquardt with optx.ImplicitAdjoint() (IFT gradient)
#           Tikhonov regularisation (1e-4 * joints) keeps Jacobian full-rank
#   outer : optax.AdamW + zero_nans + clip_by_global_norm
#   collision: PyTorch self-collision penalty at IK-solution configurations

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from core import Morphology, Task
from tasks.sampling.morphology_sampler import sample_morph
from methods.baselines.direct_ik_common import (
    _collision_critical_distance,
    _preprocess_lengths,
)
from kinematics.kinematics import forward_kinematics
from logutils.csv_logger import OptimizationCSVLogger
from logutils.timing import OptimizationTimer
from validation.optimization_validation import run_optimization_validation

from paths import PROJECT_ROOT as _PROJECT_ROOT

# ── JAX / optimistix / optax imports ─────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optimistix as optx
    import optax

    # ── Forward kinematics (Python for-loop, same as NAGE utils.py) ──────────

    def _jax_transform(alpha, a, d, theta):
        ca, sa = jnp.cos(alpha), jnp.sin(alpha)
        ct, st = jnp.cos(theta), jnp.sin(theta)
        zero, one = jnp.zeros_like(alpha), jnp.ones_like(alpha)
        row1 = jnp.concatenate([ct, -st, zero, a], axis=-1)
        row2 = jnp.concatenate([st * ca, ct * ca, -sa, -d * sa], axis=-1)
        row3 = jnp.concatenate([st * sa, ct * sa, ca, d * ca], axis=-1)
        row4 = jnp.concatenate([zero, zero, zero, one], axis=-1)
        return jnp.stack([row1, row2, row3, row4], axis=-2)

    def _fk_jax(mdh, theta):
        """mdh: [..., n_links, 3], theta: [..., n_links, 1] → poses [..., n_links, 4, 4]"""
        transforms = _jax_transform(mdh[..., 0:1], mdh[..., 1:2], mdh[..., 2:3], theta)
        n = mdh.shape[-2]
        pose_shape = mdh.shape[:-2] + (4, 4)
        pose = jnp.broadcast_to(jnp.eye(4), pose_shape)
        poses = []
        for i in range(n):
            pose = pose @ transforms[..., i, :, :]
            poses.append(pose)
        return jnp.stack(poses, axis=-3)  # [..., n_links, 4, 4]

    # ── Preprocessing (same custom_vjp as before) ─────────────────────────────
    import lineax as lx

    @jax.custom_vjp
    def _squash_ste(param, threshold):
        return param * (jnp.abs(param) >= threshold).astype(param.dtype)

    def _squash_ste_fwd(param, threshold):
        return _squash_ste(param, threshold), None

    def _squash_ste_bwd(_, g):
        return g, None

    _squash_ste.defvjp(_squash_ste_fwd, _squash_ste_bwd)

    # Normaliser with explicit zero-guard backward (mirrors NAGE normaliser_bwd)
    @jax.custom_vjp
    def _normalise(param):
        l2 = jnp.hypot(param[:, 0:1], param[:, 1:2])
        norm = jnp.sum(l2, axis=0, keepdims=True)
        return param / jnp.maximum(norm, 1e-12)

    def _normalise_fwd(param):
        l2 = jnp.hypot(param[:, 0:1], param[:, 1:2])
        norm = jnp.sum(l2, axis=0, keepdims=True)
        safe_norm = jnp.maximum(norm, 1e-12)
        return param / safe_norm, (param, l2, safe_norm)

    def _normalise_bwd(res, g):
        param, l2, safe_norm = res
        safe_l2 = jnp.maximum(l2, 1e-12)
        EPS = 1e-6
        chain = jnp.where(
            jnp.any(jnp.abs(param) > EPS, axis=1, keepdims=True),
            param / safe_l2,
            jnp.zeros_like(param),
        )
        grad = (g * safe_norm - chain * jnp.sum(g * param)) / safe_norm**2
        return (grad,)

    _normalise.defvjp(_normalise_fwd, _normalise_bwd)

    def _preprocess_jax(lengths, link_radius):
        threshold = 2.0 * link_radius
        nl = _normalise(lengths)
        sq = _squash_ste(nl, threshold)
        return _normalise(sq)

    # ── IK residual with Tikhonov regularisation (prevents singular Jacobian) ─

    def _ik_residual(joints, args):
        """Residual for optx.least_squares.  joints: [n_links, 1]"""
        morph, target_pose = args
        reached = _fk_jax(morph, joints)[-1]  # [4, 4]
        t_err = reached[:3, 3] - target_pose[:3, 3]
        r_err = (reached[:3, :3] - target_pose[:3, :3]).flatten()
        reg = joints.flatten() * 1e-4  # Tikhonov: keeps J full-rank
        return jnp.concatenate([t_err, r_err, reg])

    # ── Single IK solve with IFT gradient via ImplicitAdjoint ────────────────

    def _solve_ik(morph, target_pose, init_joints):
        """morph: [n_links, 3], target_pose: [4, 4], init_joints: [n_links, 1]"""
        solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-4)
        sol = optx.least_squares(
            fn=_ik_residual,
            solver=solver,
            y0=init_joints,
            args=(morph, target_pose),
            max_steps=50,
            # ImplicitAdjoint with well_posed=False: Tikhonov reg makes the
            # residual overdetermined ([19, 7]), not square — need least-squares solve
            adjoint=optx.ImplicitAdjoint(
                linear_solver=lx.AutoLinearSolver(well_posed=False)
            ),
            throw=False,
        )
        return sol.value  # [n_links, 1]

    # vmap over (target_pose, init_joints); morph is shared
    _vmap_ik = jax.vmap(_solve_ik, in_axes=(None, 0, 0))

    # ── SE3 distance (from NAGE utils.py) ─────────────────────────────────────

    def _jax_distance(x1, x2):
        t_err_sq = jnp.sum((x1[:3, 3] - x2[:3, 3]) ** 2)
        r_err_mat = x1[:3, :3].T @ x2[:3, :3]
        trace = r_err_mat[0, 0] + r_err_mat[1, 1] + r_err_mat[2, 2]
        cos_a = jnp.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
        rot_err = jnp.arccos(cos_a)
        return jnp.sqrt(t_err_sq / 8.0 + rot_err**2 / (2.0 * jnp.pi**2) + 1e-12)

    # ── Loss function (IK + SE3, over all task poses) ─────────────────────────

    def _loss_fn(
        lengths, alpha, task_poses, init_joints, link_radius, zero_link_weight
    ):
        """
        lengths:         [n_links, 2]
        alpha:           [n_links, 1]
        task_poses:      [N, 4, 4]
        init_joints:     [N, n_links, 1]  random each call
        zero_link_weight: penalty weight for links with both a and d near zero
        """
        processed = _preprocess_jax(lengths, link_radius)
        morph = jnp.concatenate([alpha, processed], axis=1)  # [n_links, 3]

        # IK for every pose with its own random init_joints
        optimal_joints = _vmap_ik(morph, task_poses, init_joints)  # [N, n_links, 1]

        # FK at IK solutions
        bmorph = jnp.broadcast_to(morph, (task_poses.shape[0], *morph.shape))
        reached = jax.vmap(_fk_jax)(bmorph, optimal_joints)  # [N, n_links, 4, 4]
        ee_poses = reached[:, -1, :, :]  # [N, 4, 4]

        dists = jax.vmap(_jax_distance)(ee_poses, task_poses)  # [N]

        # Penalize links where both a and d are zero (degenerate pure-rotation links).
        # processed[:,0]=a, processed[:,1]=d; sum of squares → 0 only when both are 0.
        link_norm_sq = processed[:, 0] ** 2 + processed[:, 1] ** 2  # [n_links]
        zero_link_penalty = jnp.mean(jnp.exp(-link_norm_sq / (link_radius**2)))

        return jnp.mean(dists) + zero_link_weight * zero_link_penalty, (
            morph,
            optimal_joints,
        )

    _JAX_AVAILABLE = True

except ImportError as _e:
    _JAX_AVAILABLE = False
    _JAX_IMPORT_ERROR = str(_e)


# ── DOF resolution ────────────────────────────────────────────────────────────


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


# ── Main entry point ──────────────────────────────────────────────────────────


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path, list[float]]:
    """NAGE baseline morphology optimiser (JAX + optax AdamW + IFT gradient)."""
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX / optimistix / optax not installed.\n"
            "  pip install 'jax[cpu]' optimistix equinox optax\n"
            f"  (original error: {_JAX_IMPORT_ERROR})"
        )

    n_iter = int(optimization_parameters.get("num_iterations", 100))
    logging_ = bool(optimization_parameters.get("logging", True))
    eval_interval = int(optimization_parameters.get("eval_interval", 10))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1.0))
    learning_rate = float(optimization_parameters.get("learning_rate", 0.01))
    collision_weight = float(optimization_parameters.get("collision_weight", 10.0))
    collision_margin = float(optimization_parameters.get("collision_margin", 0.0))
    ik_weight = float(optimization_parameters.get("ik_weight", 1.0))
    zero_link_weight = float(optimization_parameters.get("zero_link_weight", 1.0))

    device = morph.params.device
    timer = OptimizationTimer(device, sync_jax=True)
    timer.start()

    candidate_dofs_raw = optimization_parameters.get("candidate_dofs", None)
    morph_dof = morph.n_links - 1
    dof = (
        _resolve_candidate_dofs(candidate_dofs_raw)[0]
        if candidate_dofs_raw
        else morph_dof
    )

    link_radius = morph.link_radius
    n_links = dof + 1

    if dof != morph_dof:
        if logging_:
            print(
                f"[NumericalIK-JAX] DOF mismatch: morph={morph_dof}, sampling {dof}-DOF."
            )
        torch.manual_seed(random_seed)
        sampled = sample_morph(1, dof, analytically_solvable=False, device=device)[0]
        alpha_torch = sampled[:, 0:1].clone()
        lengths_torch = sampled[:, 1:].clone()
    else:
        alpha_torch = morph.params[:, 0:1].clone().to(device)
        lengths_torch = morph.params[:, 1:].clone().to(device)

    alpha_jnp = jnp.array(alpha_torch.cpu().numpy())  # [n_links, 1]
    lengths_jnp = jnp.array(lengths_torch.cpu().numpy())  # [n_links, 2]

    if logging_:
        print(f"[NumericalIK-JAX] {dof}-DOF, {n_iter} iters, lr={learning_rate}")
        print(f"[NumericalIK-JAX] JAX backend: {jax.default_backend()}")

    scene = None

    goal_poses_torch = task.goal_poses.to(device)
    total_poses = goal_poses_torch.shape[0]
    n_sample = (
        max(1, int(round(total_poses * percentage_poses)))
        if percentage_poses <= 1.0
        else min(total_poses, int(round(percentage_poses)))
    )
    goal_poses_all_jnp = jnp.array(goal_poses_torch.cpu().numpy())  # [N, 4, 4]

    # JIT-compile loss + grad
    _loss_and_grad = eqx.filter_jit(jax.value_and_grad(_loss_fn, has_aux=True))

    # optax: zero_nans → clip_by_global_norm → adamw  (same as NAGE baseline)
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate),
    )
    opt_state = optimizer.init(lengths_jnp)

    csv_logger = OptimizationCSVLogger(root_dir=_PROJECT_ROOT)
    val_generator = torch.Generator(device=device)
    val_generator.manual_seed(random_seed + 1)
    rng = np.random.default_rng(random_seed)

    if logging_:
        print(f"[NumericalIK-JAX] CSV: {csv_logger.csv_path}")
        print(
            f"[NumericalIK-JAX] n_poses={n_sample}/{total_poses}, "
            f"collision_weight={collision_weight}"
        )

    # ── PyTorch collision penalty helper ──────────────────────────────────────

    def _collision_penalty_and_grad(lengths_np, theta_stars_np):
        lengths_pt = torch.tensor(
            lengths_np.reshape(n_links, 2),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        processed_pt, _ = _preprocess_lengths(lengths_pt, link_radius)
        mdh_pt = torch.cat([alpha_torch.to(device).float(), processed_pt], dim=1)

        if theta_stars_np is not None and len(theta_stars_np) > 0:
            theta_pt = torch.tensor(theta_stars_np, dtype=torch.float32, device=device)
            theta_pt = torch.nan_to_num(theta_pt, nan=0.0)
            N = theta_pt.shape[0]
            # theta_stars: [N, n_links, 1] → append fixed EE joint
            # theta_stars from JAX has shape [N, n_links, 1]
            theta_full = theta_pt.reshape(N, n_links, 1)
            mdh_batch = mdh_pt.unsqueeze(0).expand(N, -1, -1)
            all_poses = forward_kinematics(mdh_batch, theta_full)
            crit_dist = _collision_critical_distance(mdh_batch, all_poses, link_radius)
        else:
            zero_theta = torch.zeros(n_links, 1, device=device, dtype=torch.float32)
            zero_poses = forward_kinematics(mdh_pt, zero_theta)
            crit_dist = _collision_critical_distance(
                mdh_pt.unsqueeze(0), zero_poses.unsqueeze(0), link_radius
            )

        col_penalty = F.relu(collision_margin - crit_dist).mean()
        loss_col = collision_weight * col_penalty
        loss_col.backward()
        grad_col = lengths_pt.grad.detach().cpu().numpy().flatten().astype(np.float64)
        return float(loss_col.detach()), grad_col

    # ── Optimisation loop (Python for-loop, same as NAGE) ────────────────────

    try:
        progress_bar = tqdm(
            range(n_iter), desc=f"NumericalIK-JAX {dof}-DOF", dynamic_ncols=True
        )

        for update_idx in progress_bar:
            # Random pose subset
            idx = rng.choice(total_poses, size=n_sample, replace=False)
            poses_batch = goal_poses_all_jnp[idx]  # [N, 4, 4]

            # Random initial joints (key = step index, same as NAGE)
            key = jax.random.PRNGKey(update_idx)
            init_joints = jax.random.uniform(
                key,
                shape=(n_sample, n_links, 1),
                minval=-jnp.pi,
                maxval=jnp.pi,
            )

            # IK loss + IFT gradient
            (ik_loss, (morph_jax, theta_stars)), ik_grads = _loss_and_grad(
                lengths_jnp,
                alpha_jnp,
                poses_batch,
                init_joints,
                link_radius,
                zero_link_weight,
            )

            # Collision penalty (PyTorch)
            theta_stars_np = np.array(theta_stars)  # [N, n_links, 1]
            col_val, col_grad = _collision_penalty_and_grad(
                np.array(lengths_jnp), theta_stars_np
            )

            # Combine gradients: IFT (JAX) + collision (PyTorch)
            ik_grads_np = np.array(ik_grads)
            total_grads = jnp.array(
                ik_weight * ik_grads_np + col_grad.reshape(n_links, 2)
            )
            total_loss = ik_weight * float(ik_loss) + col_val

            # optax step (zero_nans + clip + adamw)
            updates, opt_state = optimizer.update(total_grads, opt_state, lengths_jnp)
            lengths_jnp = optax.apply_updates(lengths_jnp, updates)
            lengths_jnp = lengths_jnp.at[-1, 1].set(
                jnp.maximum(lengths_jnp[-1, 1], 0.0)
            )

            progress_bar.set_postfix(
                ik=f"{float(ik_loss):.4f}",
                col=f"{col_val:.3f}",
            )

            # Periodic validation
            validation_data = None
            if eval_interval > 0 and update_idx % eval_interval == 0 and logging_:
                proc_np = np.array(
                    jnp.concatenate(
                        [alpha_jnp, _preprocess_jax(lengths_jnp, link_radius)], axis=1
                    )
                )
                raw_np = np.array(jnp.concatenate([alpha_jnp, lengths_jnp], axis=1))
                proc_t = torch.tensor(proc_np, device=device)
                raw_t = torch.tensor(raw_np, device=device)
                with timer.validation():
                    validation_data = run_optimization_validation(
                        processed_morphology=proc_t.detach(),
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
                    f"ik={float(ik_loss):.4f}, col={col_val:.4f}, "
                    f"best_se3={best_se3:.4f}"
                )
                csv_logger.log_iteration(
                    iteration=update_idx,
                    loss=torch.tensor(total_loss),
                    reachability_probability=torch.zeros(1),
                    raw_morphology=raw_t.detach(),
                    processed_morphology=proc_t.detach(),
                    validation_data=validation_data,
                )

        # ── Final evaluation ─────────────────────────────────────────────────
        final_proc_np = np.array(
            jnp.concatenate(
                [alpha_jnp, _preprocess_jax(lengths_jnp, link_radius)], axis=1
            )
        )
        final_raw_np = np.array(jnp.concatenate([alpha_jnp, lengths_jnp], axis=1))
        final_proc_t = torch.tensor(final_proc_np, device=device)
        final_raw_t = torch.tensor(final_raw_np, device=device)

        with timer.validation():
            final_val = run_optimization_validation(
                processed_morphology=final_proc_t,
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
            loss=torch.tensor(total_loss),
            reachability_probability=torch.zeros(1),
            raw_morphology=final_raw_t,
            processed_morphology=final_proc_t,
            validation_data=final_val,
        )
        csv_logger.close()

        final_se3 = final_val["best_se3_dist_mean"].detach().cpu().item()
        print(f"[NumericalIK-JAX] Done. final_se3={final_se3:.6f}")

        return (
            Morphology(params=final_proc_t.detach(), link_radius=link_radius),
            csv_logger.csv_path,
            timer.result(),
        )

    except Exception:
        csv_logger.close()
        raise
