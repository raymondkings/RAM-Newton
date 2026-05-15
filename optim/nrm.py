# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm

# Modified work Copyright (c) 2026 Julian Arkenau
# -----------------------------------------------------------------------------
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from torch import Tensor

from optim.model import MLP
from optim.plot import plot_link_lengths, plot_link_lengths_trajectory, plot_ik_fk_trajectory
from interface import Morphology, Task
from util.kinematics import IK, FK, build_robot_dict, build_scene, pose_errors, se3_distance

EPS = 1e-4


def _se3_to_vector(pose: Tensor) -> Tensor:
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


def _load_model() -> MLP:
    device = torch.get_default_device()
    metadata = json.loads((_WEIGHTS_DIR / "metadata.json").read_text())
    model = MLP(**metadata["hyperparameter"])
    model.load_state_dict(torch.load(_WEIGHTS_DIR / "checkpoint.pth", map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


class SquasherSTE(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, param, threshold):
        mask = (param.abs() >= threshold).float()
        return param * mask

    @staticmethod
    def backward(ctx, grad_output):
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


def optimize_morphology(morph: Morphology, task: Task, optimization_parameters: dict) -> Morphology:
    """Optimize morphology link lengths (a, d) for the given task."""
    n_iter = optimization_parameters.get("num_iterations", 100)
    lr = optimization_parameters.get("learning_rate", 0.01)
    logging = optimization_parameters.get("logging", True)
    eval_interval = optimization_parameters.get("eval_interval", 1)
    ignore_ground = optimization_parameters.get("ignore_ground", False)
    ignore_obstacles = optimization_parameters.get("ignore_obstacles", False)

    device = morph.params.device
    print(f"[Info] Starting morphology optimization with {n_iter} iterations on device {device} ...")
    base_pose_inv = torch.linalg.inv(task.environment.base_pose.to(torch.float32))
    scene = build_scene(task, base_pose_inv, ignore_ground=ignore_ground, ignore_obstacles=ignore_obstacles)

    task_vec = _se3_to_vector(task.goal_poses)

    alpha = morph.params[:, 0:1].clone()
    lengths = morph.params[:, 1:].clone()
    lengths.requires_grad_(True)

    optimizer = torch.optim.AdamW([lengths], lr=lr)
    model = _load_model()
    model.train()  # cuDNN LSTM backward requires training mode

    loss_list: list[float] = []
    prob_list: list[float] = []
    lengths_history: list[Tensor] = []
    pos_err_list: list[float] = []
    rot_err_list: list[float] = []
    se3_dist_list: list[float] = []

    for i in tqdm(range(n_iter), desc="optimizing"):
        optimizer.zero_grad()

        param, _ = _preprocess(lengths, morph.link_radius)
        morph_params = torch.cat([alpha, param], dim=1)
        bmorph = morph_params.unsqueeze(0).expand(task.goal_poses.shape[0], -1, -1)
        logit = model(bmorph, task_vec)

        loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(logit, torch.ones_like(logit))
        loss.backward()
        optimizer.step()

        if i % eval_interval == 0 and logging:
            with torch.no_grad():
                param_eval, _ = _preprocess(lengths.detach(), morph.link_radius)
                eval_params = torch.cat([alpha, param_eval], dim=1)
                eval_morph = Morphology(params=eval_params.clone(), link_radius=morph.link_radius)
            robot_dict, urdf_path = build_robot_dict(eval_morph)
            ik_solver = IK(robot_dict, scene, num_seeds=32, max_batch_size=task.goal_poses.shape[0])
            fk_solver = FK(robot_dict)
            try:
                joints, _ = ik_solver.solve(task.goal_poses, base_pose_inv, device)
                reached = fk_solver.compute(joints, base_pose_inv, device)
            finally:
                os.unlink(urdf_path)
            pos_err, rot_err = pose_errors(reached, task.goal_poses)
            pos_err, rot_err = pos_err.cpu(), rot_err.cpu()
            dists = se3_distance(pos_err, rot_err)
            pos_err_list.append(pos_err.mean().item())
            rot_err_list.append(rot_err.mean().item())
            se3_dist_list.append(dists.mean().item())
            loss_list.append(loss.item())
            prob_list.append(torch.sigmoid(logit).mean().item())
            lengths_history.append(lengths.detach().clone().cpu())

    if logging:
        eval_steps = list(range(0, n_iter, eval_interval))[:len(pos_err_list)]
        print(f"\n{'step':>6}  {'pos_err':>8}  {'rot_err':>8}  {'se3_dist':>8}")
        stride = max(1, len(pos_err_list) // 10)
        for idx in range(0, len(pos_err_list), stride):
            print(f"{eval_steps[idx]:>6}  {pos_err_list[idx]:>8.4f}  {rot_err_list[idx]:>8.4f}  {se3_dist_list[idx]:>8.4f}")
        plot_ik_fk_trajectory(
            steps=eval_steps,
            pos_errs=pos_err_list,
            rot_errs=rot_err_list,
            se3_dists=se3_dist_list,
            )

        print(f"\n{'iter':>6}  {'loss':>8}  {'nrm_prob':>8}")
        for i in range(0, len(loss_list), 10):
            print(f"{i:>6}  {loss_list[i]:>8.4f}  {prob_list[i]:>8.3f}")
        plot_link_lengths(torch.stack(lengths_history), title="Raw link lengths during optimization (before normalisation/squashing)")
        plot_link_lengths_trajectory(torch.stack(lengths_history), title="Raw link lengths during optimization (before normalisation/squashing)")

    with torch.no_grad():
        param, _ = _preprocess(lengths, morph.link_radius)
        final_params = torch.cat([alpha, param], dim=1)

    return Morphology(params=final_params.detach(), link_radius=morph.link_radius)
