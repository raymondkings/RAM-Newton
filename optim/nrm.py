# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm

# Modified work Copyright (c) 2026 Julian Arkenau
# -----------------------------------------------------------------------------
import json
from pathlib import Path

import torch
from tqdm import tqdm
from torch import Tensor

from optim.model import MLP
from interface import Morphology, Task

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


def optimize_morphology(morph: Morphology, task: Task, optimization_parameters: dict, recorder=None) -> Morphology:
    """Optimize morphology link lengths (a, d) for the given task."""
    n_iter = optimization_parameters.get("num_iterations", 100)
    lr = optimization_parameters.get("learning_rate", 0.01)
    logging = optimization_parameters.get("logging", True)

    task_vec = _se3_to_vector(task.goal_poses)

    alpha = morph.params[:, 0:1].clone()
    lengths = morph.params[:, 1:].clone()
    lengths.requires_grad_(True)

    optimizer = torch.optim.AdamW([lengths], lr=lr)
    model = _load_model()
    model.train()  # cuDNN LSTM backward requires training mode

    loss_list: list[float] = []
    prob_list: list[float] = []

    for i in tqdm(range(n_iter), desc="optimizing"):
        optimizer.zero_grad()

        param, _ = _preprocess(lengths, morph.link_radius)
        morph_params = torch.cat([alpha, param], dim=1)
        bmorph = morph_params.unsqueeze(0).expand(task.goal_poses.shape[0], -1, -1)
        logit = model(bmorph, task_vec)

        loss = torch.nn.BCEWithLogitsLoss(reduction="mean")(logit, torch.ones_like(logit))
        loss.backward()
        optimizer.step()

        # Extract scalar metrics => pulls values out of the compute graph
        # prob_val is the mean predicted reachability across all goal poses (0→1)
        loss_val = loss.item()
        prob_val = torch.sigmoid(logit).detach().mean().item()

        if logging:
            loss_list.append(loss_val)
            prob_list.append(prob_val)

        if recorder is not None:
            # Snapshot the current morphology for the timelapse frame.
            # no_grad + detach().clone() so the renderer gets a plain tensor,
            # not one wired into the gradient graph.
            with torch.no_grad():
                snap_param, _ = _preprocess(lengths, morph.link_radius)
                snap_params = torch.cat([alpha, snap_param], dim=1)
            recorder.add_frame(
                Morphology(params=snap_params.detach().clone(), link_radius=morph.link_radius),
                i, n_iter, loss_val, prob_val,
            )

    if logging:
        print(f"\n{'iter':>6}  {'loss':>8}  {'nrm_prob':>8}")
        for i in range(0, len(loss_list), 10):
            print(f"{i:>6}  {loss_list[i]:>8.4f}  {prob_list[i]:>8.3f}")

    with torch.no_grad():
        param, _ = _preprocess(lengths, morph.link_radius)
        final_params = torch.cat([alpha, param], dim=1)

    return Morphology(params=final_params.detach(), link_radius=morph.link_radius)
