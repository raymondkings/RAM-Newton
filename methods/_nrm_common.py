# -----------------------------------------------------------------------------
# Original work Copyright (c) 2025 Tim Walter
# Source: https://github.com/TimWalter/nrm
#
# Modified work Copyright (c) 2026 Julian Arkenau / Shiyuan Zhang
# -----------------------------------------------------------------------------
# NRM surrogate input pipeline shared by the gradient and candidate-selection
# optimizers: checkpoint loading, SE(3) <-> 9D pose-vector conversion, and the
# normalize -> squash -> normalize morphology-length preprocessing.
#
# All helpers accept an arbitrary leading batch dimension; a single morphology
# [n_links, 2] and batched candidates [N, n_links, 2] go through the same code.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from torch import Tensor

from methods.nrm_model import MLP
from paths import WEIGHTS_DIR

EPS = 1e-4

_CHECKPOINT_PATH = WEIGHTS_DIR / "checkpoint_5-7.pth"


# ------------------------------- SE(3) helpers -------------------------------


def _se3_to_vector(pose: Tensor) -> Tensor:
    """Convert SE(3) pose matrices [..., 4, 4] to 9D NRM pose vectors."""
    rot_6d = pose[..., :3, :2].transpose(-1, -2).reshape(*pose.shape[:-2], 6)
    return torch.cat([pose[..., :3, 3], rot_6d], dim=-1)


def _rotation_6d_to_matrix(rot_6d: Tensor) -> Tensor:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1, eps=EPS)
    a2_orthogonal = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_orthogonal, dim=-1, eps=EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _vector_to_se3(vector: Tensor) -> Tensor:
    """Convert 9D pose vectors [position, rotation_6d] to SE(3) matrices."""
    poses = torch.eye(4, dtype=vector.dtype, device=vector.device).expand(
        *vector.shape[:-1],
        4,
        4,
    )
    poses = poses.clone()
    poses[..., :3, :3] = _rotation_6d_to_matrix(vector[..., 3:9])
    poses[..., :3, 3] = vector[..., 0:3]
    return poses


def _rotation_angle_between(rot_a: Tensor, rot_b: Tensor) -> Tensor:
    rel = rot_a.transpose(-1, -2) @ rot_b
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_angle = 0.5 * (trace - 1.0)
    skew = torch.stack(
        [
            rel[..., 2, 1] - rel[..., 1, 2],
            rel[..., 0, 2] - rel[..., 2, 0],
            rel[..., 1, 0] - rel[..., 0, 1],
        ],
        dim=-1,
    )
    sin_angle = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    return torch.atan2(sin_angle, cos_angle)


# ------------------------------- model loading -------------------------------


def _load_model(device: torch.device) -> MLP:
    """Load the pretrained NRM surrogate with frozen weights."""
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
    """Build raw and processed morphology tensors for (batched) candidates."""
    processed_lengths, _ = _preprocess_lengths(length_candidates, link_radius)
    raw_morphologies = torch.cat([alpha_candidates, length_candidates], dim=-1)
    processed_morphologies = torch.cat([alpha_candidates, processed_lengths], dim=-1)
    return raw_morphologies, processed_morphologies
