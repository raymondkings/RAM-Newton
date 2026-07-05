from __future__ import annotations

import math

import torch

NUM_POSES = 10
ALPHA_RANGE_DEGREES = (0.0, 180.0)

# Interprets the handwritten START_POSE as the alpha=0 orientation with the
# missing +x tool-axis entry restored in the first row.
START_POSE = torch.tensor(
    [
        [0.0, 0.0, 1.0, 0.20],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def _default_device() -> torch.device:
    try:
        return torch.get_default_device()
    except AttributeError:
        return torch.empty(()).device


def _pose_from_alpha(alpha: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Create poses on the y=0 half-circle arch."""
    alpha = alpha.to(dtype=dtype)
    s = torch.sin(alpha)
    c = torch.cos(alpha)

    poses = torch.eye(4, dtype=dtype, device=alpha.device).repeat(
        alpha.numel(),
        1,
        1,
    )
    poses[:, 0, 0] = s
    poses[:, 0, 1] = 0.0
    poses[:, 0, 2] = c
    poses[:, 1, 0] = 0.0
    poses[:, 1, 1] = -1.0
    poses[:, 1, 2] = 0.0
    poses[:, 2, 0] = c
    poses[:, 2, 1] = 0.0
    poses[:, 2, 2] = -s

    poses[:, 0, 3] = 0.45 - 0.25 * c
    poses[:, 1, 3] = 0.0
    poses[:, 2, 3] = 0.1 + 0.25 * s
    return poses


def task_sampler(
    num_poses: int = NUM_POSES,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return one deterministic y=0 arch trajectory.

    ``num_poses`` is the only sampling knob. It includes the fixed start and
    goal poses, so the trajectory is fully determined by this integer.
    """
    if num_poses < 2:
        raise ValueError("num_poses must be at least 2 to include start and goal.")

    device = torch.device(device) if device is not None else _default_device()
    alpha_start = math.radians(float(ALPHA_RANGE_DEGREES[0]))
    alpha_goal = math.radians(float(ALPHA_RANGE_DEGREES[1]))
    alpha = torch.linspace(
        alpha_start,
        alpha_goal,
        steps=num_poses,
        dtype=dtype,
        device=device,
    )
    return _pose_from_alpha(alpha, dtype=dtype)


def create_task(
    num_poses: int = NUM_POSES,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return task_sampler(
        num_poses=num_poses,
        device=device,
        dtype=dtype,
    )
