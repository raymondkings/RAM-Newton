from __future__ import annotations

import math

import torch

from tasks.sampling._pose_common import (
    ALPHA_RANGE_DEGREES,
    _default_device,
    _pose_from_alpha,
)

NUM_POSES = 10


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
