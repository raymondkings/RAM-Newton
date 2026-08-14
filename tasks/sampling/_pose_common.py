"""Shared geometry for the task-pose samplers.

Both pose_sampler.py (static task poses) and trajectory_pose_sampler.py (arch
trajectory) build poses on the same y=0 half-circle arch and share the default
device resolution and the canonical start pose. Keeping that geometry here means
the arch formula lives in exactly one place.
"""

from __future__ import annotations

import torch

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
    """Create base poses on the y=0 half-circle arch from alpha values in radians.

    The rotation block follows the user's derivation:
        [[sin(a),  0, cos(a)],
         [0,      -1, 0     ],
         [cos(a),  0, -sin(a)]]

    Translation follows the corrected formula:
        x = 0.45 - 0.25*cos(alpha)
        z = 0.1 + 0.25*sin(alpha)
    """
    alpha = alpha.to(dtype=dtype)
    s = torch.sin(alpha)
    c = torch.cos(alpha)
    n = alpha.numel()

    poses = torch.eye(4, dtype=dtype, device=alpha.device).repeat(n, 1, 1)
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
