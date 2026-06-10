"""Singularity diagnostics: *which* joints align as the Yoshikawa index → 0.

The Yoshikawa manipulability index is ``w = prod(singular_values(J))``, so it
collapses to zero exactly when the smallest singular value of the geometric
Jacobian does. The scalar ``w`` tells you *that* you are near a singularity;
the SVD tells you *why*:

  - the **right** singular vector of the smallest singular value is a
    joint-space direction whose joint-velocity combination produces (almost)
    no end-effector motion — its large components are the joints that have
    become redundant / aligned;
  - the matching **left** singular vector is the Cartesian DOF being lost.

A complementary, purely geometric check looks at the joint axes directly: a
classic revolute singularity is two axes becoming collinear (parallel *and*
sharing a line).

Pure torch — no rendering dependencies, so it is cheap to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SingularityAnalysis:
    """SVD decomposition of a single ``[6, dof]`` geometric Jacobian."""

    manipulability: float  # Yoshikawa index, prod of the singular values
    singular_values: Tensor  # [k] descending, k = min(6, dof)
    min_singular_value: float  # the one that drives w -> 0
    joint_weights: Tensor  # [dof] |right singular vec of the smallest sv|, in [0, 1]
    lost_task_direction: Tensor  # [6] left singular vec of the smallest sv (twist)
    u: Tensor  # [6, k] left singular vectors (task-space)
    vt: Tensor  # [k, dof] right singular vectors (joint-space, as rows)


def analyze_singularity(jacobian: Tensor) -> SingularityAnalysis:
    """SVD diagnostics for one ``[6, dof]`` geometric Jacobian.

    The Jacobian is the output of :func:`task.morphology_sampler.geometric_jacobian`
    for a single configuration. Computation is done in float64 for numerical
    stability near the singularity, where the smallest singular value is tiny.
    """
    if jacobian.ndim != 2 or jacobian.shape[0] != 6:
        raise ValueError(
            f"expected a single [6, dof] Jacobian, got {tuple(jacobian.shape)}"
        )

    u, s, vt = torch.linalg.svd(jacobian.to(torch.float64), full_matrices=False)
    idx = int(torch.argmin(s))
    return SingularityAnalysis(
        manipulability=float(torch.prod(s)),
        singular_values=s,
        min_singular_value=float(s[idx]),
        joint_weights=vt[idx].abs(),
        lost_task_direction=u[:, idx],
        u=u,
        vt=vt,
    )


def joint_axis_alignment(
    poses: Tensor, *, length_scale: float = 1.0
) -> tuple[Tensor, Tensor]:
    """Pairwise collinearity of revolute joint axes from FK frames.

    Two revolute axes form a singular pair when they are both parallel and
    share a common line. We measure each separately so a caller can threshold
    them independently.

    Args:
        poses: ``[n_links, 4, 4]`` base-to-link transforms (last frame is the
            end-effector); the local z-axis of each frame is the joint axis.
        length_scale: characteristic robot size (metres) used to make the
            line-distance metric dimensionless; the robot reach is a good value.

    Returns:
        ``(parallel, offset)`` each ``[dof, dof]``:
          - ``parallel[i, j] = |sin angle|`` between axes i and j (0 = parallel).
          - ``offset[i, j]`` = perpendicular distance between the two axis lines,
            normalised by ``length_scale`` (0 = the axes share a line).
        Both are zero on the diagonal.
    """
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError(f"expected [n_links, 4, 4] poses, got {tuple(poses.shape)}")

    z = poses[:-1, :3, 2]  # [dof, 3] axis directions (unit by construction)
    p = poses[:-1, :3, 3]  # [dof, 3] a point on each axis
    dof = z.shape[0]

    zi = z[:, None, :].expand(dof, dof, 3)
    zj = z[None, :, :].expand(dof, dof, 3)
    parallel = torch.linalg.cross(zi, zj, dim=-1).norm(dim=-1)

    # Perpendicular component of (p_j - p_i) w.r.t. axis i: |(p_j - p_i) x z_i|.
    # Given parallel axes, this is the distance between the two axis lines.
    dp = (p[None, :, :] - p[:, None, :]).expand(dof, dof, 3)
    offset = torch.linalg.cross(dp, zi, dim=-1).norm(dim=-1) / max(length_scale, 1e-9)
    return parallel, offset


def aligned_pairs(
    poses: Tensor,
    *,
    length_scale: float = 1.0,
    parallel_tol: float = 0.1,
    offset_tol: float = 0.05,
) -> list[tuple[int, int]]:
    """Joint index pairs whose axes are (nearly) collinear.

    ``parallel_tol`` is on ``|sin angle|`` (0.1 ≈ 5.7°); ``offset_tol`` is on the
    line distance normalised by ``length_scale``.
    """
    parallel, offset = joint_axis_alignment(poses, length_scale=length_scale)
    dof = parallel.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(dof):
        for j in range(i + 1, dof):
            if parallel[i, j] < parallel_tol and offset[i, j] < offset_tol:
                pairs.append((i, j))
    return pairs


def translational_ellipsoid(jacobian: Tensor) -> tuple[Tensor, Tensor]:
    """Principal axes and radii of the translational manipulability ellipsoid.

    The ellipsoid lives in task space; its axes are the left singular vectors
    of the translational Jacobian ``J_v = J[:3]`` and its radii are the matching
    singular values. As ``w → 0`` for a translational singularity, the smallest
    radius collapses, flattening the ellipsoid into a disk or line.

    Returns:
        ``(radii, axes)`` with ``radii`` ``[3]`` (descending, zero-padded if
        ``dof < 3``) and ``axes`` ``[3, 3]`` whose columns are unit directions.
    """
    jv = jacobian[:3].to(torch.float64)
    u, s, _ = torch.linalg.svd(jv, full_matrices=True)  # u: [3, 3]
    radii = torch.zeros(3, dtype=torch.float64)
    radii[: s.shape[0]] = s
    return radii, u
