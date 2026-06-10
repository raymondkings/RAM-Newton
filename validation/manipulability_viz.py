"""Visualise *which joints become aligned* as the Yoshikawa index → 0.

Streams a kinematic (FK-only) view of a robot over a joint-space trajectory to a
viser scene. A slider scrubs through the waypoints; at every waypoint the scene
shows, on top of the link skeleton:

  * **Joint axes coloured by the singularity's right singular vector** — the
    joints that light up red are the ones whose motions cancel at the
    end-effector, i.e. the joints that have become aligned / redundant.
  * **Highlighted axis pairs** that are geometrically collinear (white tubes
    joining their origins), the classic cause of a revolute singularity.
  * **The translational manipulability ellipsoid** at the end-effector, which
    flattens from a ball into a disk/line as the index collapses; its shortest
    principal axis points along the Cartesian DOF being lost.
  * A live markdown readout of ``w``, the smallest singular value, the top
    contributing joints, and any collinear pairs.

Run ``python -m validation.manipulability_viz`` for a self-contained demo, or
call :func:`visualize_singularities` with your own morphology and path (the same
``list[Tensor]`` of joint configs that :func:`validation.visualize.animate_plan`
consumes).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from interface import Morphology
from util.kinematics import compute_link_world_poses
from util.manipulability import (
    aligned_pairs,
    analyze_singularity,
    translational_ellipsoid,
)
from task.morphology_sampler import geometric_jacobian

# Visual tuning.
_AXIS_HALF_LENGTH = 0.12  # joint-axis arrow half-length [m]
_ELLIPSOID_GAIN = 0.06  # metres of ellipsoid radius per unit singular value
_SKELETON_COLOR = (90, 90, 110)
_PAIR_COLOR = (255, 255, 255)


def _heat(weight: float) -> tuple[int, int, int]:
    """Light grey (idle joint) → saturated red (joint drives the singularity)."""
    w = float(np.clip(weight, 0.0, 1.0))
    r = int(round(210 + (235 - 210) * w))
    g = int(round(210 + (35 - 210) * w))
    b = int(round(215 + (35 - 215) * w))
    return r, g, b


def _matrix_to_wxyz(rot: np.ndarray) -> np.ndarray:
    """SVD axes -> a proper-rotation quaternion in viser's wxyz order."""
    rot = rot.copy()
    if np.linalg.det(rot) < 0:  # mirror -> flip one axis to make it a rotation
        rot[:, -1] *= -1.0
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return np.array([w, x, y, z], dtype=np.float64)


class _FrameCache:
    """Per-waypoint geometry and singularity diagnostics, computed once."""

    def __init__(self, morph: Morphology, path: list[torch.Tensor], length_scale: float):
        self.length_scale = length_scale
        self.origins: list[np.ndarray] = []  # [n_links, 3] frame origins
        self.axes: list[np.ndarray] = []  # [dof, 3] joint-axis directions
        self.weights: list[np.ndarray] = []  # [dof] normalised joint weights
        self.ee: list[np.ndarray] = []  # [3] end-effector position
        self.ell_radii: list[np.ndarray] = []  # [3] ellipsoid semi-axes
        self.ell_rot: list[np.ndarray] = []  # [3, 3] ellipsoid axes (columns)
        self.manip: list[float] = []
        self.min_sv: list[float] = []
        self.pairs: list[list[tuple[int, int]]] = []

        n_joints = morph.n_links - 1
        device = morph.params.device
        for q in path:
            q = q.detach().to(device)[:n_joints]  # planner configs may carry extra entries
            poses = compute_link_world_poses(morph, q)
            jac = geometric_jacobian(poses)  # [6, dof]
            info = analyze_singularity(jac)
            radii, ell_axes = translational_ellipsoid(jac)

            w = info.joint_weights
            w = w / w.max().clamp_min(1e-12)  # normalise for colouring

            self.origins.append(poses[:, :3, 3].cpu().numpy())
            self.axes.append(poses[:-1, :3, 2].cpu().numpy())
            self.weights.append(w.cpu().numpy())
            self.ee.append(poses[-1, :3, 3].cpu().numpy())
            self.ell_radii.append(radii.cpu().numpy())
            self.ell_rot.append(ell_axes.cpu().numpy())
            self.manip.append(info.manipulability)
            self.min_sv.append(info.min_singular_value)
            self.pairs.append(
                aligned_pairs(poses, length_scale=length_scale)
            )


def visualize_singularities(
    morph: Morphology,
    path: list[torch.Tensor],
    *,
    port: int = 8081,
    length_scale: float | None = None,
    autoplay: bool = True,
    fps: int = 20,
) -> None:
    """Open a viser scene that shows joint alignment as ``w`` collapses.

    Args:
        morph: robot morphology (MDH params).
        path: list of joint-config tensors ``(n_joints,)`` — the trajectory to
            scrub; densify with ``interpolate_path`` for a smooth sweep.
        port: viser server port.
        length_scale: characteristic reach [m] for the collinearity test;
            defaults to the sum of |a| and |d| over the links.
        autoplay: advance the slider automatically on a loop until disconnected.
        fps: autoplay frame rate.
    """
    import viser

    if len(path) == 0:
        raise ValueError("path is empty")

    if length_scale is None:
        length_scale = float(
            morph.a.abs().sum() + morph.d.abs().sum()
        ) or 1.0

    cache = _FrameCache(morph, path, length_scale)
    n_frames = len(path)

    server = viser.ViserServer(port=port)
    server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))

    slider = server.gui.add_slider(
        "waypoint", min=0, max=n_frames - 1, step=1, initial_value=0
    )
    play = server.gui.add_checkbox("autoplay", initial_value=autoplay)
    readout = server.gui.add_markdown("")

    def draw(frame: int) -> None:
        origins = cache.origins[frame]
        axes = cache.axes[frame]
        weights = cache.weights[frame]
        ee = cache.ee[frame]

        # Link skeleton: polyline through consecutive frame origins.
        seg = np.stack([origins[:-1], origins[1:]], axis=1)  # [n_seg, 2, 3]
        server.scene.add_line_segments(
            "/robot/skeleton",
            points=seg.astype(np.float32),
            colors=np.array(_SKELETON_COLOR, np.uint8),
            line_width=4.0,
        )

        # Joint axes as arrows, coloured by the v_min weight of each joint.
        starts = origins[:-1] - axes * _AXIS_HALF_LENGTH
        ends = origins[:-1] + axes * _AXIS_HALF_LENGTH
        server.scene.add_arrows(
            "/robot/joint_axes",
            points=np.stack([starts, ends], axis=1).astype(np.float32),
            colors=np.array([_heat(float(w)) for w in weights], np.uint8),
            shaft_radius=0.006,
            head_radius=0.016,
            head_length=0.03,
        )

        # Collinear axis pairs -> white tube between their origins.
        pairs = cache.pairs[frame]
        if pairs:
            pseg = np.stack(
                [[origins[i], origins[j]] for i, j in pairs], axis=0
            )
            server.scene.add_line_segments(
                "/robot/aligned_pairs",
                points=pseg.astype(np.float32),
                colors=np.array(_PAIR_COLOR, np.uint8),
                line_width=8.0,
            )
        else:
            server.scene.add_line_segments(
                "/robot/aligned_pairs",
                points=np.zeros((1, 2, 3), np.float32),
                colors=np.array(_PAIR_COLOR, np.uint8),
                line_width=0.1,
            )

        # Translational manipulability ellipsoid at the end-effector.
        radii = np.maximum(cache.ell_radii[frame] * _ELLIPSOID_GAIN, 1e-4)
        server.scene.add_icosphere(
            "/robot/ellipsoid",
            radius=1.0,
            scale=(float(radii[0]), float(radii[1]), float(radii[2])),
            wxyz=_matrix_to_wxyz(cache.ell_rot[frame]),
            position=tuple(float(v) for v in ee),
            color=(80, 170, 255),
            opacity=0.35,
            flat_shading=False,
        )

        # Live diagnostics.
        order = np.argsort(-weights)
        top = ", ".join(f"j{int(k)} ({weights[k]:.2f})" for k in order[:3])
        pair_txt = (
            ", ".join(f"j{i}–j{j}" for i, j in pairs) if pairs else "none"
        )
        readout.content = (
            f"**waypoint {frame + 1}/{n_frames}**\n\n"
            f"- Yoshikawa `w` = `{cache.manip[frame]:.3e}`\n"
            f"- σ_min = `{cache.min_sv[frame]:.3e}`\n"
            f"- driving joints: {top}\n"
            f"- collinear axes: {pair_txt}"
        )

    @slider.on_update
    def _(_event: object) -> None:
        draw(int(slider.value))

    draw(0)
    print(f"Open http://localhost:{port}  ({n_frames} waypoints)")

    try:
        while True:
            if play.value and n_frames > 1:
                slider.value = (int(slider.value) + 1) % n_frames
                draw(int(slider.value))
            time.sleep(1.0 / max(fps, 1))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


def _demo_morphology() -> Morphology:
    """A planar 3R arm whose end-effector ellipsoid collapses when extended."""
    # MDH rows [alpha, a, d]; last row is the end-effector frame.
    params = torch.tensor(
        [
            [0.0, 0.30, 0.0],
            [0.0, 0.30, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.00, 0.0],
        ],
        dtype=torch.float32,
    )
    return Morphology(params=params)


def _demo_path(n: int = 120) -> list[torch.Tensor]:
    """Sweep from a folded elbow to fully stretched (a boundary singularity)."""
    path = []
    for t in np.linspace(0.0, 1.0, n):
        # Elbow joints straighten to 0 -> arm fully extended, ellipsoid flattens.
        bend = (1.0 - t) * 1.4
        path.append(torch.tensor([0.4 * t, -bend, bend], dtype=torch.float32))
    return path


if __name__ == "__main__":
    visualize_singularities(_demo_morphology(), _demo_path())
