import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import mpl_toolkits.mplot3d
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import torch

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses

# Layout colours
_BG       = "#1a1a2e"  # main background
_PANEL_BG = "#12122a"  # metric panel background
_GRID     = "#2a2a4a"  # grid lines and pane edges

# Foreground / accent colours
_C_LINK     = "#4fc3f7"  # robot link segments (light blue)
_C_EEF      = "#ffd700"  # end-effector highlight (gold)
_C_GOAL     = "#ff6b6b"  # goal pose markers (red)
_C_LOSS     = "#ff6b6b"  # loss curve — same red as goals, both represent target reachability
_C_PROB     = "#4ecdc4"  # NRM probability curve (teal)
_C_OBSTACLE = "#8888aa"  # obstacle wireframe (grey-blue)
_C_AXIS     = "#aaaacc"  # 3D axis labels
_C_TICK_3D  = "#777799"  # 3D axis tick labels
_C_TICK_2D  = "#888899"  # metric panel tick labels


class TimeLapseRecorder:
    """Collects rendered frames during morphology optimization and writes a video."""

    def __init__(self, cfg: dict, task: Task) -> None:
        self.task = task
        self.stride: int = cfg.get("frame_every_n_steps", 5)
        self.fps: int = cfg.get("fps", 8)
        self.dpi: int = cfg.get("dpi", 120)
        self.fmt: str = cfg.get("format", "gif").lower()

        out_dir = Path(cfg.get("output_dir", "timelapse"))
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        # File name uses a sortable compact format; display label is human-readable
        self.output_path: Path = out_dir / f"{now.strftime('%Y-%m-%d_%H.%M.%S')}.{self.fmt}"
        self.timestamp_label: str = now.strftime("%d %B %Y  ·  %H:%M:%S")

        self.frames: list[np.ndarray] = []
        self._loss_hist: list[float] = []
        self._prob_hist: list[float] = []

    def add_frame(
        self,
        morph: Morphology,
        iteration: int,
        n_iter: int,
        loss: float,
        prob: float,
    ) -> None:
        # Always record metrics; only render a frame every `stride` steps (and always on the last iter).
        self._loss_hist.append(loss)
        self._prob_hist.append(prob)
        if iteration % self.stride == 0 or iteration == n_iter - 1:
            try:
                frame = self._render_frame(morph, iteration, n_iter, loss, prob)
                self.frames.append(frame)
            except Exception as exc:
                print(f"[timelapse] Frame {iteration} skipped: {exc}")

    def _render_frame(
        self,
        morph: Morphology,
        iteration: int,
        n_iter: int,
        loss: float,
        prob: float,
    ) -> np.ndarray:
        fig = Figure(figsize=(18, 7), dpi=self.dpi, facecolor=_BG)
        canvas = FigureCanvasAgg(fig)

        # Figure-level title: run identity + per-frame metrics
        fig.suptitle(
            f"NRM Morphology Optimization  ·  {self.timestamp_label}\n"
            f"Iter {iteration + 1} / {n_iter}   |   Loss {loss:.4f}   |   NRM prob {prob:.3f}",
            color="white",
            fontsize=11,
            y=0.98,
        )

        # Layout: 3D scene spans full left column; right column has 3 stacked panels
        gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.55, wspace=0.3)
        ax3d    = fig.add_subplot(gs[:, 0], projection="3d")
        ax_loss = fig.add_subplot(gs[0, 1])
        ax_prob = fig.add_subplot(gs[1, 1])
        ax_mdh  = fig.add_subplot(gs[2, 1])

        ax3d.set_facecolor(_BG)

        _draw_scene(ax3d, morph, self.task)
        _draw_metric(ax_loss, self._loss_hist, "Loss", _C_LOSS, ylim=(0.0, None))
        _draw_metric(ax_prob, self._prob_hist, "NRM prob", _C_PROB, ylim=(0.0, 1.0))
        _draw_mdh_table(ax_mdh, morph)

        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.97)
        canvas.draw()

        buf = canvas.buffer_rgba()
        img = np.asarray(buf)[..., :3].copy()
        return img

    def save(self) -> Path:
        if not self.frames:
            print("[timelapse] No frames recorded — skipping save.")
            return self.output_path

        print(f"[timelapse] Encoding {len(self.frames)} frames → {self.output_path}")
        if self.fmt == "gif":
            _save_gif(self.frames, self.output_path, self.fps)
        elif self.fmt in ("mp4", "mov"):
            _save_mp4(self.frames, self.output_path, self.fps)
        else:
            fallback = self.output_path.with_suffix(".gif")
            print(f"[timelapse] Unknown format '{self.fmt}', saving as GIF: {fallback}")
            _save_gif(self.frames, fallback, self.fps)
        print(f"[timelapse] Saved → {self.output_path.resolve()}")
        return self.output_path


# ---------------------------------------------------------------------------
# Scene drawing
# ---------------------------------------------------------------------------

def _draw_scene(ax, morph: Morphology, task: Task) -> None:
    base = task.environment.base_pose.cpu()

    # --- Morphology (robot arm) ---
    # Run FK at rest pose (all joints = 0) to get each link frame's world position.
    # We only care about the origin of each frame, which gives us the joint positions.
    with torch.no_grad():
        link_poses = compute_link_world_poses(morph).cpu()  # [n_links, 4, 4]
    world_poses = (base.unsqueeze(0) @ link_poses)  # transform from robot frame → world frame

    # Collect joint positions: robot base origin + one point per link frame
    pts = [base[:3, 3].numpy()]
    for i in range(world_poses.shape[0]):
        pts.append(world_poses[i, :3, 3].numpy())
    pts = np.array(pts)  # [n_links+1, 3]

    # Draw links as connected line segments with dots at each joint
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "o-", color=_C_LINK,
            linewidth=2.5, markersize=4, zorder=5)
    # Highlight the end-effector (last link frame) in gold
    ax.scatter(*pts[-1], color=_C_EEF, s=70, zorder=6, depthshade=False)

    # --- Task: goal poses ---
    # Show only the translation component of each SE(3) goal as a red dot.
    # Orientation is not visualised here — the arm just needs its tip to reach these points.
    for i in range(task.goal_poses.shape[0]):
        p = task.goal_poses[i, :3, 3].cpu().numpy()
        ax.scatter(*p, color=_C_GOAL, s=22, alpha=0.85, depthshade=False, zorder=4)

    # --- Task: environment obstacles ---
    # Draw each box obstacle as a grey wireframe so the arm geometry is still readable through it.
    for obs in task.environment.obstacles:
        if obs.kind == "box":
            _draw_box_wireframe(ax, obs.center.cpu().numpy(), obs.half_extents.cpu().numpy())

    # Fixed axis limits and view angle so the camera doesn't jump between frames
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.view_init(elev=22, azim=-55)

    ax.set_xlabel("X", color=_C_AXIS, fontsize=8, labelpad=2)
    ax.set_ylabel("Y", color=_C_AXIS, fontsize=8, labelpad=2)
    ax.set_zlabel("Z", color=_C_AXIS, fontsize=8, labelpad=2)
    ax.tick_params(colors=_C_TICK_3D, labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4)


def _draw_box_wireframe(ax, center: np.ndarray, half: np.ndarray) -> None:
    # Build the 8 corners of the box from center ± half_extents, then draw all 12 edges.
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=float)
    corners = center + signs * half
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    segs = [[corners[i], corners[j]] for i, j in edges]
    ax.add_collection3d(Line3DCollection(segs, colors=_C_OBSTACLE, linewidths=0.9, alpha=0.55))


# ---------------------------------------------------------------------------
# MDH parameter table
# ---------------------------------------------------------------------------

def _draw_mdh_table(ax, morph: Morphology) -> None:
    """Render the 7×3 MDH parameter tensor as a plain text table.

    Columns are [α, a, d]. α is frozen throughout optimisation (only a and d
    are updated by AdamW), so it is visually separated by a vertical line.
    """
    data = morph.params.detach().cpu().numpy()  # [n_links, 3]
    n_links = data.shape[0]

    col_labels = ["α  (frozen)", "a", "d"]
    cell_text = [[f"{data[row, col]:.4f}" for col in range(3)] for row in range(n_links)]
    row_labels = [f"link {i}" for i in range(n_links)]

    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style every cell to match the dark theme
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(_PANEL_BG)
        cell.set_edgecolor(_GRID)
        cell.set_text_props(color="white")

    ax.set_title("MDH Parameters", color="white", fontsize=9, pad=6)


# ---------------------------------------------------------------------------
# Metric drawing
# ---------------------------------------------------------------------------

def _draw_metric(
    ax,
    history: list[float],
    label: str,
    color: str,
    ylim: tuple = (None, None),
) -> None:
    ax.set_facecolor(_PANEL_BG)

    # Plot the full history up to the current iteration as a line.
    # history grows by one entry per optimization step, so the x-axis is iteration number.
    if history:
        ax.plot(history, color=color, linewidth=1.6)

    ax.set_ylabel(label, color="white", fontsize=8)

    # Always start the x-axis at 0 so the plot doesn't shift as history grows
    ax.set_xlim(left=0)

    # Apply fixed y-axis bounds when provided — e.g. (0, 1) for NRM prob keeps
    # the scale stable across frames so the curve doesn't appear to jump around
    if any(v is not None for v in ylim):
        ax.set_ylim(*ylim)

    # Style to match the dark theme
    ax.tick_params(colors=_C_TICK_2D, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4, linestyle="--")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio
    # imageio expects duration in milliseconds per frame
    imageio.mimsave(str(path), frames, format="GIF", loop=0, duration=1000 // fps)


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    h, w = frames[0].shape[:2]
    # libx264 requires even dimensions
    w -= w % 2
    h -= h % 2
    frames = [f[:h, :w] for f in frames]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "22",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(b"".join(f.tobytes() for f in frames))
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{stderr.decode()}")
