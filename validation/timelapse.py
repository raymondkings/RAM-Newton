import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import pyglet
pyglet.options['headless'] = True

import newton
import warp as wp
from pyglet.math import Vec3 as PyVec3

from interface import Morphology, Task
from validation.render import build_scene_builder

# Layout colours
_BG       = "#1a1a2e"  # main background
_PANEL_BG = "#12122a"  # metric panel background
_GRID     = "#2a2a4a"  # grid lines and pane edges

# Foreground / accent colours
_C_LOSS     = "#ff6b6b"  # loss curve
_C_PROB     = "#4ecdc4"  # NRM probability curve (teal)
_C_TICK_2D  = "#888899"  # metric panel tick labels

# Camera tuning — adjust these to frame the scene.
#
# The L-environment scene spans roughly:
#   X: -0.3 → 0.8 m   Y: -0.3 → 0.3 m   Z: 0 → 0.7 m
# Scene centre ≈ (0.25, 0, 0.3).
#
# _CAM_POS  — world-space camera position (metres).
# _CAM_TARGET — world-space point the camera looks at.
#
# To retune: move _CAM_POS further from _CAM_TARGET to zoom out, or change
# the position to change the viewing angle.
_CAM_POS    = PyVec3(0.95, -0.85, 1.0)   # camera position
_CAM_TARGET = (0.10, 0.0, 0.30)          # look-at point (scene centre)


class TimeLapseRecorder:
    """Collects rendered frames during morphology optimization and writes a video."""

    def __init__(self, cfg: dict, task: Task) -> None:
        self.task = task
        self.stride: int = cfg.get("frame_every_n_steps", 5)
        self.fps: int = cfg.get("fps", 8)
        self.dpi: int = cfg.get("dpi", 120)
        self.fmt: str = cfg.get("format", "gif").lower()

        # Frame dimensions (pixels)
        self._scene_w: int = cfg.get("scene_width", 1200)
        self._scene_h: int = cfg.get("scene_height", 900)
        self._metrics_w: int = cfg.get("metrics_width", 750)

        out_dir = Path(cfg.get("output_dir", "timelapse"))
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        self.output_path: Path = out_dir / f"{now.strftime('%Y-%m-%d_%H.%M.%S')}.{self.fmt}"
        self.timestamp_label: str = now.strftime("%d %B %Y  ·  %H:%M:%S")

        self.frames: list[np.ndarray] = []
        self._loss_hist: list[float] = []
        self._prob_hist: list[float] = []

        # Headless OpenGL viewer — created once and reused across frames.
        self._gl_viewer = newton.viewer.ViewerGL(
            headless=True,
            width=self._scene_w,
            height=self._scene_h,
        )

    def add_frame(
        self,
        morph: Morphology,
        iteration: int,
        n_iter: int,
        loss: float,
        prob: float,
    ) -> None:
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
        scene   = self._render_scene_gl(morph)
        metrics = self._render_metrics_mpl(morph, iteration, n_iter, loss, prob)
        return np.concatenate([scene, metrics], axis=1)

    # ------------------------------------------------------------------
    # Scene panel — rendered by the headless Newton/OpenGL viewer
    # ------------------------------------------------------------------

    def _render_scene_gl(self, morph: Morphology) -> np.ndarray:
        """Render robot + environment via ViewerGL and return an [H, W, 3] uint8 array."""
        builder = build_scene_builder(morph, self.task)
        model   = builder.finalize()
        state   = model.state()

        self._gl_viewer.set_model(model)

        # Re-apply camera every frame — set_model may internally touch camera state.
        self._gl_viewer.camera.pos = _CAM_POS
        self._gl_viewer.camera.look_at(_CAM_TARGET)

        self._gl_viewer.begin_frame(0.0)
        self._gl_viewer.log_state(state)
        self._gl_viewer.end_frame()

        return self._gl_viewer.get_frame().numpy()  # [scene_h, scene_w, 3] uint8

    # ------------------------------------------------------------------
    # Metrics panel — rendered by matplotlib
    # ------------------------------------------------------------------

    def _render_metrics_mpl(
        self,
        morph: Morphology,
        iteration: int,
        n_iter: int,
        loss: float,
        prob: float,
    ) -> np.ndarray:
        """Render loss / prob / MDH panels and return an [H, W, 3] uint8 array."""
        w_in = self._metrics_w / self.dpi
        h_in = self._scene_h   / self.dpi

        fig = Figure(figsize=(w_in, h_in), dpi=self.dpi, facecolor=_BG)
        canvas = FigureCanvasAgg(fig)

        fig.suptitle(
            f"NRM Morphology Optimization  ·  {self.timestamp_label}\n"
            f"Iter {iteration + 1} / {n_iter}   |   Loss {loss:.4f}   |   NRM prob {prob:.3f}",
            color="white",
            fontsize=9,
            y=0.98,
        )

        gs = fig.add_gridspec(3, 1, hspace=0.6)
        ax_loss = fig.add_subplot(gs[0])
        ax_prob = fig.add_subplot(gs[1])
        ax_mdh  = fig.add_subplot(gs[2])

        _draw_metric(ax_loss, self._loss_hist, "Loss",     _C_LOSS, ylim=(0.0, None))
        _draw_metric(ax_prob, self._prob_hist, "NRM prob", _C_PROB, ylim=(0.0, 1.0))
        _draw_mdh_table(ax_mdh, morph)

        fig.subplots_adjust(top=0.88, bottom=0.06, left=0.18, right=0.97)
        canvas.draw()

        buf = canvas.buffer_rgba()
        return np.asarray(buf)[..., :3].copy()  # [scene_h, metrics_w, 3] uint8

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def save(self) -> Path:
        self._gl_viewer.close()

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
# MDH parameter table
# ---------------------------------------------------------------------------

def _draw_mdh_table(ax, morph: Morphology) -> None:
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

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(_PANEL_BG)
        cell.set_edgecolor(_GRID)
        cell.set_text_props(color="white")


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

    if history:
        ax.plot(history, color=color, linewidth=1.6)

    ax.set_ylabel(label, color="white", fontsize=8)
    ax.set_xlim(left=0)

    if any(v is not None for v in ylim):
        ax.set_ylim(*ylim)

    ax.tick_params(colors=_C_TICK_2D, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4, linestyle="--")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _save_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio
    imageio.mimsave(str(path), frames, format="GIF", loop=0, duration=1000 // fps)


def _save_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    h, w = frames[0].shape[:2]
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
