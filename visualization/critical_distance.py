"""Self-collision critical-distance (d_crit) visualization.

d_crit is defined as the smallest surface gap between two non-ignored links, measured on
cuRobo's collision spheres (the same model as the planner checks).

Shown as a plotly timeline, a live label, and a 3-D highlight of the critical pair.
"""

import time

import numpy as np

from visualization import critical_distance_style as style

# Severity thresholds (m): gap > SAFE -> green, > WARN -> orange, else red.
# Colors and other styling live in critical_distance_style.
_CLEAR_SAFE = 0.10
_CLEAR_WARN = 0.02


def _short_link_label(name: str) -> str:
    """Abbreviate a collision-link name ("link_3" -> "L3")."""
    if name == "base_link":
        return "base"
    if name.startswith("link_"):
        return "L" + name[len("link_") :]
    return name


def _link_collision_meta(curobo_planner):
    """Helper function to read the robot's collision layout.

    Returns:
        link_labels: short name per link (e.g. "L3"), for plot/label display.
        block_starts: index where each link's spheres begin in the flat array,
            so a sphere range can be collapsed back to its link.
        ignore_mask: (n_links, n_links) bool, True for link pairs whose contact
            is not a real collision (a link with itself, or pairs the planner
            ignores); these are excluded from d_crit.
        n_spheres: total sphere count, to sanity-check the live FK output.
    """
    sphere_dict = curobo_planner._sphere_dict
    link_names = list(sphere_dict.keys())
    counts = np.array([len(sphere_dict[name]) for name in link_names], dtype=np.int64)
    n_links = len(link_names)

    # Start index of each link's block: [0, count0, count0+count1, ...].
    block_starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)

    # Diagonal True = a link never collides with itself; then add the planner's
    # ignored pairs. Names absent from the sphere dict are skipped.
    name_to_idx = {name: i for i, name in enumerate(link_names)}
    ignore_mask = np.eye(n_links, dtype=bool)
    for a, others in curobo_planner._self_collision_ignore.items():
        ia = name_to_idx.get(a)
        if ia is None:
            continue
        for b in others:
            ib = name_to_idx.get(b)
            if ib is not None:
                ignore_mask[ia, ib] = True
                ignore_mask[ib, ia] = True

    link_labels = [_short_link_label(name) for name in link_names]
    return link_labels, block_starts, ignore_mask, int(counts.sum())


def precompute_self_collision_data(
    spheres_per_frame, block_starts, ignore_mask, n_links
):
    """Worst link clearances over a trajectory of (S, 4) world-frame spheres.

    Sphere gap = ``||p_i - p_j|| - r_i - r_j``; link clearance = min over the
    sphere pairs spanning the two links; ignored pairs are forced to ``+inf``.

    Returns:
        link_clearance_matrix: (n_links, n_links) min clearance over all frames.
        per_frame_worst: ``(d_crit, li, lj, si, sj)`` per frame — worst gap with
            its link and sphere pair; ``(+inf, -1, -1, -1, -1)`` if all ignored.
    """
    link_clearance_matrix = np.full((n_links, n_links), np.inf, dtype=np.float64)
    per_frame_worst = []
    if not spheres_per_frame:
        return link_clearance_matrix, per_frame_worst

    # Sphere -> link ids, so one argmin yields both the link and sphere pair.
    n_spheres = spheres_per_frame[0].shape[0]
    link_ids = np.zeros(n_spheres, dtype=np.int64)
    link_ids[block_starts[1:]] = 1
    link_ids = np.cumsum(link_ids)
    sphere_ignore = ignore_mask[link_ids[:, None], link_ids[None, :]]

    for spheres in spheres_per_frame:
        centers = spheres[:, :3].astype(np.float64)
        radii = spheres[:, 3].astype(np.float64)

        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        clearance = dist - radii[:, None] - radii[None, :]

        # Spheres with radius <= 0 are disabled; never let them win the min.
        invalid = radii <= 0.0
        if invalid.any():
            clearance[invalid, :] = np.inf
            clearance[:, invalid] = np.inf
        clearance[sphere_ignore] = np.inf

        # Sphere -> link: min over each contiguous block, rows then columns.
        link_clear = np.minimum.reduceat(
            np.minimum.reduceat(clearance, block_starts, axis=0),
            block_starts,
            axis=1,
        )

        flat = int(np.argmin(clearance))
        si, sj = np.unravel_index(flat, clearance.shape)
        d_crit = float(clearance[si, sj])
        if np.isfinite(d_crit):
            per_frame_worst.append(
                (d_crit, int(link_ids[si]), int(link_ids[sj]), int(si), int(sj))
            )
        else:
            per_frame_worst.append((np.inf, -1, -1, -1, -1))

        np.minimum(link_clearance_matrix, link_clear, out=link_clearance_matrix)

    return link_clearance_matrix, per_frame_worst


def _downsample_min(values: np.ndarray, max_points: int):
    """Downsample keeping each bucket's minimum, so dips always survive.

    Returns (kept_indices, values[kept_indices]); all-NaN buckets keep a NaN.
    """
    n = values.shape[0]
    if n <= max_points:
        return np.arange(n), values
    starts = (np.arange(max_points) * n) // max_points
    ends = np.append(starts[1:], n)
    idx = np.empty(max_points, dtype=np.int64)
    for k, (s, e) in enumerate(zip(starts, ends, strict=False)):
        block = values[s:e]
        idx[k] = s + (int(np.nanargmin(block)) if np.isfinite(block).any() else 0)
    return idx, values[idx]


def build_crit_distance_timeline(per_frame_worst, link_labels):
    """d_crit curve over danger bands, with global-min marker and cursor.

    The cursor must stay the figure's *last* shape (``_set_timeline_cursor``).
    """
    import plotly.graph_objects as go

    d = np.array([w[0] for w in per_frame_worst], dtype=np.float64)
    d[~np.isfinite(d)] = np.nan  # all-ignored frames -> gaps in the curve
    x, y = _downsample_min(d, 600)

    any_finite = bool(np.isfinite(d).any())
    y_lo = min(-0.01, float(np.nanmin(d)) - 0.02) if any_finite else -0.01
    y_hi = max(0.12, float(np.nanmax(d)) + 0.02) if any_finite else 0.12

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": style.CURVE_COLOR, "width": style.CURVE_WIDTH},
            hovertemplate="frame %{x}<br>d_crit %{y:.3f} m<extra></extra>",
        )
    )

    bands = [
        (y_lo, 0.0, style.BAND_PENETRATION),
        (0.0, _CLEAR_WARN, style.BAND_DANGER),
        (_CLEAR_WARN, _CLEAR_SAFE, style.BAND_WARN),
        (_CLEAR_SAFE, y_hi, style.BAND_SAFE),
    ]
    for lo, hi, fill in bands:
        if hi > lo:
            fig.add_shape(
                type="rect",
                xref="paper",
                x0=0,
                x1=1,
                yref="y",
                y0=lo,
                y1=hi,
                fillcolor=fill,
                line={"width": 0},
                layer="below",
            )
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=0.0,
        y1=0.0,
        line={
            "color": style.ZERO_LINE_COLOR,
            "width": style.ZERO_LINE_WIDTH,
            "dash": "dot",
        },
        layer="below",
    )

    subtitle = None
    if any_finite:
        i_min = int(np.nanargmin(d))
        d_min, li, lj = per_frame_worst[i_min][:3]
        value = f"{d_min:+.3f} m".replace("-", "−")
        pair = f"{link_labels[li]}↔{link_labels[lj]}"
        subtitle = f"worst over trajectory: {value} ({pair}) at frame {i_min}"
        fig.add_trace(
            go.Scatter(
                x=[i_min],
                y=[float(d_min)],
                mode="markers",
                marker={
                    "size": style.MARKER_SIZE,
                    "color": style.MARKER_COLOR,
                    "symbol": style.MARKER_SYMBOL,
                },
                hovertemplate=f"min %{{y:.3f}} m ({pair})<extra></extra>",
            )
        )

    # Playback cursor — must stay the LAST shape.
    fig.add_shape(
        type="line",
        xref="x",
        x0=0,
        x1=0,
        yref="paper",
        y0=0,
        y1=1,
        line={"color": style.CURSOR_COLOR, "width": style.CURSOR_WIDTH},
    )

    title = {
        "text": "Critical self-collision distance d_crit",
        "font": {"size": style.TITLE_FONT_SIZE},
    }
    if subtitle is not None:
        title["subtitle"] = {
            "text": subtitle,
            "font": {"size": style.SUBTITLE_FONT_SIZE, "color": style.SUBTITLE_COLOR},
        }
    fig.update_layout(
        title=title,
        xaxis={
            "title": "frame",
            "range": [0, max(len(d) - 1, 1)],
            "fixedrange": True,
        },
        yaxis={"title": "d_crit (m)", "range": [y_lo, y_hi], "fixedrange": True},
        margin=style.MARGIN,
        showlegend=False,
        plot_bgcolor=style.PLOT_BGCOLOR,
        dragmode=False,
    )
    return fig


def _set_timeline_cursor(handle, figure, frame_idx: int) -> None:
    """Move the cursor (the last shape) and push the whole figure to clients."""
    cursor = figure.layout.shapes[-1]
    cursor.x0 = cursor.x1 = frame_idx
    handle.figure = figure


def _format_crit_label(worst, link_labels) -> str:
    """Format one ``per_frame_worst`` entry as the live markdown label."""
    d, li, lj = worst[:3]
    if not np.isfinite(d) or li < 0:
        return "**d_crit = n/a**  (all pairs ignored)"
    dot = (
        style.DOT_SAFE
        if d > _CLEAR_SAFE
        else style.DOT_WARN
        if d > _CLEAR_WARN
        else style.DOT_DANGER
    )
    value = f"{d:+.3f} m".replace("-", "−")  # "+" makes negative gaps obvious
    return f"**d_crit = {value}**  {dot}  [{link_labels[li]} ↔ {link_labels[lj]}]"


def _severity_color(d: float) -> tuple[int, int, int]:
    """Clearance -> RGB, same thresholds as the label and timeline bands."""
    if d > _CLEAR_SAFE:
        return style.SEVERITY_SAFE
    if d > _CLEAR_WARN:
        return style.SEVERITY_WARN
    return style.SEVERITY_DANGER


def _update_crit_highlight(
    worst, spheres, sphere_handles, link_spheres, gap_handle, state
) -> None:
    """Color the critical pair's spheres and draw the gap ruler (severity color).

    The ruler spans surface to surface, so its length is d_crit. ``state``
    holds the applied ``(li, lj, color)``; colors are only pushed on change.
    """
    d, li, lj, si, sj = worst
    color = _severity_color(d) if li >= 0 else None
    prev_li, prev_lj, prev_color = state[0]
    if (li, lj, color) != (prev_li, prev_lj, prev_color):
        if prev_li >= 0:
            for k in link_spheres[prev_li] + link_spheres[prev_lj]:
                sphere_handles[k].color = style.SPHERE_DEFAULT_COLOR
        if li >= 0:
            for k in link_spheres[li] + link_spheres[lj]:
                sphere_handles[k].color = color
        state[0] = (li, lj, color)

    if li < 0:
        if gap_handle.visible:
            gap_handle.visible = False
        return
    pi = np.asarray(spheres[si][:3], dtype=np.float64)
    pj = np.asarray(spheres[sj][:3], dtype=np.float64)
    delta = pj - pi
    norm = float(np.linalg.norm(delta))
    if norm > 1e-9:
        u = delta / norm
        a = pi + u * float(spheres[si][3])
        b = pj - u * float(spheres[sj][3])
    else:
        a, b = pi, pj
    gap_handle.points = np.array([[a, b]], dtype=np.float32)
    gap_handle.colors = np.tile(np.array(color, dtype=np.uint8), (1, 2, 1))
    gap_handle.visible = True


class CriticalDistanceMonitor:
    """Owns the d_crit GUI elements; ``update`` refreshes them each frame."""

    def __init__(
        self,
        server,
        per_frame_worst,
        link_labels,
        block_starts,
        n_spheres,
        sphere_handles,
    ):
        self._per_frame_worst = per_frame_worst
        self._link_labels = link_labels
        self._sphere_handles = sphere_handles
        self._cursor_state = [-1, 0.0]  # [last pushed frame_idx, last push time]
        self._highlight_state = [(-1, -1, None)]  # (li, lj, color) applied

        self._timeline_fig = build_crit_distance_timeline(per_frame_worst, link_labels)
        self._timeline_handle = server.gui.add_plotly(
            self._timeline_fig, aspect=style.TIMELINE_ASPECT
        )
        self._label_handle = server.gui.add_markdown(
            _format_crit_label(per_frame_worst[0], link_labels)
        )
        # Sphere indices per link, for recoloring the critical pair.
        block_ends = np.append(block_starts[1:], n_spheres)
        self._link_spheres = [
            list(range(int(s), int(e)))
            for s, e in zip(block_starts, block_ends, strict=False)
        ]
        self._gap_handle = server.scene.add_line_segments(
            "/curobo/crit_gap",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=np.zeros((1, 2, 3), dtype=np.uint8),
            line_width=style.GAP_LINE_WIDTH,
            visible=False,
        )
        self._toggle = server.gui.add_checkbox(
            "Highlight critical pair (3D)", initial_value=True
        )

    def update(self, frame_idx: int, spheres) -> None:
        """Refresh label, 3-D highlight and (rate-limited) timeline cursor."""
        worst = self._per_frame_worst[frame_idx]
        _update_crit_highlight(
            worst
            if self._toggle.value
            else (np.inf, -1, -1, -1, -1),  # to turn off visualization
            spheres,
            self._sphere_handles,
            self._link_spheres,
            self._gap_handle,
            self._highlight_state,
        )
        self._label_handle.content = _format_crit_label(worst, self._link_labels)
        now = time.monotonic()
        if frame_idx != self._cursor_state[0] and now - self._cursor_state[1] >= 0.15:
            self._cursor_state[0] = frame_idx
            self._cursor_state[1] = now
            _set_timeline_cursor(self._timeline_handle, self._timeline_fig, frame_idx)


def build_critical_distance_monitor(
    server, curobo_planner, path, n_joints, sphere_handles
):
    """Precompute d_crit for ``path`` and build the monitor. Never raises.

    Returns ``None`` (with a console note) if plotly is missing or the sphere
    model is inconsistent — diagnostics must never break playback.
    """
    try:
        link_labels, block_starts, ignore_mask, n_spheres_exp = _link_collision_meta(
            curobo_planner
        )
        spheres_per_frame = [
            curobo_planner.robot_spheres_world(q[:n_joints]) for q in path
        ]
        got = spheres_per_frame[0].shape[0] if spheres_per_frame else 0
        if got != n_spheres_exp or len(link_labels) < 2:
            print(
                "[viewer] Skipping self-collision timeline "
                f"(spheres {got} vs expected {n_spheres_exp}, "
                f"{len(link_labels)} links)."
            )
            return None
        print("Precomputing self-collision clearances ...")
        _, per_frame_worst = precompute_self_collision_data(
            spheres_per_frame, block_starts, ignore_mask, len(link_labels)
        )
        return CriticalDistanceMonitor(
            server,
            per_frame_worst,
            link_labels,
            block_starts,
            n_spheres_exp,
            sphere_handles,
        )
    except ImportError as exc:
        print(
            f"[viewer] Self-collision timeline needs plotly ({exc}); "
            "install it with `uv pip install plotly`."
        )
    except Exception as exc:  # noqa: BLE001 - timeline is diagnostics only
        print(f"[viewer] Self-collision timeline unavailable: {exc}")
    return None
