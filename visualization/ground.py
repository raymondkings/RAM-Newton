"""Ground plane helpers: visual grid and origin frame."""

import numpy as np
import warp as wp


def make_origin_axes(axis_length: float = 0.1):
    """Return (begins, ends, colors) warp arrays for an XYZ origin frame.
    Args:
        axis_length: Length of each axis arrow in metres.
    Returns:
        Tuple of three wp.array(dtype=wp.vec3) ready for viewer.log_lines().
    """
    begins = wp.array([wp.vec3(0.0, 0.0, 0.0)] * 3, dtype=wp.vec3)
    ends = wp.array(
        [
            wp.vec3(axis_length, 0.0, 0.0),
            wp.vec3(0.0, axis_length, 0.0),
            wp.vec3(0.0, 0.0, axis_length),
        ],
        dtype=wp.vec3,
    )
    colors = wp.array(
        [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],
        dtype=wp.vec3,
    )
    return begins, ends, colors


def add_ground_grid_to_viser(
    server,
    name: str = "/ground_grid",
    grid_size: float = 10.0,
    divisions: int = 10,
    color: tuple[int, int, int] = (180, 180, 180),
    line_width: float = 1.0,
) -> None:
    """Add a visual wireframe grid at z=0 to a Viser scene.
    Args:
        server:     viser.ViserServer (access via viewer._server for Newton's ViewerViser).
        name:       Scene node name.
        grid_size:  Total size of the grid in X and Y (metres).
        divisions:  Number of cells along each axis.
        color:      RGB colour of the grid lines, 0–255.
        line_width: Line width in pixels.
    """
    half = grid_size / 2.0
    step = grid_size / divisions

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(divisions + 1):
        y = -half + i * step
        segments.append((np.array([-half, y, 0.0]), np.array([half, y, 0.0])))
    for i in range(divisions + 1):
        x = -half + i * step
        segments.append((np.array([x, -half, 0.0]), np.array([x, half, 0.0])))

    points = np.array(segments, dtype=np.float32)  # (N, 2, 3)
    colors = np.tile(
        np.array(color, dtype=np.uint8), (len(segments), 2, 1)
    )  # (N, 2, 3)

    server.scene.add_line_segments(
        name=name,
        points=points,
        colors=colors,
        line_width=line_width,
    )
