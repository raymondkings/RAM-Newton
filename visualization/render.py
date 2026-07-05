import math
import socket
import time

import newton
import numpy as np
import torch
import warp as wp

from core import Morphology, Task
from kinematics.kinematics import (
    compute_link_world_poses,
    forward_kinematics,
)
from kinematics.mdh import add_robot_to_builder
from kinematics.self_collision import get_joint_limits
from tasks.sampling.morphology_sampler import yoshikawa_manipulability
from visualization.critical_distance import build_critical_distance_monitor
from visualization.critical_distance_style import SPHERE_DEFAULT_COLOR
from visualization.ground import add_ground_grid_to_viser, make_origin_axes


def _trajectory_manipulability(curobo_planner, path: list, n_joints: int) -> np.ndarray:
    """Yoshikawa manipulability index at every waypoint of a joint-space path.

    The end-effector Jacobian is read directly off cuRobo's kinematics (see
    ``CuroboPlanner.tool_jacobian``); the index is the product of the singular
    values of J — equal to ``sqrt(det(J Jᵀ))`` for dof ≥ 6, and falling back to
    the singular-value product for dof < 6 where ``det(J Jᵀ)`` is degenerate.

    Returns a float64 array of shape (len(path),).
    """
    qs = torch.stack([p[:n_joints] for p in path])  # [N, dof]
    jac = curobo_planner.tool_jacobian(qs)  # [N, 6, dof]
    manip = yoshikawa_manipulability(jac, soft=True)  # [N]
    return manip.detach().cpu().double().numpy()


def _trajectory_svd_values(curobo_planner, path: list, n_joints: int) -> np.ndarray:
    """Singular values of J at every waypoint. Returns [N, k] float64, k = min(6, dof)."""
    qs = torch.stack([p[:n_joints] for p in path])
    jac = curobo_planner.tool_jacobian(qs)  # [N, 6, dof]
    _, s, _ = torch.linalg.svd(jac)  # [N, min(6, dof)]
    return s.detach().cpu().double().numpy()


def _eef_world_positions(morph: "Morphology", path: list) -> np.ndarray:
    """EEF world position at every waypoint. Returns [N, 3] float32."""
    n_links = morph.n_links
    n_joints = n_links - 1
    N = len(path)
    qs = torch.stack([p[:n_joints] for p in path])
    thetas = torch.zeros(
        N, n_links, 1, device=morph.params.device, dtype=morph.params.dtype
    )
    thetas[:, :n_joints, 0] = qs
    mdh = morph.params.unsqueeze(0).expand(N, -1, -1)
    poses = forward_kinematics(mdh, thetas)  # [N, n_links, 4, 4]
    return poses[:, -1, :3, 3].detach().cpu().float().numpy()


def _densify_path(path: list[torch.Tensor], step_dq: float) -> list[torch.Tensor]:
    """Resample a joint-space path at uniform arc length so frame speed is constant.

    cuRobo returns a velocity-profiled trajectory (bell-shaped: slow near goals,
    fast mid-transit), so rendering one waypoint per frame produces visibly
    uneven speed. We accumulate joint-space arc length and re-sample at uniform
    steps of ``step_dq`` rad — every consecutive pair in the returned list
    differs by exactly ``step_dq`` (the last step may be shorter). The first
    and last waypoints are preserved exactly.
    """
    if len(path) < 2 or step_dq <= 0:
        return list(path)

    qs = torch.stack(list(path))  # [N, dof]
    seg = (qs[1:] - qs[:-1]).norm(dim=-1)  # [N-1]
    s = torch.cat([torch.zeros(1, device=qs.device, dtype=qs.dtype), seg.cumsum(0)])
    total = float(s[-1])
    if total < step_dq:
        return [path[0], path[-1]]

    n_steps = int(math.ceil(total / step_dq))
    s_target = torch.linspace(0.0, total, n_steps + 1, device=qs.device, dtype=qs.dtype)

    # For each target arc length, find the original segment that contains it
    # (right-side bucket so s_target == s[i] picks segment i) and interpolate.
    idx = torch.searchsorted(s, s_target, right=True).clamp_(1, len(s) - 1)
    s0 = s[idx - 1]
    s1 = s[idx]
    alpha = (
        ((s_target - s0) / (s1 - s0).clamp_min(1e-12)).clamp_(0.0, 1.0).unsqueeze(-1)
    )
    q0 = qs[idx - 1]
    q1 = qs[idx]
    out_qs = q0 * (1.0 - alpha) + q1 * alpha
    return [out_qs[i] for i in range(out_qs.shape[0])]


def _resample_uniform_arc(
    pts: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at uniform Cartesian arc length, keeping the same count.

    Returns (new_pts, new_values) with the same shape as the inputs. If the path
    has zero total length, the inputs are returned unchanged.
    """
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(s[-1])
    if total < 1e-9:
        return pts, values
    s_target = np.linspace(0.0, total, len(pts))
    new_pts = np.stack(
        [np.interp(s_target, s, pts[:, d]) for d in range(pts.shape[1])], axis=1
    ).astype(pts.dtype)
    new_values = np.interp(s_target, s, values).astype(values.dtype)
    return new_pts, new_values


def _manip_colormap(values: np.ndarray) -> np.ndarray:
    """Map manipulability values to RGB uint8 colors: red=low, green=high. Returns [N, 3]."""
    v_min, v_max = values.min(), values.max()
    t = (values - v_min) / max(float(v_max - v_min), 1e-9)
    colors = np.zeros((len(values), 3), dtype=np.uint8)
    colors[:, 0] = (255 * (1.0 - t)).astype(np.uint8)
    colors[:, 1] = (255 * t).astype(np.uint8)
    return colors


def add_curobo_scene_to_viser(server, scene) -> None:
    """Draw cuRobo collision geometry as wireframe overlays in the Viser scene.

    cuRobo and the viewer use the same world/base frame.
    """
    if scene is None:
        return

    _COLOR = (255, 200, 0)  # bright yellow

    for c in scene.cuboid or []:
        dims = tuple(float(d) for d in c.dims)
        pos = tuple(float(v) for v in c.pose[:3])
        q = tuple(float(v) for v in c.pose[3:7])
        server.scene.add_box(
            f"/curobo/{c.name}",
            color=_COLOR,
            dimensions=dims,
            wxyz=q,
            position=pos,
        )

    for s in scene.sphere or []:
        server.scene.add_icosphere(
            f"/curobo/{s.name}",
            radius=float(s.radius),
            color=_COLOR,
            position=tuple(float(v) for v in s.pose[:3]),
        )


# X/Y/Z axes are drawn red/green/blue.
_AXIS_COLORS = [
    wp.vec3(1.0, 0.0, 0.0),
    wp.vec3(0.0, 1.0, 0.0),
    wp.vec3(0.0, 0.0, 1.0),
]


def _pose_axis_lines(poses_pos_rot, axis_length: float):
    """Build (begins, ends, colors) warp arrays for an RGB triad at each pose.

    poses_pos_rot: iterable of (pos, R) pairs as plain Python lists, where pos is
    a length-3 position and R is a 3x3 rotation (world frame).
    """
    begins_list, ends_list, colors_list = [], [], []
    for pos, R in poses_pos_rot:
        for j in range(3):
            tip = [pos[k] + R[k][j] * axis_length for k in range(3)]
            begins_list.append(wp.vec3(*pos))
            ends_list.append(wp.vec3(*tip))
            colors_list.append(_AXIS_COLORS[j])

    return (
        wp.array(begins_list, dtype=wp.vec3),
        wp.array(ends_list, dtype=wp.vec3),
        wp.array(colors_list, dtype=wp.vec3),
    )


def make_goal_pose_axes(goal_poses, axis_length: float = 0.05):
    """Return (begins, ends, colors) warp arrays for EEF frame axes at each goal pose.

    Draws X/Y/Z axes as red/green/blue line segments of length axis_length.
    goal_poses: [N, 4, 4] SE3 matrices in world frame.
    """
    pairs = [
        (goal_poses[i, :3, 3].cpu().tolist(), goal_poses[i, :3, :3].cpu().tolist())
        for i in range(goal_poses.shape[0])
    ]
    return _pose_axis_lines(pairs, axis_length)


def make_eef_pose_axes(morph, q_joints: torch.Tensor, axis_length: float = 0.05):
    """Return (begins, ends, colors) warp arrays for an RGB coordinate triad at the EEF.

    X/Y/Z are drawn as red/green/blue line segments, matching the goal-pose triad style.
    """
    n_links = morph.params.shape[0]
    n_joints = n_links - 1
    theta = torch.zeros(
        n_links, 1, device=morph.params.device, dtype=morph.params.dtype
    )
    theta[:n_joints, 0] = q_joints[:n_joints].detach()
    poses = forward_kinematics(morph.params, theta)
    eef_world = poses[-1].cpu()

    pos = eef_world[:3, 3].tolist()
    R = eef_world[:3, :3].tolist()
    return _pose_axis_lines([(pos, R)], axis_length)


_GHOST_COLOR = (160, 60, 255)  # purple — best IK approximation
_GHOST_OPACITY = 0.45

_GOAL_COLOR_SUCCESS = (50, 180, 50)  # 🟢 green — reached
_GOAL_COLOR_FAILED = (240, 140, 0)  # 🟠 orange — first unreachable goal
_GOAL_COLOR_UNREACHED = (210, 40, 40)  # 🔴 red — never attempted
_GOAL_COLOR_DEFAULT = (190, 190, 190)  # ⚪ light grey — unknown status
_GOAL_FRAME_AXIS_LENGTH = 0.08
_EEF_FRAME_AXIS_LENGTH = 0.08


def _goal_color(i: int, failed_at_goal: int | None) -> tuple:
    if failed_at_goal is None:
        return _GOAL_COLOR_SUCCESS
    if i < failed_at_goal:
        return _GOAL_COLOR_SUCCESS
    if i == failed_at_goal:
        return _GOAL_COLOR_FAILED
    return _GOAL_COLOR_UNREACHED


def _add_ghost_robot_to_viser(server, curobo_planner, best_ik_q, n_joints: int) -> list:
    """Add semi-transparent ghost robot at best_ik_q using collision spheres.

    Returns a list of viser IcosphereHandles (empty if no ghost is shown).
    """
    if curobo_planner is None or best_ik_q is None:
        return []
    spheres = curobo_planner.robot_spheres_world(best_ik_q[:n_joints])
    handles = []
    for i, (x, y, z, r) in enumerate(spheres):
        h = server.scene.add_icosphere(
            f"/ghost_robot/sphere_{i}",
            radius=float(r),
            color=_GHOST_COLOR,
            opacity=_GHOST_OPACITY,
            position=(float(x), float(y), float(z)),
            visible=False,
        )
        handles.append(h)
    return handles


def _add_ghost_toggle(server, ghost_handles: list):
    """Add a viser checkbox that shows/hides best_ik ghost robot spheres."""
    if not ghost_handles:
        return None
    toggle = server.gui.add_checkbox("Show best_ik", initial_value=False)

    @toggle.on_update
    def _on_toggle(_event) -> None:
        for h in ghost_handles:
            h.visible = toggle.value

    return toggle


def build_scene_builder(morph: Morphology, q=None) -> newton.ModelBuilder:
    """Construct a ModelBuilder for the robot."""
    builder = newton.ModelBuilder()

    poses = compute_link_world_poses(morph, q=q)
    poses = poses.cpu()
    add_robot_to_builder(builder, morph, poses, label="robot")

    return builder


def add_goals_to_viser(server, task, failed_at_goal) -> None:
    n_goals = task.goal_poses.shape[0]
    for i in range(n_goals):
        pos = tuple(float(v) for v in task.goal_poses[i, :3, 3].cpu().tolist())
        color = (
            _GOAL_COLOR_DEFAULT
            if failed_at_goal == "unknown"
            else _goal_color(i, failed_at_goal)
        )
        server.scene.add_icosphere(
            f"/goals/sphere_{i}",
            radius=0.02,
            color=color,
            position=pos,
        )


def _add_goal_legend(server) -> None:
    server.gui.add_markdown(
        "**Goal status**\n\n"
        "🟢 &nbsp;Reached\n\n"
        "🟠 &nbsp;First failure\n\n"
        "🔴 &nbsp;Not attempted\n\n"
        "⚪ &nbsp;Unknown\n\n"
        "---\n\n"
        "**Coordinate frames**\n\n"
        "🟥 &nbsp;X-axis\n\n"
        "🟩 &nbsp;Y-axis\n\n"
        "🟦 &nbsp;Z-axis\n\n"
        "---"
    )


def _get_joint_limits(morph: Morphology, curobo_planner) -> np.ndarray:
    """Return [n_joints, 2] array of (lower, upper) limits.

    Prefer cuRobo's limits when available (those are what the planner enforces);
    otherwise derive them from the morphology.
    """
    if curobo_planner is not None:
        lims = curobo_planner._planner.kinematics.get_joint_limits().position
        return lims.detach().cpu().numpy().T.astype(np.float32)  # [n_joints, 2]
    jl = (
        get_joint_limits(morph.params).detach().cpu().numpy()
    )  # [n_links, 2] = [range, offset]
    n_joints = morph.n_links - 1
    lower = jl[:n_joints, 1]
    upper = jl[:n_joints, 1] + jl[:n_joints, 0]
    return np.stack([lower, upper], axis=-1).astype(np.float32)


def _add_joint_limit_panel(
    server,
    morph: Morphology,
    curobo_planner,
    initial_q: torch.Tensor | None = None,
    on_change=None,
    folder_label: str = "Joint limits",
) -> tuple[list, np.ndarray, "object"]:
    """Add a viser GUI panel of sliders showing each joint within its limits.

    If `on_change` is provided, sliders are interactive; the callback is invoked
    with the full joint vector (np.ndarray) whenever any slider changes.

    Returns (slider_handles, limits) so callers can update values per frame.
    """
    n_joints = morph.n_links - 1
    limits = _get_joint_limits(morph, curobo_planner)

    if initial_q is not None:
        q0 = initial_q.detach().cpu().numpy().astype(np.float32)[:n_joints]
    else:
        q0 = np.zeros(n_joints, dtype=np.float32)

    interactive = on_change is not None
    handles: list = []
    folder = server.gui.add_folder(folder_label, expand_by_default=False)
    with folder:
        for i in range(n_joints):
            lo, hi = float(limits[i, 0]), float(limits[i, 1])
            # Slider needs lo < hi; pad degenerate ranges so viser accepts them.
            if hi - lo < 1e-4:
                lo, hi = lo - 1e-3, hi + 1e-3
            v = float(np.clip(q0[i], lo, hi))
            h = server.gui.add_slider(
                f"j{i}",
                min=lo,
                max=hi,
                step=(hi - lo) / 200.0,
                initial_value=v,
                disabled=not interactive,
                hint=f"limits: [{lo:.3f}, {hi:.3f}] rad",
            )
            handles.append(h)

    if interactive:

        def _emit(_event) -> None:
            q = np.array([h.value for h in handles], dtype=np.float32)
            on_change(q)

        for h in handles:
            h.on_update(_emit)

    return handles, limits, folder


def _update_joint_limit_panel(
    handles: list, limits: np.ndarray, q: torch.Tensor
) -> None:
    """Update the per-joint sliders to reflect current joint config q."""
    if not handles:
        return
    n = len(handles)
    q_np = q.detach().cpu().numpy().astype(np.float32)[:n]
    for i, h in enumerate(handles):
        lo, hi = float(limits[i, 0]), float(limits[i, 1])
        h.value = float(np.clip(q_np[i], lo, hi))


def _setup_viewer(
    model: newton.Model, port: int, share: bool, curobo_planner=None
) -> newton.viewer.ViewerViser:
    viewer = newton.viewer.ViewerViser(port=port, share=share, verbose=False)
    viewer.set_model(model)
    add_ground_grid_to_viser(viewer._server, grid_size=4.0, divisions=8)
    viewer._server.scene.add_icosphere(
        "/unit_sphere", radius=1.0, color=(180, 180, 255), opacity=0.08
    )
    _add_goal_legend(viewer._server)

    if curobo_planner is not None:
        add_curobo_scene_to_viser(viewer._server, curobo_planner.scene)

    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        host_ip = "localhost"
    print(
        f"Viewer running — local: http://localhost:{port}  remote: http://{host_ip}:{port}"
    )
    if share and viewer._share_url:
        print(f"Public share URL: {viewer._share_url}")

    return viewer


def render_scene(
    morph: Morphology,
    task: Task,
    port: int = 8080,
    share: bool = False,
    curobo_planner=None,
    q=None,
    failed_at_goal: int | None = "unknown",
    best_ik_q=None,
    start_q=None,
) -> None:
    """Render morphology + task environment in the Newton viewer (static)."""
    builder = build_scene_builder(morph, q=q)
    model = builder.finalize()
    state = model.state()

    n_joints = morph.n_links - 1
    if q is not None:
        arr = q.detach().cpu().float().numpy()[:n_joints]
        state.joint_q.assign(arr)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    viewer = _setup_viewer(model, port, share, curobo_planner)
    add_goals_to_viser(viewer._server, task, failed_at_goal)
    ghost_handles = _add_ghost_robot_to_viser(
        viewer._server, curobo_planner, best_ik_q, n_joints
    )
    _add_ghost_toggle(viewer._server, ghost_handles)
    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)
    goal_axes_b, goal_axes_e, goal_axes_c = make_goal_pose_axes(
        task.goal_poses, axis_length=_GOAL_FRAME_AXIS_LENGTH
    )
    eef_q = (
        start_q.to(dtype=morph.params.dtype, device=morph.params.device)
        if start_q is not None and hasattr(start_q, "to")
        else torch.zeros(
            morph.n_links - 1, dtype=morph.params.dtype, device=morph.params.device
        )
    )

    eef_b, eef_e, eef_c = make_eef_pose_axes(
        morph, eef_q, axis_length=_EEF_FRAME_AXIS_LENGTH
    )

    ghost_available = bool(ghost_handles) and curobo_planner is not None

    # Decide what the sliders should target at startup. If a best-IK ghost is
    # available, drive it (most useful when the trajectory halted); otherwise
    # fall back to the builder geometry.
    target = "ghost" if ghost_available else "builder"

    # Initial config for each target (kept independent so switching back to one
    # restores its prior pose).
    if start_q is not None:
        builder_q = (
            start_q.detach().cpu().numpy().astype(np.float32)[:n_joints]
            if hasattr(start_q, "detach")
            else np.asarray(start_q, dtype=np.float32)[:n_joints]
        )
    else:
        builder_q = np.zeros(n_joints, dtype=np.float32)
    ghost_q = (
        best_ik_q.detach().cpu().numpy().astype(np.float32)[:n_joints]
        if ghost_available
        else builder_q.copy()
    )
    state.joint_q.assign(builder_q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    def _apply_to_builder(q: np.ndarray) -> None:
        state.joint_q.assign(q[:n_joints])
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
        viewer.end_frame()

    def _apply_to_ghost(q: np.ndarray) -> None:
        spheres = curobo_planner.robot_spheres_world(torch.from_numpy(q[:n_joints]))
        for h, (x, y, z, *_) in zip(ghost_handles, spheres, strict=False):
            h.position = (float(x), float(y), float(z))

    def _on_joint_change(q: np.ndarray) -> None:
        if target == "ghost" and ghost_available:
            ghost_q[:] = q[:n_joints]
            _apply_to_ghost(q)
        else:
            builder_q[:] = q[:n_joints]
            _apply_to_builder(q)

    handles, limits, _ = _add_joint_limit_panel(
        viewer._server,
        morph,
        curobo_planner,
        initial_q=best_ik_q if target == "ghost" else None,
        on_change=_on_joint_change,
    )

    if ghost_available:
        # Ghost starts as the slider target, so make it visible up-front.
        for h in ghost_handles:
            h.visible = True
        target_picker = viewer._server.gui.add_button_group(
            "Sliders drive", ("best_ik", "Builder geometry")
        )

        @target_picker.on_click
        def _on_target_change(_event) -> None:
            nonlocal target
            new_target = "ghost" if target_picker.value == "best_ik" else "builder"
            if new_target == target:
                return
            target = new_target
            # Auto-show ghost when it becomes the slider target so edits are visible.
            if target == "ghost":
                for h in ghost_handles:
                    h.visible = True
            # Re-sync slider values to the newly selected target's current config.
            q_target = ghost_q if target == "ghost" else builder_q
            for i, h in enumerate(handles):
                lo, hi = float(limits[i, 0]), float(limits[i, 1])
                h.value = float(np.clip(q_target[i], lo, hi))

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
    viewer.log_lines("/goals/frames", goal_axes_b, goal_axes_e, goal_axes_c, width=0.04)
    viewer.log_lines("/eef_frame", eef_b, eef_e, eef_c, width=0.04)
    viewer.end_frame()

    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        viewer.close()


def animate_plan(
    morph: Morphology,
    task: Task,
    path: list,
    port: int = 8080,
    share: bool = False,
    fps: int = 30,
    hold_seconds: float = 2.0,
    loop: bool = True,
    startup_delay: float = 5.0,
    max_joint_speed: float = math.pi / 4,
    curobo_planner=None,
    failed_at_goal: int | None = "unknown",
    best_ik_q=None,
) -> None:
    """Execute a planned joint-space trajectory in Newton physics and stream it to the viewer.

    Args:
        path: list of joint-config tensors (n_joints,), expected to already be
            dense (e.g. cuRobo's interpolated plan) for smooth target tracking.
        fps: rendered frames per second — lower values mean slower playback.
        startup_delay: seconds to hold the initial pose before playback starts,
            giving time to open the browser tab.
        max_joint_speed: joint-space playback speed in rad/s. The cuRobo path
            is resampled at uniform arc length so every frame advances by
            ``max_joint_speed * frame_dt`` rad — i.e. this is the actual joint
            speed during playback (the GUI slider further scales wall time).
    """
    n_joints = morph.n_links - 1

    builder = build_scene_builder(morph)
    model = builder.finalize()
    state = model.state()

    frame_dt = 1.0 / fps
    # Resample the cuRobo path at uniform joint-space arc length so every frame
    # advances by the same dq. This replaces cuRobo's bell-shaped velocity
    # profile with a constant joint-space speed of max_joint_speed, giving
    # visually uniform playback even across long, goal-sparse transits.
    path = _densify_path(path, max_joint_speed * frame_dt)

    viewer = _setup_viewer(model, port, share, curobo_planner)
    add_goals_to_viser(viewer._server, task, failed_at_goal)
    ghost_handles = _add_ghost_robot_to_viser(
        viewer._server, curobo_planner, best_ik_q, n_joints
    )
    _add_ghost_toggle(viewer._server, ghost_handles)

    speed_slider = viewer._server.gui.add_slider(
        "Playback speed", min=0.0, max=4.0, step=0.05, initial_value=1.0
    )
    viewer._server.gui.add_markdown("---")

    joint_limit_handles, joint_limit_bounds, _ = _add_joint_limit_panel(
        viewer._server, morph, curobo_planner, initial_q=path[0]
    )

    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)
    goal_axes_b, goal_axes_e, goal_axes_c = make_goal_pose_axes(
        task.goal_poses, axis_length=_GOAL_FRAME_AXIS_LENGTH
    )

    # Yoshikawa manipulability over the whole trajectory. The Jacobian comes from
    # cuRobo, so the signal is only available when a planner is set up. Values are
    # precomputed in a single batched cuRobo kinematics call.
    manip_values = (
        _trajectory_manipulability(curobo_planner, path, n_joints)
        if curobo_planner is not None
        else None
    )
    manip_readout = None
    manip_plot = None
    if manip_values is not None:
        from viser import uplot

        manip_folder = viewer._server.gui.add_folder("Yoshikawa Manipulability")
        with manip_folder:
            manip_readout = viewer._server.gui.add_markdown("—")
            xs = np.arange(len(manip_values), dtype=np.float64)
            ys_full = np.asarray(manip_values, dtype=np.float64)
            # The curve grows as playback advances: values past the current
            # frame are NaN so they aren't drawn, giving a visual indication of
            # progress through the trajectory.
            ys0 = np.full_like(ys_full, np.nan)
            ys0[0] = ys_full[0]
            manip_plot = viewer._server.gui.add_uplot(
                data=(xs, ys0),
                series=(
                    uplot.Series(label="frame"),
                    uplot.Series(label="manipulability", stroke="#3b82f6", width=2),
                ),
                scales={
                    "x": uplot.Scale(time=False),
                    "y": uplot.Scale(
                        auto=False,
                        min=float(ys_full.min()),
                        max=float(ys_full.max()),
                    ),
                },
                aspect=2.0,
            )

    # Feature 1: EEF trajectory colored green→red by manipulability.
    # Painted once as a static point cloud so it is visible throughout playback.
    if manip_values is not None:
        _eef_pts, _eef_manip = _resample_uniform_arc(
            _eef_world_positions(morph, path), manip_values
        )
        _eef_colors = _manip_colormap(_eef_manip)
        viewer._server.scene.add_point_cloud(
            "/manip_trajectory",
            points=_eef_pts,
            colors=_eef_colors,
            point_size=0.015,
            point_shape="circle",
        )

    # Feature 3: per-singular-value progress bars — one bar per σᵢ of J,
    # normalized to the largest singular value seen across the full trajectory.
    svd_values = (
        _trajectory_svd_values(curobo_planner, path, n_joints)
        if curobo_planner is not None
        else None
    )
    sv_max = float(svd_values.max()) + 1e-9 if svd_values is not None else 1.0
    sv_bars: list = []
    sv_labels: list = []
    if svd_values is not None:
        k = svd_values.shape[1]
        sv_folder = viewer._server.gui.add_folder(
            "Singular values  σ₁ ≥ … ≥ σₖ", expand_by_default=False
        )
        with sv_folder:
            viewer._server.gui.add_markdown(
                "<small>Bars normalized to the global max σ₁. "
                "Bottom bar collapsing to zero signals a singularity.</small>"
            )
            for i in range(k):
                lbl = viewer._server.gui.add_markdown(f"σ{i + 1}")
                bar = viewer._server.gui.add_progress_bar(
                    float(svd_values[0, i] / sv_max)
                )
                sv_labels.append(lbl)
                sv_bars.append(bar)

    def _sleep(dt: float) -> None:
        """Sleep for one frame; spin-wait if playback is paused (speed == 0)."""
        while speed_slider.value == 0.0 and viewer.is_running():
            time.sleep(0.05)
        if viewer.is_running():
            time.sleep(dt / speed_slider.value)

    print(f"({len(path)} frames — starting in {startup_delay:.0f}s)")

    hold_frames = int(hold_seconds * fps)

    # Pre-create one icosphere handle per robot collision sphere so we can
    # cheaply update their positions each frame instead of recreating them.
    robot_sphere_handles = []
    if curobo_planner is not None:
        q0 = path[0]
        spheres0 = curobo_planner.robot_spheres_world(q0[:n_joints])
        for i, (x, y, z, r) in enumerate(spheres0):
            h = viewer._server.scene.add_icosphere(
                f"/curobo/robot/sphere_{i}",
                radius=float(r),
                color=SPHERE_DEFAULT_COLOR,
                position=(float(x), float(y), float(z)),
            )
            robot_sphere_handles.append(h)

    # Per-frame critical self-collision distances
    crit_monitor = None
    if curobo_planner is not None:
        crit_monitor = build_critical_distance_monitor(
            viewer._server, curobo_planner, path, n_joints, robot_sphere_handles
        )

    def render_q(q, t: float, frame_idx: int | None = None) -> None:
        if manip_values is not None and frame_idx is not None:
            value = float(manip_values[frame_idx])
            if manip_readout is not None:
                manip_readout.content = f"{value:.6g}"
            if manip_plot is not None:
                grown = np.full_like(ys_full, np.nan)
                grown[: frame_idx + 1] = ys_full[: frame_idx + 1]
                manip_plot.data = (xs, grown)
            for i, (lbl, bar) in enumerate(zip(sv_labels, sv_bars, strict=False)):
                sv = float(svd_values[frame_idx, i])
                lbl.content = f"σ{i + 1} = {sv:.4g}"
                bar.value = sv / sv_max
        arr = q.detach().cpu().numpy().astype(np.float32)[:n_joints]
        state.joint_q.assign(arr)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        if curobo_planner is not None and robot_sphere_handles:
            spheres = curobo_planner.robot_spheres_world(q[:n_joints])
            for h, (x, y, z, *_) in zip(robot_sphere_handles, spheres, strict=False):
                h.position = (float(x), float(y), float(z))
            if crit_monitor is not None and frame_idx is not None:
                crit_monitor.update(frame_idx, spheres)
        eef_b, eef_e, eef_c = make_eef_pose_axes(
            morph, q, axis_length=_EEF_FRAME_AXIS_LENGTH
        )
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
        viewer.log_lines(
            "/goals/frames", goal_axes_b, goal_axes_e, goal_axes_c, width=0.04
        )
        viewer.log_lines("/eef_frame", eef_b, eef_e, eef_c, width=0.04)
        viewer.end_frame()
        _update_joint_limit_panel(joint_limit_handles, joint_limit_bounds, q)

    try:
        t = 0.0

        last_idx = len(path) - 1

        # Hold the initial pose during startup so the browser can connect
        stop = time.monotonic() + startup_delay
        while time.monotonic() < stop and viewer.is_running():
            render_q(path[0], t, 0)
            t += frame_dt
            _sleep(frame_dt)

        last_idx = len(path) - 1
        while viewer.is_running():
            for _ in range(hold_frames):
                render_q(path[0], t, 0)
                t += frame_dt
                _sleep(frame_dt)
                if not viewer.is_running():
                    break

            prev_q = path[0]
            for frame_idx, q in enumerate(path):
                if not viewer.is_running():
                    break
                dq = float((q - prev_q).norm())
                frame_time = max(frame_dt, dq / max_joint_speed)
                render_q(q, t, frame_idx)
                t += frame_time
                _sleep(frame_time)
                prev_q = q

            for _ in range(hold_frames):
                render_q(path[-1], t, last_idx)
                t += frame_dt
                _sleep(frame_dt)
                if not viewer.is_running():
                    break

            if not loop:
                while viewer.is_running():
                    time.sleep(0.1)
                break
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
