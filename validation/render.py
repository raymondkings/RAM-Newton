import math
import socket
import time
import numpy as np
import torch
import warp as wp
import newton

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses, forward_kinematics
from util.mdh import add_robot_to_builder
from util.self_collision import get_joint_limits
from validation.ground import add_ground_grid_to_viser, make_origin_axes

# PD gains for joint position control during Newton simulation
_KE = 500.0  # position stiffness  [N·m/rad]
_KD = 50.0  # velocity damping    [N·m·s/rad]


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


# Jiyao: new function to visulize direction, blue shows Z-direction
def make_goal_pose_axes(goal_poses, axis_length: float = 0.05):
    """Return (begins, ends, colors) warp arrays for EEF frame axes at each goal pose.

    Draws X/Y/Z axes as red/green/blue line segments of length axis_length.
    goal_poses: [N, 4, 4] SE3 matrices in world frame.
    """
    begins_list, ends_list, colors_list = [], [], []
    axis_colors = [
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    ]
    #                          r                            g                            b
    for i in range(goal_poses.shape[0]):
        pos = goal_poses[i, :3, 3].cpu().tolist()
        R = goal_poses[i, :3, :3].cpu().tolist()
        for j in range(3):
            tip = [pos[k] + R[k][j] * axis_length for k in range(3)]
            begins_list.append(wp.vec3(*pos))
            ends_list.append(wp.vec3(*tip))
            colors_list.append(axis_colors[j])

    return (
        wp.array(begins_list, dtype=wp.vec3),
        wp.array(ends_list, dtype=wp.vec3),
        wp.array(colors_list, dtype=wp.vec3),
    )


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

    axis_colors = [
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    ]
    begins_list, ends_list, colors_list = [], [], []
    for j in range(3):
        tip = [pos[k] + R[k][j] * axis_length for k in range(3)]
        begins_list.append(wp.vec3(*pos))
        ends_list.append(wp.vec3(*tip))
        colors_list.append(axis_colors[j])

    return (
        wp.array(begins_list, dtype=wp.vec3),
        wp.array(ends_list, dtype=wp.vec3),
        wp.array(colors_list, dtype=wp.vec3),
    )


_GHOST_COLOR = (160, 60, 255)  # purple — best IK approximation
_GHOST_OPACITY = 0.45

_GOAL_COLOR_SUCCESS = (50, 180, 50)  # 🟢 green — reached
_GOAL_COLOR_FAILED = (240, 140, 0)  # 🟠 orange — first unreachable goal
_GOAL_COLOR_UNREACHED = (210, 40, 40)  # 🔴 red — never attempted
_GOAL_COLOR_DEFAULT = (190, 190, 190)  # ⚪ light grey — unknown status
_GOAL_FRAME_AXIS_LENGTH = 0.08
_EEF_FRAME_AXIS_LENGTH = 0.08
_POSE_FRAME_LINE_WIDTH = 0.035


def _goal_color(i: int, failed_at_goal: int | None, n_goals: int) -> tuple:
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


def build_scene_builder(morph: Morphology, task: Task, q=None) -> newton.ModelBuilder:
    """Construct a ModelBuilder for the robot and static scene markers."""
    builder = newton.ModelBuilder()

    region = task.reachable_region
    if region is not None:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                p=wp.vec3(
                    (region.x_min + region.x_max) / 2,
                    (region.y_min + region.y_max) / 2,
                    (region.z_min + region.z_max) / 2,
                ),
                q=wp.quat_identity(),
            ),
            hx=(region.x_max - region.x_min) / 2,
            hy=(region.y_max - region.y_min) / 2,
            hz=(region.z_max - region.z_min) / 2,
            as_site=True,
            color=wp.vec3(0.0, 0.8, 0.2),
            label="reachable_region",
        )

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
            else _goal_color(i, failed_at_goal, n_goals)
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
        "🟦 &nbsp;Z-axis"
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
    folder = server.gui.add_folder(folder_label)
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
    builder = build_scene_builder(morph, task, q=q)
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
        for h, (x, y, z, *_) in zip(ghost_handles, spheres):
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
    sim_substeps: int = 4,
    startup_delay: float = 5.0,
    max_joint_speed: float = math.pi,
    curobo_planner=None,
    failed_at_goal: int | None = "unknown",
    best_ik_q=None,
) -> None:
    """Execute a planned joint-space trajectory in Newton physics and stream it to the viewer.

    Each animation frame sets the next position target and advances the Newton
    simulation by `sim_substeps` steps using PD position control, so contact
    forces with obstacles are fully resolved.

    Args:
        path: list of joint-config tensors (n_joints,), expected to already be
            dense (e.g. cuRobo's interpolated plan) for smooth target tracking.
        fps: rendered frames per second — lower values mean slower playback.
        sim_substeps: physics substeps per rendered frame.
        startup_delay: seconds to hold the initial pose before playback starts,
            giving time to open the browser tab.
        max_joint_speed: maximum joint speed in rad/s. Frames with large joint
            displacements are paced slower so speed never exceeds this limit,
            without affecting the physics timestep.
    """
    n_joints = morph.n_links - 1

    builder = build_scene_builder(morph, task)
    model = builder.finalize()
    state = model.state()

    frame_dt = 1.0 / fps

    viewer = _setup_viewer(model, port, share, curobo_planner)
    add_goals_to_viser(viewer._server, task, failed_at_goal)
    ghost_handles = _add_ghost_robot_to_viser(
        viewer._server, curobo_planner, best_ik_q, n_joints
    )
    _add_ghost_toggle(viewer._server, ghost_handles)
    joint_limit_handles, joint_limit_bounds, _ = _add_joint_limit_panel(
        viewer._server, morph, curobo_planner, initial_q=path[0]
    )

    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)
    goal_axes_b, goal_axes_e, goal_axes_c = make_goal_pose_axes(
        task.goal_poses, axis_length=_GOAL_FRAME_AXIS_LENGTH
    )

    speed_slider = viewer._server.gui.add_slider(
        "Playback speed", min=0.0, max=4.0, step=0.05, initial_value=1.0
    )

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
                color=(0, 200, 255),
                position=(float(x), float(y), float(z)),
            )
            robot_sphere_handles.append(h)

    def render_q(q, t: float) -> None:
        arr = q.detach().cpu().numpy().astype(np.float32)[:n_joints]
        state.joint_q.assign(arr)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        if curobo_planner is not None and robot_sphere_handles:
            spheres = curobo_planner.robot_spheres_world(q[:n_joints])
            for h, (x, y, z, *_) in zip(robot_sphere_handles, spheres):
                h.position = (float(x), float(y), float(z))
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

        # Hold the initial pose during startup so the browser can connect
        stop = time.monotonic() + startup_delay
        while time.monotonic() < stop and viewer.is_running():
            render_q(path[0], t)
            t += frame_dt
            _sleep(frame_dt)

        while viewer.is_running():
            for _ in range(hold_frames):
                render_q(path[0], t)
                t += frame_dt
                _sleep(frame_dt)
                if not viewer.is_running():
                    break

            prev_q = path[0]
            for q in path:
                if not viewer.is_running():
                    break
                dq = float((q - prev_q).norm())
                frame_time = max(frame_dt, dq / max_joint_speed)
                render_q(q, t)
                t += frame_time
                _sleep(frame_time)
                prev_q = q

            for _ in range(hold_frames):
                render_q(path[-1], t)
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
