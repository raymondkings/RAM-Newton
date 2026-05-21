import math
import socket
import time
import numpy as np
import warp as wp
import newton
from scipy.spatial.transform import Rotation

import torch
from interface import Morphology, Task
from util.kinematics import compute_link_world_poses, forward_kinematics
from util.mdh import add_robot_to_builder
from validation.ground import add_ground_grid_to_viser, make_origin_axes

# PD gains for joint position control during Newton simulation
_KE = 500.0  # position stiffness  [N·m/rad]
_KD = 50.0  # velocity damping    [N·m·s/rad]


def add_curobo_scene_to_viser(server, scene, base_pose_f32) -> None:
    """Draw cuRobo collision geometry as wireframe overlays in the Viser scene.

    cuRobo stores obstacles in robot-local frame; base_pose_f32 (4×4 world←robot)
    is used to transform them back to world frame before rendering.
    """
    R_base = base_pose_f32[:3, :3].float().cpu().numpy()
    t_base = base_pose_f32[:3, 3].float().cpu().numpy()

    def _world_pos(p_local):
        return R_base @ np.asarray(p_local, dtype=np.float64) + t_base

    def _world_wxyz(wxyz_local):
        # wxyz_local: [qw, qx, qy, qz]
        xyzw = np.array([wxyz_local[1], wxyz_local[2], wxyz_local[3], wxyz_local[0]])
        R_world = R_base @ Rotation.from_quat(xyzw).as_matrix()
        xyzw_w = Rotation.from_matrix(R_world).as_quat()
        return (float(xyzw_w[3]), float(xyzw_w[0]), float(xyzw_w[1]), float(xyzw_w[2]))

    _COLOR = (255, 200, 0)  # bright yellow

    for c in scene.cuboid or []:
        p = _world_pos(c.pose[:3])
        q = _world_wxyz(c.pose[3:7])
        dims = tuple(float(d) for d in c.dims)
        pos = tuple(float(v) for v in p)
        server.scene.add_box(
            f"/curobo/{c.name}",
            color=_COLOR,
            dimensions=dims,
            wxyz=q,
            position=pos,
        )

    for s in scene.sphere or []:
        p = _world_pos(s.pose[:3])
        server.scene.add_icosphere(
            f"/curobo/{s.name}",
            radius=float(s.radius),
            color=_COLOR,
            position=tuple(float(v) for v in p),
        )

#Jiyao: new function to visulize direction, blue shows Z-direction
def make_goal_pose_axes(goal_poses, axis_length: float = 0.05):
    """Return (begins, ends, colors) warp arrays for EEF frame axes at each goal pose.

    Draws X/Y/Z axes as red/green/blue line segments of length axis_length.
    goal_poses: [N, 4, 4] SE3 matrices in world frame.
    """
    begins_list, ends_list, colors_list = [], [], []
    axis_colors = [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)]
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
#################################################################################
#Jiyao: added EEF direction arrows, black for -Z(+z is the assemble direction),## 
#       purple for +X(gripper opening direction)                              ###
#################################################################################
def _make_eef_arrow(
    morph, q_joints: torch.Tensor, base_pose: torch.Tensor,
    col_idx: int, negate: bool, color: wp.vec3, axis_length: float = 0.08,
):
    """Generic EEF axis arrow.  col_idx: 0=X, 1=Y, 2=Z.  negate flips direction."""
    n_links = morph.params.shape[0]
    n_joints = n_links - 1
    theta = torch.zeros(n_links, 1, device=morph.params.device, dtype=morph.params.dtype)
    theta[:n_joints, 0] = q_joints[:n_joints].detach()
    poses = forward_kinematics(morph.params, theta)
    eef_world = (base_pose @ poses[-1]).cpu()

    pos   = np.array(eef_world[:3, 3].tolist(), dtype=np.float64)
    axis  = np.array(eef_world[:3, col_idx].tolist(), dtype=np.float64)
    axis /= np.linalg.norm(axis) + 1e-8
    if negate:
        axis = -axis
    tip = pos + axis * axis_length

    head_len   = axis_length * 0.25
    head_width = axis_length * 0.12
    ref   = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp1 = np.cross(ref, axis);  perp1 /= np.linalg.norm(perp1) + 1e-8
    perp2 = np.cross(axis, perp1); perp2 /= np.linalg.norm(perp2) + 1e-8
    head_base = tip - axis * head_len
    head_pts  = [
        head_base + perp1 * head_width,
        head_base - perp1 * head_width,
        head_base + perp2 * head_width,
        head_base - perp2 * head_width,
    ]

    begins = [wp.vec3(*pos.tolist())] + [wp.vec3(*tip.tolist())] * 4
    ends   = [wp.vec3(*tip.tolist())] + [wp.vec3(*hp.tolist()) for hp in head_pts]
    colors = [color] * 5
    return (
        wp.array(begins, dtype=wp.vec3),
        wp.array(ends,   dtype=wp.vec3),
        wp.array(colors, dtype=wp.vec3),
    )


def make_eef_z_arrow(morph, q_joints: torch.Tensor, base_pose: torch.Tensor, axis_length: float = 0.08):
    """Black arrow along EEF -Z (arm body direction)."""
    return _make_eef_arrow(morph, q_joints, base_pose, col_idx=2, negate=True,
                           color=wp.vec3(0.0, 0.0, 0.0), axis_length=axis_length)


def make_eef_x_arrow(morph, q_joints: torch.Tensor, base_pose: torch.Tensor, axis_length: float = 0.08):
    """Purple arrow along EEF +X direction."""
    return _make_eef_arrow(morph, q_joints, base_pose, col_idx=0, negate=False,
                           color=wp.vec3(0.6, 0.0, 0.9), axis_length=axis_length)


_GHOST_COLOR = (160, 60, 255)  # purple — best IK approximation
_GHOST_OPACITY = 0.45

_GOAL_COLOR_SUCCESS = (50, 180, 50)  # 🟢 green — reached
_GOAL_COLOR_FAILED = (240, 140, 0)  # 🟠 orange — first unreachable goal
_GOAL_COLOR_UNREACHED = (210, 40, 40)  # 🔴 red — never attempted
_GOAL_COLOR_DEFAULT = (190, 190, 190)  # ⚪ light grey — unknown status


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


def _add_ghost_toggle(server, ghost_handles: list) -> None:
    """Add a viser checkbox that shows/hides ghost robot spheres."""
    if not ghost_handles:
        return
    toggle = server.gui.add_checkbox("Show ghost robot (best IK)", initial_value=False)

    @toggle.on_update
    def _on_toggle(_event) -> None:
        for h in ghost_handles:
            h.visible = toggle.value


def build_scene_builder(morph: Morphology, task: Task) -> newton.ModelBuilder:
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

    for i in range(task.goal_poses.shape[0]):
        pos = task.goal_poses[i, :3, 3].cpu()
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(p=wp.vec3(*pos.tolist()), q=wp.quat_identity()),
            radius=0.01,
            as_site=True,
            color=wp.vec3(1.0, 0.2, 0.2),
        )

    poses = compute_link_world_poses(morph)
    poses = (task.environment.base_pose.unsqueeze(0) @ poses).cpu()
    add_robot_to_builder(builder, morph, poses, label="robot")

    return builder


def add_obstacles_to_viser(server, task) -> None:
    for i, obs in enumerate(task.environment.obstacles):
        if obs.kind == "box":
            center = tuple(float(v) for v in obs.center.cpu().tolist())
            dims = tuple(float(obs.half_extents[j].item() * 2) for j in range(3))
            server.scene.add_box(
                f"/obstacles/box_{i}",
                color=(170, 170, 170),
                dimensions=dims,
                position=center,
            )


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
            radius=0.03,
            color=color,
            position=pos,
        )


def _add_goal_legend(server) -> None:
    server.gui.add_markdown(
        "**Goal status**\n\n"
        "🟢 &nbsp;Reached\n\n"
        "🟠 &nbsp;First failure\n\n"
        "🔴 &nbsp;Not attempted\n\n"
        "⚪ &nbsp;Unknown"
    )


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
        add_curobo_scene_to_viser(
            viewer._server, curobo_planner.scene, curobo_planner._base_pose_f32
        )

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
    failed_at_goal: int | None = "unknown",
    best_ik_q=None,
) -> None:
    """Render morphology + task environment in the Newton viewer (static)."""
    builder = build_scene_builder(morph, task)
    model = builder.finalize()
    state = model.state()

    viewer = _setup_viewer(model, port, share, curobo_planner)
    add_obstacles_to_viser(viewer._server, task)
    add_goals_to_viser(viewer._server, task, failed_at_goal)
    n_joints = morph.n_links - 1
    ghost_handles = _add_ghost_robot_to_viser(
        viewer._server, curobo_planner, best_ik_q, n_joints
    )
    _add_ghost_toggle(viewer._server, ghost_handles)
    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)
    goal_axes_b, goal_axes_e, goal_axes_c = make_goal_pose_axes(task.goal_poses)
    zero_q = torch.zeros(morph.n_links - 1, dtype=morph.params.dtype, device=morph.params.device)
    
    eef_zb, eef_ze, eef_zc = make_eef_z_arrow(morph, zero_q, task.environment.base_pose)
    eef_xb, eef_xe, eef_xc = make_eef_x_arrow(morph, zero_q, task.environment.base_pose)

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
    viewer.log_lines("/goal_frames", goal_axes_b, goal_axes_e, goal_axes_c)
    viewer.log_lines("/eef_z", eef_zb, eef_ze, eef_zc)
    viewer.log_lines("/eef_x", eef_xb, eef_xe, eef_xc)
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
        path: list of joint-config tensors (n_joints,), already densified with
            `interpolate_path` for smooth target tracking.
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
    add_obstacles_to_viser(viewer._server, task)
    add_goals_to_viser(viewer._server, task, failed_at_goal)
    ghost_handles = _add_ghost_robot_to_viser(
        viewer._server, curobo_planner, best_ik_q, n_joints
    )
    _add_ghost_toggle(viewer._server, ghost_handles)
    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)
    goal_axes_b, goal_axes_e, goal_axes_c = make_goal_pose_axes(task.goal_poses)

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
        eef_zb, eef_ze, eef_zc = make_eef_z_arrow(morph, q, task.environment.base_pose)
        eef_xb, eef_xe, eef_xc = make_eef_x_arrow(morph, q, task.environment.base_pose)
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
        viewer.log_lines("/goal_frames", goal_axes_b, goal_axes_e, goal_axes_c)
        viewer.log_lines("/eef_z", eef_zb, eef_ze, eef_zc)
        viewer.log_lines("/eef_x", eef_xb, eef_xe, eef_xc)
        viewer.end_frame()

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
