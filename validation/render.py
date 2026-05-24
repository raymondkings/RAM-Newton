import math
import socket
import time
import numpy as np
import warp as wp
import newton
from scipy.spatial.transform import Rotation

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses
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


def build_scene_builder(morph: Morphology, task: Task, q=None) -> newton.ModelBuilder:
    """Construct a ModelBuilder with obstacles, goal markers, and the robot."""
    builder = newton.ModelBuilder()

    for obs in task.environment.obstacles:
        if obs.kind == "box":
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    p=wp.vec3(*obs.center.cpu().tolist()), q=wp.quat_identity()
                ),
                hx=obs.half_extents[0].item(),
                hy=obs.half_extents[1].item(),
                hz=obs.half_extents[2].item(),
            )

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
            radius=0.03,
            as_site=True,
            color=wp.vec3(1.0, 0.2, 0.2),
        )

    poses = compute_link_world_poses(morph, q=q)
    poses = (task.environment.base_pose.unsqueeze(0) @ poses).cpu()
    add_robot_to_builder(builder, morph, poses, label="robot")

    return builder


def _setup_viewer(
    model: newton.Model, port: int, share: bool, curobo_planner=None
) -> newton.viewer.ViewerViser:
    viewer = newton.viewer.ViewerViser(port=port, share=share, verbose=False)
    viewer.set_model(model)
    add_ground_grid_to_viser(viewer._server, grid_size=4.0, divisions=8)
    viewer._server.scene.add_icosphere(
        "/unit_sphere", radius=1.0, color=(180, 180, 255), opacity=0.08
    )

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
    q=None,
) -> None:
    """Render morphology + task environment in the Newton viewer (static)."""
    builder = build_scene_builder(morph, task, q=q)
    model = builder.finalize()
    state = model.state()

    viewer = _setup_viewer(model, port, share, curobo_planner)
    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
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
    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)

    speed_slider = viewer._server.gui.add_slider(
        "Playback speed", min=0.1, max=4.0, step=0.1, initial_value=1.0
    )

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
        if robot_sphere_handles:
            spheres = curobo_planner.robot_spheres_world(q[:n_joints])
            for h, (x, y, z, *_) in zip(robot_sphere_handles, spheres):
                h.position = (float(x), float(y), float(z))
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
        viewer.end_frame()

    try:
        t = 0.0

        # Hold the initial pose during startup so the browser can connect
        stop = time.monotonic() + startup_delay
        while time.monotonic() < stop and viewer.is_running():
            render_q(path[0], t)
            t += frame_dt
            time.sleep(frame_dt / speed_slider.value)

        while viewer.is_running():
            for _ in range(hold_frames):
                render_q(path[0], t)
                t += frame_dt
                time.sleep(frame_dt / speed_slider.value)
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
                time.sleep(frame_time / speed_slider.value)
                prev_q = q

            for _ in range(hold_frames):
                render_q(path[-1], t)
                t += frame_dt
                time.sleep(frame_dt / speed_slider.value)
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
