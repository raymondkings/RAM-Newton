import socket
import time
import numpy as np
import warp as wp
import newton

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses
from validation.mdh_to_newton import add_robot_to_builder
from validation.ground import add_ground_grid_to_viser, make_origin_axes

# PD gains for joint position control during Newton simulation
_KE = 500.0   # position stiffness  [N·m/rad]
_KD = 50.0    # velocity damping    [N·m·s/rad]


def _build_scene_builder(morph: Morphology, task: Task) -> newton.ModelBuilder:
    """Construct a ModelBuilder with obstacles, goal markers, and the robot."""
    builder = newton.ModelBuilder()

    for obs in task.environment.obstacles:
        if obs.kind == "box":
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(*obs.center.tolist()), q=wp.quat_identity()),
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
        pos = task.goal_poses[i, :3, 3]
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(p=wp.vec3(*pos.tolist()), q=wp.quat_identity()),
            radius=0.03,
            as_site=True,
            color=wp.vec3(1.0, 0.2, 0.2),
        )

    poses = compute_link_world_poses(morph)
    poses = task.environment.base_pose.unsqueeze(0) @ poses
    add_robot_to_builder(builder, morph, poses, label="robot")

    return builder


def _setup_viewer(model: newton.Model, port: int, share: bool) -> newton.viewer.ViewerViser:
    viewer = newton.viewer.ViewerViser(port=port, share=share, verbose=False)
    viewer.set_model(model)
    add_ground_grid_to_viser(viewer._server, grid_size=4.0, divisions=8)
    viewer._server.scene.add_icosphere("/unit_sphere", radius=1.0, color=(180, 180, 255), opacity=0.08)

    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        host_ip = "localhost"
    print(f"Viewer running — local: http://localhost:{port}  remote: http://{host_ip}:{port}")
    if share and viewer._share_url:
        print(f"Public share URL: {viewer._share_url}")

    return viewer


def render_scene(morph: Morphology, task: Task, port: int = 8080, share: bool = False) -> None:
    """Render morphology + task environment in the Newton viewer (static)."""
    builder = _build_scene_builder(morph, task)
    model = builder.finalize()
    state = model.state()

    viewer = _setup_viewer(model, port, share)
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
    fps: int = 10,
    hold_seconds: float = 2.0,
    loop: bool = True,
    sim_substeps: int = 4,
    startup_delay: float = 5.0,
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
    """
    n_joints = morph.n_links - 1

    builder = _build_scene_builder(morph, task)
    model = builder.finalize()
    state = model.state()

    frame_dt = 1.0 / fps

    viewer = _setup_viewer(model, port, share)
    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)

    print(f"({len(path)} frames — starting in {startup_delay:.0f}s)")

    hold_frames = int(hold_seconds * fps)

    def render_q(q, t: float) -> None:
        arr = q.detach().cpu().numpy().astype(np.float32)[:n_joints]
        state.joint_q.assign(arr)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
        viewer.end_frame()

    try:
        t = 0.0

        # Hold the initial pose during startup so the browser can connect
        deadline = time.monotonic() + startup_delay
        while time.monotonic() < deadline and viewer.is_running():
            render_q(path[0], t)
            t += frame_dt
            time.sleep(frame_dt)

        while viewer.is_running():
            for _ in range(hold_frames):
                render_q(path[0], t)
                t += frame_dt
                time.sleep(frame_dt)
                if not viewer.is_running():
                    break

            for q in path:
                if not viewer.is_running():
                    break
                render_q(q, t)
                t += frame_dt
                time.sleep(frame_dt)

            for _ in range(hold_frames):
                render_q(path[-1], t)
                t += frame_dt
                time.sleep(frame_dt)
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
