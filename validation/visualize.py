import time
import numpy as np
import torch
import warp as wp
import newton
from newton import JointTargetMode

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses
from util.mdh import add_robot_to_builder
from validation.ground import add_ground_grid_to_viser, make_origin_axes

# PD gains for joint position control during Newton simulation
_KE = 500.0  # position stiffness  [N·m/rad]
_KD = 50.0  # velocity damping    [N·m·s/rad]


def animate_plan(
    morph: Morphology,
    task: Task,
    path: list[torch.Tensor],
    port: int = 8080,
    fps: int = 30,
    hold_seconds: float = 1.0,
    loop: bool = True,
    sim_substeps: int = 4,
) -> None:
    """Execute the planned path in Newton physics and stream it to a viewer.

    Each animation frame sets the next joint-position target and advances the
    Newton simulation by `sim_substeps` steps with PD position control, so
    contact forces are fully resolved.  The viewer loops until the tab is closed.

    Args:
        path: list of joint-config tensors (n_joints,).  Should be densified
            with `interpolate_path` for smooth target tracking.
        sim_substeps: physics substeps per rendered frame.
    """
    n_joints = morph.n_links - 1

    builder = newton.ModelBuilder()

    for obs in task.environment.obstacles:
        if obs.kind == "box":
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    p=wp.vec3(*obs.center.tolist()), q=wp.quat_identity()
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
        pos = task.goal_poses[i, :3, 3]
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(p=wp.vec3(*pos.tolist()), q=wp.quat_identity()),
            radius=0.03,
            as_site=True,
            color=wp.vec3(1.0, 0.2, 0.2),
        )

    rest_poses = compute_link_world_poses(morph)
    add_robot_to_builder(builder, morph, rest_poses, label="robot")

    # Enable PD position control on every revolute DOF
    for i in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[i] = _KE
        builder.joint_target_kd[i] = _KD
        builder.joint_target_mode[i] = int(JointTargetMode.POSITION)

    # Start at the first waypoint so the robot doesn't lurch from rest
    builder.joint_q = path[0].detach().cpu().numpy().astype(np.float32).tolist()

    model = builder.finalize()
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    solver = newton.solvers.SolverMuJoCo(model)

    frame_dt = 1.0 / fps
    sim_dt = frame_dt / sim_substeps

    viewer = newton.viewer.ViewerViser(port=port)
    viewer.set_model(model)
    add_ground_grid_to_viser(viewer._server, grid_size=4.0, divisions=8)

    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)

    print(f"Open http://localhost:{port}  ({len(path)} frames)")

    hold_frames = int(hold_seconds * fps)

    def step_to(q: torch.Tensor) -> None:
        nonlocal state_0, state_1
        arr = q.detach().cpu().numpy().astype(np.float32)[:n_joints]
        control.joint_target_pos.assign(arr)
        for _ in range(sim_substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

    def render(t: float) -> None:
        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
        viewer.end_frame()

    try:
        t = 0.0
        while viewer.is_running():
            for _ in range(hold_frames):
                step_to(path[0])
                render(t)
                t += frame_dt
                time.sleep(frame_dt)
                if not viewer.is_running():
                    break

            for q in path:
                if not viewer.is_running():
                    break
                step_to(q)
                render(t)
                t += frame_dt
                time.sleep(frame_dt)

            for _ in range(hold_frames):
                step_to(path[-1])
                render(t)
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
