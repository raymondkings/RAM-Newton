import time
import numpy as np
import torch
import warp as wp
import newton

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses
from validation.mdh_to_newton import add_robot_to_builder
from validation.ground import add_ground_grid_to_viser, make_origin_axes


def animate_plan(
    morph: Morphology,
    task: Task,
    path: list[torch.Tensor],
    port: int = 8080,
    fps: int = 30,
    hold_seconds: float = 1.0,
    loop: bool = True,
) -> None:
    """Open a Newton viewer and play the path. Loops until the user closes the tab.
    Args:
        path: list of joint-config tensors (n_joints,). Should already be densified
            with `interpolate_path` for a smooth animation.
        hold_seconds: how long to pause at start and end of each loop.
    """
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

    rest_poses = compute_link_world_poses(morph)
    rest_poses = task.environment.base_pose.unsqueeze(0) @ rest_poses
    add_robot_to_builder(builder, morph, rest_poses, label="robot")

    model = builder.finalize()
    state = model.state()

    viewer = newton.viewer.ViewerViser(port=port)
    viewer.set_model(model)
    add_ground_grid_to_viser(viewer._server, grid_size=4.0, divisions=8)

    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)

    print(f"Open http://localhost:{port}  ({len(path)} frames)")

    frame_dt = 1.0 / fps
    hold_frames = int(hold_seconds * fps)

    def render_q(q: torch.Tensor, t: float) -> None:
        arr = q.detach().cpu().numpy().astype(np.float32).reshape(-1)
        state.joint_q.assign(arr)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        viewer.begin_frame(t)
        viewer.log_state(state)
        viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
        viewer.end_frame()

    try:
        t = 0.0
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