import time
import warp as wp
import newton

from interface import Morphology, Task
from util.kinematics import compute_link_world_poses
from validation.mdh_to_newton import add_robot_to_builder
from validation.ground import add_ground_grid_to_viser, make_origin_axes


def render_scene(morph: Morphology, task: Task, port: int = 8080) -> None:
    """Render morphology + task environment in the Newton viewer."""
    builder = newton.ModelBuilder()

    # Walls
    for obs in task.environment.obstacles:
        if obs.kind == "box":
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=wp.vec3(*obs.center.tolist()),
                                   q=wp.quat_identity()),
                hx=obs.half_extents[0].item(),
                hy=obs.half_extents[1].item(),
                hz=obs.half_extents[2].item(),
            )

    # Reachable region (optional)
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

    # Goal poses
    for i in range(task.goal_poses.shape[0]):
        pos = task.goal_poses[i, :3, 3]
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(p=wp.vec3(*pos.tolist()), q=wp.quat_identity()),
            radius=0.03,
            as_site=True,
            color=wp.vec3(1.0, 0.2, 0.2),
        )

    # Robot
    poses = compute_link_world_poses(morph)
    poses = task.environment.base_pose.unsqueeze(0) @ poses
    add_robot_to_builder(builder, morph, poses, label="robot")

    model = builder.finalize()
    state = model.state()

    viewer = newton.viewer.ViewerViser(port=port)
    viewer.set_model(model)
    add_ground_grid_to_viser(viewer._server, grid_size=4.0, divisions=8)
    viewer._server.scene.add_icosphere("/unit_sphere", radius=1.0, color=(180, 180, 255), opacity=0.08)

    axes_begins, axes_ends, axes_colors = make_origin_axes(axis_length=0.1)

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.log_lines("/origin_frame", axes_begins, axes_ends, axes_colors)
    viewer.end_frame()

    print(f"Open http://localhost:{port}")
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        viewer.close()