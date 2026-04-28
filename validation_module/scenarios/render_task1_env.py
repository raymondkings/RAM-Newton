import time
import torch
import warp as wp
import newton
from validation_module.scenarios.task1 import make_task1


def main():
    task = make_task1()
    
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    
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
    
    # Reachable region — site only (no collision), bright color
    region = task.reachable_region
    region_center = wp.vec3(
        (region.x_min + region.x_max) / 2,
        (region.y_min + region.y_max) / 2,
        (region.z_min + region.z_max) / 2,
    )

    builder.add_shape_box(
        body=-1,
        xform=wp.transform(p=region_center, q=wp.quat_identity()),
        hx=(region.x_max - region.x_min) / 2,
        hy=(region.y_max - region.y_min) / 2,
        hz=(region.z_max - region.z_min) / 2,
        as_site=True,
        color=wp.vec3(0.0, 0.8, 0.2),
        label="reachable_region",
    )
    
    # Goal poses as small spheres
    for i in range(task.goal_poses.shape[0]):
        pos = task.goal_poses[i, :3, 3]
        builder.add_shape_sphere(
            body=-1,
            xform=wp.transform(p=wp.vec3(*pos.tolist()), q=wp.quat_identity()),
            radius=0.03,
        )
    
    model = builder.finalize()
    state = model.state()
    
    viewer = newton.viewer.ViewerViser(port=8080)
    viewer.set_model(model)
    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.end_frame()
    
    print("Open http://localhost:8080 in your browser")
    
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    main()