import time
import warp as wp
import newton
from validation_module.scenarios.task1_environment import make_task1_environment


def main():
    env = make_task1_environment()
    
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    
    for obs in env.obstacles:
        if obs.kind == "box":
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    p=wp.vec3(*obs.center.tolist()),
                    q=wp.quat_identity(),
                ),
                hx=obs.half_extents[0].item(),
                hy=obs.half_extents[1].item(),
                hz=obs.half_extents[2].item(),
            )
    
    model = builder.finalize()
    state = model.state()
    
    viewer = newton.viewer.ViewerViser(port=8080)
    viewer.set_model(model)
    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.end_frame()
    
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        viewer.close()


if __name__ == "__main__":
    main()