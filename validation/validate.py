import newton
import warp as wp
from interface import Morphology, Task, ValidationResult
from validation.mdh_to_newton import add_robot_to_builder
from util.kinematics import compute_link_world_poses
from validation.collision_check import build_self_collision_ignore_pairs, check_collisions
from validation.ground import add_ground_collision


def validate(morph: Morphology, task: Task, debug: bool = False) -> ValidationResult:
    builder = newton.ModelBuilder()
    add_ground_collision(builder)

    for i, obs in enumerate(task.environment.obstacles):
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
                label=f"obstacle_{i}",
            )

    poses = compute_link_world_poses(morph)
    poses = task.environment.base_pose.unsqueeze(0) @ poses
    add_robot_to_builder(builder, morph, poses)

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    # ground box is always shape 0 — add_ground_collision() is called first
    ignore_pairs = build_self_collision_ignore_pairs(model)
    counts = check_collisions(
        model, state,
        ignore_self_pairs=ignore_pairs,
        base_body=0,
        ground_shape=0,
        debug=debug,
    )

    return ValidationResult(
        self_collision_free=counts["n_self_collisions"] == 0,
        env_collision_free=counts["n_env_collisions"] == 0,
        n_self_collisions=counts["n_self_collisions"],
        n_env_collisions=counts["n_env_collisions"],
        n_samples=1,
    )