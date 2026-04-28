from validation_module.kinematics import compute_link_world_poses
from validation_module.scenarios.task1 import make_task1
from validation_module.scenarios.test_morphology import make_test_morphology
from validation_module.validate import validate
import newton, warp as wp
from validation_module.mdh_to_newton import add_robot_to_builder

task = make_task1()
morph = make_test_morphology()

# Show actual poses used in validation (with base_pose applied)
poses_raw = compute_link_world_poses(morph)
poses_world = task.environment.base_pose.unsqueeze(0) @ poses_raw

print("base_pose z offset:", task.environment.base_pose[2, 3].item())
print("Link world poses AFTER base_pose:")
for i in range(morph.n_links):
    print(f"  link {i}: z = {poses_world[i, 2, 3]:.3f}")
print()

result = validate(morph, task)
print(result)

builder2 = newton.ModelBuilder()
builder2.add_ground_plane()
for obs in task.environment.obstacles:
    if obs.kind == "box":
        builder2.add_shape_box(body=-1,
            xform=wp.transform(p=wp.vec3(*obs.center.tolist()), q=wp.quat_identity()),
            hx=obs.half_extents[0].item(), hy=obs.half_extents[1].item(), hz=obs.half_extents[2].item())
poses2 = compute_link_world_poses(morph)
poses2 = task.environment.base_pose.unsqueeze(0) @ poses2
add_robot_to_builder(builder2, morph, poses2)
model2 = builder2.finalize()
state2 = model2.state()
newton.eval_fk(model2, state2.joint_q, state2.joint_qd, state2)
contacts2 = model2.collide(state2)
n = int(contacts2.rigid_contact_count.numpy()[0])
shape_body = model2.shape_body.numpy()
s0 = contacts2.rigid_contact_shape0.numpy()[:n]
s1 = contacts2.rigid_contact_shape1.numpy()[:n]
print(f"Contacts: {n}")
for i in range(n):
    print(f"  shape {s0[i]} (body {shape_body[s0[i]]}) vs shape {s1[i]} (body {shape_body[s1[i]]})")