# This notebook aims to build a really really toy example of a 4 revolute joint robot...
# (a total of 4 visible parts, with one 2DOF joint)
#
# Mi = [alpha_{i-1}, a_{i-1}, d_i]
# Example:
# M1 = [0,      0, 1]
# M2 = [-pi/2,  1, 0]
# M3 = [ pi/2,  1, 1]
# M4 = [ pi/2,  0, 0]   # dummy stage for the 2DOF joint
# M5 = [0,      1, 0]

from pathlib import Path
import math

import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


# -----------------------------
# helper rotations
# -----------------------------
def qx(angle):
    return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)

def qy(angle):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)

def qz(angle):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)


# Create a new model builder
builder = newton.ModelBuilder()

# Add a ground plane (infinite static plane at z=0)
builder.add_ground_plane()

# MODIFIED:
CAPSULE_RADIUS = 0.10

# MODIFIED:
UNIT_HALF = 0.5


# =============================
# m1 = [0, 0, 1]
# =============================
# link/body --> correspond to frame 0
link0 = builder.add_link(
    xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),
        q=wp.quat_identity(),
    ),
    label="link0",
)

# geometry of M1: d = 1 -> one capsule along local z
builder.add_shape_capsule(
    body=link0,
    xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 0.5),
        q=wp.quat_identity(),   # default capsule axis = local z
    ),
    radius=CAPSULE_RADIUS,
    half_height=UNIT_HALF,      # MODIFIED: was 1.0
)

# fixed to the world
j0 = builder.add_joint_fixed(
    parent=-1,   # world
    child=link0,
    parent_xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),
        q=wp.quat_identity(),
    ),
    child_xform=wp.transform_identity(),
    label="world_to_link0_fixed",
)


# =============================
# m2 = [-pi/2, 1, 0]
# =============================
# link/body --> correspond to frame 1
link1 = builder.add_link(
    xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),
        q=wp.quat_identity(),
    ),
    label="link1",
)

# geometry of M2: a = 1 -> one capsule along local x
builder.add_shape_capsule(
    body=link1,
    xform=wp.transform(
        p=wp.vec3(0.5, 0.0, 0.0),
        q=qy(math.pi / 2.0),   # z -> x
    ),
    radius=CAPSULE_RADIUS,
    half_height=UNIT_HALF,     # MODIFIED: was 1.0
)

# joint of link0 and link1
j1 = builder.add_joint_revolute(
    parent=link0,
    child=link1,
    axis=wp.vec3(0.0, 0.0, 1.0),
    parent_xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 1.0),
        q=qx(-math.pi / 2.0),
    ),
    child_xform=wp.transform_identity(),
    label="link0_to_link1_rotational",
)


# =============================
# m3 = [pi/2, 1, 1]
# =============================
# link/body --> correspond to frame 2
link2 = builder.add_link(
    xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),
        q=wp.quat_identity(),
    ),
    label="link2",
)

# geometry of M3:
# a = 1 -> one capsule along local x
builder.add_shape_capsule(
    body=link2,
    xform=wp.transform(
        p=wp.vec3(0.5, 0.0, 0.0),
        q=qy(math.pi / 2.0),   # z -> x
    ),
    radius=CAPSULE_RADIUS,
    half_height=UNIT_HALF,     # MODIFIED
)

# d = 1 -> second capsule along local z, attached at x = 1
builder.add_shape_capsule(
    body=link2,
    xform=wp.transform(
        p=wp.vec3(1.0, 0.0, 0.5),
        q=wp.quat_identity(),   # MODIFIED: keep it along local z
    ),
    radius=CAPSULE_RADIUS,
    half_height=UNIT_HALF,      # MODIFIED
)

# joint of link1 and link2
j2 = builder.add_joint_revolute(
    parent=link1,
    child=link2,
    axis=wp.vec3(0.0, 0.0, 1.0),   # MODIFIED
    parent_xform=wp.transform(
        p=wp.vec3(1.0, 0.0, 0.0),
        q=qx(math.pi / 2.0),
    ),
    child_xform=wp.transform_identity(),
    label="link1_to_link2_rotational",
)


# =============================
# m4 = [pi/2, 0, 0]
# dummy link for the 2DOF joint
# =============================
# link/body --> correspond to frame 3
link3 = builder.add_link(
    xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),   # MODIFIED
        q=wp.quat_identity(),
    ),
    label="link3_dummy",
)

# no capsule added, this is the dummy link for a 2DOF rotational joint

# first axis of the 2DOF joint
j3 = builder.add_joint_revolute(
    parent=link2,
    child=link3,
    axis=wp.vec3(0.0, 0.0, 1.0),   # MODIFIED: first axis
    parent_xform=wp.transform(
        p=wp.vec3(1.0, 0.0, 1.0),
        q=qx(math.pi / 2.0),
    ),
    child_xform=wp.transform_identity(),
    label="link2_to_link3_rotational",
)


# =============================
# m5 = [0, 1, 0]
# =============================
# link/body --> correspond to frame 4
link4 = builder.add_link(
    xform=wp.transform(
        p=wp.vec3(0.0, 0.0, 0.0),   # MODIFIED
        q=wp.quat_identity(),
    ),
    label="link4",
)

# geometry of M5: a = 1 -> one capsule along local x
builder.add_shape_capsule(
    body=link4,
    xform=wp.transform(
        p=wp.vec3(0.5, 0.0, 0.0),
        q=qy(math.pi / 2.0),   # z -> x
    ),
    radius=CAPSULE_RADIUS,
    half_height=UNIT_HALF,     # MODIFIED
)

# second axis of the 2DOF joint, same point as j3
j4 = builder.add_joint_revolute(
    parent=link3,
    child=link4,
    axis=wp.vec3(0.0, 1.0, 0.0),
    parent_xform=wp.transform_identity(),
    child_xform=wp.transform_identity(),
    label="link3_to_link4_rotational",
)


# articulation
builder.add_articulation([j0, j1, j2, j3, j4], label="simple_robot")

model = builder.finalize()

state = model.state()

# all 0
state.joint_q.numpy()[:] = 0.0
state.joint_qd.numpy()[:] = 0.0

# Forward kinematics
newton.eval_fk(model, state.joint_q, state.joint_qd, state)

viewer = newton.viewer.ViewerGL()
viewer.set_model(model)

# MODIFIED:
if hasattr(viewer, "camera_pos"):
    viewer.camera_pos = wp.vec3(3.8, -4.2, 3.0)
if hasattr(viewer, "camera_target"):
    viewer.camera_target = wp.vec3(0.9, 0.0, 0.9)

sim_time = 0.0

# ---------------- STATIC VIEW ----------------
while viewer.is_running():
    viewer.begin_frame(sim_time)
    viewer.log_state(state)
    viewer.end_frame()

# ---------------- PROGRAMMATIC PAN EXAMPLE (KEEP COMMENTED FOR NOW) ----------------
# pan = wp.vec3(0.0, 1.0, 0.0)   # move camera sideways
# if hasattr(viewer, "camera_pos") and hasattr(viewer, "camera_target"):
#     viewer.camera_pos = viewer.camera_pos + pan
#     viewer.camera_target = viewer.camera_target + pan

# ---------------- DYNAMIC VERSION (KEEP COMMENTED FOR NOW) ----------------
# dt = 1.0 / 60.0
# while viewer.is_running():
#     q = state.joint_q.numpy()
#     q[:] = 0.0
#     if q.shape[0] >= 4:
#         q[0] = 0.25 * math.sin(sim_time)
#         q[1] = 0.20 * math.sin(sim_time + 0.8)
#         q[2] = 0.15 * math.sin(sim_time + 1.6)
#         q[3] = 0.20 * math.sin(sim_time + 2.2)
#
#     state.joint_qd.numpy()[:] = 0.0
#     newton.eval_fk(model, state.joint_q, state.joint_qd, state)
#
#     viewer.begin_frame(sim_time)
#     viewer.log_state(state)
#     viewer.end_frame()
#
#     sim_time += dt