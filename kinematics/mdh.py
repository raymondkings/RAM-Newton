import math
import xml.etree.ElementTree as ET

import newton
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from core.morphology import Morphology
from kinematics.self_collision import get_joint_limits

_LINK_COLORS = [
    wp.vec3(0.13, 0.32, 0.50),
    wp.vec3(0.55, 0.20, 0.12),
    wp.vec3(0.18, 0.42, 0.18),
    wp.vec3(0.38, 0.23, 0.52),
    wp.vec3(0.55, 0.43, 0.08),
    wp.vec3(0.08, 0.43, 0.42),
    wp.vec3(0.50, 0.16, 0.29),
    wp.vec3(0.28, 0.28, 0.28),
]
_JOINT_MARKER_COLOR = wp.vec3(0.0, 0.0, 0.0)
_JOINT_MARKER_RADIUS_SCALE = 1.25


def _link_color(i: int) -> wp.vec3:
    return _LINK_COLORS[i % len(_LINK_COLORS)]


def to_urdf(morph: Morphology) -> str:
    """Convert an mdh description of a morphology to a URDF string.

    Joint limits are derived from morphology geometry to avoid self-collisions
    between the link capsules adjacent to each joint.
    """
    n = morph.n_links
    n_joints = n - 1  # revolute DOFs

    joint_limits = get_joint_limits(morph.params)  # [n_links, 2] = [range, offset]

    robot = ET.Element("robot", name="mdh_robot")
    ET.SubElement(robot, "link", name="base_link")
    for i in range(n):
        ET.SubElement(robot, "link", name=f"link_{i}")

        alpha = morph.alpha[i].item()
        a = morph.a[i].item()
        d = morph.d[i].item()
        sa, ca = math.sin(alpha), math.cos(alpha)

        parent = "base_link" if i == 0 else f"link_{i - 1}"
        joint_type = "fixed" if i == n_joints else "revolute"

        jnt = ET.SubElement(robot, "joint", name=f"joint_{i}", type=joint_type)
        ET.SubElement(jnt, "parent", link=parent)
        ET.SubElement(jnt, "child", link=f"link_{i}")
        ET.SubElement(
            jnt,
            "origin",
            xyz=f"{a:.8f} {-sa * d:.8f} {ca * d:.8f}",
            rpy=f"{alpha:.8f} 0.0 0.0",
        )
        ET.SubElement(jnt, "axis", xyz="0 0 1")
        if joint_type == "revolute":
            joint_range = joint_limits[i, 0].item()
            offset = joint_limits[i, 1].item()
            lower = offset
            upper = offset + joint_range
            ET.SubElement(
                jnt,
                "limit",
                lower=f"{lower:.6f}",
                upper=f"{upper:.6f}",
                effort="1000",
                velocity=f"{math.pi:.6f}",
            )

    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode")


def add_robot_to_builder(
    builder: newton.ModelBuilder,
    morph: Morphology,
    pose: torch.Tensor,
    label: str = "robot",
) -> tuple[list[int], list[int]]:
    """Add robot links and joints to an existing builder.

    Joint structure follows MDH convention:
    - Joint i (revolute, axis z) connects body i-1 (or world for i=0) to body i,
      and corresponds to theta_i.
    - The last MDH transform usually has theta = 0 (EE frame), so the last joint
      is added as a fixed joint and does not contribute to joint_q.

    Args:
        builder: Existing ModelBuilder (may already contain environment shapes).
        morph: Morphology with shape (n_links, 3) MDH parameters.
        pose: World poses for each link, shape (n_links, 4, 4) — typically from FK
            with all joint angles zero.
        label: Articulation label.

    Returns:
        (link_indices, joint_indices). joint_q has length n_links - 1
        (the last joint is fixed and contributes no DOF).
    """
    links = []
    joints = []
    n = morph.n_links

    for i in range(n):
        link_xform = wp.transform(
            p=pose[i, :3, 3],
            q=Rotation.from_matrix(pose[i, :3, :3]).as_quat(),
        )

        link = builder.add_link(mass=0.0, inertia=None, xform=link_xform)
        links.append(link)
        builder.add_shape_sphere(
            body=link,
            radius=morph.link_radius * _JOINT_MARKER_RADIUS_SCALE,
            xform=wp.transform_identity(),
            as_site=True,
            color=_JOINT_MARKER_COLOR,
            label=f"joint_{i}_marker",
        )

        d_val = morph.d[i].item()
        a_val = morph.a[i].item()
        link_color = _link_color(i)

        # d-capsule: represents Trans_z(d_i), applied AFTER R_z(theta_i).
        # So it rotates with theta_i — attached to body i, along body i's local z.
        if abs(d_val) > 2 * morph.link_radius:
            builder.add_shape_capsule(
                body=link,
                radius=morph.link_radius,
                half_height=abs(d_val) / 2,
                xform=wp.transform(
                    p=wp.vec3(0.0, 0.0, -d_val / 2),
                    q=wp.quat_identity(),
                ),
                color=link_color,
                label=f"link_{i}_d",
            )

        # a-capsule: represents Trans_x(a_i), applied BEFORE R_z(theta_i).
        # So it does NOT rotate with theta_i — attached to body i-1 (the parent),
        # along body i-1's local x. For i=0 the parent is the world/base frame.
        if abs(a_val) > 2 * morph.link_radius:
            parent_body = -1 if i == 0 else links[i - 1]
            builder.add_shape_capsule(
                body=parent_body,
                radius=morph.link_radius,
                half_height=abs(a_val) / 2,
                xform=wp.transform(
                    p=wp.vec3(a_val / 2, 0.0, 0.0),
                    q=Rotation.from_euler("y", 90, degrees=True).as_quat(),
                ),
                color=link_color,
                label=f"link_{i}_a",
            )

        if i == 0:
            parent_idx = -1
            pose_rel = pose[0]
        else:
            parent_idx = links[i - 1]
            pose_rel = torch.linalg.inv(pose[i - 1]) @ pose[i]

        parent_xform = wp.transform(
            p=pose_rel[:3, 3],
            q=Rotation.from_matrix(pose_rel[:3, :3]).as_quat(),
        )

        if i == n - 1:
            joints.append(
                builder.add_joint_fixed(
                    parent=parent_idx,
                    child=link,
                    parent_xform=parent_xform,
                    child_xform=wp.transform_identity(),
                )
            )
        else:
            joints.append(
                builder.add_joint_revolute(
                    parent=parent_idx,
                    child=link,
                    parent_xform=parent_xform,
                    child_xform=wp.transform_identity(),
                    axis=(0.0, 0.0, 1.0),
                )
            )

    if joints:
        builder.add_articulation(joints, label=label)

    return links, joints


def build_mdh_newton_model(
    morph: Morphology,
    pose: torch.Tensor,
    add_ground_plane: bool = True,
    label: str = "robot",
):
    """Convenience wrapper: builds a model with just the robot (no environment)."""
    builder = newton.ModelBuilder()
    builder.validate_inertia_detailed = True
    if add_ground_plane:
        add_ground_collision(builder)

    add_robot_to_builder(builder, morph, pose, label=label)

    model = builder.finalize()
    state = model.state()
    return model, state
