import math
import torch
import newton
from scipy.spatial.transform import Rotation
from interface import Morphology, Task
import warp as wp
import xml.etree.ElementTree as ET
from interface.morphology import Morphology
from util.kinematics import transformation_matrix, forward_kinematics
from util.self_collision import get_capsules

EPS = 1e-4

def get_joint_limits(morph: Morphology) -> torch.Tensor:
    """
    Compute joint limits based on the morphology to avoid self-collisions.

    Args:
        morph: MDH parameters encoding the robot geometry.

    Returns:
        Joint limits
    """
    params = morph.params
    joint_limits = torch.zeros(*params.shape[:-1], 2, device=params.device)

    extended_morph = torch.cat([torch.zeros_like(params[..., :1, :]), params], dim=-2)
    alpha0, a0, d0 = extended_morph[..., :-2, :].split(1, dim=-1)
    alpha1, a1, d1 = extended_morph[..., 1:-1, :].split(1, dim=-1)

    coordinate_fix = torch.eye(4, device=params.device, dtype=params.dtype).repeat(*params.shape[:-2], params.shape[-2] - 1,
                                                                                   1, 1)
    wrist = (a1[..., 0] == 0) & (d1[..., 0] == 0)
    coordinate_fix[wrist] = transformation_matrix(alpha0, a0, d0, torch.zeros_like(d0))[wrist]

    plane_normal = torch.stack([
        torch.zeros_like(alpha1),
        -torch.sin(alpha1),
        torch.cos(alpha1),
        torch.zeros_like(alpha1)], dim=-1)
    plane_anchor = torch.stack([
        a1,
        -d1 * torch.sin(alpha1),
        d1 * torch.cos(alpha1),
        torch.ones_like(alpha1)], dim=-1)

    plane_normal = torch.sum(coordinate_fix * plane_normal, dim=-1)[..., :3]
    plane_anchor = torch.sum(coordinate_fix * plane_anchor, dim=-1)[..., :3]

    stacked_morph = torch.stack([extended_morph[..., :-2, :], extended_morph[..., 1:-1, :], extended_morph[..., 2:, :]],
                                dim=-2)
    stacked_morph[~wrist, 0, :] = 0.0
    stacked_poses = forward_kinematics(stacked_morph, torch.zeros(*stacked_morph.shape[:-1], 1, device=params.device))
    start, end = get_capsules(stacked_morph, stacked_poses)
    capsules = end - start

    # Get closest non-zero capsule before joint
    pre_capsule = capsules[..., 3, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 2, :]
    pre_capsule[mask] = capsules[mask := pre_capsule.norm(dim=-1) < 1e-6, 1, :]

    # Get closest non-zero capsule after joint
    post_capsule = capsules[..., -2, :]
    post_capsule[mask] = capsules[mask := post_capsule.norm(dim=-1) < 1e-6, -1, :]

    in_plane = ((pre_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6
    in_plane &= ((post_capsule - plane_anchor) * plane_normal).sum(dim=-1).abs() < 1e-6

    limited = (pre_capsule.norm(dim=-1) > EPS) & (post_capsule.norm(dim=-1) > EPS) & in_plane

    mask = post_capsule.norm(dim=-1) > pre_capsule.norm(dim=-1)
    arc = torch.arcsin(2 * morph.link_radius / post_capsule.norm(dim=-1))
    arc[mask] = torch.arcsin(2 * morph.link_radius / pre_capsule.norm(dim=-1))[mask]

    joint_limits[..., :-1, 0] = torch.where(limited, 2 * torch.pi - 2 * arc, 2 * torch.pi)  # Range
    angle = torch.atan2(torch.sum(torch.cross(pre_capsule, post_capsule, dim=-1) * plane_normal, dim=-1),
                        torch.sum(pre_capsule * post_capsule, dim=-1))
    # if their angle becomes pi, they collide and are antiparallel
    angle = torch.atan2(torch.sin(torch.pi - angle), torch.cos(torch.pi - angle))
    joint_limits[..., :-1, 1] = torch.where(limited, angle + arc, -torch.pi)  # Offset

    return joint_limits

def to_urdf(morph: Morphology, joint_lims: list[tuple[float, float]]) -> str:
    """Convert a mhd description of a morphology to a URDF string""" 
    n = morph.n_links
    n_joints = n - 1  # revolute DOFs

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
            jnt, "origin",
            xyz=f"{a:.8f} {-sa * d:.8f} {ca * d:.8f}",
            rpy=f"{alpha:.8f} 0.0 0.0",
        )
        ET.SubElement(jnt, "axis", xyz="0 0 1")
        if joint_type == "revolute":
            span, offset = joint_lims[i]
            ET.SubElement(
                jnt, "limit",
                lower=f"{offset:.6f}",
                upper=f"{offset + span:.6f}",
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

        d_val = morph.d[i].item()
        a_val = morph.a[i].item()

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
                label=f"link_{i}_d",
            )

        # a-capsule: represents Trans_x(a_i), applied BEFORE R_z(theta_i).
        # So it does NOT rotate with theta_i — attached to body i-1 (the parent),
        # along body i-1's local x. For i=0 the parent is the world: a static
        # placement would be needed there; none of our morphs use a_0 > 0 yet.
        if abs(a_val) > 2 * morph.link_radius and i > 0:
            builder.add_shape_capsule(
                body=links[i - 1],
                radius=morph.link_radius,
                half_height=abs(a_val) / 2,
                xform=wp.transform(
                    p=wp.vec3(a_val / 2, 0.0, 0.0),
                    q=Rotation.from_euler("y", 90, degrees=True).as_quat(),
                ),
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