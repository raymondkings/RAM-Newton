import time
import torch
import warp as wp
import newton
from scipy.spatial.transform import Rotation

from interface import Morphology
from validation.ground import add_ground_collision


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


def show_model_in_notebook(model, state, port: int = 8080):
    """Create a viewer, attach a model, and render the state."""
    viewer = newton.viewer.ViewerViser(port=port)
    viewer.set_model(model)
    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.end_frame()
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        viewer.close()
    return viewer