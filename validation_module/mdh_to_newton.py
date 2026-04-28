import time
import torch
import warp as wp
import newton
from scipy.spatial.transform import Rotation

from interface import Morphology


def add_robot_to_builder(
    builder: newton.ModelBuilder,
    morph: Morphology,
    pose: torch.Tensor,
    label: str = "robot",
) -> tuple[list[int], list[int]]:
    """Add robot links and joints to an existing builder.
    
    Args:
        builder: Existing ModelBuilder (may already contain environment shapes).
        morph: Morphology with shape (n_links, 3) MDH parameters.
        pose: World poses for each link, shape (n_links, 4, 4) — typically from FK.
        label: Articulation label.
    
    Returns:
        (link_indices, joint_indices)
    """
    links = []
    joints = []
    
    for i in range(morph.n_links):
        link_xform = wp.transform(
            p=pose[i, :3, 3],
            q=Rotation.from_matrix(pose[i, :3, :3]).as_quat(),
        )
        
        link = builder.add_link(mass=0.0, inertia=None, xform=link_xform)
        links.append(link)
        
        d_val = morph.d[i].item()
        a_val = morph.a[i].item()
        
        # d-capsule (along local z)
        builder.add_shape_capsule(
            body=link,
            radius=morph.link_radius,
            half_height=abs(d_val) / 2,
            xform=wp.transform(
                p=torch.tensor([0.0, 0.0, -d_val / 2]),
                q=Rotation.identity().as_quat(),
            ),
        )
        
        # a-capsule (along local x)
        builder.add_shape_capsule(
            body=link,
            radius=morph.link_radius,
            half_height=abs(a_val) / 2,
            xform=wp.transform(
                p=torch.tensor([-a_val / 2, 0.0, -d_val]),
                q=Rotation.from_euler("y", 90, degrees=True).as_quat(),
            ),
        )
        
        if i != 0:
            pose_rel = torch.linalg.inv(pose[i - 1]) @ pose[i]
            joints.append(
                builder.add_joint_revolute(
                    parent=links[i - 1],
                    child=link,
                    parent_xform=wp.transform(
                        p=pose_rel[:3, 3],
                        q=Rotation.from_matrix(pose_rel[:3, :3]).as_quat(),
                    ),
                    child_xform=wp.transform_identity(),
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
        builder.add_ground_plane()
    
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