import time
import torch
import warp as wp
import newton
from scipy.spatial.transform import Rotation


def build_mdh_newton_model(
    morph: torch.Tensor,
    pose: torch.Tensor,
    link_radius: float,
    validate_inertia_detailed: bool = True,
    add_ground_plane: bool = True,
    label: str = "simple_robot",
):
    """Build a Newton model from MDH morphology parameters.

    Args:
        morph: Tensor of shape (N, 3) containing MDH link parameters.
        pose: Tensor of shape (N, 4, 4) containing link poses in world coordinates.
        link_radius: Radius for both capsule shapes attached to each link.
        validate_inertia_detailed: If True, enable detailed inertia validation.
        add_ground_plane: If True, add a ground plane to the builder.
        label: Articulation label for the final model.

    Returns:
        model: The finalized Newton model.
        state: The model state object.
    """
    builder = newton.ModelBuilder()
    builder.validate_inertia_detailed = validate_inertia_detailed
    if add_ground_plane:
        builder.add_ground_plane()

    links = []
    joints = []
    for i in range(morph.shape[0]):
        link_xform = wp.transform(
            p=pose[i, :3, 3],
            q=Rotation.from_matrix(pose[i, :3, :3]).as_quat(),
        )

        link = builder.add_link(mass=0.0, inertia=None, xform=link_xform)
        links.append(link)

        builder.add_shape_capsule(
            body=link,
            radius=link_radius,
            half_height=abs(morph[i, 2].item()) / 2,
            xform=wp.transform(
                p=torch.tensor([0.0, 0.0, -morph[i, 2].item() / 2]),
                q=Rotation.identity().as_quat(),
            ),
        )

        builder.add_shape_capsule(
            body=link,
            radius=link_radius,
            half_height=abs(morph[i, 1].item()) / 2,
            xform=wp.transform(
                p=torch.tensor([-morph[i, 1].item() / 2, 0.0, -morph[i, 2].item()]),
                q=Rotation.from_euler("y", 90, degrees=True).as_quat(),
            ),
        )

        if i != 0:
            pose_rel = torch.linalg.inv(pose[i - 1]) @ pose[i]
            joints.append(
                builder.add_joint_fixed(
                    parent=links[i - 1],
                    child=link,
                    parent_xform=wp.transform(
                        p=pose_rel[:3, 3],
                        q=Rotation.from_matrix(pose_rel[:3, :3]).as_quat(),
                    ),
                    child_xform=wp.transform_identity(),
                )
            )

    builder.add_articulation(joints, label=label)
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
