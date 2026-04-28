import torch
from mdh_to_newton import build_mdh_newton_model, show_model_in_notebook


def simple_mdh_morphology():
    """Return a minimal MDH morphology and world poses for a toy chain."""
    morph = torch.tensor(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.8, 0.4],
            [0.0, 0.8, 0.3],
        ],
        dtype=torch.float32,
    )

    pose0 = torch.eye(4, dtype=torch.float32)
    pose1 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.9],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pose2 = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.7],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    pose = torch.stack([pose0, pose1, pose2])
    return morph, pose


if __name__ == "__main__":
    morph, pose = simple_mdh_morphology()
    model, state = build_mdh_newton_model(
        morph,
        pose,
        link_radius=0.05,
        validate_inertia_detailed=False,
        add_ground_plane=True,
        label="simple_toy_mdh_robot",
    )
    print("Built Newton model with", morph.shape[0], "links")
    print("If running in a notebook, open http://localhost:8080 to view the model.")
    show_model_in_notebook(model, state, port=8080)
