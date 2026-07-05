from dataclasses import dataclass

import torch
from jaxtyping import Float

from .environment import Environment


@dataclass
class Task:
    """A complete task specification.
    `start_q` is the joint configuration the robot is in at the start of the
    task — the planner finds a collision-free path from here to a config that
    reaches one of `goal_poses`. If None, defaults to all-zeros (rest pose).
    Goals are visited in the order they appear in `goal_poses`.
    """

    environment: Environment
    goal_poses: Float[torch.Tensor, "n_goals 4 4"]
    start_q: Float[torch.Tensor, "n_joints"] | None = None
