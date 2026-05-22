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
    """

    environment: Environment
    goal_poses: Float[torch.Tensor, "n_goals 4 4"]
    reachable_region: "ReachableRegion | None" = None
    start_q: Float[torch.Tensor, "n_joints"] | None = None


@dataclass
class ReachableRegion:
    """Axis-aligned volume from which goal poses are drawn (e.g. Task 1's room interior)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, point: torch.Tensor) -> bool:
        x, y, z = point
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )
