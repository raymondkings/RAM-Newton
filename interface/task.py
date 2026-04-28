from dataclasses import dataclass, field
import torch
from jaxtyping import Float
from .environment import Environment

@dataclass
class Task:
    """A complete task specification."""
    environment: Environment
    goal_poses: Float[torch.Tensor, "n_goals 4 4"]
    reachable_region: "ReachableRegion | None" = None


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
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)