"""Path data types and interpolation utilities."""
from dataclasses import dataclass
import torch


@dataclass
class PlanResult:
    success: bool
    path: list[torch.Tensor]   # joint configs along the path; empty if not success
    n_iterations: int
    n_nodes: int
    kinematic_only: bool = False  # retained for API compatibility; always False


def interpolate_path(path: list[torch.Tensor], step: float = 0.02) -> list[torch.Tensor]:
    """Densify path with at most `step` joint-distance between consecutive frames."""
    if len(path) < 2:
        return path
    out = [path[0]]
    for q_a, q_b in zip(path[:-1], path[1:]):
        delta = q_b - q_a
        n = max(1, int(torch.ceil(delta.norm() / step).item()))
        for k in range(1, n + 1):
            out.append(q_a + (k / n) * delta)
    return out
