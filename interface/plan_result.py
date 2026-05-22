"""Path data types and interpolation utilities."""

from dataclasses import dataclass
import torch


@dataclass
class PlanResult:
    success: bool
    path: list[torch.Tensor]  # joint configs along the path; empty if not success
    n_iterations: int
    n_nodes: int
    kinematic_only: bool = False  # retained for API compatibility; always False
