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
    failed_at_goal: int | None = None  # index of first goal that could not be reached
    best_ik_q: "torch.Tensor | None" = None  # best IK joint config when planning failed
    reachable_ratio: float = (
        0.0  # fraction of sequence goals reached before first failure
    )
