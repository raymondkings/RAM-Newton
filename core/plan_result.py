"""Path data types and interpolation utilities."""

from dataclasses import dataclass

import torch


@dataclass
class PlanResult:
    success: bool
    path: list[torch.Tensor]  # joint configs along the path; empty if not success
    failed_at_goal: int | None = None  # index of first goal that could not be reached
    best_ik_q: "torch.Tensor | None" = None  # best IK joint config when planning failed
