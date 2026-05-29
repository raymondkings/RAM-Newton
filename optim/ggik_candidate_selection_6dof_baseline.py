from __future__ import annotations

from pathlib import Path

from interface import Morphology, Task
from util.ggik_baseline_common import optimize_candidate_selection


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """GGIK candidate-selection baseline restricted to DOF 6."""
    return optimize_candidate_selection(
        morph,
        task,
        optimization_parameters,
        default_candidate_dofs=(6,),
        label="GGIK DOF6 candidate baseline",
    )
