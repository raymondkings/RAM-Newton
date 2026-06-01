from __future__ import annotations

from pathlib import Path

from interface import Morphology, Task
from util.ggik_baseline_common import optimize_candidate_selection


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """GGIK candidate-selection baseline for DOF 5-7.

    Alpha candidates are fixed. For each candidate, this baseline optimizes
    [a, d] using GGIK-predicted joints and FK SE3 error. Final selection is
    based on the top fraction ranked by GGIK/FK mean SE3 error.
    """
    return optimize_candidate_selection(
        morph,
        task,
        optimization_parameters,
        default_candidate_dofs=(7, 6, 5),
        label="GGIK DOF7-5 candidate baseline",
    )
