from __future__ import annotations

from pathlib import Path

from interface import Morphology, Task
from util.ggik_baseline_common import optimize_single_morphology


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """GGIK baseline for one morphology with continuous alpha relaxation.

    During optimization alpha, a, and d are continuous. The final alpha values
    are mapped back to {-pi/2, 0, pi/2}; final validation uses cuRobo when the
    project validation stack is available.
    """
    return optimize_single_morphology(
        morph,
        task,
        optimization_parameters,
        optimize_alpha=True,
        quantize_alpha_at_end=True,
        validate_with_curobo=True,
        label="GGIK alpha-relaxed baseline",
    )
