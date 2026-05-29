from __future__ import annotations

from pathlib import Path

from interface import Morphology, Task
from util.ggik_baseline_common import optimize_single_morphology


def optimize_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
) -> tuple[Morphology, Path]:
    """GGIK baseline for one fixed-alpha initial morphology.

    This optimizes only continuous MDH lengths [a, d]. The objective uses GGIK
    to predict joint configurations, then differentiates FK pose error and a
    self-collision penalty with respect to morphology.
    """
    return optimize_single_morphology(
        morph,
        task,
        optimization_parameters,
        optimize_alpha=False,
        quantize_alpha_at_end=False,
        validate_with_curobo=False,
        label="GGIK fixed-alpha baseline",
    )
