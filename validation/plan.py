"""Motion planning via cuRobo (GPU-accelerated TrajOpt + graph search)."""
import torch

from interface import Morphology, Task
from validation.planner import PlanResult
from validation.curobo_planner import CUROBO_AVAILABLE, CuroboPlanner


def plan_to_pose(
    morph: Morphology,
    task: Task,
    goal_pose: torch.Tensor,
    start_q: torch.Tensor | None = None,
    max_attempts: int = 5,
) -> tuple[PlanResult, torch.Tensor, torch.Tensor | None]:
    """Plan a collision-free trajectory to goal_pose using cuRobo.

    Returns (result, start_q, goal_q).  goal_q is None on failure.
    Requires CUDA and the curobo package.
    """
    n_joints = morph.n_links - 1
    dtype = morph.params.dtype
    if start_q is None:
        start_q = task.start_q.to(dtype) if task.start_q is not None else torch.zeros(n_joints, dtype=dtype)

    if not CUROBO_AVAILABLE or not torch.cuda.is_available():
        print("cuRobo unavailable (requires CUDA + the curobo package).")
        return PlanResult(success=False, path=[], n_iterations=0, n_nodes=0), start_q, None

    print("Planning with cuRobo (GPU TrajOpt + graph search)...")
    try:
        planner = CuroboPlanner(morph, task)
        result, goal_q = planner.plan(goal_pose, start_q, max_attempts=max_attempts)
        return result, start_q, goal_q
    except Exception as e:
        print(f"  cuRobo error: {e}")
        return PlanResult(success=False, path=[], n_iterations=0, n_nodes=0), start_q, None
