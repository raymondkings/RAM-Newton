"""Motion planning via cuRobo (GPU-accelerated TrajOpt + graph search)."""
import torch

from interface import Morphology, Task
from validation.planner import PlanResult
from validation.curobo_planner import CUROBO_AVAILABLE, CuroboPlanner


def plan_to_poses(
    morph: Morphology,
    task: Task,
    goal_poses: torch.Tensor,
    start_q: torch.Tensor | None = None,
    max_attempts: int = 5,
) -> tuple[PlanResult, torch.Tensor, torch.Tensor | None]:
    """Plan a collision-free trajectory through all goal_poses using cuRobo.

    Chains plans sequentially: start_q → goal[0] → goal[1] → … → goal[N-1].
    Returns (result, start_q, final_goal_q).  final_goal_q is None on failure.
    Requires CUDA and the curobo package.
    """
    n_joints = morph.n_links - 1
    dtype = morph.params.dtype
    if start_q is None:
        start_q = task.start_q.to(dtype) if task.start_q is not None else torch.zeros(n_joints, dtype=dtype)

    if not CUROBO_AVAILABLE or not torch.cuda.is_available():
        print("cuRobo unavailable (requires CUDA + the curobo package).")
        return PlanResult(success=False, path=[], n_iterations=0, n_nodes=0), start_q, None

    print(f"Planning sequence of {goal_poses.shape[0]} goals with cuRobo (GPU TrajOpt + graph search)...")
    planner = CuroboPlanner(morph, task)
    result, final_q = planner.plan_sequence(goal_poses, start_q, max_attempts=max_attempts)
    return result, start_q, final_q
