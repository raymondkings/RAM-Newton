"""High-level entry point: plan a collision-free path from a start joint config
to a configuration that reaches a goal pose, using RRT-Connect."""
import torch

from interface import Morphology, Task

from util.kinematics import (
    forward_kinematics,
    numerical_inverse_kinematics,
    pure_analytical_inverse_kinematics,
)
from validation.collision_oracle import CollisionOracle
from validation.planner import PlanResult, rrt_connect, shortcut


def _ee_pose_error(
    morph_params: torch.Tensor,
    full_joints: torch.Tensor,
    goal_pose: torch.Tensor,
) -> tuple[float, float]:
    """Returns (position_err, rotation_frobenius_err) for a single joint config."""
    morph_b = morph_params.unsqueeze(0)
    fj = full_joints.unsqueeze(0)
    ee_pose = forward_kinematics(morph_b, fj)[0, -1]
    pos_err = (ee_pose[:3, 3] - goal_pose[:3, 3]).norm().item()
    rot_err = ((ee_pose[:3, :3] - goal_pose[:3, :3]) ** 2).sum().sqrt().item()
    return pos_err, rot_err


def find_goal_config(
    morph_params: torch.Tensor,
    goal_pose: torch.Tensor,
    oracle: CollisionOracle | None,
    start_q: torch.Tensor,
    pos_tol: float = 0.03,
    rot_tol: float | None = None,
    n_numerical_seeds: int = 64,
) -> torch.Tensor | None:
    """Find a joint config that reaches goal_pose and is close to start_q.

    Strategy: try analytical IK (cheap when applicable); always run numerical
    IK as well; collect all candidates; filter; pick closest to start_q.

    oracle: collision oracle. Pass None to skip collision filtering — useful
    for testing whether a morphology is *kinematically* capable of reaching
    a pose, independent of obstacles.

    rot_tol: Frobenius-norm tolerance on rotation matrix difference. None means
    skip orientation check (position-only IK). Many morphologies have structural
    twists that prevent reaching arbitrary orientations.
    """
    dtype = morph_params.dtype
    candidates: list[torch.Tensor] = []

    try:
        sols = pure_analytical_inverse_kinematics(
            morph_params.to(torch.float64), goal_pose.to(torch.float64).unsqueeze(0)
        )
        if sols and sols[0].shape[0] > 0:
            for k in range(sols[0].shape[0]):
                candidates.append(sols[0][k].to(dtype))
    except RuntimeError:
        pass

    num_rot_weight = 0.5 if rot_tol is not None else 0.0
    num = numerical_inverse_kinematics(
        morph_params,
        goal_pose.to(dtype),
        n_seeds=n_numerical_seeds,
        rot_weight=num_rot_weight,
    )
    for k in range(num.shape[0]):
        candidates.append(num[k])

    valid: list[torch.Tensor] = []
    for cand in candidates:
        pos_err, rot_err = _ee_pose_error(morph_params, cand, goal_pose.to(dtype))
        if pos_err > pos_tol:
            continue
        if rot_tol is not None and rot_err > rot_tol:
            continue
        q = cand[:-1, 0]
        if oracle is not None and not oracle.is_collision_free(q):
            continue
        valid.append(q)

    if not valid:
        return None

    dists = torch.stack([(q - start_q).norm() for q in valid])
    return valid[int(dists.argmin().item())]


def plan_to_pose(
    morph: Morphology,
    task: Task,
    goal_pose: torch.Tensor,
    start_q: torch.Tensor | None = None,
    max_iter: int = 2000,
    step_size: float = 0.15,
    smooth: bool = True,
    seed: int | None = 0,
    fallback_kinematic_only: bool = True,
) -> tuple[PlanResult, torch.Tensor, torch.Tensor | None]:
    """Plan a path from start_q to a joint config that reaches goal_pose.

    Returns (result, start_q, goal_q).

    If `fallback_kinematic_only` is True and either (a) no collision-free IK
    solution exists, or (b) the planner cannot connect, we fall back to a
    kinematic-only IK that ignores environment collisions and return a
    straight-line path in joint space. The result will have
    `kinematic_only=True` to signal that the path is for inspection only —
    it may crash through obstacles.

    goal_q is None only if the morphology is *kinematically* incapable of
    reaching the pose (no IK solution at all, even ignoring collisions).
    """
    n_joints = morph.n_links - 1
    dtype = morph.params.dtype
    if start_q is None:
        if task.start_q is not None:
            start_q = task.start_q.to(dtype)
        else:
            start_q = torch.zeros(n_joints, dtype=dtype)

    oracle = CollisionOracle(morph, task)

    if not oracle.is_collision_free(start_q):
        print("  [plan_to_pose] start configuration is in collision — skipping plan.")
        empty = PlanResult(success=False, path=[], n_iterations=0, n_nodes=0)
        return empty, start_q, None

    # IK runs in the morphology's local frame; goal_pose is given in world coords.
    base_pose = task.environment.base_pose.to(dtype)
    goal_pose_local = torch.linalg.inv(base_pose) @ goal_pose.to(dtype)

    goal_q = find_goal_config(morph.params, goal_pose_local, oracle, start_q)

    if goal_q is not None:
        result = rrt_connect(
            start_q, goal_q, oracle, max_iter=max_iter, step_size=step_size, seed=seed
        )
        if result.success and smooth:
            result.path = shortcut(result.path, oracle, n_attempts=100, step_size=step_size)
        if result.success:
            return result, start_q, goal_q
        # planner failed despite collision-free IK existing — fall through to fallback

    if not fallback_kinematic_only:
        empty = PlanResult(success=False, path=[], n_iterations=0, n_nodes=0)
        return empty, start_q, goal_q

    kin_goal_q = find_goal_config(morph.params, goal_pose_local, None, start_q)
    if kin_goal_q is None:
        empty = PlanResult(success=False, path=[], n_iterations=0, n_nodes=0)
        return empty, start_q, None

    fallback = PlanResult(
        success=True,
        path=[start_q, kin_goal_q],
        n_iterations=0,
        n_nodes=2,
        kinematic_only=True,
    )
    return fallback, start_q, kin_goal_q