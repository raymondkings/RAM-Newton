import os

import torch

from interface import Morphology, Task
from util.kinematics import (
    IK,
    FK,
    build_robot_dict,
    build_scene,
    pose_errors,
    se3_distance,
)


def build_optimization_validation_context(
    task: Task,
    device: torch.device,
    ignore_ground: bool = False,
    ignore_obstacles: bool = False,
):
    """Build reusable validation context for optimization.

    Returns:
        base_pose_inv:
            [4, 4] inverse of task.environment.base_pose.
        scene:
            cuRobo Scene built in the robot-local frame.
    """
    base_pose_inv = torch.linalg.inv(
        task.environment.base_pose.to(device=device, dtype=torch.float32)
    )

    scene = build_scene(
        task,
        base_pose_inv,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    return base_pose_inv, scene


def _num_sampled_poses(total_poses: int, percentage_poses: float) -> int:
    """Convert percentage_poses to number of sampled poses.

    If percentage_poses is in (0, 1], it is treated as a fraction.
    If percentage_poses > 1, it is treated as an absolute number of poses.
    """
    if percentage_poses <= 0:
        raise ValueError("percentage_poses must be positive.")

    if percentage_poses <= 1.0:
        return max(1, int(round(total_poses * percentage_poses)))

    return min(total_poses, int(round(percentage_poses)))


def _sample_goal_poses(
    goal_poses: torch.Tensor,
    percentage_poses: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample goal poses for validation.

    Returns:
        sampled_pose_indices:
            [P] indices into the original goal_poses.
        sampled_goal_poses:
            [P, 4, 4] sampled goal poses.
    """
    total_poses = goal_poses.shape[0]
    n_sample = _num_sampled_poses(total_poses, percentage_poses)

    if n_sample == total_poses:
        sampled_pose_indices = torch.arange(
            total_poses,
            device=goal_poses.device,
        )
    else:
        sampled_pose_indices = torch.randperm(
            total_poses,
            generator=generator,
            device=goal_poses.device,
        )[:n_sample]

    sampled_goal_poses = goal_poses[sampled_pose_indices]

    return sampled_pose_indices, sampled_goal_poses


def _ik_success_pose_rate(success: torch.Tensor, num_poses: int) -> torch.Tensor:
    """Compute the fraction of goal poses with successful IK solutions."""
    success = success.detach().cpu().bool()

    if success.numel() == 0 or num_poses <= 0:
        return torch.tensor(0.0, dtype=torch.float32)

    if success.ndim == 0:
        return success.to(dtype=torch.float32)

    if success.shape[0] == num_poses or success.numel() % num_poses == 0:
        count = success.reshape(num_poses, -1).any(dim=1).sum()
        return count.to(dtype=torch.float32) / float(num_poses)

    return success.reshape(-1).float().mean()


def run_optimization_validation(
    processed_morphology: torch.Tensor,
    morph: Morphology,
    task: Task,
    scene,
    base_pose_inv: torch.Tensor,
    device: torch.device,
    percentage_poses: float,
    number_random_seed: int,
    pose_sampling_generator: torch.Generator,
) -> dict:
    """Run one IK/FK validation step during morphology optimization.

    Logic:
        1. Build a temporary cuRobo robot from the current processed morphology.
        2. Randomly sample a subset of task.goal_poses.
        3. Run cuRobo IK on the sampled poses.
        4. Run FK for the returned joint solutions.
        5. Compute pose errors.
        6. Compute sampled goal pose IK success rate.
        7. Return data directly compatible with OptimizationCSVLogger.
    """
    eval_morph = Morphology(
        params=processed_morphology.detach().clone(),
        link_radius=morph.link_radius,
    )

    sampled_pose_indices, sampled_goal_poses = _sample_goal_poses(
        goal_poses=task.goal_poses.to(device),
        percentage_poses=percentage_poses,
        generator=pose_sampling_generator,
    )

    robot_dict, urdf_path = build_robot_dict(eval_morph)

    try:
        ik_solver = IK(
            robot_dict,
            scene,
            num_seeds=number_random_seed,
            max_batch_size=sampled_goal_poses.shape[0],
        )
        fk_solver = FK(robot_dict)

        joints, ik_success = ik_solver.solve(
            sampled_goal_poses,
            base_pose_inv,
            device,
        )

        reached = fk_solver.compute(
            joints,
            base_pose_inv,
            device,
        )

    finally:
        os.unlink(urdf_path)

    pos_err, rot_err = pose_errors(
        reached,
        sampled_goal_poses,
    )
    se3_dist = se3_distance(pos_err, rot_err)
    ik_success_rate = _ik_success_pose_rate(
        ik_success,
        num_poses=sampled_goal_poses.shape[0],
    )

    return {
        "sampled_pose_indices": sampled_pose_indices,
        "sampled_goal_poses": sampled_goal_poses,
        "ik_success_pose_rate": ik_success_rate,
        "best_joints": joints,
        "fk_reached_poses_best": reached,
        "best_pos_err_mean": pos_err.mean(),
        "best_rot_err_mean": rot_err.mean(),
        "best_se3_dist_mean": se3_dist.mean(),
        "best_pos_err_per_pose": pos_err,
        "best_rot_err_per_pose": rot_err,
        "best_se3_dist_per_pose": se3_dist,
    }
