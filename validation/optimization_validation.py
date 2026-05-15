import os

import torch

from interface import Morphology, Task
from util.kinematics import (
    IK,
    FK,
    build_robot_dict,
    build_scene,
    select_best_seed_solutions,
)


def build_optimization_validation_context(
    task: Task,
    device: torch.device,
    ignore_ground: bool = False,
    ignore_obstacles: bool = False,
):
    """Build reusable validation context for optimization.

    This is separated from optim/nrm.py so other optimization algorithms can
    reuse the same IK/FK validation logic.

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
        3. Run cuRobo IK with multiple seeds/candidates.
        4. Run FK for all candidate joint solutions.
        5. For each sampled pose, choose the seed with the smallest SE3 error.
        6. Return a dictionary directly compatible with OptimizationCSVLogger.

    The returned mean errors are:
        mean over sampled poses of each pose's best seed/candidate error.
    """
    eval_morph = Morphology(
        params=processed_morphology.detach().clone(),
        link_radius=morph.link_radius,
    )

    robot_dict, urdf_path = build_robot_dict(eval_morph)

    try:
        ik_solver = IK(
            robot_dict,
            scene,
            num_seeds=number_random_seed,
            max_batch_size=task.goal_poses.shape[0],
        )
        fk_solver = FK(robot_dict)

        joints_all, success, sampled_pose_indices, sampled_goal_poses = ik_solver.solve(
            goal_poses_world=task.goal_poses.to(device),
            percentage_poses=percentage_poses,
            base_pose_inv=base_pose_inv,
            device=device,
            generator=pose_sampling_generator,
        )

        best = select_best_seed_solutions(
            fk_solver=fk_solver,
            joints_all=joints_all,
            sampled_goal_poses=sampled_goal_poses,
            base_pose_inv=base_pose_inv,
            device=device,
        )

    finally:
        os.unlink(urdf_path)

    best_pos_err = best["best_pos_err"]
    best_rot_err = best["best_rot_err"]
    best_se3_dist = best["best_se3_dist"]

    return {
        "sampled_pose_indices": sampled_pose_indices,
        "sampled_goal_poses": sampled_goal_poses,

        "best_seed_indices": best["best_seed_indices"],
        "best_joints": best["best_joints"],
        "fk_reached_poses_best": best["reached_best"],

        "best_pos_err_mean": best_pos_err.mean(),
        "best_rot_err_mean": best_rot_err.mean(),
        "best_se3_dist_mean": best_se3_dist.mean(),

        "best_pos_err_per_pose": best_pos_err,
        "best_rot_err_per_pose": best_rot_err,
        "best_se3_dist_per_pose": best_se3_dist,
    }
