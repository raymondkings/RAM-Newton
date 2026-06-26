import json
from pathlib import Path

import torch

from optim.nrm_alpha_random_selection_trajectory import (
    optimize_morphology_and_trajectory,
)
from interface import Task
from task.environment import l_environment
from task.task_pose_sampler_trajectory_ver import NUM_POSES, create_task
from util.pipeline_common import (
    build_arg_parser,
    resolve_initial_morphology,
    run_plan,
    run_postprocess,
    set_global_seed,
    setup_device,
    warn_ignored_config_keys,
)


DEFAULT_CONFIG = Path(__file__).parent / "config.json"
IGNORED_CONFIG_KEYS = (
    "num_iterations",
    "eval_interval",
    "timelapse",
)


def parse_args():
    return build_arg_parser(
        "Candidate selection pipeline with alternating morphology/trajectory "
        "optimization.",
        DEFAULT_CONFIG,
    )


def main() -> None:
    args = parse_args()
    warn_ignored_config_keys(args, IGNORED_CONFIG_KEYS, "candidate trajectory")

    seed = int(getattr(args, "seed", 0))
    learning_rate_length = float(getattr(args, "learning_rate_length", 0.01))
    learning_rate_pose = float(
        getattr(args, "learning_rate_angle", learning_rate_length)
    )
    number_random_seed = int(getattr(args, "number_random_seed", 32))
    percentage_poses = float(getattr(args, "percentage_poses", 1))
    ignore_ground = bool(getattr(args, "ignore_ground", False))
    ignore_obstacles = bool(getattr(args, "ignore_obstacles", False))
    visualize = bool(getattr(args, "visualize", True))
    debug = bool(getattr(args, "debug", True))

    set_global_seed(seed)
    initial_morphology_dof = int(getattr(args, "dof", 6))

    device = setup_device()

    print("[Info] Config:", json.dumps(vars(args), indent=2))

    plan_goal_start = bool(getattr(args, "plan_goal_start", False))
    if plan_goal_start:
        print(
            "[Info] plan_goal_start enabled: optimization uses the full optimized "
            "trajectory; final planner uses only optimized start and goal poses."
        )

    trajectory_poses = create_task(
        num_poses=int(getattr(args, "num_poses", NUM_POSES)),
        device=device,
    )

    task = Task(
        environment=l_environment(),
        goal_poses=trajectory_poses,
        reachable_region=None,
        start_q=None,
    )
    print(f"[Info] Task trajectory poses: {task.goal_poses.shape[0]}")

    morph, cached_csv_path, used_cache = resolve_initial_morphology(
        args, seed, initial_morphology_dof, device, Path(__file__).parent
    )

    print(
        f"[Info] Initial morphology params:\n{morph.params} "
        f"\nlink_radius={morph.link_radius}"
    )

    if used_cache:
        optimized_morph, csv_path, optimization_timing = (
            morph,
            cached_csv_path,
            [0.0, 0.0],
        )
        optimized_trajectory = task.goal_poses
    else:
        optimization_parameters = {
            "learning_rate": learning_rate_length,
            "learning_rate_pose": learning_rate_pose,
            "logging": debug,
            "random_seed": seed,
            "number_random_seed": number_random_seed,
            "percentage_poses": percentage_poses,
            "candidate_batch_size": getattr(args, "candidate_batch_size", 64),
            "distribution_batch_size": getattr(args, "distribution_batch_size", 128),
            "ignore_ground": ignore_ground,
            "ignore_obstacles": ignore_obstacles,
        }
        if hasattr(args, "num_alpha_candidates"):
            optimization_parameters["num_alpha_candidates"] = getattr(
                args,
                "num_alpha_candidates",
            )

        optimized_morph, optimized_trajectory, csv_path, optimization_timing = (
            optimize_morphology_and_trajectory(
                morph=morph,
                task=task,
                optimization_parameters=optimization_parameters,
            )
        )

    print(
        f"[Info] Optimized morphology params:\n{optimized_morph.params} "
        f"\nlink_radius={optimized_morph.link_radius}"
    )
    print(f"[Info] Optimized trajectory poses: {optimized_trajectory.shape[0]}")
    print(f"[Info] Optimization CSV: {csv_path}")
    print(
        "[Info] Optimization timing "
        f"T=[t_opt_or_baseline, t_validation_curobo] seconds: {optimization_timing}"
    )

    run_postprocess(Path(csv_path), args)

    if plan_goal_start:
        plan_task = Task(
            environment=task.environment,
            goal_poses=torch.stack(
                [optimized_trajectory[0], optimized_trajectory[-1]],
                dim=0,
            ),
            reachable_region=task.reachable_region,
            start_q=task.start_q,
        )
    else:
        plan_task = Task(
            environment=task.environment,
            goal_poses=optimized_trajectory,
            reachable_region=task.reachable_region,
            start_q=task.start_q,
        )

    run_plan(
        optimized_morph,
        plan_task,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
        debug=debug,
        visualize=visualize,
    )


if __name__ == "__main__":
    main()
