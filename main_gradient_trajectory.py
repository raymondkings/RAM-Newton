import json
from pathlib import Path

import torch

from optim.nrm_trajectory import optimize_morphology_and_trajectory
from interface import Task
from task.environment import l_environment
from task.task_pose_sampler_trajectory_ver import (
    NUM_POSES,
    create_task,
)
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
IGNORED_TRAJECTORY_CONFIG_KEYS = (
    "candidate_batch_size",
    "distribution_batch_size",
    "num_iterations",
    "timelapse",
)


def parse_args():
    return build_arg_parser(
        "Main pipeline for task generation, morphology optimization, and validation.",
        DEFAULT_CONFIG,
    )


def main() -> None:
    args = parse_args()
    warn_ignored_config_keys(args, IGNORED_TRAJECTORY_CONFIG_KEYS, "trajectory")

    seed = int(getattr(args, "seed", 0))
    learning_rate_length = float(getattr(args, "learning_rate_length", 0.01))
    learning_rate_pose = float(
        getattr(args, "learning_rate_angle", learning_rate_length)
    )
    eval_interval = int(getattr(args, "eval_interval", 1))
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
            "[Info] plan_goal_start enabled: optimization uses all sampled poses; "
            "final planner uses only the optimized start and goal poses."
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

    # NOTE: for the updated candidate selection algorithm, the initial morphology is only used to get the link radius and the device
    morph, cached_csv_path, used_cache = resolve_initial_morphology(
        args, seed, initial_morphology_dof, device, Path(__file__).parent
    )

    print(
        f"[Info] Initial morphology params:\n{morph.params} \nlink_radius={morph.link_radius}"
    )

    if used_cache:
        optimized_morph, csv_path, optimization_timing = (
            morph,
            cached_csv_path,
            [0.0, 0.0],
        )
        optimized_trajectory = task.goal_poses
    else:
        optimized_morph, optimized_trajectory, csv_path, optimization_timing = (
            optimize_morphology_and_trajectory(
                morph=morph,
                task=task,
                optimization_parameters={
                    "learning_rate": learning_rate_length,
                    "learning_rate_pose": learning_rate_pose,
                    "logging": debug,
                    "eval_interval": eval_interval,
                    "random_seed": seed,
                    "number_random_seed": number_random_seed,
                    "percentage_poses": percentage_poses,
                },
            )
        )

    print(
        f"[Info] Optimized morphology params:\n{optimized_morph.params} \nlink_radius={optimized_morph.link_radius}"
    )
    print(f"[Info] Optimized trajectory poses: {optimized_trajectory.shape[0]}")
    print(f"[Info] Optimization CSV: {csv_path}")
    print(
        "[Info] Optimization timing "
        f"T=[t_opt_or_baseline, t_validation_curobo] seconds: {optimization_timing}"
    )
    print(f"[Benchmark] optim_seconds={optimization_timing[0]:.2f}")

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
