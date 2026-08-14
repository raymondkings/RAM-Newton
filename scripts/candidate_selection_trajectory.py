import json

from core import Task
from methods.candidate_selection.trajectory import (
    optimize_morphology_and_trajectory,
)
from paths import DEFAULT_CONFIG, PROJECT_ROOT
from pipeline.common import (
    build_arg_parser,
    build_trajectory_plan_task,
    cached_optimization_result,
    finalize_and_report,
    resolve_initial_morphology,
    run_candidate_plans,
    set_global_seed,
    setup_device,
    warn_ignored_config_keys,
)
from tasks.environment import l_environment
from tasks.sampling.fixed_alpha_candidates import (
    DEFAULT_DIRECT_PRESAMPLING_BATCH_SIZE,
    DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
)
from tasks.sampling.trajectory_pose_sampler import NUM_POSES, create_task

IGNORED_CONFIG_KEYS = (
    "num_iterations",
    "eval_interval",
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
    csv_logging = bool(getattr(args, "csv_logging", True))

    set_global_seed(seed)
    initial_morphology_dof = int(getattr(args, "dof", 6))

    device = setup_device()

    print("[Info] Config:", json.dumps(vars(args), indent=2))

    num_plan_candidates = int(getattr(args, "num_plan_candidates", 1))
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
        start_q=None,
    )
    print(f"[Info] Task trajectory poses: {task.goal_poses.shape[0]}")

    morph, cached_csv_path, used_cache = resolve_initial_morphology(
        args, seed, initial_morphology_dof, device, PROJECT_ROOT
    )

    print(
        f"[Info] Initial morphology params:\n{morph.params} "
        f"\nlink_radius={morph.link_radius}"
    )

    if used_cache:
        optimized_morph, csv_path, optimization_timing = cached_optimization_result(
            morph, cached_csv_path
        )
        optimized_trajectory = task.goal_poses
        candidates = [(optimized_morph, optimized_trajectory, None)]
    else:
        optimization_parameters = {
            "learning_rate": learning_rate_length,
            "learning_rate_pose": learning_rate_pose,
            "logging": debug,
            "csv_logging": csv_logging,
            "random_seed": seed,
            "number_random_seed": number_random_seed,
            "percentage_poses": percentage_poses,
            "candidate_batch_size": getattr(args, "candidate_batch_size", 64),
            "distribution_batch_size": getattr(args, "distribution_batch_size", 128),
            "direct_presampling_batch_size": getattr(
                args,
                "direct_presampling_batch_size",
                DEFAULT_DIRECT_PRESAMPLING_BATCH_SIZE,
            ),
            "dynamic_rejection_batch_size": getattr(
                args,
                "dynamic_rejection_batch_size",
                DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
            ),
            "ignore_ground": ignore_ground,
            "ignore_obstacles": ignore_obstacles,
            "num_plan_candidates": num_plan_candidates,
        }
        if hasattr(args, "num_alpha_candidates"):
            optimization_parameters["num_alpha_candidates"] = args.num_alpha_candidates
        if hasattr(args, "log_root_dir"):
            optimization_parameters["log_root_dir"] = args.log_root_dir

        (
            optimized_morph,
            optimized_trajectory,
            csv_path,
            optimization_timing,
            candidates,
        ) = optimize_morphology_and_trajectory(
            morph=morph,
            task=task,
            optimization_parameters=optimization_parameters,
        )

    finalize_and_report(
        optimized_morph,
        csv_path,
        optimization_timing,
        args,
        used_cache,
        optimized_trajectory=optimized_trajectory,
    )

    run_candidate_plans(
        [
            (
                candidate_morph,
                build_trajectory_plan_task(task, candidate_trajectory, plan_goal_start),
                ik_success_rate,
            )
            for candidate_morph, candidate_trajectory, ik_success_rate in candidates
        ],
        num_plan_candidates=num_plan_candidates,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
        debug=debug,
        visualize=visualize,
    )


if __name__ == "__main__":
    main()
