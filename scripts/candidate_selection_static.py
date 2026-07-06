import json

from core import Task
from methods.candidate_selection.static import optimize_morphology
from paths import DEFAULT_CONFIG, PROJECT_ROOT
from pipeline.common import (
    build_arg_parser,
    cached_optimization_result,
    finalize_and_report,
    resolve_initial_morphology,
    run_candidate_plans,
    set_global_seed,
    setup_device,
    warn_ignored_config_keys,
)

DEFAULT_CONFIG = Path(__file__).parent / "config.json"
IGNORED_CONFIG_KEYS = (
    "learning_rate_angle",
    "timelapse",
)


def parse_args():
    return build_arg_parser(
        "Main pipeline for task generation, morphology optimization, and validation.",
        DEFAULT_CONFIG,
    )


def main() -> None:
    args = parse_args()
    warn_ignored_config_keys(args, IGNORED_CONFIG_KEYS, "candidate static")
    set_global_seed(args.seed)
    initial_morphology_dof = int(getattr(args, "dof", 6))

    device = setup_device()

    print("[Info] Config:", json.dumps(vars(args), indent=2))

    num_plan_candidates = int(getattr(args, "num_plan_candidates", 1))
    plan_goal_start = bool(getattr(args, "plan_goal_start", False))
    if plan_goal_start:
        print(
            "[Info] plan_goal_start enabled: optimization uses all sampled poses; "
            "final planner uses only start pose and first-path max-alpha goal pose."
        )

    optimizer_goal_poses, planner_goal_poses = create_task_pose_sets(
        seed=args.seed,
        start_pose=START_POSE,
        device=device,
        num_samples=int(getattr(args, "num_samples", NUM_SAMPLES)),
        num_line_samples=int(getattr(args, "num_line_samples", NUM_LINE_SAMPLES)),
        num_extra_paths=int(getattr(args, "num_extra_paths", NUM_EXTRA_PATHS)),
        repeat=int(getattr(args, "repeat_start_goal", REPEAT_START_GOAL)),
    )

    task = Task(
        environment=l_environment(),
        goal_poses=optimizer_goal_poses,
        start_q=None,
    )
    planner_task = Task(
        environment=task.environment,
        goal_poses=planner_goal_poses,
        start_q=task.start_q,
    )
    print(
        "[Info] Task poses: "
        f"optimizer={task.goal_poses.shape[0]}, "
        f"planner_main_path={planner_task.goal_poses.shape[0]}"
    )

    # NOTE: for the updated candidate selection algorithm, the initial morphology is only used to get the link radius and the device
    morph, cached_csv_path, used_cache = resolve_initial_morphology(
        args, args.seed, initial_morphology_dof, device, PROJECT_ROOT
    )

    print(
        f"[Info] Initial morphology params:\n{morph.params} \nlink_radius={morph.link_radius}"
    )

    if used_cache:
        optimized_morph, csv_path, optimization_timing = cached_optimization_result(
            morph, cached_csv_path
        )
        candidates = [(optimized_morph, None)]
    else:
        optimization_parameters = {
            "num_iterations": args.num_iterations,
            "learning_rate": args.learning_rate_length,
            "logging": args.debug,
            "csv_logging": bool(getattr(args, "csv_logging", True)),
            "eval_interval": args.eval_interval,
            "random_seed": args.seed,
            "number_random_seed": args.number_random_seed,
            "percentage_poses": args.percentage_poses,
            "candidate_batch_size": getattr(args, "candidate_batch_size", 64),
            "distribution_batch_size": getattr(args, "distribution_batch_size", 128),
            "ignore_ground": args.ignore_ground,
            "ignore_obstacles": args.ignore_obstacles,
            "num_plan_candidates": num_plan_candidates,
        }
        if hasattr(args, "log_root_dir"):
            optimization_parameters["log_root_dir"] = args.log_root_dir

        optimized_morph, csv_path, optimization_timing, candidates = (
            optimize_morphology(
                morph=morph,
                task=task,
                optimization_parameters=optimization_parameters,
            )
        )

    finalize_and_report(
        optimized_morph, csv_path, optimization_timing, args, used_cache
    )

    if plan_goal_start:
        plan_task = Task(
            environment=task.environment,
            goal_poses=create_start_goal_poses(
                start_pose=START_POSE,
                device=device,
            ),
            start_q=task.start_q,
        )
    else:
        plan_task = planner_task

    run_candidate_plans(
        [
            (candidate_morph, plan_task, ik_success_rate)
            for candidate_morph, ik_success_rate in candidates
        ],
        num_plan_candidates=num_plan_candidates,
        ignore_ground=args.ignore_ground,
        ignore_obstacles=args.ignore_obstacles,
        debug=args.debug,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
