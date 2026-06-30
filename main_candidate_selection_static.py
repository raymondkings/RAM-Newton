import json
from pathlib import Path

from optim.nrm_alpha_random_selection import optimize_morphology
from interface import Task
from task.environment import l_environment
from task.task_pose_sampler import (
    START_POSE,
    create_start_goal_poses,
    create_task_pose_sets,
)
from util.optimization_timing import OptimizationTiming
from util.pipeline_common import (
    build_arg_parser,
    report_optimization_timing,
    resolve_initial_morphology,
    run_plan,
    run_postprocess,
    set_global_seed,
    setup_device,
)


DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def parse_args():
    return build_arg_parser(
        "Main pipeline for task generation, morphology optimization, and validation.",
        DEFAULT_CONFIG,
    )


def main() -> None:
    args = parse_args()
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
    )

    task = Task(
        environment=l_environment(),
        goal_poses=optimizer_goal_poses,
        reachable_region=None,
        start_q=None,
    )
    planner_task = Task(
        environment=task.environment,
        goal_poses=planner_goal_poses,
        reachable_region=task.reachable_region,
        start_q=task.start_q,
    )
    print(
        "[Info] Task poses: "
        f"optimizer={task.goal_poses.shape[0]}, "
        f"planner_main_path={planner_task.goal_poses.shape[0]}"
    )

    # NOTE: for the updated candidate selection algorithm, the initial morphology is only used to get the link radius and the device
    morph, cached_csv_path, used_cache = resolve_initial_morphology(
        args, args.seed, initial_morphology_dof, device, Path(__file__).parent
    )

    print(
        f"[Info] Initial morphology params:\n{morph.params} \nlink_radius={morph.link_radius}"
    )

    if used_cache:
        optimized_morph, csv_path, optimization_timing = (
            morph,
            cached_csv_path,
            OptimizationTiming(0.0, 0.0),
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
            optimization_parameters["log_root_dir"] = getattr(args, "log_root_dir")

        optimized_morph, csv_path, optimization_timing, candidates = (
            optimize_morphology(
                morph=morph,
                task=task,
                optimization_parameters=optimization_parameters,
            )
        )

    print(
        f"[Info] Optimized morphology params:\n{optimized_morph.params} \nlink_radius={optimized_morph.link_radius}"
    )
    print(f"[Info] Optimization CSV: {csv_path}")
    report_optimization_timing(optimization_timing)

    if used_cache or bool(getattr(args, "csv_logging", True)):
        run_postprocess(Path(csv_path), args)
    else:
        print("[Info] csv_logging disabled: skipping CSV-based postprocessing.")

    if plan_goal_start:
        plan_task = Task(
            environment=task.environment,
            goal_poses=create_start_goal_poses(
                start_pose=START_POSE,
                device=device,
            ),
            reachable_region=task.reachable_region,
            start_q=task.start_q,
        )
    else:
        plan_task = planner_task

    successes = []
    for idx, (candidate_morph, ik_success_rate) in enumerate(candidates):
        if len(candidates) > 1:
            print(
                f"[Info] Planning candidate {idx + 1}/{len(candidates)} "
                f"(ik_success_pose_rate={ik_success_rate})"
            )
        success = run_plan(
            candidate_morph,
            plan_task,
            ignore_ground=args.ignore_ground,
            ignore_obstacles=args.ignore_obstacles,
            debug=args.debug,
            visualize=args.visualize and idx == 0,
        )
        successes.append(success)

    if len(candidates) > 1:
        print(
            f"[Info] success@{num_plan_candidates}: "
            f"tried={len(candidates)} any_success={any(successes)}"
        )


if __name__ == "__main__":
    main()
