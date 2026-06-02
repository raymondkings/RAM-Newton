import argparse
import json
import random
from pathlib import Path

import torch

from task.morphology_sampler import sample_dof6_initial_morphologies
from optim.nrm_alpha_random_selection_5_7 import optimize_morphology
from interface import Morphology, Task
from task.environment import l_environment
from validation.curobo_planner import CuroboPlanner, interpolate_path
from validation.render import animate_plan, render_scene
from util.csv_log_reader import load_latest_optimized_morphology

# target selecting
# from task.target1 import create_task
# from task.target1plus import create_task
from task.task_pose_sampler import START_POSE, create_start_goal_poses, create_task
# from task.target2 import create_task


DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def set_global_seed(seed: int) -> None:
    """Set random seeds used by this pipeline.
    Later modules can also reuse this seed if they support deterministic behavior.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def load_config(path: Path | None = None) -> argparse.Namespace:
    config_path = path or DEFAULT_CONFIG
    with open(config_path) as f:
        data = json.load(f)
    return argparse.Namespace(**data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main pipeline for task generation, morphology optimization, and validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file (default: config.json next to main.py).",
    )
    args = parser.parse_args()
    return load_config(args.config)


def run_plan(
    morph: Morphology,
    task: Task,
    ignore_ground: bool = False,
    ignore_obstacles: bool = False,
    debug: bool = False,
    visualize: bool = True,
) -> None:
    dtype = morph.params.dtype
    device = morph.params.device
    planner = CuroboPlanner(
        morph,
        task,
        device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )
    start_q = (
        task.start_q.to(dtype)
        if task.start_q is not None
        else planner.default_start_q().to(dtype)
    )

    task.start_q = start_q

    print(f"[Info] Start configuration: {start_q.tolist()}")

    if not planner.check_start_feasibility(start_q):
        raise RuntimeError(
            f"Start configuration is in collision (self or world) — aborting.\n"
            f"  start_q = {start_q.tolist()}\n"
            "  Check that IK candidate poses are reachable for this morphology, or run with "
            "--ignore-ground / --ignore-obstacles to diagnose."
        )

    n_goals = task.goal_poses.shape[0]
    result, final_q = planner.plan_sequence(task.goal_poses, start_q)

    if not result.success:
        failed_at = result.failed_at_goal
        if failed_at is not None:
            print(f"[cuRobo] Planning failed at goal {failed_at}/{n_goals}.")
        else:
            print("[cuRobo] Planning failed.")
        if result.path:
            print(
                f"[cuRobo] Executing partial plan: {len(result.path)} waypoints up to goal {failed_at}."
            )
            if visualize:
                dense = interpolate_path(result.path, step=0.03)
                print(
                    f"Animating partial plan — {len(dense)} frames (failure at goal {failed_at}/{n_goals}) ..."
                )
                animate_plan(
                    morph,
                    task,
                    dense,
                    curobo_planner=planner,
                    failed_at_goal=failed_at,
                    best_ik_q=result.best_ik_q,
                )
        elif debug and visualize:
            print("Rendering static scene for debugging.")
            render_scene(
                morph,
                task,
                curobo_planner=planner,
                failed_at_goal=failed_at,
                best_ik_q=result.best_ik_q,
                start_q=start_q,
            )
        return

    print(f"\nSequence complete: {len(result.path)} waypoints through {n_goals} goals.")
    if visualize:
        dense = interpolate_path(result.path, step=0.03)
        print(f"Animating — {len(dense)} frames ...")
        animate_plan(
            morph,
            task,
            dense,
            curobo_planner=planner,
            failed_at_goal=None,
            best_ik_q=None,
        )

        # To visualize the static scene instead of animating, replace the 3 lines above with:
        # render_scene(morph, task, curobo_planner=planner, q=start_q)


# def run_postprocess(csv_path: Path, task: Task, args: argparse.Namespace) -> None:
#     """Run optional CSV-based plotting and timelapse generation."""
#     plot_cfg = getattr(args, "plot", {})
#     if isinstance(plot_cfg, dict) and plot_cfg.get("enabled", True):
#         from postprocess.plot import create_plots_from_csv

#         output_dir = plot_cfg.get("output_dir", "output/figures")
#         paths = create_plots_from_csv(csv_path, output_dir=output_dir)
#         for path in paths:
#             print(f"[postprocess] Plot saved: {path}")

#     tl_cfg = getattr(args, "timelapse", None)
#     if isinstance(tl_cfg, dict) and tl_cfg.get("enabled", False):
#         from postprocess.timelapse import create_timelapse_from_csv

#         video_path = create_timelapse_from_csv(csv_path, task, tl_cfg)
#         print(f"[postprocess] Timelapse saved: {video_path}")


def run_postprocess(csv_path: Path, task: Task, args: argparse.Namespace) -> None:
    """Run candidate-selection CSV plotting. This is for the candidate selection algorithm

    This reuses the existing config['plot'] section.
    The old timelapse postprocess is intentionally skipped because the new CSV
    stores final candidate rows, not an optimization trajectory.
    """
    plot_cfg = getattr(args, "plot", {})

    if isinstance(plot_cfg, dict) and plot_cfg.get("enabled", True):
        from postprocess.plot_candidate_selection import (
            create_candidate_selection_plots,
        )

        output_dir = plot_cfg.get("output_dir", "output/figures")

        paths = create_candidate_selection_plots(
            csv_path=csv_path,
            output_dir=output_dir,
        )

        for path in paths:
            print(f"[postprocess] Candidate plot saved: {path}")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print("[Info] Config:", json.dumps(vars(args), indent=2))

    plan_goal_start = bool(getattr(args, "plan_goal_start", False))
    if plan_goal_start:
        print(
            "[Info] plan_goal_start enabled: optimization uses all sampled poses; "
            "final planner uses only start pose and first-path max-alpha goal pose."
        )

    task = Task(
        environment=l_environment(),
        goal_poses=create_task(
            seed=args.seed,
            start_pose=START_POSE,
            device=device,
        ),
        reachable_region=None,
        start_q=None,
    )

    # NOTE: for the updated candidate selection algorithm, the initial morphology is only used to get the link radius and the device
    # possible TODO: update the structure to decude redundancy
    use_cached_optimized_morphology = getattr(
        args, "use_cached_optimized_morphology", False
    )
    cached_optimization_csv = getattr(args, "cached_optimization_csv", None)

    if use_cached_optimized_morphology:
        csv_source = (
            Path(cached_optimization_csv)
            if cached_optimization_csv is not None
            else Path(__file__).parent / "output"
        )
        optimized_morph, csv_path = load_latest_optimized_morphology(
            csv_source,
            device=device,
        )
        morph = optimized_morph
    else:
        initial_morphologies = sample_dof6_initial_morphologies(
            num_initial_samples=1,
            seed=args.seed,
            device=device,
            analytically_solvable=False,
            as_list=False,
        )

        morph = Morphology(params=initial_morphologies[0])

    print(
        f"[Info] Initial morphology params:\n{morph.params} \nlink_radius={morph.link_radius}"
    )

    if not use_cached_optimized_morphology:
        optimized_morph, csv_path = optimize_morphology(
            morph=morph,
            task=task,
            optimization_parameters={
                "num_iterations": args.num_iterations,
                "learning_rate": args.learning_rate_length,
                "logging": args.debug,
                "eval_interval": args.eval_interval,
                "random_seed": args.seed,
                "number_random_seed": args.number_random_seed,
                "percentage_poses": args.percentage_poses,
                "candidate_batch_size": getattr(args, "candidate_batch_size", 64),
                "distribution_batch_size": getattr(
                    args, "distribution_batch_size", 128
                ),
                "ignore_ground": args.ignore_ground,
                "ignore_obstacles": args.ignore_obstacles,
            },
        )

    # # for testing the alpha(different learning rate, different input)
    # optimized_morph, csv_path = optimize_morphology(
    #     morph=morph,
    #     task=task,
    #     optimization_parameters={
    #         "num_iterations": args.num_iterations,
    #         "learning_rate_angle": args.learning_rate_angle,
    #         "learning_rate_length": args.learning_rate_length,
    #         "logging": args.debug,
    #         "eval_interval": args.eval_interval,
    #         "random_seed": args.seed,
    #         "number_random_seed": args.number_random_seed,
    #         "percentage_poses": args.percentage_poses,
    #         "ignore_ground": args.ignore_ground,
    #         "ignore_obstacles": args.ignore_obstacles,
    #     },
    # )

    print(
        f"[Info] Optimized morphology params:\n{optimized_morph.params} \nlink_radius={optimized_morph.link_radius}"
    )
    print(f"[Info] Optimization CSV: {csv_path}")

    run_postprocess(Path(csv_path), task, args)

    plan_task = task
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

    run_plan(
        optimized_morph,
        plan_task,
        ignore_ground=args.ignore_ground,
        ignore_obstacles=args.ignore_obstacles,
        debug=args.debug,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
