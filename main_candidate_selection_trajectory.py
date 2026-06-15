import argparse
import json
import random
from pathlib import Path

import torch

from task.morphology_sampler import sample_initial_morphologies
from optim.nrm_alpha_random_selection_trajectory import (
    optimize_morphology_and_trajectory,
)
from interface import Morphology, Task
from task.environment import l_environment
from task.task_pose_sampler_trajectory_ver import NUM_POSES, create_task
from validation.curobo_planner import CuroboPlanner
from validation.render import animate_plan, render_scene
from util.csv_log_reader import load_latest_optimized_morphology


DEFAULT_CONFIG = Path(__file__).parent / "config.json"
IGNORED_CONFIG_KEYS = (
    "num_iterations",
    "eval_interval",
    "timelapse",
)


def set_global_seed(seed: int) -> None:
    """Set random seeds used by this pipeline."""
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
        description=(
            "Candidate selection pipeline with alternating morphology/trajectory "
            "optimization."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file (default: config.json next to this script).",
    )
    args = parser.parse_args()
    return load_config(args.config)


def warn_ignored_config_keys(args: argparse.Namespace) -> None:
    ignored_keys = [key for key in IGNORED_CONFIG_KEYS if hasattr(args, key)]
    if ignored_keys:
        print(
            "[Info] Ignored config keys in candidate trajectory pipeline: "
            + ", ".join(ignored_keys)
        )


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

    if not planner.is_q_feasible(start_q):
        print(
            f"Start configuration is in collision (self or world) -- aborting.\n"
            f"  start_q = {start_q.tolist()}\n"
            "  Check that IK candidate poses are reachable for this morphology, or run with "
            "--ignore-ground / --ignore-obstacles to diagnose."
        )
        return

    n_goals = task.goal_poses.shape[0]
    result, _ = planner.plan_sequence(task.goal_poses, start_q)

    if not result.success:
        failed_at = result.failed_at_goal
        if result.path:
            print(
                f"[cuRobo] Executing partial plan: {len(result.path)} "
                f"waypoints up to goal {failed_at}."
            )
            if visualize:
                animate_plan(
                    morph,
                    task,
                    result.path,
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
        print(f"Animating -- {len(result.path)} frames ...")
        animate_plan(
            morph,
            task,
            result.path,
            curobo_planner=planner,
            failed_at_goal=None,
            best_ik_q=None,
        )


def run_postprocess(csv_path: Path, args: argparse.Namespace) -> None:
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
    warn_ignored_config_keys(args)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

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

    use_cached_optimized_morphology = getattr(
        args,
        "use_cached_optimized_morphology",
        False,
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
        optimized_trajectory = task.goal_poses
    else:
        initial_morphologies = sample_initial_morphologies(
            num_initial_samples=1,
            dof=initial_morphology_dof,
            seed=seed,
            device=device,
            analytically_solvable=False,
            as_list=False,
        )
        morph = Morphology(params=initial_morphologies[0])

    print(
        f"[Info] Initial morphology params:\n{morph.params} "
        f"\nlink_radius={morph.link_radius}"
    )

    if not use_cached_optimized_morphology:
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

        optimized_morph, optimized_trajectory, csv_path = (
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
