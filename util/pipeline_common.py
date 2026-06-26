"""Shared boilerplate for the main_*.py pipeline entry points.

Each main_*.py script differs in its task construction and optimizer call; this
module holds the surrounding plumbing (seeding, config loading, device setup,
cuRobo planning/animation, postprocessing, and the cached-morphology bootstrap)
that was previously duplicated verbatim across all of them.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from interface import Morphology, Task
from task.morphology_sampler import sample_initial_morphologies
from util.csv_log_reader import load_latest_optimized_morphology
from validation.curobo_planner import CuroboPlanner
from validation.render import animate_plan, render_scene


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


def load_config(path: Path | None, default_config: Path) -> argparse.Namespace:
    config_path = path or default_config
    with open(config_path) as f:
        data = json.load(f)
    return argparse.Namespace(**data)


def build_arg_parser(description: str, default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config JSON file (default: config.json next to this script).",
    )
    args = parser.parse_args()
    return load_config(args.config, default_config)


def warn_ignored_config_keys(
    args: argparse.Namespace, ignored_keys: tuple[str, ...], label: str
) -> None:
    present = [key for key in ignored_keys if hasattr(args, key)]
    if present:
        print(f"[Info] Ignored config keys in {label} pipeline: " + ", ".join(present))


def setup_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    return device


def resolve_initial_morphology(
    args: argparse.Namespace,
    seed: int,
    dof: int,
    device: torch.device,
    base_dir: Path,
) -> tuple[Morphology, Path | None, bool]:
    """Return the initial Morphology, plus the cached CSV path and a cache-hit flag.

    If `use_cached_optimized_morphology` is set, loads the latest optimized
    morphology from `cached_optimization_csv` (or `base_dir / "output"`) and
    returns it with `used_cache=True`, so the caller can skip optimization
    entirely. Otherwise samples a fresh initial morphology and returns
    `used_cache=False`.
    """
    use_cached_optimized_morphology = getattr(
        args, "use_cached_optimized_morphology", False
    )
    if use_cached_optimized_morphology:
        cached_optimization_csv = getattr(args, "cached_optimization_csv", None)
        csv_source = (
            Path(cached_optimization_csv)
            if cached_optimization_csv is not None
            else base_dir / "output"
        )
        morph, csv_path = load_latest_optimized_morphology(csv_source, device=device)
        return morph, csv_path, True

    initial_morphologies = sample_initial_morphologies(
        num_initial_samples=1,
        dof=dof,
        seed=seed,
        device=device,
        analytically_solvable=False,
        as_list=False,
    )
    return Morphology(params=initial_morphologies[0]), None, False


def run_plan(
    morph: Morphology,
    task: Task,
    ignore_ground: bool = False,
    ignore_obstacles: bool = False,
    debug: bool = False,
    visualize: bool = True,
) -> None:
    plan_start = time.perf_counter()
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
        print(f"[Benchmark] plan_seconds={time.perf_counter() - plan_start:.2f}")
        return

    n_goals = task.goal_poses.shape[0]
    result, _ = planner.plan_sequence(task.goal_poses, start_q)

    if not result.success:
        failed_at = result.failed_at_goal
        if result.path:
            print(
                f"[cuRobo] Executing partial plan: {len(result.path)} waypoints up to goal {failed_at}."
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
        print(f"[Benchmark] plan_seconds={time.perf_counter() - plan_start:.2f}")
        return

    print(f"\nSequence complete: {len(result.path)} waypoints through {n_goals} goals.")
    print(f"[Benchmark] plan_seconds={time.perf_counter() - plan_start:.2f}")
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
    """Run candidate-selection CSV plotting.

    This reuses the existing config['plot'] section.
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
