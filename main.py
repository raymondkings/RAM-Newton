import argparse
import json
import random
from pathlib import Path
import math

import torch

from task.morphology_sampler import sample_dof6_initial_morphologies
from optim.nrm import optimize_morphology
from interface import Morphology, Task
from task.environment import l_environment
from task.target import simple_targets
from validation.curobo_planner import CuroboPlanner, interpolate_path
from validation.render import animate_plan, render_scene

DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def set_global_seed(seed: int) -> None:
    """Set random seeds used by this pipeline.

    Currently this mainly controls initial morphology sampling.
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

def run_plan(morph: Morphology, task: Task, ignore_env_collision: bool = False) -> None:
    n_joints = morph.n_links - 1
    dtype = morph.params.dtype
    start_q = task.start_q.to(dtype) if task.start_q is not None else torch.zeros(n_joints, dtype=dtype)

    planner = CuroboPlanner(morph, task, morph.params.device, ignore_env_collision=ignore_env_collision)
    result, final_q = planner.plan_sequence(task.goal_poses, start_q)

    if final_q is None or not result.success:
        print("\nPlanning failed. Rendering static scene for debugging.")
        render_scene(morph, task, curobo_planner=planner)
        return

    print(f"\nSequence complete: {len(result.path)} waypoints through {task.goal_poses.shape[0]} goals.")
    dense = interpolate_path(result.path, step=0.03)
    print(f"Animating — {len(dense)} frames ...")
    animate_plan(morph, task, dense, curobo_planner=planner)


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print("Config:", json.dumps(vars(args), indent=2))

    initial_morphologies = sample_dof6_initial_morphologies(
        num_initial_samples=1,
        seed=args.seed,
        device=device,
        analytically_solvable=False,
        as_list=False,
    )
    
    start_q = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    task = Task(
        environment=l_environment(),
        goal_poses=simple_targets(),
        reachable_region=None,
        start_q=start_q,
    )

    # get initial sampled morphology
    morph = Morphology(params=initial_morphologies[0])

    print(f"Initial morphology params:\n{morph.params} \nlink_radius={morph.link_radius}")
    
    # optimize morphology for the task
    if args.optimize:
        optimized_morph = optimize_morphology(
            morph=morph,
            task=task,
            optimization_parameters = {
                "num_iterations": 100,
                "learning_rate": 0.01,
                "logging": args.debug,
            },
        )
        print(f"Optimized morphology params:\n{optimized_morph.params} \nlink_radius={optimized_morph.link_radius}")
    else:
        optimized_morph = morph

    run_plan(optimized_morph, task, ignore_env_collision=getattr(args, "ignore_env_collision", False))


if __name__ == "__main__":
    main()