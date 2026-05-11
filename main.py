import argparse
import json
import random
from pathlib import Path
import math

import torch

from task.morphology_sampler import sample_dof6_initial_morphologies
from optim.nrm import optimize_morphology
from interface import Morphology, Task
from interface.environment import Box, Capsule, Environment, Sphere
from task.environment import l_environment
from task.target import simple_targets
from validation.plan import plan_to_poses
from validation.planner import interpolate_path
from validation.render import animate_plan

DEFAULT_CONFIG = Path(__file__).parent / "config.json"

def _to_cpu(morph: Morphology, task: Task) -> tuple[Morphology, Task]:
    cpu_morph = Morphology(params=morph.params.cpu(), link_radius=morph.link_radius)
    cpu_obstacles = []
    for obs in task.environment.obstacles:
        if isinstance(obs, Box):
            cpu_obstacles.append(Box(center=obs.center.cpu(), half_extents=obs.half_extents.cpu(), rotation=obs.rotation.cpu()))
        elif isinstance(obs, Capsule):
            cpu_obstacles.append(Capsule(center=obs.center.cpu(), half_height=obs.half_height, radius=obs.radius, rotation=obs.rotation.cpu()))
        else:
            cpu_obstacles.append(Sphere(center=obs.center.cpu(), radius=obs.radius))
    env = task.environment
    cpu_task = Task(
        environment=Environment(obstacles=cpu_obstacles, base_pose=env.base_pose.cpu()),
        goal_poses=task.goal_poses.cpu(),
        reachable_region=task.reachable_region,
        start_q=task.start_q.cpu() if task.start_q is not None else None,
    )
    return cpu_morph, cpu_task


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

def run_plan(morph: Morphology, task: Task) -> None:
    result, start_q, final_q = plan_to_poses(morph, task, task.goal_poses)

    if final_q is None or not result.success:
        print("\nPlanning failed. Skipping animation.")
        return

    print(f"\nSequence complete: {len(result.path)} waypoints through {task.goal_poses.shape[0]} goals.")
    dense = interpolate_path(result.path, step=0.03)
    print(f"Animating — {len(dense)} frames ...")
    animate_plan(morph, task, dense)


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

    cpu_morph, cpu_task = _to_cpu(optimized_morph, task)
    run_plan(cpu_morph, cpu_task)


if __name__ == "__main__":
    main()