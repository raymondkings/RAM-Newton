import argparse
import random
from typing import Any

import torch

from morphology_sampler import sample_dof6_initial_morphologies


OPTIMIZATION_PARAMETER_CHOICES = ("ad", "alpha", "all")


def positive_int(value: str) -> int:
    value_int = int(value)
    if value_int <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value_int


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Main pipeline for task generation, morphology optimization, and validation."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Currently used for initial morphology sampling.",
    )
    parser.add_argument(
        "--num-reach",
        type=positive_int,
        default=500,
        help="Number of poses/regions to reach.",
    )
    parser.add_argument(
        "--num-avoid",
        type=positive_int,
        default=500,
        help="Number of poses/regions to avoid.",
    )
    parser.add_argument(
        "--optimization-parameters",
        type=str,
        choices=OPTIMIZATION_PARAMETER_CHOICES,
        default="ad",
        help='Which morphology parameters to optimize: "ad", "alpha", or "all".',
    )
    parser.add_argument(
        "--num-initial-samples",
        type=positive_int,
        default=1,
        help="Number of initial DOF=6 morphologies to sample.",
    )

    return parser.parse_args(argv)


def run_task_module(
    num_reach_poses: int,
    num_avoid_poses: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Call the Task Module.

    Expected output:
        poses_to_reach: shape [num_reach_poses, 9]
        poses_to_avoid: shape [num_avoid_poses, 9]
        collision_environment: agreed environment representation

    TODO: Replace generate_task with Jiyao's final function name if needed.
    """
    try:
        from task import generate_task
    except ImportError as exc:
        raise NotImplementedError(
            "Task module is not ready yet. TODO: create task.py and implement "
            "generate_task(num_reach_poses, num_avoid_poses, seed)."
        ) from exc

    return generate_task(
        num_reach_poses=num_reach_poses,
        num_avoid_poses=num_avoid_poses,
        seed=seed,
    )


def run_optimization_module(
    poses_to_reach: Any,
    poses_to_avoid: Any,
    initial_morphologies: torch.Tensor,
    optimization_parameters: str,
) -> torch.Tensor:
    """Call the Optimization Module.

    Expected input:
        poses_to_reach: shape [num_reach, 9]
        poses_to_avoid: shape [num_avoid, 9]
        initial_morphologies: Tensor [num_initial_samples, 7, 3]
        optimization_parameters: "ad", "alpha", or "all"

    Expected output:
        optimized_morphologies: Tensor [num_initial_samples, 7, 3]

    TODO: Replace optimize_morphology with Julian's final function name if needed.
    """
    try:
        from optimization import optimize_morphology
    except ImportError as exc:
        raise NotImplementedError(
            "Optimization module is not ready yet. TODO: create optimization.py and implement "
            "optimize_morphology(poses_to_reach, poses_to_avoid, initial_morphologies, "
            "optimization_parameters)."
        ) from exc

    return optimize_morphology(
        poses_to_reach=poses_to_reach,
        poses_to_avoid=poses_to_avoid,
        initial_morphologies=initial_morphologies,
        optimization_parameters=optimization_parameters,
    )


def run_validation_module(
    optimized_morphologies: torch.Tensor,
    collision_environment: Any,
    poses_to_reach: Any,
    poses_to_avoid: Any,
) -> Any:
    """Call the Validation Module.

    Expected input:
        optimized_morphologies: Tensor [num_initial_samples, 7, 3]
        collision_environment: environment representation from Task Module
        poses_to_reach: shape [num_reach, 9]
        poses_to_avoid: shape [num_avoid, 9]

    Expected output:
        validation result, e.g. dict with self-collision and environment-collision information.

    TODO: Confirm Raymond's final validate interface.
    """
    try:
        from validation import validate
    except ImportError as exc:
        raise NotImplementedError(
            "Validation module is not ready yet. TODO: create validation.py and implement validate(...)."
        ) from exc

    return validate(
        optimized_morphologies=optimized_morphologies,
        collision_environment=collision_environment,
        poses_to_reach=poses_to_reach,
        poses_to_avoid=poses_to_avoid,
    )


def run_pipeline(args: argparse.Namespace) -> Any:
    """Run the normal main pipeline."""
    set_global_seed(args.seed)

    print("Pipeline starts...")
    print(f"seed = {args.seed}")
    print(f"num_reach = {args.num_reach}, num_avoid = {args.num_avoid}")
    print(f"optimization_parameters = {args.optimization_parameters}")
    print(f"num_initial_samples = {args.num_initial_samples}")

    initial_morphologies = sample_dof6_initial_morphologies(
        num_initial_samples=args.num_initial_samples,
        seed=args.seed,
        device=None,
        analytically_solvable=False,
        cpu_output=True,
        as_list=False,
    )
    print(f"initial_morphologies.shape = {tuple(initial_morphologies.shape)}")

    poses_to_reach, poses_to_avoid, collision_environment = run_task_module(
        num_reach_poses=args.num_reach,
        num_avoid_poses=args.num_avoid,
        seed=args.seed,
    )

    optimized_morphologies = run_optimization_module(
        poses_to_reach=poses_to_reach,
        poses_to_avoid=poses_to_avoid,
        initial_morphologies=initial_morphologies,
        optimization_parameters=args.optimization_parameters,
    )
    print(f"optimized_morphologies.shape = {tuple(optimized_morphologies.shape)}")

    validation_result = run_validation_module(
        optimized_morphologies=optimized_morphologies,
        collision_environment=collision_environment,
        poses_to_reach=poses_to_reach,
        poses_to_avoid=poses_to_avoid,
    )

    print("Validation result:")
    print(validation_result)

    return validation_result


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()