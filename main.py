import argparse
import json
import random
from pathlib import Path

import torch

from task.morphology_sampler import sample_dof6_initial_morphologies
from optim.nrm import optimize_morphology
from interface import Morphology, Task
from task.environment import l_environment
from task.target import create_task
from validation.curobo_planner import CuroboPlanner, interpolate_path
from validation.render import animate_plan, render_scene

DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def find_self_collision_free_start_q(
    morph: Morphology,
    task: Task,
    device: torch.device,
    ignore_ground: bool = False,
    ignore_obstacles: bool = False,
) -> torch.Tensor:
    import os
    from util.kinematics import build_robot_dict, build_scene, IK

    robot_dict, urdf_path = build_robot_dict(morph)
    base_pose_inv = torch.linalg.inv(task.environment.base_pose.to(device))
    scene = build_scene(
        task,
        base_pose_inv,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )

    ik_solver = IK(
        robot_dict=robot_dict,
        scene=scene,
        num_seeds=32,
        max_batch_size=1,
        self_collision_check=True,
    )

    # Home pose: directly above base (base is at z=0.2, so 0.55 gives 35 cm clearance)
    home_pose = torch.eye(4, device=device)
    home_pose[2, 3] = 0.55

    joints, success = ik_solver.solve(home_pose.unsqueeze(0), base_pose_inv, device)

    try:
        os.unlink(urdf_path)
    except OSError:
        pass

    if success[0]:
        print("[Info] Self Collision-free start configuration found via IK.")
        return joints[0].to(morph.params.dtype)

    print("[Warning] Home-pose IK failed — falling back to zero start configuration.")
    return torch.zeros(morph.n_links - 1, dtype=morph.params.dtype, device=device)


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
    n_joints = morph.n_links - 1
    dtype = morph.params.dtype
    start_q = (
        task.start_q.to(dtype)
        if task.start_q is not None
        else torch.zeros(n_joints, dtype=dtype)
    )

    planner = CuroboPlanner(
        morph,
        task,
        morph.params.device,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )
    result, final_q = planner.plan_sequence(task.goal_poses, start_q)

    if final_q is None or not result.success:
        print("\nPlanning failed.")
        if debug and visualize:
            print("Rendering static scene for debugging.")
            render_scene(morph, task, curobo_planner=planner)
        return

    print(
        f"\nSequence complete: {len(result.path)} waypoints through {task.goal_poses.shape[0]} goals."
    )
    if visualize:
        dense = interpolate_path(result.path, step=0.03)
        print(f"Animating — {len(dense)} frames ...")
        animate_plan(morph, task, dense, curobo_planner=planner)

        # To only visualize the static scene without animation, comment out the 3 lines above and call the following funct
        # render_scene(morph, task, curobo_planner=planner, q=start_q)


def run_postprocess(csv_path: Path, task: Task, args: argparse.Namespace) -> None:
    """Run optional CSV-based plotting and timelapse generation."""
    plot_cfg = getattr(args, "plot", {})
    if isinstance(plot_cfg, dict) and plot_cfg.get("enabled", True):
        from postprocess.plot import create_plots_from_csv

        output_dir = plot_cfg.get("output_dir", "output/figures")
        paths = create_plots_from_csv(csv_path, output_dir=output_dir)
        for path in paths:
            print(f"[postprocess] Plot saved: {path}")

    tl_cfg = getattr(args, "timelapse", None)
    if isinstance(tl_cfg, dict) and tl_cfg.get("enabled", False):
        from postprocess.timelapse import create_timelapse_from_csv

        video_path = create_timelapse_from_csv(csv_path, task, tl_cfg)
        print(f"[postprocess] Timelapse saved: {video_path}")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print("[Info] Config:", json.dumps(vars(args), indent=2))

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
        goal_poses=create_task(),
        reachable_region=None,
        start_q=start_q,
    )

    morph = Morphology(params=initial_morphologies[0])

    print(
        f"[Info] Initial morphology params:\n{morph.params} \nlink_radius={morph.link_radius}"
    )

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

    # from util.csv_log_reader import load_middle_start_q_from_last_validation
    # task.start_q = load_middle_start_q_from_last_validation(csv_path=csv_path, device=optimized_morph.params.device)
    # print(task.start_q)

    task.start_q = find_self_collision_free_start_q(
        optimized_morph,
        task,
        device,
        ignore_ground=args.ignore_ground,
        ignore_obstacles=args.ignore_obstacles,
    )

    print(f"[Info] : Start Configuration : {task.start_q.tolist()}")

    run_plan(
        optimized_morph,
        task,
        ignore_ground=args.ignore_ground,
        ignore_obstacles=args.ignore_obstacles,
        debug=args.debug,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
