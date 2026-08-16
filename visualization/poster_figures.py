#!/usr/bin/env python3
"""
Poster figures for the NRM-Newton candidate-selection pipeline.

Each grid/env/plan command launches a Viser viewer — open the printed URL and
screenshot it (no headless capture in this repo).

  grid   Fig 1     curated grid of candidate morphologies (no env).
  env    Fig 1     the task: wall + goal arc, no robot.
  plan   Figs 2-4  one morphology attempting the task; the goal arc is colored by
                   the cuRobo *sequence* planner: green = reached in order,
                   orange = first failure, red = not attempted. Ghost/frames off.
  scan   headless  plan several morphs and print reached/total (no viewer) to
                   pick a clean fail (Fig 2) or partial (Fig 3).

Note: "reached 34/52" = waypoints chained before the first failure, NOT a claim
that the red poses are individually unreachable (per-pose IK is a separate metric).

Examples (CSV = output/<run>/morphology_history.csv from
main_candidate_selection_static.py with csv_logging enabled):

  python visualization/poster_figures.py grid --dof 6 --n 20
  python visualization/poster_figures.py env
  python visualization/poster_figures.py scan --source csv --csv CSV --indices 2,9,16
  python visualization/poster_figures.py plan --source naive --dof 6 --seed 1    # Fig 2
  python visualization/poster_figures.py plan --source csv --csv CSV --index 9   # Fig 3
  python visualization/poster_figures.py plan --source csv --csv CSV --index 2   # Fig 4
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

import newton
import torch
import warp as wp
from interface import Morphology, Task
from task.environment import l_environment
from task.morphology_sampler import sample_initial_morphologies
from task.task_pose_sampler import START_POSE, create_task_pose_sets
from util.csv_log_reader import read_optimization_csv
from util.kinematics import build_scene, compute_link_world_poses
from util.mdh import add_robot_to_builder
from util.pipeline_common import run_plan, set_global_seed, setup_device
from validation.curobo_planner import CuroboPlanner
from validation.ground import add_ground_grid_to_viser
from validation.render import (
    add_curobo_scene_to_viser,
    add_goals_to_viser,
    make_goal_pose_axes,
    render_scene,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_CACHE = PROJECT_ROOT / "initial_candidates"
LINK_RADIUS = 0.025


# --------------------------------------------------------------------------- #
# Task + morphology sources
# --------------------------------------------------------------------------- #
def build_planner_task(task_seed: int, device) -> Task:
    """Recreate the exact environment + goal arc the main pipeline plans against."""
    _optimizer_goal_poses, planner_goal_poses = create_task_pose_sets(
        seed=task_seed, start_pose=START_POSE, device=device
    )
    return Task(
        environment=l_environment(),
        goal_poses=planner_goal_poses,
        reachable_region=None,
        start_q=None,
    )


def naive_morph(dof: int, seed: int, device) -> Morphology:
    """An un-optimized initial morphology (Fig 2 — 'before our method')."""
    params = sample_initial_morphologies(
        num_initial_samples=1,
        dof=dof,
        seed=seed,
        device=device,
        analytically_solvable=False,
        as_list=False,
    )[0]
    return Morphology(params=params, link_radius=LINK_RADIUS)


def csv_candidate_rows(csv_path: Path) -> list[dict]:
    """Logged candidates (with a processed morphology) from an optimization CSV."""
    rows = read_optimization_csv(csv_path)
    return [r for r in rows if r.get("processed_morphology_json") is not None]


def morph_from_row(row: dict, device) -> Morphology:
    params = torch.tensor(
        row["processed_morphology_json"], dtype=torch.float32, device=device
    )
    return Morphology(params=params, link_radius=LINK_RADIUS)


def pool_morphs(dof: int, seed: int, device) -> torch.Tensor:
    """Cached pre-optimization candidate pool, shape [M, dof+1, 3]."""
    path = CANDIDATE_CACHE / f"DOF{dof}_seed{seed}" / "candidates.json"
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in CANDIDATE_CACHE.glob("DOF*")))
        raise FileNotFoundError(f"No candidate cache at {path}. Available: {available}")
    data = json.loads(path.read_text())
    return torch.tensor(data["morphologies"], dtype=torch.float32, device=device)


# --------------------------------------------------------------------------- #
# Planning helpers (shared by scan + plan)
# --------------------------------------------------------------------------- #
def _resolve_start_q(planner: CuroboPlanner, task: Task):
    if task.start_q is not None:
        return task.start_q
    return planner.default_start_q()


def plan_morph(
    morph,
    task,
    ignore_ground,
    ignore_obstacles,
    max_attempts=5,
    num_ik_seeds=32,
    num_trajopt_seeds=4,
):
    """Build a cuRobo planner and plan the goal sequence.

    Returns (result, start_q, planner). result is None if no feasible start
    config exists for this morphology.
    """
    device = morph.params.device
    planner = CuroboPlanner(
        morph,
        task,
        device,
        num_ik_seeds=num_ik_seeds,
        num_trajopt_seeds=num_trajopt_seeds,
        ignore_ground=ignore_ground,
        ignore_obstacles=ignore_obstacles,
    )
    try:
        start_q = _resolve_start_q(planner, task)
    except RuntimeError as exc:
        print(f"    no feasible start config: {exc}")
        return None, None, planner
    result, _final_q = planner.plan_sequence(
        task.goal_poses, start_q, max_attempts=max_attempts
    )
    return result, start_q, planner


def _reached_of_total(result, n_total: int) -> int:
    """Number of goals reached in order (goals 0..failed-1)."""
    return n_total if result.success else int(result.failed_at_goal)


def _free(planner) -> None:
    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Viewer utilities
# --------------------------------------------------------------------------- #
def _print_url(port: int, label: str = "viewer") -> None:
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "localhost"
    print(
        f"[{label}] running — local: http://localhost:{port}   remote: http://{ip}:{port}"
    )
    print(f"[{label}] open the URL, frame your shot, screenshot. Ctrl-C here to stop.")


def _spin(viewer) -> None:
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        viewer.close()


def _round_hundred(x: int) -> int:
    return int(round(x / 100.0) * 100)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_grid(args, device) -> None:
    """Fig 1 hero: a curated grid of candidate skeletons, no env/goals."""
    morphs = pool_morphs(args.dof, args.pool_seed, device)
    total = morphs.shape[0]
    n = min(args.n, total)
    # Evenly spaced across the pool (cached in alpha-signature order) -> variety.
    idx = torch.linspace(0, total - 1, n).round().long().tolist()
    cols = args.cols or math.ceil(math.sqrt(n))
    spacing = args.spacing

    builder = newton.ModelBuilder()
    for k, gi in enumerate(idx):
        morph = Morphology(params=morphs[gi], link_radius=LINK_RADIUS)
        poses = compute_link_world_poses(morph).detach().cpu()
        row, col = divmod(k, cols)
        poses[:, 0, 3] += col * spacing
        poses[:, 1, 3] += -row * spacing
        add_robot_to_builder(builder, morph, poses, label=f"robot_{k}")

    model = builder.finalize()
    state = model.state()

    viewer = newton.viewer.ViewerViser(port=args.port, share=False, verbose=False)
    viewer.set_model(model)
    if args.ground:
        add_ground_grid_to_viser(
            viewer._server, grid_size=max(4.0, cols * spacing), divisions=8
        )
    viewer._server.gui.add_markdown(
        f"**Candidate pool (Fig 1)** — {n} of {total} DOF-{args.dof} skeletons, "
        f"pre-optimization.\n\nPoster caption: "
        f"“~{_round_hundred(total)} candidate skeletons”"
    )

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    viewer.end_frame()
    _print_url(args.port, "grid")
    _spin(viewer)


def cmd_env(args, device) -> None:
    """Environment only: wall + neutral goal arc, no robot (the task setup)."""
    task = build_planner_task(args.task_seed, device)
    scene = build_scene(
        task, ignore_ground=args.ignore_ground, ignore_obstacles=args.ignore_obstacles
    )

    # Minimal model so the viewer has something to set; the wall + goals are
    # added as viser scene geometry, no robot.
    builder = newton.ModelBuilder()
    builder.add_shape_sphere(
        body=-1,
        xform=wp.transform_identity(),
        radius=0.01,
        as_site=True,
        color=wp.vec3(0.1, 0.1, 0.1),
    )
    model = builder.finalize()
    state = model.state()

    viewer = newton.viewer.ViewerViser(port=args.port, share=False, verbose=False)
    viewer.set_model(model)
    if args.ground:
        add_ground_grid_to_viser(viewer._server, grid_size=4.0, divisions=8)
    add_curobo_scene_to_viser(viewer._server, scene)  # yellow wall
    if args.show_goals:
        add_goals_to_viser(viewer._server, task, "unknown")  # neutral grey arc
    viewer._server.gui.add_markdown(
        f"**The task** — trace {task.goal_poses.shape[0]} end-effector poses "
        "over the wall (no robot shown)."
    )

    viewer.begin_frame(0.0)
    viewer.log_state(state)
    if args.show_frames and args.show_goals:
        gb, ge, gc = make_goal_pose_axes(task.goal_poses, axis_length=0.08)
        viewer.log_lines("/goals/frames", gb, ge, gc, width=0.04)
    viewer.end_frame()
    _print_url(args.port, "env")
    _spin(viewer)


def cmd_scan(args, device) -> None:
    """Headless: plan several morphologies and print reached/total."""
    task = build_planner_task(args.task_seed, device)
    n_total = int(task.goal_poses.shape[0])
    print(
        f"[scan] goal arc = {n_total} waypoints | "
        f"ignore_ground={args.ignore_ground} ignore_obstacles={args.ignore_obstacles}"
    )

    entries: list[tuple[str, Morphology]] = []
    if args.source == "naive":
        dofs = [int(x) for x in args.dofs.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        for d in dofs:
            for s in seeds:
                entries.append((f"naive dof{d} seed{s}", naive_morph(d, s, device)))
    else:  # csv
        if args.csv is None:
            raise SystemExit("scan --source csv requires --csv PATH")
        all_rows = list(enumerate(csv_candidate_rows(args.csv)))
        if args.indices:
            want = {int(x) for x in args.indices.split(",")}
            all_rows = [(i, r) for i, r in all_rows if i in want]
        for i, row in all_rows:
            ik = row.get("ik_success_pose_rate")
            ik_str = f"{ik:.3f}" if ik is not None else "n/a"
            label = f"csv[{i:02d}] marker{row['iteration']} ikpose={ik_str}"
            entries.append((label, morph_from_row(row, device)))

    results: list[tuple[str, int | None]] = []
    for label, morph in entries:
        print(f"\n[scan] === {label} ===")
        result, _start_q, planner = plan_morph(
            morph,
            task,
            args.ignore_ground,
            args.ignore_obstacles,
            args.max_attempts,
            args.num_ik_seeds,
            args.num_trajopt_seeds,
        )
        if result is None:
            results.append((label, None))
            _free(planner)
            continue
        reached = _reached_of_total(result, n_total)
        status = "ALL GREEN" if result.success else f"fail@goal {result.failed_at_goal}"
        print(
            f"[scan] {label}: reached {reached}/{n_total} "
            f"({100 * reached / n_total:.0f}%) — {status}"
        )
        results.append((label, reached))
        _free(planner)

    print("\n[scan] ===================== summary =====================")
    for label, reached in sorted(results, key=lambda x: (x[1] is None, x[1] or 0)):
        if reached is None:
            print(f"  {label:34s}  no feasible start")
        else:
            print(
                f"  {label:34s}  {reached:3d}/{n_total}  ({100 * reached / n_total:3.0f}%)"
            )
    print(
        "\n[scan] pick: Fig 2 = a low % with a feasible start; "
        "Fig 3 = a partial (~30-70%); Fig 4 = 100%."
    )


def _load_plan_morph(args, device) -> Morphology:
    if args.source == "naive":
        print(f"[plan] naive morphology: dof{args.dof} seed{args.seed} (un-optimized)")
        return naive_morph(args.dof, args.seed, device)

    if args.csv is None:
        raise SystemExit("plan --source csv requires --csv PATH")
    rows = csv_candidate_rows(args.csv)
    if args.index is not None:
        row = rows[args.index]
    elif args.marker is not None:
        matches = [r for r in rows if r["iteration"] == args.marker]
        if not matches:
            raise SystemExit(f"No candidate with marker {args.marker} in {args.csv}")
        row = matches[0]
    else:
        markers = [r["iteration"] for r in rows]
        target = 2 if 2 in markers else max(markers)  # selected candidate
        row = next(r for r in rows if r["iteration"] == target)
    ik = row.get("ik_success_pose_rate")
    print(
        f"[plan] csv candidate: marker={row['iteration']} "
        f"ik_success_pose_rate={ik} (marker 2 = selected)"
    )
    return morph_from_row(row, device)


def cmd_plan(args, device) -> None:
    """Figs 2-4: plan one morphology and render the green/orange/red goal arc."""
    task = build_planner_task(args.task_seed, device)
    morph = _load_plan_morph(args, device)
    n_total = int(task.goal_poses.shape[0])

    if args.animate:
        # Loop the trajectory in the viewer (same path the main pipeline takes).
        ok = run_plan(
            morph,
            task,
            ignore_ground=args.ignore_ground,
            ignore_obstacles=args.ignore_obstacles,
            debug=True,
            visualize=True,
        )
        print(f"[plan] success={ok}")
        return

    # Static poster shot: plan once, then render a single frame.
    result, start_q, planner = plan_morph(
        morph,
        task,
        args.ignore_ground,
        args.ignore_obstacles,
        args.max_attempts,
        args.num_ik_seeds,
        args.num_trajopt_seeds,
    )
    if result is None:
        print("[plan] no feasible start config; nothing meaningful to render.")
        return

    reached = _reached_of_total(result, n_total)
    status = (
        "ALL GREEN" if result.success else f"failed at goal {result.failed_at_goal}"
    )
    print(
        f"[plan] reached {reached}/{n_total} ({100 * reached / n_total:.0f}%) — {status}"
    )

    if result.success:
        display_q = result.path[-1] if result.path else start_q
        failed_at = None
    else:
        display_q = result.best_ik_q if result.best_ik_q is not None else start_q
        failed_at = result.failed_at_goal

    # cuRobo returns joint configs on CPU; move to the morph's device.
    if display_q is not None:
        display_q = display_q.to(morph.params.device)

    # Build the robot at rest (q=None) and pose it to display_q via start_q, so
    # the *solid* arm shows its best attempt (fail) or final config (success).
    # Ghost + reference frames are off by default for a clean poster still.
    render_scene(
        morph,
        task,
        port=args.port,
        curobo_planner=planner,
        q=None,
        start_q=display_q,
        failed_at_goal=failed_at,
        best_ik_q=result.best_ik_q,
        show_ghost=args.show_ghost,
        show_frames=args.show_frames,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser, planning: bool = False) -> None:
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--task-seed", type=int, default=0)
    if planning:
        p.add_argument(
            "--ignore-ground", dest="ignore_ground", action="store_true", default=True
        )
        p.add_argument("--no-ignore-ground", dest="ignore_ground", action="store_false")
        p.add_argument(
            "--ignore-obstacles",
            dest="ignore_obstacles",
            action="store_true",
            default=False,
        )
        p.add_argument("--max-attempts", type=int, default=5)
        p.add_argument("--num-ik-seeds", type=int, default=32)
        p.add_argument("--num-trajopt-seeds", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("grid", help="Fig 1 hero: curated grid of skeletons (no env).")
    _add_common(pg)
    pg.add_argument("--dof", type=int, default=6)
    pg.add_argument("--pool-seed", type=int, default=0)
    pg.add_argument("--n", type=int, default=20)
    pg.add_argument("--cols", type=int, default=0, help="0 = auto (~sqrt).")
    pg.add_argument("--spacing", type=float, default=1.5)
    pg.add_argument("--ground", action="store_true", help="Add a faint ground grid.")

    pe = sub.add_parser("env", help="Environment only: wall + goal arc, no robot.")
    pe.add_argument("--port", type=int, default=8080)
    pe.add_argument("--task-seed", type=int, default=0)
    pe.add_argument(
        "--ignore-ground", dest="ignore_ground", action="store_true", default=True
    )
    pe.add_argument("--no-ignore-ground", dest="ignore_ground", action="store_false")
    pe.add_argument(
        "--ignore-obstacles",
        dest="ignore_obstacles",
        action="store_true",
        default=False,
    )
    pe.add_argument("--ground", action="store_true", help="Add a faint ground grid.")
    pe.add_argument(
        "--no-goals",
        dest="show_goals",
        action="store_false",
        default=True,
        help="Hide the goal arc (wall only).",
    )
    pe.add_argument(
        "--frames",
        dest="show_frames",
        action="store_true",
        default=False,
        help="Show goal coordinate frames.",
    )

    pscan = sub.add_parser(
        "scan", help="Headless: plan several morphs, print reached/total."
    )
    _add_common(pscan, planning=True)
    pscan.add_argument("--source", choices=["naive", "csv"], required=True)
    pscan.add_argument("--dofs", default="6,7", help="naive: comma DOFs, e.g. 6,7")
    pscan.add_argument("--seeds", default="0,1,2,3,4", help="naive: comma seeds")
    pscan.add_argument("--csv", type=Path, default=None)
    pscan.add_argument(
        "--indices", default=None, help="csv: comma row indices to scan (default: all)."
    )

    pplan = sub.add_parser("plan", help="Figs 2-4: plan one morph, render the arc.")
    _add_common(pplan, planning=True)
    pplan.add_argument("--source", choices=["naive", "csv"], required=True)
    pplan.add_argument("--dof", type=int, default=7)
    pplan.add_argument("--seed", type=int, default=0)
    pplan.add_argument("--csv", type=Path, default=None)
    pplan.add_argument(
        "--marker", type=int, default=None, help="CSV iteration marker (2=selected)."
    )
    pplan.add_argument(
        "--index", type=int, default=None, help="CSV row index override."
    )
    pplan.add_argument(
        "--animate", action="store_true", help="Loop the trajectory instead of a still."
    )
    pplan.add_argument(
        "--ghost",
        dest="show_ghost",
        action="store_true",
        default=False,
        help="Show the translucent best-IK ghost robot (off by default).",
    )
    pplan.add_argument(
        "--frames",
        dest="show_frames",
        action="store_true",
        default=False,
        help="Show goal / EEF / origin coordinate frames (off by default).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_global_seed(args.task_seed)
    device = setup_device()

    dispatch = {
        "grid": cmd_grid,
        "env": cmd_env,
        "scan": cmd_scan,
        "plan": cmd_plan,
    }
    dispatch[args.cmd](args, device)


if __name__ == "__main__":
    main()
