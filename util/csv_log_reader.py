import csv
import json
from pathlib import Path
from typing import Any

import torch

from interface import Morphology


def _parse_json_cell(value: str) -> Any | None:
    """Parse one JSON cell from the optimization CSV.

    Empty strings are treated as missing values.
    """
    if value is None or value == "":
        return None
    return json.loads(value)


def _parse_float_cell(value: str) -> float | None:
    """Parse one scalar CSV cell.

    Empty strings are treated as missing values.
    """
    if value is None or value == "":
        return None
    return float(value)


def read_optimization_csv(csv_path: str | Path) -> list[dict]:
    """Read optimization CSV rows into Python dictionaries.

    This helper keeps all JSON arrays as Python lists. Plot/timelapse utilities
    can convert them to torch.Tensor when needed.
    """
    csv_path = Path(csv_path)

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)

            parsed["iteration"] = int(row["iteration"])
            parsed["loss"] = _parse_float_cell(row.get("loss", ""))
            parsed["reachability_probability"] = _parse_float_cell(
                row.get("reachability_probability", "")
            )
            parsed["ik_success_pose_rate"] = _parse_float_cell(
                row.get("ik_success_pose_rate", "")
            )

            for key in [
                "raw_morphology_json",
                "processed_morphology_json",
                "sampled_pose_indices_json",
                "sampled_goal_poses_json",
                "best_joints_json",
                "fk_reached_poses_best_json",
                "best_pos_err_per_pose_json",
                "best_rot_err_per_pose_json",
                "best_se3_dist_per_pose_json",
            ]:
                if key in parsed:
                    parsed[key] = _parse_json_cell(row.get(key, ""))

            for key in [
                "best_pos_err_mean",
                "best_rot_err_mean",
                "best_se3_dist_mean",
            ]:
                if key in parsed:
                    parsed[key] = _parse_float_cell(row.get(key, ""))

            rows.append(parsed)

    return rows


def tensor_from_json_cell(value) -> torch.Tensor | None:
    """Convert a parsed JSON cell to tensor, or return None if missing."""
    if value is None:
        return None
    return torch.tensor(value, dtype=torch.float32)


def load_middle_start_q_from_last_validation(
    csv_path: str | Path,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    goal_pose_index: int = 1,
) -> torch.Tensor:
    """Load the joint solution for one original goal pose from validation CSV data.

    The CSV stores sampled_pose_indices_json in the same order as best_joints_json.
    By default, this picks the joint solution for task.goal_poses[1].
    """
    rows = read_optimization_csv(csv_path)

    validation_rows = [
        row
        for row in rows
        if row.get("best_joints_json") is not None
        and row.get("sampled_pose_indices_json") is not None
    ]

    if not validation_rows:
        raise ValueError(
            f"No validation row with best_joints_json and sampled_pose_indices_json "
            f"found in CSV: {csv_path}"
        )

    max_iteration = max(row["iteration"] for row in validation_rows)
    final_rows = [row for row in validation_rows if row["iteration"] == max_iteration]
    last_validation_row = final_rows[-1]
    best_joints = tensor_from_json_cell(last_validation_row["best_joints_json"])
    sampled_pose_indices = last_validation_row["sampled_pose_indices_json"]

    if best_joints is None:
        raise ValueError(
            f"Last validation row has empty best_joints_json in CSV: {csv_path}"
        )

    if best_joints.ndim != 2:
        raise ValueError(
            f"Expected best_joints shape [num_poses, dof], "
            f"got {tuple(best_joints.shape)}"
        )

    num_poses = best_joints.shape[0]
    if num_poses == 0:
        raise ValueError("best_joints_json contains zero joint solutions.")

    if len(sampled_pose_indices) != num_poses:
        raise ValueError(
            f"sampled_pose_indices length ({len(sampled_pose_indices)}) does not "
            f"match best_joints rows ({num_poses}) in CSV: {csv_path}"
        )

    try:
        selected_idx = [int(idx) for idx in sampled_pose_indices].index(goal_pose_index)
    except ValueError as exc:
        raise ValueError(
            f"Goal pose index {goal_pose_index} was not sampled in selected "
            f"validation row. sampled_pose_indices={sampled_pose_indices}"
        ) from exc

    start_q = best_joints[selected_idx].to(dtype=dtype)

    if device is not None:
        start_q = start_q.to(device)

    print(
        f"[Info] Loaded start_q from selected validation row: "
        f"iteration={last_validation_row['iteration']}, "
        f"best_joints_shape={tuple(best_joints.shape)}, "
        f"goal_pose_index={goal_pose_index}, "
        f"selected_sampled_pose_index={selected_idx}"
    )

    return start_q


def load_latest_optimized_morphology(
    csv_source: str | Path,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    link_radius: float = 0.025,
) -> tuple[Morphology, Path]:
    """Load the latest optimized Morphology from a CSV file or output directory.

    If csv_source is a directory, log_*.csv files are searched newest-first until a
    file with processed_morphology_json is found.
    """
    csv_source = Path(csv_source)
    if csv_source.is_dir():
        candidates = sorted(
            csv_source.glob("log_*.csv"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            raise FileNotFoundError(
                f"No optimization CSV files found in directory: {csv_source}"
            )
        csv_path = None
        morphology_rows = None
        for candidate in reversed(candidates):
            rows = read_optimization_csv(candidate)
            candidate_rows = [
                row for row in rows if row.get("processed_morphology_json") is not None
            ]
            if candidate_rows:
                csv_path = candidate
                morphology_rows = candidate_rows
                break
        if csv_path is None or morphology_rows is None:
            raise ValueError(
                f"No row with processed_morphology_json found in any optimization CSV under: {csv_source}"
            )
    elif csv_source.is_file():
        csv_path = csv_source
        rows = read_optimization_csv(csv_path)
        morphology_rows = [
            row for row in rows if row.get("processed_morphology_json") is not None
        ]
    else:
        raise FileNotFoundError(f"CSV path does not exist: {csv_source}")

    if not morphology_rows:
        raise ValueError(
            f"No row with processed_morphology_json found in optimization CSV: {csv_path}"
        )

    # Select by highest iteration, not file order. The candidate-selection optimizer
    # (nrm_alpha_random_selection) writes the finally selected morphology with
    # iteration=2 but does NOT place it last — later rows are iteration=0/1 candidates.
    # The trajectory optimizers (continuous/ste) write the final morphology with the
    # largest iteration, which is also the last row, so max-iteration matches both.
    max_iteration = max(row["iteration"] for row in morphology_rows)
    final_rows = [row for row in morphology_rows if row["iteration"] == max_iteration]
    latest_row = final_rows[-1]
    processed = tensor_from_json_cell(latest_row["processed_morphology_json"])
    if processed is None:
        raise ValueError(
            f"Selected morphology row has empty processed_morphology_json in CSV: {csv_path}"
        )

    params = processed.to(dtype=dtype)
    if device is not None:
        params = params.to(device)

    morph = Morphology(params=params, link_radius=link_radius)
    print(
        f"[Info] Loaded optimized morphology from CSV: {csv_path} "
        f"(iteration={latest_row['iteration']})"
    )
    return morph, csv_path
