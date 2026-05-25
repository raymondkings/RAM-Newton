import csv
import json
from pathlib import Path
from typing import Any

import torch


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


# This is for loading a specific joint config from the IK solver. (the middle one, can be changed to set a specific one)
# TODO: if we want to visualize the final joint config, you can pick a specific pose and its joint config
def load_middle_start_q_from_last_validation(
    csv_path: str | Path,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load middle joint solution from the last CSV row that contains validation data.

    The selected row is the last row with non-empty best_joints_json.
    If best_joints has shape [P, dof], choose index P // 2.
    Example:
        P = 8 -> index 4
        P = 3 -> index 1
    """
    rows = read_optimization_csv(csv_path)

    validation_rows = [row for row in rows if row.get("best_joints_json") is not None]

    if not validation_rows:
        raise ValueError(
            f"No validation row with best_joints_json found in CSV: {csv_path}"
        )

    last_validation_row = validation_rows[-1]
    best_joints = tensor_from_json_cell(last_validation_row["best_joints_json"])

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

    middle_idx = num_poses // 2
    start_q = best_joints[middle_idx].to(dtype=dtype)

    if device is not None:
        start_q = start_q.to(device)

    print(
        f"[Info] Loaded start_q from last validation row: "
        f"iteration={last_validation_row['iteration']}, "
        f"best_joints_shape={tuple(best_joints.shape)}, "
        f"selected_index={middle_idx}"
    )

    return start_q
