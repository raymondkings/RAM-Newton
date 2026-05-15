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
            parsed["reachability_probability"] = _parse_float_cell(row.get("reachability_probability", ""))

            for key in [
                "raw_morphology_json",
                "normalized_morphology_json",
                "processed_morphology_json",
                "sampled_pose_indices_json",
                "sampled_goal_poses_json",
                "best_seed_indices_json",
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
