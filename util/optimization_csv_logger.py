import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor


class OptimizationCSVLogger:
    """CSV logger for NRM morphology optimization.

    The CSV file is intended to be the source for later plotting and timelapse
    reconstruction. Therefore, optimization itself does not directly plot or
    record videos.

    Rows without IK/FK validation use empty strings in validation-related fields.
    """

    FIELDNAMES = [
        "iteration",
        "loss",
        "reachability_probability",

        "raw_morphology_json",
        "processed_morphology_json",

        "sampled_pose_indices_json",
        "sampled_goal_poses_json",

        "best_seed_indices_json",
        "best_joints_json",
        "fk_reached_poses_best_json",

        "best_pos_err_mean",
        "best_rot_err_mean",
        "best_se3_dist_mean",

        "best_pos_err_per_pose_json",
        "best_rot_err_per_pose_json",
        "best_se3_dist_per_pose_json",
    ]

    def __init__(self, root_dir: str | Path, run_time: str | None = None) -> None:
        """Create output/log_<run_time>.csv.

        Args:
            root_dir:
                Project root. The output directory will be root_dir/output.
            run_time:
                Optional timestamp string. If None, a new timestamp is created.
                This value is useful later as an identifier for plotting/video scripts.
        """
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_time = run_time or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"log_{self.run_time}.csv"

        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    @staticmethod
    def _to_json(value) -> str:
        """Convert tensors/lists/scalars to JSON for storing in one CSV cell."""
        if value is None:
            return ""

        if isinstance(value, Tensor):
            value = value.detach().cpu().tolist()

        return json.dumps(value)

    @staticmethod
    def _to_scalar(value) -> float | str:
        """Convert a scalar tensor/value to float, or empty string when missing."""
        if value is None:
            return ""

        if isinstance(value, Tensor):
            return float(value.detach().cpu().item())

        return float(value)

    def log_iteration(
        self,
        iteration: int,
        loss: float | Tensor | None,
        reachability_probability: float | Tensor | None,
        raw_morphology: Tensor,
        processed_morphology: Tensor,
        validation_data: dict | None = None,
    ) -> None:
        """Write one optimization iteration to the CSV.

        Args:
            iteration:
                Iteration index. Iteration 0 is usually the initial morphology.
            loss:
                NRM BCE loss. Can be None for rows without loss.
            reachability_probability:
                Mean sigmoid(logit) over all task goal poses.
            raw_morphology:
                [7, 3] morphology before normalize/squash/normalize.
            processed_morphology:
                [7, 3] morphology after normalize -> squash -> normalize.
            validation_data:
                Optional dictionary produced by nrm._run_validation(...).
                Missing validation fields are written as empty strings.
        """
        validation_data = validation_data or {}

        row = {
            "iteration": iteration,
            "loss": self._to_scalar(loss),
            "reachability_probability": self._to_scalar(reachability_probability),

            "raw_morphology_json": self._to_json(raw_morphology),
            "processed_morphology_json": self._to_json(processed_morphology),

            "sampled_pose_indices_json": self._to_json(validation_data.get("sampled_pose_indices")),
            "sampled_goal_poses_json": self._to_json(validation_data.get("sampled_goal_poses")),

            "best_seed_indices_json": self._to_json(validation_data.get("best_seed_indices")),
            "best_joints_json": self._to_json(validation_data.get("best_joints")),
            "fk_reached_poses_best_json": self._to_json(validation_data.get("fk_reached_poses_best")),

            "best_pos_err_mean": self._to_scalar(validation_data.get("best_pos_err_mean")),
            "best_rot_err_mean": self._to_scalar(validation_data.get("best_rot_err_mean")),
            "best_se3_dist_mean": self._to_scalar(validation_data.get("best_se3_dist_mean")),

            "best_pos_err_per_pose_json": self._to_json(validation_data.get("best_pos_err_per_pose")),
            "best_rot_err_per_pose_json": self._to_json(validation_data.get("best_rot_err_per_pose")),
            "best_se3_dist_per_pose_json": self._to_json(validation_data.get("best_se3_dist_per_pose")),
        }

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        self._file.close()

    def __enter__(self) -> "OptimizationCSVLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
