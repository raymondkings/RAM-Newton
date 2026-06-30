import csv
import json
from datetime import datetime
from pathlib import Path

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
        "ik_success_pose_rate",
        "best_joints_json",
        "fk_reached_poses_best_json",
        "best_pos_err_mean",
        "best_rot_err_mean",
        "best_se3_dist_mean",
        "best_pos_err_per_pose_json",
        "best_rot_err_per_pose_json",
        "best_se3_dist_per_pose_json",
    ]

    def __init__(
        self,
        root_dir: str | Path,
        run_time: str | None = None,
        output_subdir: str | None = "output",
        enabled: bool = True,
    ) -> None:
        """Create <output_dir>/<run_time>/morphology_history.csv.

        Args:
            root_dir:
                Project root (or other base directory). The output directory
                will be root_dir/output_subdir, or root_dir itself when
                output_subdir is None.
            run_time:
                Optional timestamp string. If None, a new timestamp is created.
                This value is useful later as an identifier for plotting/video scripts,
                and names the per-run subfolder holding this run's CSVs.
            output_subdir:
                Subdirectory under root_dir to write into. Pass None to write
                directly into root_dir (e.g. when root_dir is already a
                dedicated per-run output directory, as in the benchmark harness).
            enabled:
                When False, skip creating the output directory/file entirely
                and make log_iteration() a no-op. Lets callers turn off the
                per-iteration disk writes (a real cost over many iterations
                or many benchmark subprocesses) while keeping csv_path
                pointing at the path that would have been used.
        """
        self.enabled = enabled
        self.root_dir = Path(root_dir)
        self.output_dir = (
            self.root_dir / output_subdir if output_subdir else self.root_dir
        )

        self.run_time = run_time or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.run_time
        self.csv_path = self.run_dir / "morphology_history.csv"

        self._file = None
        self._writer = None
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
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
        if not self.enabled:
            return

        validation_data = validation_data or {}

        row = {
            "iteration": iteration,
            "loss": self._to_scalar(loss),
            "reachability_probability": self._to_scalar(reachability_probability),
            "raw_morphology_json": self._to_json(raw_morphology),
            "processed_morphology_json": self._to_json(processed_morphology),
            "sampled_pose_indices_json": self._to_json(
                validation_data.get("sampled_pose_indices")
            ),
            "sampled_goal_poses_json": self._to_json(
                validation_data.get("sampled_goal_poses")
            ),
            "ik_success_pose_rate": self._to_scalar(
                validation_data.get("ik_success_pose_rate")
            ),
            "best_joints_json": self._to_json(validation_data.get("best_joints")),
            "fk_reached_poses_best_json": self._to_json(
                validation_data.get("fk_reached_poses_best")
            ),
            "best_pos_err_mean": self._to_scalar(
                validation_data.get("best_pos_err_mean")
            ),
            "best_rot_err_mean": self._to_scalar(
                validation_data.get("best_rot_err_mean")
            ),
            "best_se3_dist_mean": self._to_scalar(
                validation_data.get("best_se3_dist_mean")
            ),
            "best_pos_err_per_pose_json": self._to_json(
                validation_data.get("best_pos_err_per_pose")
            ),
            "best_rot_err_per_pose_json": self._to_json(
                validation_data.get("best_rot_err_per_pose")
            ),
            "best_se3_dist_per_pose_json": self._to_json(
                validation_data.get("best_se3_dist_per_pose")
            ),
        }

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> "OptimizationCSVLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class InternalOptimizationCSVLogger:
    """Small CSV logger for optimizer-internal aggregate metrics.

    Writes <suffix>.csv next to parent_csv_path, i.e. into the same per-run
    subfolder as the main OptimizationCSVLogger CSV.
    """

    def __init__(
        self,
        parent_csv_path: str | Path,
        fieldnames: list[str],
        suffix: str = "convergence_metrics",
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        parent_csv_path = Path(parent_csv_path)
        self.csv_path = parent_csv_path.parent / f"{suffix}.csv"
        self.fieldnames = fieldnames

        self._file = None
        self._writer = None
        if self.enabled:
            self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
            self._writer.writeheader()
            self._file.flush()

    @staticmethod
    def _to_cell(value) -> float | int | str:
        if value is None:
            return ""
        if isinstance(value, Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
            return json.dumps(value.tolist())
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float | str):
            return value
        return json.dumps(value)

    def log_row(self, **values) -> None:
        if not self.enabled:
            return

        row = {field: self._to_cell(values.get(field)) for field in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> "InternalOptimizationCSVLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
