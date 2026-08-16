import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

from core import Morphology
from logutils.csv_logger import OptimizationCSVLogger


def _raise_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_size_limit()


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

    This helper keeps all JSON arrays as Python lists. Plotting utilities
    can convert them to torch.Tensor when needed.
    """
    csv_path = Path(csv_path)

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)

            # Parse each cell by the type OptimizationCSVLogger declares for it,
            # so the reader stays in sync with the writer's columns automatically.
            parsed["iteration"] = int(row["iteration"])

            for key in OptimizationCSVLogger.scalar_fieldnames():
                if key in parsed:
                    parsed[key] = _parse_float_cell(row.get(key, ""))

            for key in OptimizationCSVLogger.json_fieldnames():
                if key in parsed:
                    parsed[key] = _parse_json_cell(row.get(key, ""))

            rows.append(parsed)

    return rows


def tensor_from_json_cell(value) -> torch.Tensor | None:
    """Convert a parsed JSON cell to tensor, or return None if missing."""
    if value is None:
        return None
    return torch.tensor(value, dtype=torch.float32)


def load_latest_optimized_morphology(
    csv_source: str | Path,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    link_radius: float = 0.025,
) -> tuple[Morphology, Path]:
    """Load the latest optimized Morphology from a CSV file or output directory.

    If csv_source is a directory, each run's <run_time>/morphology_history.csv is
    searched newest-first until a file with processed_morphology_json is found.
    """
    csv_source = Path(csv_source)
    if csv_source.is_dir():
        candidates = sorted(
            csv_source.glob("*/morphology_history.csv"), key=lambda p: p.stat().st_mtime
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
    # (candidate_selection.static) writes the finally selected morphology with
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
