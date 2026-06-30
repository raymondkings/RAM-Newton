"""
Benchmark the optimization+planning pipeline across N random seeds,
with and without collision avoidance.

How it works:
    Each algorithm entry in benchmarks/config.json names an optim_algo, which selects
    which main_*.py pipeline to benchmark: candidate_selection_static
    (today's heuristic over static poses, main_candidate_selection_static.py),
    candidate_selection_trajectory (our heuristic, candidate-selection
    morphology+trajectory search, main_candidate_selection_trajectory.py), or
    gradient_trajectory (the alternating gradient-based morphology+trajectory
    optimizer baseline, main_gradient_trajectory.py). Each optim_algo has its
    own sweepable sampler params (see ENTRY_POINTS).

    The sweep is the cartesian product of seeds x sampler-param combos x
    CONDITIONS (obstacle collision on/off; ground collision is always
    ignored). For each combination, build_config() patches the project's
    config.json with
    the seed, condition flags, and sampler overrides, writes it to
    <output_dir>/configs/, and run_single() shells out to
    `uv run python <optim_algo's script> --config <that file>` as a fresh
    subprocess (so a crash or hang in one run can't take down the sweep).

    Each result row is appended to the results CSV immediately and the file
    is flushed, so the run is crash-safe and resumable: pass --resume
    <csv> (or rerun against the same --output-dir) to skip any
    (seed, condition, *sampler_values) tuple already present via
    load_completed().

    The algorithms to benchmark, and the sampler-param tuples to sweep for
    each, are loaded from benchmarks/config.json next to this file: a list of
    "algorithms" entries, each pinned to its own optim_algo, sampler param
    tuples ("configs"), and output_dir. The script always runs every entry
    in that list sequentially, one after another, in a single invocation.

    Once all runs for an algorithm finish (or with --replot against its
    --output-dir of existing CSVs), generate_report() prints a per-condition
    success/failure breakdown and _make_figure() renders it to a PNG — one
    subfigure per sampler-param combo when more than one was swept. This
    report+figure step repeats once per algorithm.

    For candidate_selection_static and candidate_selection_trajectory,
    num_plan_candidates (abbrev "k") is itself a swept sampler param. With
    num_plan_candidates=1 (the default), "success" is single-candidate
    success as before. With num_plan_candidates=k>1, the corresponding
    main_*.py script runs full planning for each of the optimizer's top-k
    validated candidates and the process succeeds if any of them plans
    successfully — i.e. the reported rate is success@k, not single-candidate
    success. Sweep both a k=1 row and a k>1 row in benchmarks/config.json to
    compare them side by side; don't read a k>1 row as if it were the
    single-candidate rate. gradient_trajectory (the gradient-descent
    baseline) has no discrete candidate pool to rank, so it has no
    num_plan_candidates param and always reports single-candidate success.

    To compare our heuristic against the alternating gradient baseline, bundle
    both trajectory optim_algo values as separate "algorithms" entries in
    benchmarks/config.json so they run back-to-back into their own output_dirs, then
    compare the resulting CSVs/figures.

Usage:
    uv run python benchmarks/benchmark.py                        # run every algorithm in benchmarks/config.json, one after another
    uv run python benchmarks/benchmark.py --num-seeds 5          # quick smoke test
    uv run python benchmarks/benchmark.py --seeds-start 50       # resume from seed 50
    uv run python benchmarks/benchmark.py --resume results.csv   # skip already-done rows (single-algorithm benchmarks/config.json only)

Results are written to benchmark_results/ after each run (crash-safe).
A summary + figure are generated once all runs complete.
"""

import argparse
import csv
import math
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from task.task_pose_sampler import (
    NUM_EXTRA_PATHS,
    NUM_LINE_SAMPLES,
    NUM_SAMPLES,
    REPEAT_START_GOAL,
)
from task.task_pose_sampler_trajectory_ver import NUM_POSES

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Each entry point pairs a main_*.py pipeline script with the sampler-param
# keys that can be swept for it. Keyed by a short name for the optim
# algorithm, which is also the value expected in config.json's "optim_algo"
# field.
ENTRY_POINTS = {
    "candidate_selection_static": {
        "script": "main_candidate_selection_static.py",
        "sampler_params": {
            "num_samples": {"default": NUM_SAMPLES, "abbrev": "nsp"},
            "num_line_samples": {"default": NUM_LINE_SAMPLES, "abbrev": "nls"},
            "num_extra_paths": {"default": NUM_EXTRA_PATHS, "abbrev": "nep"},
            "repeat_start_goal": {"default": REPEAT_START_GOAL, "abbrev": "rsg"},
            "num_plan_candidates": {"default": 1, "abbrev": "k"},
        },
    },
    "candidate_selection_trajectory": {
        "script": "main_candidate_selection_trajectory.py",
        "sampler_params": {
            "num_poses": {"default": NUM_POSES, "abbrev": "npo"},
            "num_plan_candidates": {"default": 1, "abbrev": "k"},
        },
    },
    "gradient_trajectory": {
        "script": "main_gradient_trajectory.py",
        "sampler_params": {
            "num_poses": {"default": NUM_POSES, "abbrev": "npo"},
        },
    },
}

# Populated by _configure_entry_point() once optim_algo is known.
SAMPLER_PARAMS: dict[str, dict] = {}
SAMPLER_KEYS: list[str] = []
SAMPLER_DEFAULTS: dict[str, int] = {}
SAMPLER_ABBREV: dict[str, str] = {}
RESULT_FIELDS: list[str] = []
_ROW_PARSERS: dict[str, Callable[[str], Any]] = {}
ENTRY_SCRIPT: str = ""
OPTIM_ALGO: str = ""


def _configure_entry_point(optim_algo: str) -> None:
    """Set the ENTRY_SCRIPT/OPTIM_ALGO/SAMPLER_* globals (and the fields/parsers
    derived from them) for the chosen optim_algo."""
    global SAMPLER_PARAMS, SAMPLER_KEYS, SAMPLER_DEFAULTS, SAMPLER_ABBREV
    global RESULT_FIELDS, _ROW_PARSERS, ENTRY_SCRIPT, OPTIM_ALGO

    entry = ENTRY_POINTS[optim_algo]
    ENTRY_SCRIPT = entry["script"]
    OPTIM_ALGO = optim_algo
    SAMPLER_PARAMS = entry["sampler_params"]
    SAMPLER_KEYS = list(SAMPLER_PARAMS)
    SAMPLER_DEFAULTS = {k: v["default"] for k, v in SAMPLER_PARAMS.items()}
    SAMPLER_ABBREV = {k: v["abbrev"] for k, v in SAMPLER_PARAMS.items()}
    RESULT_FIELDS = [
        "seed",
        "condition",
        *SAMPLER_KEYS,
        "success",
        "failure_reason",
        "duration_seconds",
        "optim_duration_seconds",
        "validation_duration_seconds",
        "plan_duration_seconds",
        "returncode",
    ]
    _ROW_PARSERS = {
        "seed": int,
        "condition": str,
        "success": lambda v: v == "True",
        "failure_reason": str,
        "duration_seconds": float,
        "optim_duration_seconds": lambda v: float(v) if v else None,
        "validation_duration_seconds": lambda v: float(v) if v else None,
        "plan_duration_seconds": lambda v: float(v) if v else None,
        "returncode": int,
        **{k: int for k in SAMPLER_KEYS},
    }


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


with open(CONFIG_PATH) as f:
    _presets_config = json.load(f)
CONDITIONS = list(_presets_config["conditions"].items())
# Each entry in "algorithms" is pinned to an optim_algo (an ENTRY_POINTS key)
# with its own sampler-param "configs" tuples and "output_dir" (resolved
# relative to the project root). Algorithms run sequentially, one after
# another, in a single invocation. Top-level "num_seeds" can be overridden
# per-algorithm.
DEFAULT_NUM_SEEDS = _presets_config.get("num_seeds")


def _expand_configs(
    configs: list[tuple], num_plan_candidates: list[int] | None
) -> list[tuple]:
    """Cross "configs" with a separate "num_plan_candidates" sweep list, if given.

    Lets benchmarks/config.json sweep top-k candidate planning independently
    of an algorithm's other sampler params, instead of having to repeat every
    base config tuple once per k value. The algorithm's sampler_params (see
    ENTRY_POINTS) must list num_plan_candidates last for the appended value to
    land in the right position.
    """
    if not num_plan_candidates:
        return configs
    return [c + (k,) for c in configs for k in num_plan_candidates]


ALGORITHMS = [
    {
        "optim_algo": algo["optim_algo"],
        "configs": _expand_configs(
            [tuple(c) for c in algo["configs"]],
            algo.get("num_plan_candidates"),
        ),
        "output_dir": PROJECT_DIR
        / algo.get("output_dir", f"benchmark_results/{algo['optim_algo']}"),
        "num_seeds": algo.get("num_seeds"),
    }
    for algo in _presets_config["algorithms"]
]


def _extract_error_detail(stderr: str, stdout: str) -> str:
    """Pull the last 'SomeError: ...' line out of stderr (or stdout fallback)."""
    for text in (stderr, stdout):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in reversed(lines):
            if "Error" in line and ":" in line:
                return line[:120]
    lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    return lines[-1][:120] if lines else ""


def classify_output(stdout: str, returncode: int, stderr: str = "") -> tuple[bool, str]:
    """Return (success, failure_reason) by parsing the entry point's stdout/stderr."""
    if returncode != 0:
        detail = _extract_error_detail(stderr, stdout)
        reason = f"optimization_error: {detail}" if detail else "optimization_error"
        return False, reason
    if "Sequence complete:" in stdout:
        return True, ""
    if "Executing partial plan" in stdout:
        return False, "partial_plan"
    if "Start configuration is in collision" in stdout:
        return False, "start_collision"
    if "Rendering static scene" in stdout:
        return False, "no_path_found"
    return False, "unknown"


def effective_sampler(overrides: dict, base_config: dict) -> dict[str, int]:
    """Resolve each sampler key to its effective int value: override > config > default."""
    out = {}
    for key in SAMPLER_KEYS:
        val = overrides.get(key)
        if val is None:
            val = base_config.get(key, SAMPLER_DEFAULTS[key])
        out[key] = int(val)
    return out


def combo_key(row: dict) -> tuple:
    return tuple(row[k] for k in SAMPLER_KEYS)


def group_by(rows: list[dict], key_fn: Callable[[dict], Any]) -> dict[Any, list[dict]]:
    grouped: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[key_fn(r)].append(r)
    return grouped


def build_config(
    seed: int,
    condition_flags: dict,
    sampler_overrides: dict,
    base_config: dict,
) -> dict:
    cfg = dict(base_config)
    cfg["seed"] = seed
    cfg["visualize"] = False
    cfg["debug"] = False
    cfg["plot"] = {"enabled": False, "output_dir": "output/figures"}
    cfg["timelapse"] = {"enabled": False}
    cfg.update(condition_flags)
    for key, value in sampler_overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


_BENCHMARK_LINE = re.compile(r"^\[Benchmark\] (\w+)=([\d.]+)$")


def _extract_benchmark_metrics(stdout: str) -> dict[str, float]:
    """Parse every `[Benchmark] name=value` line in stdout into {name: value}.

    A name can appear more than once (e.g. `plan_seconds` is emitted once per
    planning attempt when num_plan_candidates > 1), in which case the values
    are summed so the metric reflects the total cost across all attempts.
    """
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        match = _BENCHMARK_LINE.match(line)
        if match:
            name, value = match.group(1), float(match.group(2))
            metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def run_single(
    seed: int,
    condition_name: str,
    condition_flags: dict,
    sampler_overrides: dict,
    base_config: dict,
    config_dir: Path,
    timeout: int,
) -> dict:
    cfg = build_config(seed, condition_flags, sampler_overrides, base_config)
    # Route this run's optimization CSV logs (output/<run_time>/morphology_history.csv
    # and its siblings) under the benchmark's own output_dir instead of the
    # project root, so log_root_dir/<run_time>/ lands next to this run's
    # results/figure/configs.
    cfg["log_root_dir"] = str(config_dir.parent)
    suffix = "_".join(
        f"{SAMPLER_ABBREV[k]}"
        + (
            "default"
            if sampler_overrides[k] is None
            else str(int(sampler_overrides[k]))
        )
        for k in SAMPLER_KEYS
    )
    config_path = config_dir / f"seed{seed:04d}_{condition_name}_{suffix}.json"
    config_path.write_text(json.dumps(cfg, indent=2))

    start = time.time()
    try:
        proc = subprocess.run(
            ["uv", "run", "python", ENTRY_SCRIPT, "--config", str(config_path)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start
        success, reason = classify_output(proc.stdout, proc.returncode, proc.stderr)
        returncode = proc.returncode
        metrics = _extract_benchmark_metrics(proc.stdout)
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        success, reason, returncode = False, "timeout", -1
        metrics = {}
    except Exception as e:
        duration = time.time() - start
        success, reason, returncode = False, f"runner_error: {e}", -2
        metrics = {}

    def _phase(name: str) -> float | str:
        value = metrics.get(name)
        return round(value, 1) if value is not None else ""

    return {
        "seed": seed,
        "condition": condition_name,
        **effective_sampler(sampler_overrides, base_config),
        "success": success,
        "failure_reason": reason,
        "duration_seconds": round(duration, 1),
        "optim_duration_seconds": _phase("optim_seconds"),
        "validation_duration_seconds": _phase("validation_seconds"),
        "plan_duration_seconds": _phase("plan_seconds"),
        "returncode": returncode,
    }


def load_completed(csv_path: Path) -> set[tuple]:
    """Return set of (seed, condition, *sampler_values) already present in results CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {
            (int(r["seed"]), r["condition"], *(int(r[k]) for k in SAMPLER_KEYS))
            for r in csv.DictReader(f)
        }


def _stats(rows: list[dict]) -> tuple[int, int, float, float]:
    """Return (total, n_success, avg_duration_seconds, total_duration_seconds)."""
    durations = [r["duration_seconds"] for r in rows]
    n_success = sum(1 for r in rows if r["success"])
    return len(rows), n_success, np.mean(durations), sum(durations)


def generate_report(results: list[dict], output_dir: Path) -> None:
    by_condition = group_by(results, lambda r: r["condition"])

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for cond, rows in sorted(by_condition.items()):
        total, n_success, avg_duration, total_duration = _stats(rows)
        print(
            f"\n{cond}:  {n_success}/{total} successful  ({100 * n_success / total:.1f}%)"
        )
        print(
            f"  avg duration: {avg_duration:.0f}s  |  total: {total_duration / 3600:.1f}h"
        )
        failures = Counter(r["failure_reason"] for r in rows if not r["success"])
        for reason, count in failures.most_common():
            print(f"  {reason}: {count}")

    _print_sampler_sweep_summary(results)
    _make_figure(results, output_dir)


def _print_sampler_sweep_summary(results: list[dict]) -> None:
    by_sampler = group_by(results, combo_key)
    if len(by_sampler) <= 1:
        return

    print("\n" + "-" * 50)
    print("SAMPLER PARAM SWEEP (" + ", ".join(SAMPLER_KEYS) + ")")
    print("-" * 50)
    for key in sorted(by_sampler.keys()):
        total, n_success, avg_duration, _ = _stats(by_sampler[key])
        params = "  ".join(
            f"{SAMPLER_ABBREV[k]}={v:<4}" for k, v in zip(SAMPLER_KEYS, key)
        )
        print(
            f"  {params}  {n_success}/{total} successful  "
            f"({100 * n_success / total:.1f}%)  avg {avg_duration:.0f}s"
        )


FAILURE_COLORS = {
    "partial_plan": "#C44545",
    "optimization_error": "#D94A4A",
    "start_collision": "#8E44AD",
    "no_path_found": "#F1C40F",
    "timeout": "#E67E22",
    "unknown": "#7F8C8D",
}
SUCCESS_COLOR = "#4C8DC9"


def _plot_outcome_breakdown(ax, rows: list[dict]) -> None:
    by_condition = group_by(rows, lambda r: r["condition"])
    conditions = sorted(by_condition.keys())
    all_reasons = sorted(
        {
            r["failure_reason"]
            for rs in by_condition.values()
            for r in rs
            if not r["success"]
        }
    )
    bottoms = np.zeros(len(conditions))
    totals = [len(by_condition[c]) for c in conditions]
    for reason in all_reasons:
        counts = [
            sum(
                1
                for r in by_condition[c]
                if not r["success"] and r["failure_reason"] == reason
            )
            for c in conditions
        ]
        pcts = [100 * cnt / tot for cnt, tot in zip(counts, totals)]
        label = reason if len(reason) <= 50 else reason[:47] + "..."
        ax.bar(
            conditions,
            pcts,
            bottom=bottoms,
            label=label,
            color=FAILURE_COLORS.get(reason, "#CCCCCC"),
        )
        bottoms += np.array(pcts)
    success_pcts = [
        100 * sum(r["success"] for r in by_condition[c]) / len(by_condition[c])
        for c in conditions
    ]
    ax.bar(
        conditions, success_pcts, bottom=bottoms, label="success", color=SUCCESS_COLOR
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage of runs (%)")
    ax.set_title("Outcome Breakdown by Condition", pad=5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        fontsize=8,
        ncol=1,
        frameon=False,
    )
    for i, rate in enumerate(success_pcts):
        ax.text(i, 102, f"{rate:.1f}% success", ha="center", fontsize=10)


def _make_figure(results: list[dict], output_dir: Path) -> None:
    by_combo = group_by(results, combo_key)
    combos = sorted(by_combo.keys())
    optim_algo = OPTIM_ALGO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"benchmark_{optim_algo}_{timestamp}.png"
    title = f"Optimization + Planning Benchmark\nAlgorithm: {optim_algo}"

    if len(combos) <= 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        _plot_outcome_breakdown(ax, results)
        plt.tight_layout()
    else:
        ncols = 3
        nrows = math.ceil(len(combos) / ncols)
        fig, axes_grid = plt.subplots(
            nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows + 1.2), squeeze=False
        )
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
        abbrev_legend = "   |   ".join(
            f"{abbr} = {key}" for key, abbr in SAMPLER_ABBREV.items()
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=6.0)
        fig.text(
            0.5,
            0.965,
            abbrev_legend,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
        )
        axes_flat = axes_grid.flatten()
        for ax, key in zip(axes_flat, combos):
            combo_rows = by_combo[key]
            title = "   ".join(
                f"{SAMPLER_ABBREV[k]}={v}" for k, v in zip(SAMPLER_KEYS, key)
            )
            _plot_outcome_breakdown(ax, combo_rows)
            ax.set_title(
                f"{title}  ({len(combo_rows)} runs)", fontsize=10, fontweight="bold"
            )
        for ax in axes_flat[len(combos) :]:
            ax.set_visible(False)

    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds to evaluate (default: 100, or config.json's/"
        "algorithm's default)",
    )
    parser.add_argument(
        "--seeds-start", type=int, default=0, help="First seed value (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent dir under which each algorithm in config.json gets its "
        "own <output-dir>/<optim_algo> subdirectory (default: each "
        "algorithm's own output_dir from config.json).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-run timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing results CSV, skipping completed "
        "(seed, condition, *sampler_values) tuples. Only supported when "
        "config.json has a single algorithm.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Load all CSVs from --output-dir and regenerate the figure without running benchmarks",
    )
    parser.add_argument(
        "--no-results-csv",
        action="store_true",
        help="Run the sweep without writing benchmark_<optim_algo>_<timestamp>.csv "
        "or the report/figure. Disables --resume and crash-safety for this run; "
        "use only for a quick/disposable sweep.",
    )
    return parser.parse_args()


def _load_results_csvs(csv_paths: list[Path]) -> list[dict]:
    results = []
    for csv_path in csv_paths:
        with open(csv_path, newline="") as f:
            results.extend(
                {k: _ROW_PARSERS[k](v) for k, v in row.items()}
                for row in csv.DictReader(f)
            )
    return results


def _run_replot(output_dir: Path) -> None:
    csv_files = sorted(output_dir.glob("benchmark_*.csv"))
    if not csv_files:
        print(f"[Benchmark] No CSV files found in {output_dir}")
        return
    all_results = _load_results_csvs(csv_files)
    if all_results:
        generate_report(all_results, output_dir)
    else:
        print("[Benchmark] No results to report.")


def _resolve_results_csv(
    args: argparse.Namespace, output_dir: Path, optim_module: str
) -> Path:
    if args.resume and args.resume.exists():
        print(f"[Benchmark] Resuming from {args.resume}")
        return args.resume
    results_csv = (
        output_dir / f"benchmark_{optim_module}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    print(f"[Benchmark] Writing results to {results_csv}")
    return results_csv


def _run_sweep(
    args: argparse.Namespace,
    configs: list[tuple],
    base_config: dict,
    results_csv: Path | None,
    config_dir: Path,
) -> None:
    """Run the seed x condition x configs sweep.

    When results_csv is None (--no-results-csv), nothing is written to disk:
    no resume-tracking of completed runs, no crash-safety. Use only for a
    quick/disposable sweep.
    """
    completed = load_completed(results_csv) if results_csv is not None else {}
    if completed:
        print(f"[Benchmark] Skipping {len(completed)} already-completed runs")

    seeds = range(args.seeds_start, args.seeds_start + args.num_seeds)
    total = len(seeds) * len(configs) * len(CONDITIONS)
    done = 0

    def _run_combos(writer, f) -> None:
        nonlocal done
        for seed in seeds:
            for combo in configs:
                overrides = dict(zip(SAMPLER_KEYS, combo))
                effective = effective_sampler(overrides, base_config)
                sweep_label = " ".join(
                    f"{SAMPLER_ABBREV[k]}={effective[k]}" for k in SAMPLER_KEYS
                )
                for condition_name, condition_flags in CONDITIONS:
                    done += 1
                    key = (seed, condition_name, *(effective[k] for k in SAMPLER_KEYS))
                    if key in completed:
                        print(
                            f"[{done}/{total}] seed={seed} {condition_name} "
                            f"{sweep_label} — skipped (already done)"
                        )
                        continue

                    print(
                        f"[{done}/{total}] seed={seed} {condition_name} {sweep_label} ...",
                        flush=True,
                    )
                    result = run_single(
                        seed,
                        condition_name,
                        condition_flags,
                        overrides,
                        base_config,
                        config_dir,
                        args.timeout,
                    )
                    status = (
                        "OK"
                        if result["success"]
                        else f"FAIL ({result['failure_reason']})"
                    )
                    print(f"  -> {status}  ({result['duration_seconds']:.0f}s)")

                    if writer is not None:
                        writer.writerow(result)
                        f.flush()

    if results_csv is None:
        _run_combos(writer=None, f=None)
        return

    csv_exists = results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not csv_exists:
            writer.writeheader()
        _run_combos(writer, f)


def _run_algorithm(
    args: argparse.Namespace,
    optim_algo: str,
    configs: list[tuple],
    num_seeds: int,
    output_dir: Path,
) -> None:
    """Configure the entry point for one optim_algo and run its full
    seed x condition x configs sweep, then report on its results."""
    if optim_algo not in ENTRY_POINTS:
        raise SystemExit(
            f"config.json algorithm optim_algo {optim_algo!r} is not one of "
            f"{sorted(ENTRY_POINTS)}"
        )
    _configure_entry_point(optim_algo)

    if any(len(c) != len(SAMPLER_KEYS) for c in configs):
        raise SystemExit(
            f"config.json configs for optim_algo {optim_algo!r} don't match "
            f"its sampler params {SAMPLER_KEYS}"
        )

    config_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    print(f"\n{'#' * 60}\n[Benchmark] optim_algo={optim_algo}\n{'#' * 60}")

    if args.replot:
        _run_replot(output_dir)
        return

    with open(PROJECT_DIR / "config.json") as f:
        base_config = json.load(f)

    sweep_args = argparse.Namespace(**vars(args))
    sweep_args.num_seeds = num_seeds

    if args.no_results_csv:
        print("[Benchmark] --no-results-csv: not writing results CSV or report")
        results_csv = None
    else:
        results_csv = _resolve_results_csv(sweep_args, output_dir, optim_algo)
    _run_sweep(sweep_args, configs, base_config, results_csv, config_dir)

    if results_csv is None:
        return

    all_results = _load_results_csvs([results_csv])
    if all_results:
        generate_report(all_results, output_dir)
    else:
        print("[Benchmark] No results to report.")


def main() -> None:
    args = parse_args()

    if args.resume and len(ALGORITHMS) > 1:
        raise SystemExit(
            "--resume isn't supported when config.json has more than one "
            "algorithm; rerun against a single algorithm's --output-dir instead"
        )

    for algo_entry in ALGORITHMS:
        output_dir = (
            args.output_dir / algo_entry["optim_algo"]
            if args.output_dir
            else algo_entry["output_dir"]
        )
        num_seeds = (
            args.num_seeds or algo_entry["num_seeds"] or DEFAULT_NUM_SEEDS or 100
        )
        _run_algorithm(
            args, algo_entry["optim_algo"], algo_entry["configs"], num_seeds, output_dir
        )


if __name__ == "__main__":
    main()
