"""
Benchmark the optimization+planning pipeline across N random seeds,
with and without collision avoidance.

How it works:
    The sweep is the cartesian product of seeds x sampler-param combos x
    CONDITIONS (obstacle collision on/off; ground collision is always
    ignored). For each combination, build_config() patches config.json with
    the seed, condition flags, and sampler overrides, writes it to
    <output_dir>/configs/, and run_single() shells out to
    `uv run python main.py --config <that file>` as a fresh subprocess
    (so a crash or hang in one run can't take down the sweep).

    Each result row is appended to the results CSV immediately and the file
    is flushed, so the run is crash-safe and resumable: pass --resume
    <csv> (or rerun against the same --output-dir) to skip any
    (seed, condition, *sampler_values) tuple already present via
    load_completed().

    Sampler params (num_samples, num_line_samples, num_extra_paths,
    repeat_start_goal) can be swept directly with --num-samples 10,20,30
    etc., or via a named --preset, loaded from presets.json next to this
    file, that bundles a list of param tuples with its own
    --num-seeds/--output-dir defaults.

    Once all runs finish (or with --replot against an --output-dir of
    existing CSVs), generate_report() prints a per-condition success/failure
    breakdown and _make_figure() renders it to a PNG — one subfigure per
    sampler-param combo when more than one was swept.

Usage:
    uv run python benchmarks/pipeline_benchmark.py                        # 100 seeds, both conditions
    uv run python benchmarks/pipeline_benchmark.py --num-seeds 5          # quick smoke test
    uv run python benchmarks/pipeline_benchmark.py --seeds-start 50       # resume from seed 50
    uv run python benchmarks/pipeline_benchmark.py --resume results.csv   # skip already-done rows

Results are written to benchmark_results/ after each run (crash-safe).
A summary + figure are generated once all runs complete.
"""

import argparse
import ast
import csv
import itertools
import math
import json
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

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "benchmark_results"

SAMPLER_PARAMS = {
    "num_samples": {"default": NUM_SAMPLES, "abbrev": "nsp"},
    "num_line_samples": {"default": NUM_LINE_SAMPLES, "abbrev": "nls"},
    "num_extra_paths": {"default": NUM_EXTRA_PATHS, "abbrev": "nep"},
    "repeat_start_goal": {"default": REPEAT_START_GOAL, "abbrev": "rsg"},
}
SAMPLER_KEYS = list(SAMPLER_PARAMS)
SAMPLER_DEFAULTS = {k: v["default"] for k, v in SAMPLER_PARAMS.items()}
SAMPLER_ABBREV = {k: v["abbrev"] for k, v in SAMPLER_PARAMS.items()}

PRESETS_PATH = Path(__file__).resolve().parent / "presets.json"


with open(PRESETS_PATH) as f:
    _presets_config = json.load(f)
CONDITIONS = list(_presets_config["conditions"].items())
# Each preset's "configs" is a list of (num_samples, num_line_samples,
# num_extra_paths, repeat_start_goal) tuples; "output_dir" is resolved
# relative to the project root.
PRESETS = {
    name: {
        "configs": [tuple(c) for c in spec["configs"]],
        "num_seeds": spec["num_seeds"],
        "output_dir": PROJECT_DIR / spec["output_dir"],
    }
    for name, spec in _presets_config["presets"].items()
}

RESULT_FIELDS = [
    "seed",
    "condition",
    *SAMPLER_KEYS,
    "success",
    "failure_reason",
    "duration_seconds",
    "optim_duration_seconds",
    "plan_duration_seconds",
    "returncode",
]

# Type map used when reloading CSV rows for the final report.
_ROW_PARSERS: dict[str, Callable[[str], Any]] = {
    "seed": int,
    "condition": str,
    "success": lambda v: v == "True",
    "failure_reason": str,
    "duration_seconds": float,
    "optim_duration_seconds": lambda v: float(v) if v else None,
    "plan_duration_seconds": lambda v: float(v) if v else None,
    "returncode": int,
    **{k: int for k in SAMPLER_KEYS},
}


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
    """Return (success, failure_reason) by parsing main.py stdout/stderr."""
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


def _extract_phase_timings(stdout: str) -> tuple[float | None, float | None]:
    optim, plan = None, None
    for line in stdout.splitlines():
        if line.startswith("[Benchmark] optim_seconds="):
            optim = float(line.split("=")[1])
        elif line.startswith("[Benchmark] plan_seconds="):
            plan = float(line.split("=")[1])
    return optim, plan


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
            ["uv", "run", "python", "main.py", "--config", str(config_path)],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start
        success, reason = classify_output(proc.stdout, proc.returncode, proc.stderr)
        returncode = proc.returncode
        optim_dur, plan_dur = _extract_phase_timings(proc.stdout)
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        success, reason, returncode = False, "timeout", -1
        optim_dur, plan_dur = None, None
    except Exception as e:
        duration = time.time() - start
        success, reason, returncode = False, f"runner_error: {e}", -2
        optim_dur, plan_dur = None, None

    return {
        "seed": seed,
        "condition": condition_name,
        **effective_sampler(sampler_overrides, base_config),
        "success": success,
        "failure_reason": reason,
        "duration_seconds": round(duration, 1),
        "optim_duration_seconds": round(optim_dur, 1) if optim_dur is not None else "",
        "plan_duration_seconds": round(plan_dur, 1) if plan_dur is not None else "",
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


def _detect_optim_module() -> str:
    """Return the optim module name imported in main.py (e.g. 'nrm_alpha_random_selection')."""
    src = (PROJECT_DIR / "main.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("optim.")
        ):
            return node.module[len("optim.") :]
    return "unknown"


def _make_figure(results: list[dict], output_dir: Path) -> None:
    by_combo = group_by(results, combo_key)
    combos = sorted(by_combo.keys())
    optim_module = _detect_optim_module()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"benchmark_{optim_module}_{timestamp}.png"
    title = f"Optimization + Planning Benchmark\nAlgorithm: {optim_module}"

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


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Use a named sampler-param sweep. Sets defaults for --num-seeds and "
        "--output-dir; overrides the per-param list flags. Presets are defined in "
        "presets.json next to this file.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of seeds to evaluate (default: 100, or preset's default)",
    )
    parser.add_argument(
        "--seeds-start", type=int, default=0, help="First seed value (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results (default: benchmark_results, or preset's default)",
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
        "(seed, condition, *sampler_values) tuples",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Load all CSVs from --output-dir and regenerate the figure without running benchmarks",
    )
    for key in SAMPLER_KEYS:
        parser.add_argument(
            f"--{key.replace('_', '-')}",
            type=_parse_int_list,
            default=[None],
            dest=key,
            help=f"Comma-separated list of {key} values to sweep "
            "(default: use config.json value)",
        )
    return parser.parse_args()


def _resolve_run_settings(args: argparse.Namespace) -> tuple[dict | None, Path]:
    """Apply preset defaults to --num-seeds/--output-dir where the user didn't override them."""
    preset = PRESETS[args.preset] if args.preset else None
    if args.num_seeds is None:
        args.num_seeds = preset["num_seeds"] if preset else 100
    if args.output_dir is None:
        args.output_dir = preset["output_dir"] if preset else DEFAULT_OUTPUT_DIR
    return preset, args.output_dir


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


def _build_sweep(args: argparse.Namespace, preset: dict | None) -> list[tuple]:
    """Cartesian product of sampler-param values to sweep, from a preset's
    fixed combos or from the per-param --num-samples/etc. list flags."""
    if preset:
        return list(preset["configs"])
    sweep_lists = [getattr(args, k) for k in SAMPLER_KEYS]
    return list(itertools.product(*sweep_lists))


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
    preset: dict | None,
    base_config: dict,
    results_csv: Path,
    config_dir: Path,
) -> None:
    completed = load_completed(results_csv)
    if completed:
        print(f"[Benchmark] Skipping {len(completed)} already-completed runs")

    seeds = range(args.seeds_start, args.seeds_start + args.num_seeds)
    sweep = _build_sweep(args, preset)
    total = len(seeds) * len(sweep) * len(CONDITIONS)
    done = 0

    csv_exists = results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not csv_exists:
            writer.writeheader()

        for seed in seeds:
            for combo in sweep:
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

                    writer.writerow(result)
                    f.flush()


def main() -> None:
    args = parse_args()
    preset, output_dir = _resolve_run_settings(args)

    config_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    if args.replot:
        _run_replot(output_dir)
        return

    with open(PROJECT_DIR / "config.json") as f:
        base_config = json.load(f)

    optim_module = _detect_optim_module()
    results_csv = _resolve_results_csv(args, output_dir, optim_module)
    _run_sweep(args, preset, base_config, results_csv, config_dir)

    all_results = _load_results_csvs([results_csv])
    if all_results:
        generate_report(all_results, output_dir)
    else:
        print("[Benchmark] No results to report.")


if __name__ == "__main__":
    main()
