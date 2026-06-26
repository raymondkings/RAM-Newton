"""
Benchmark the optimization+planning pipeline across N random seeds,
with and without collision avoidance.

Usage:
    uv run python benchmark/benchmark.py                        # 100 seeds, both conditions
    uv run python benchmark/benchmark.py --num-seeds 5          # quick smoke test
    uv run python benchmark/benchmark.py --seeds-start 50       # resume from seed 50
    uv run python benchmark/benchmark.py --resume results.csv   # skip already-done rows

Results are written to benchmark_results/ after each run (crash-safe).
A summary + figure are generated once all runs complete.
"""

import argparse
import csv
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

# Ground collisions are intentionally ignored in both conditions; only the
# obstacle-collision check is toggled.
CONDITIONS = [
    ("obstacles_off_ground_off", dict(ignore_obstacles=True, ignore_ground=True)),
    ("obstacles_on_ground_off", dict(ignore_obstacles=False, ignore_ground=True)),
]

SAMPLER_PARAMS = [
    ("num_samples", NUM_SAMPLES),
    ("num_line_samples", NUM_LINE_SAMPLES),
    ("num_extra_paths", NUM_EXTRA_PATHS),
    ("repeat_start_goal", REPEAT_START_GOAL),
]
SAMPLER_KEYS = [k for k, _ in SAMPLER_PARAMS]
SAMPLER_DEFAULTS = dict(SAMPLER_PARAMS)
SAMPLER_ABBREV = {
    "num_samples": "ns",
    "num_line_samples": "nls",
    "num_extra_paths": "nep",
    "repeat_start_goal": "rsg",
}

# Hardcoded sampler-param sweeps. Each entry is
# (num_samples, num_line_samples, num_extra_paths, repeat_start_goal).
PRESETS = {
    "main": {
        # Halve num_samples each step from the default (50); line_samples fixed
        # at 0; vary num_extra_paths; repeat_start_goal disabled.
        "configs": [
            (50, 0, 0, 0),
            (50, 0, 2, 0),
            (50, 0, 4, 0),
            (25, 0, 0, 0),
            (25, 0, 2, 0),
            (25, 0, 4, 0),
            (12, 0, 0, 0),
            (12, 0, 2, 0),
            (12, 0, 4, 0),
            (6, 0, 0, 0),
            (6, 0, 2, 0),
            (6, 0, 4, 0),
        ],
        "num_seeds": 20,
        "output_dir": PROJECT_DIR / "benchmark_results_main",
    },
    "small": {
        "configs": [
            (10, 10, 0, 4),
            (25, 10, 0, 4),
        ],
        "num_seeds": 3,
        "output_dir": PROJECT_DIR / "benchmark_results_small",
    },
}

RESULT_FIELDS = [
    "seed",
    "condition",
    *SAMPLER_KEYS,
    "success",
    "failure_reason",
    "duration_seconds",
    "returncode",
]

# Type map used when reloading CSV rows for the final report.
_ROW_PARSERS: dict[str, Callable[[str], Any]] = {
    "seed": int,
    "condition": str,
    "success": lambda v: v == "True",
    "failure_reason": str,
    "duration_seconds": float,
    "returncode": int,
    **{k: int for k in SAMPLER_KEYS},
}


def _extract_error_detail(stderr: str, stdout: str) -> str:
    """Pull the last exception line out of stderr (or stdout fallback)."""
    for text in (stderr, stdout):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in reversed(lines):
            if any(
                line.startswith(exc)
                for exc in ("RuntimeError:", "ValueError:", "AssertionError:", "Error:")
            ):
                return line[:120]
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


def _slug(value: int | None) -> str:
    return "default" if value is None else str(int(value))


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
        f"{SAMPLER_ABBREV[k]}{_slug(sampler_overrides[k])}" for k in SAMPLER_KEYS
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
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        success, reason, returncode = False, "timeout", -1
    except Exception as e:
        duration = time.time() - start
        success, reason, returncode = False, f"runner_error: {e}", -2

    return {
        "seed": seed,
        "condition": condition_name,
        **effective_sampler(sampler_overrides, base_config),
        "success": success,
        "failure_reason": reason,
        "duration_seconds": round(duration, 1),
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


def generate_report(results: list[dict], output_dir: Path) -> None:
    by_condition = group_by(results, lambda r: r["condition"])

    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for cond, rows in sorted(by_condition.items()):
        total = len(rows)
        n_success = sum(1 for r in rows if r["success"])
        durations = [r["duration_seconds"] for r in rows]
        print(
            f"\n{cond}:  {n_success}/{total} successful  ({100 * n_success / total:.1f}%)"
        )
        print(
            f"  avg duration: {np.mean(durations):.0f}s  |  total: {sum(durations) / 3600:.1f}h"
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
        rows = by_sampler[key]
        total = len(rows)
        n_success = sum(1 for r in rows if r["success"])
        durations = [r["duration_seconds"] for r in rows]
        params = "  ".join(
            f"{SAMPLER_ABBREV[k]}={v:<4}" for k, v in zip(SAMPLER_KEYS, key)
        )
        print(
            f"  {params}  {n_success}/{total} successful  "
            f"({100 * n_success / total:.1f}%)  avg {np.mean(durations):.0f}s"
        )


CONDITION_COLORS = {
    "no_collision": "#4C8DC9",
    "with_collision": "#D94A4A",
    "obstacles_off_ground_off": "#4C8DC9",
    "obstacles_on_ground_off": "#D94A4A",
    "obstacles_off_ground_on": "#2CA02C",
    "obstacles_on_ground_on": "#FF7F0E",
}
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


def _plot_cumulative_success(ax, rows: list[dict]) -> None:
    by_condition = group_by(rows, lambda r: r["condition"])
    for cond in sorted(by_condition.keys()):
        cond_rows = sorted(by_condition[cond], key=lambda r: r["seed"])
        cumulative = np.cumsum([r["success"] for r in cond_rows])
        seed_indices = np.array([r["seed"] for r in cond_rows])
        denom = np.arange(1, len(cond_rows) + 1)
        ax.plot(
            seed_indices,
            100 * cumulative / denom,
            label=cond,
            color=CONDITION_COLORS.get(cond, "#888"),
            linewidth=2,
        )
    ax.axhline(100, color="#888", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Cumulative success rate (%)")
    ax.set_title("Convergence of Success Rate", pad=5)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _make_figure(results: list[dict], output_dir: Path) -> None:
    by_combo = group_by(results, combo_key)
    combos = sorted(by_combo.keys())
    fig_path = output_dir / "benchmark_results.png"

    if len(combos) <= 1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle(
            "Optimization + Planning Benchmark", fontsize=14, fontweight="bold"
        )
        _plot_outcome_breakdown(axes[0], results)
        _plot_cumulative_success(axes[1], results)
        plt.tight_layout()
    else:
        fig = plt.figure(figsize=(11, 4.2 * len(combos) + 1.2), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.25, hspace=0.0)
        fig.suptitle(
            "Optimization + Planning Benchmark — sampler param sweep",
            fontsize=14,
            fontweight="bold",
            y=1.005,
        )
        subfigs = fig.subfigures(len(combos), 1, hspace=0.0)
        for subfig, key in zip(subfigs, combos):
            combo_rows = by_combo[key]
            title = ", ".join(f"{k}={v}" for k, v in zip(SAMPLER_KEYS, key))
            subfig.suptitle(
                f"{title}  ({len(combo_rows)} runs)",
                fontsize=12,
                fontweight="bold",
            )
            axes = subfig.subplots(1, 2)
            _plot_outcome_breakdown(axes[0], combo_rows)
            _plot_cumulative_success(axes[1], combo_rows)

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
        help="Use a hardcoded sampler-param sweep. Sets defaults for --num-seeds and "
        "--output-dir; overrides the per-param list flags. Configs are defined in "
        "PRESETS at the top of this file.",
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


def main() -> None:
    args = parse_args()

    preset = PRESETS[args.preset] if args.preset else None
    if args.num_seeds is None:
        args.num_seeds = preset["num_seeds"] if preset else 100
    if args.output_dir is None:
        args.output_dir = preset["output_dir"] if preset else DEFAULT_OUTPUT_DIR

    output_dir: Path = args.output_dir
    config_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    with open(PROJECT_DIR / "config.json") as f:
        base_config = json.load(f)

    if args.resume and args.resume.exists():
        results_csv = args.resume
        print(f"[Benchmark] Resuming from {results_csv}")
    else:
        results_csv = output_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.csv"
        print(f"[Benchmark] Writing results to {results_csv}")

    completed = load_completed(results_csv)
    if completed:
        print(f"[Benchmark] Skipping {len(completed)} already-completed runs")

    seeds = range(args.seeds_start, args.seeds_start + args.num_seeds)
    if preset:
        sweep = list(preset["configs"])
    else:
        sweep_lists = [getattr(args, k) for k in SAMPLER_KEYS]
        sweep = [
            (a, b, c, d)
            for a in sweep_lists[0]
            for b in sweep_lists[1]
            for c in sweep_lists[2]
            for d in sweep_lists[3]
        ]
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

    with open(results_csv, newline="") as f:
        all_results = [
            {k: _ROW_PARSERS[k](v) for k, v in row.items()} for row in csv.DictReader(f)
        ]

    if all_results:
        generate_report(all_results, output_dir)
    else:
        print("[Benchmark] No results to report.")


if __name__ == "__main__":
    main()
