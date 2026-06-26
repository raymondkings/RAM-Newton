"""
Benchmark the optimization+planning pipeline across N random seeds,
with and without collision avoidance.

Usage:
    uv run python benchmark.py                        # 100 seeds, both conditions
    uv run python benchmark.py --num-seeds 5          # quick smoke test
    uv run python benchmark.py --seeds-start 50       # resume from seed 50
    uv run python benchmark.py --resume results.csv   # skip already-done rows

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

import matplotlib.pyplot as plt
import numpy as np

from task.task_pose_sampler import (
    NUM_EXTRA_PATHS,
    NUM_LINE_SAMPLES,
    NUM_SAMPLES,
    REPEAT_START_GOAL,
)

PROJECT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "benchmark_results"

CONDITIONS = [
    ("no_collision", dict(ignore_obstacles=True, ignore_ground=True)),
    ("with_collision", dict(ignore_obstacles=False, ignore_ground=True)),
]

SAMPLER_PARAMS = [
    ("num_samples", NUM_SAMPLES),
    ("num_line_samples", NUM_LINE_SAMPLES),
    ("num_extra_paths", NUM_EXTRA_PATHS),
    ("repeat_start_goal", REPEAT_START_GOAL),
]

RESULT_FIELDS = [
    "seed",
    "condition",
    "num_samples",
    "num_line_samples",
    "num_extra_paths",
    "repeat_start_goal",
    "success",
    "failure_reason",
    "duration_seconds",
    "returncode",
]


def _extract_error_detail(stderr: str, stdout: str) -> str:
    """Pull the last exception line out of stderr (or stdout fallback)."""
    for text in (stderr, stdout):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Walk backwards looking for a recognisable exception line
        for line in reversed(lines):
            if any(
                line.startswith(exc)
                for exc in ("RuntimeError:", "ValueError:", "AssertionError:", "Error:")
            ):
                return line[:120]
            if "Error" in line and ":" in line:
                return line[:120]
    # Fall back to the last non-empty stderr line
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


def _effective_sampler_value(key: str, override: int | None, base_config: dict) -> int:
    if override is not None:
        return int(override)
    if key in base_config:
        return int(base_config[key])
    return int(dict(SAMPLER_PARAMS)[key])


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
    suffix_parts = [
        f"ns{_slug(sampler_overrides['num_samples'])}",
        f"nls{_slug(sampler_overrides['num_line_samples'])}",
        f"nep{_slug(sampler_overrides['num_extra_paths'])}",
        f"rsg{_slug(sampler_overrides['repeat_start_goal'])}",
    ]
    suffix = "_".join(suffix_parts)
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
        "num_samples": _effective_sampler_value(
            "num_samples", sampler_overrides["num_samples"], base_config
        ),
        "num_line_samples": _effective_sampler_value(
            "num_line_samples", sampler_overrides["num_line_samples"], base_config
        ),
        "num_extra_paths": _effective_sampler_value(
            "num_extra_paths", sampler_overrides["num_extra_paths"], base_config
        ),
        "repeat_start_goal": _effective_sampler_value(
            "repeat_start_goal", sampler_overrides["repeat_start_goal"], base_config
        ),
        "success": success,
        "failure_reason": reason,
        "duration_seconds": round(duration, 1),
        "returncode": returncode,
    }


def _slug(value: int | None) -> str:
    return "default" if value is None else str(int(value))


def load_completed(
    csv_path: Path,
) -> set[tuple[int, str, int, int, int, int]]:
    """Return set of (seed, condition, num_samples, num_line_samples,
    num_extra_paths, repeat_start_goal) already present in results CSV."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        completed = set()
        for r in reader:
            completed.add(
                (
                    int(r["seed"]),
                    r["condition"],
                    int(r["num_samples"]),
                    int(r["num_line_samples"]),
                    int(r["num_extra_paths"]),
                    int(r["repeat_start_goal"]),
                )
            )
        return completed


def generate_report(results: list[dict], output_dir: Path) -> None:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

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
    by_sampler: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for r in results:
        key = (
            r["num_samples"],
            r["num_line_samples"],
            r["num_extra_paths"],
            r["repeat_start_goal"],
        )
        by_sampler[key].append(r)

    if len(by_sampler) <= 1:
        return

    print("\n" + "-" * 50)
    print(
        "SAMPLER PARAM SWEEP "
        "(num_samples, num_line_samples, num_extra_paths, repeat_start_goal)"
    )
    print("-" * 50)
    for key in sorted(by_sampler.keys()):
        rows = by_sampler[key]
        total = len(rows)
        n_success = sum(1 for r in rows if r["success"])
        durations = [r["duration_seconds"] for r in rows]
        ns, nls, nep, rsg = key
        print(
            f"  ns={ns:<4} nls={nls:<4} nep={nep:<3} rsg={rsg:<4}  "
            f"{n_success}/{total} successful  "
            f"({100 * n_success / total:.1f}%)  "
            f"avg {np.mean(durations):.0f}s"
        )


CONDITION_COLORS = {"no_collision": "#5B7A99", "with_collision": "#E07A5F"}
FAILURE_COLORS = {
    "partial_plan": "#E07A5F",
    "optimization_error": "#B5523A",
    "start_collision": "#C97E4D",
    "no_path_found": "#E5B25D",
    "timeout": "#9AA0A6",
    "unknown": "#C7C9CC",
}
SUCCESS_COLOR = "#5B7A99"


def _group_by_condition(rows: list[dict]) -> dict[str, list[dict]]:
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_condition[r["condition"]].append(r)
    return by_condition


def _plot_outcome_breakdown(ax, rows: list[dict]) -> None:
    by_condition = _group_by_condition(rows)
    conditions = sorted(by_condition.keys())
    all_reasons = sorted(
        {
            r["failure_reason"]
            for rows in by_condition.values()
            for r in rows
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
        ax.bar(
            conditions,
            pcts,
            bottom=bottoms,
            label=reason,
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
    ax.legend(loc="lower right", fontsize=8)
    for i, rate in enumerate(success_pcts):
        ax.text(i, 102, f"{rate:.1f}% success", ha="center", fontsize=10)


def _plot_cumulative_success(ax, rows: list[dict]) -> None:
    by_condition = _group_by_condition(rows)
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
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Cumulative success rate (%)")
    ax.set_title("Convergence of Success Rate", pad=5)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _make_figure(results: list[dict], output_dir: Path) -> None:
    by_combo: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for r in results:
        key = (
            r["num_samples"],
            r["num_line_samples"],
            r["num_extra_paths"],
            r["repeat_start_goal"],
        )
        by_combo[key].append(r)

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
        fig = plt.figure(figsize=(11, 5 * len(combos) + 1.5), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.3, hspace=0.05)
        fig.suptitle(
            "Optimization + Planning Benchmark — sampler param sweep",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        subfigs = fig.subfigures(len(combos), 1, hspace=0.1)
        for subfig, key in zip(subfigs, combos):
            ns, nls, nep, rsg = key
            combo_rows = by_combo[key]
            subfig.suptitle(
                f"num_samples={ns}, num_line_samples={nls}, "
                f"num_extra_paths={nep}, repeat_start_goal={rsg}  "
                f"({len(combo_rows)} runs)",
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
        "--num-seeds",
        type=int,
        default=100,
        help="Number of seeds to evaluate (default: 100)",
    )
    parser.add_argument(
        "--seeds-start", type=int, default=0, help="First seed value (default: 0)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results",
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
        help=(
            "Resume from an existing results CSV, skipping completed "
            "(seed, condition, num_samples, num_line_samples, num_extra_paths, "
            "repeat_start_goal) tuples"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=_parse_int_list,
        default=[None],
        help=(
            "Comma-separated list of num_samples values to sweep "
            "(default: use config.json value)"
        ),
    )
    parser.add_argument(
        "--num-line-samples",
        type=_parse_int_list,
        default=[None],
        help=(
            "Comma-separated list of num_line_samples values to sweep; "
            "use 0 to disable the line path (default: use config.json value)"
        ),
    )
    parser.add_argument(
        "--num-extra-paths",
        type=_parse_int_list,
        default=[None],
        help=(
            "Comma-separated list of num_extra_paths values to sweep "
            "(default: use config.json value)"
        ),
    )
    parser.add_argument(
        "--repeat-start-goal",
        type=_parse_int_list,
        default=[None],
        help=(
            "Comma-separated list of repeat_start_goal values to sweep "
            "(default: use config.json value)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir
    config_dir = output_dir / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    base_config_path = PROJECT_DIR / "config.json"
    with open(base_config_path) as f:
        base_config = json.load(f)

    # Resolve results CSV path
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
    sweep = [
        (ns, nls, nep, rsg)
        for ns in args.num_samples
        for nls in args.num_line_samples
        for nep in args.num_extra_paths
        for rsg in args.repeat_start_goal
    ]
    total = len(seeds) * len(sweep) * len(CONDITIONS)
    done = 0

    csv_exists = results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not csv_exists:
            writer.writeheader()

        for seed in seeds:
            for ns, nls, nep, rsg in sweep:
                sampler_overrides = {
                    "num_samples": ns,
                    "num_line_samples": nls,
                    "num_extra_paths": nep,
                    "repeat_start_goal": rsg,
                }
                effective_ns = _effective_sampler_value("num_samples", ns, base_config)
                effective_nls = _effective_sampler_value(
                    "num_line_samples", nls, base_config
                )
                effective_nep = _effective_sampler_value(
                    "num_extra_paths", nep, base_config
                )
                effective_rsg = _effective_sampler_value(
                    "repeat_start_goal", rsg, base_config
                )
                sweep_label = (
                    f"ns={effective_ns} nls={effective_nls} "
                    f"nep={effective_nep} rsg={effective_rsg}"
                )
                for condition_name, condition_flags in CONDITIONS:
                    done += 1
                    key = (
                        seed,
                        condition_name,
                        effective_ns,
                        effective_nls,
                        effective_nep,
                        effective_rsg,
                    )
                    if key in completed:
                        print(
                            f"[{done}/{total}] seed={seed} {condition_name} "
                            f"{sweep_label} — skipped (already done)"
                        )
                        continue

                    print(
                        f"[{done}/{total}] seed={seed} {condition_name} "
                        f"{sweep_label} ...",
                        flush=True,
                    )
                    result = run_single(
                        seed,
                        condition_name,
                        condition_flags,
                        sampler_overrides,
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

    # Collect all results for report (including previously completed ones)
    all_results = []
    with open(results_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_results.append(
                {
                    "seed": int(row["seed"]),
                    "condition": row["condition"],
                    "num_samples": int(row["num_samples"]),
                    "num_line_samples": int(row["num_line_samples"]),
                    "num_extra_paths": int(row["num_extra_paths"]),
                    "repeat_start_goal": int(row["repeat_start_goal"]),
                    "success": row["success"] == "True",
                    "failure_reason": row["failure_reason"],
                    "duration_seconds": float(row["duration_seconds"]),
                    "returncode": int(row["returncode"]),
                }
            )

    if all_results:
        generate_report(all_results, output_dir)
    else:
        print("[Benchmark] No results to report.")


if __name__ == "__main__":
    main()
