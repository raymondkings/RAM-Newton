# -----------------------------------------------------------------------------
# Candidate-selection CSV plots for candidate_selection/static.py
#
# Expected CSV convention:
#   iteration = 0  -> validated top-probability candidate, not final-tier
#   iteration = 1  -> final-tier candidate but not finally selected by tiebreak
#   iteration = 2  -> final selected candidate
#
# Outputs:
#   1. Rotating 3D MP4 scatter per DOF: a, d, alpha for all validated
#      candidates. Each link index has its own color.
#   2. 2D PNG scatter: NRM probability vs validation SE3 error.
#      marker color: 0=blue, 1=orange, 2=red.
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from logutils.csv_reader import read_optimization_csv


def _csv_time_suffix(csv_path: str | Path) -> str:
    """Return the run timestamp from output/<run_time>/morphology_history.csv."""
    return Path(csv_path).parent.name


def _load_validated_candidate_rows(csv_path: str | Path) -> list[dict]:
    rows = read_optimization_csv(csv_path)

    candidate_rows = []
    for row in rows:
        if row.get("processed_morphology_json") is None:
            continue
        if row.get("best_se3_dist_mean") is None:
            continue
        if row.get("reachability_probability") is None:
            continue
        candidate_rows.append(row)

    if not candidate_rows:
        raise ValueError(
            f"No validated candidate rows found in {csv_path}. "
            "Expected rows with processed_morphology_json, "
            "reachability_probability, and best_se3_dist_mean."
        )

    return candidate_rows


def _candidate_dof(row: dict) -> int:
    morphology = row.get("processed_morphology_json")
    if morphology is None:
        raise ValueError("Candidate row is missing processed_morphology_json.")
    return len(morphology) - 1


def _group_rows_by_dof(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(_candidate_dof(row), []).append(row)
    return dict(sorted(grouped.items()))


def _create_candidate_morphology_3d_mp4_for_rows(
    rows: list[dict],
    output_path: Path,
    *,
    dof: int,
    fps: int,
    num_frames: int,
    dpi: int,
) -> Path:
    """Create one rotating 3D scatter MP4 for candidate rows with one DOF."""
    morphs = np.array([row["processed_morphology_json"] for row in rows], dtype=float)
    markers = np.array([int(row["iteration"]) for row in rows], dtype=int)

    if morphs.ndim != 3 or morphs.shape[-1] != 3:
        raise ValueError(
            f"Expected morphology array shape [num_candidates, seq_len, 3], "
            f"got {morphs.shape}."
        )

    num_candidates, seq_len, _ = morphs.shape
    expected_seq_len = dof + 1
    if seq_len != expected_seq_len:
        raise ValueError(
            f"Expected DOF{dof} morphology seq_len={expected_seq_len}, "
            f"got seq_len={seq_len}."
        )

    sizes = _marker_sizes(markers, base_size=26.0)

    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.colormaps.get_cmap("tab10")
    for link_idx in range(seq_len):
        a = morphs[:, link_idx, 1]
        d = morphs[:, link_idx, 2]
        alpha_deg = morphs[:, link_idx, 0] * 180.0 / np.pi
        color = cmap(link_idx % 10)
        ax.scatter(
            a,
            d,
            alpha_deg,
            s=sizes,
            alpha=0.78,
            color=color,
            label=f"link {link_idx}",
            depthshade=True,
        )

    ax.set_title(
        f"DOF{dof} top validated candidate morphologies "
        f"({num_candidates} candidates, {seq_len} links)"
    )
    ax.set_xlabel("a")
    ax.set_ylabel("d")
    ax.set_zlabel("alpha [deg]")
    ax.set_zticks([-90, 0, 90])

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    fig.tight_layout()

    def update(frame: int):
        azim = 360.0 * frame / num_frames
        elev = 22.0 + 8.0 * np.sin(2.0 * np.pi * frame / num_frames)
        ax.view_init(elev=elev, azim=azim)
        return (ax,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000.0 / fps,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    return output_path


def create_candidate_morphology_3d_mp4s(
    csv_path: str | Path,
    output_dir: str | Path = "output/candidate_plots",
    filename: str | None = None,
    fps: int = 24,
    num_frames: int = 180,
    dpi: int = 160,
) -> list[Path]:
    """Create rotating 3D scatter MP4s of candidate morphology parameters.

    Mixed-DOF CSVs are split into one MP4 per DOF. Single-DOF CSVs keep the old
    default filename for backwards compatibility.

    Axes:
        x = a
        y = d
        z = alpha in degrees

    Point color:
        link index

    Point size:
        iteration marker 0 -> base size
        iteration marker 1 -> 6.0x size
        iteration marker 2 -> 10.0x size
    """
    rows = _load_validated_candidate_rows(csv_path)
    grouped_rows = _group_rows_by_dof(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    single_dof = len(grouped_rows) == 1
    suffix = _csv_time_suffix(csv_path)
    for dof, dof_rows in grouped_rows.items():
        if filename is None:
            if single_dof:
                dof_filename = f"candidate_morphology_3d_{suffix}.mp4"
            else:
                dof_filename = f"candidate_morphology_3d_DOF{dof}_{suffix}.mp4"
        else:
            path = Path(filename)
            if single_dof:
                dof_filename = path.name
            else:
                dof_filename = f"{path.stem}_DOF{dof}{path.suffix or '.mp4'}"

        paths.append(
            _create_candidate_morphology_3d_mp4_for_rows(
                dof_rows,
                output_dir / dof_filename,
                dof=dof,
                fps=fps,
                num_frames=num_frames,
                dpi=dpi,
            )
        )

    return paths


def create_candidate_ranking_chart(
    csv_path: str | Path,
    output_dir: str | Path = "output/candidate_plots",
    filename: str | None = None,
    dpi: int = 200,
) -> Path:
    """Create a two-panel ranked dot chart of validated candidates.

    Left panel: NRM reachability probability (higher = better).
    Right panel: Validation SE3 error (lower = better).
    Candidates are sorted by selection tier (final selected first),
    then by probability descending within each tier.
    """
    rows = _load_validated_candidate_rows(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"candidate_ranking_{_csv_time_suffix(csv_path)}.png"
    output_path = output_dir / filename

    rows = sorted(
        rows,
        key=lambda r: (-int(r["iteration"]), -float(r["reachability_probability"])),
    )
    n = len(rows)

    probs = np.array([row["reachability_probability"] for row in rows], dtype=float)
    se3 = np.array([row["best_se3_dist_mean"] for row in rows], dtype=float)
    tiers = np.array([int(row["iteration"]) for row in rows], dtype=int)
    dofs = np.array([_candidate_dof(row) for row in rows], dtype=int)

    multi_dof = len(set(dofs.tolist())) > 1

    tier_counts: dict[int, int] = {}
    y_labels = []
    for row in rows:
        tier = int(row["iteration"])
        dof = _candidate_dof(row)
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        rank = tier_counts[tier]
        if tier == 2:
            label = "final" if rank == 1 else f"final #{rank}"
        else:
            label = f"#{rank}"
        if multi_dof:
            label = f"DOF{dof} {label}"
        y_labels.append(label)

    y = np.arange(n)

    tier_specs = [
        (2, "tab:red", "final selected"),
        (1, "tab:orange", "best SE3 tier"),
        (0, "tab:blue", "validated pool"),
    ]

    fig_h = max(3.5, 0.45 * n + 1.5)
    fig, (ax_prob, ax_se3) = plt.subplots(1, 2, figsize=(9.0, fig_h), sharey=True)

    tier_boundaries = [i - 0.5 for i in range(1, n) if tiers[i] != tiers[i - 1]]
    for ax in (ax_prob, ax_se3):
        for boundary in tier_boundaries:
            ax.axhline(
                boundary, color="gray", linewidth=0.8, linestyle="--", alpha=0.45
            )

    for tier_val, color, label in tier_specs:
        mask = tiers == tier_val
        if not mask.any():
            continue
        ypos = y[mask]
        ax_prob.hlines(ypos, 0, probs[mask], colors=color, alpha=0.22, linewidth=1.2)
        ax_prob.scatter(
            probs[mask], ypos, color=color, s=52, zorder=3, alpha=0.90, label=label
        )
        ax_se3.hlines(ypos, 0, se3[mask], colors=color, alpha=0.22, linewidth=1.2)
        ax_se3.scatter(se3[mask], ypos, color=color, s=52, zorder=3, alpha=0.90)

    ax_prob.set_xlabel("NRM mean reachability probability")
    ax_prob.set_xlim(0, 1)
    ax_prob.set_title("Probability  ▶ higher is better")

    max_se3 = se3.max() * 1.1 if se3.max() > 0 else 1.0
    ax_se3.set_xlabel("Validation mean SE3 error  [m + rad]")
    ax_se3.set_xlim(0, max_se3)
    ax_se3.set_title("SE3 error  ◀ lower is better")

    ax_prob.set_yticks(y)
    ax_prob.set_yticklabels(y_labels, fontsize=8)
    ax_prob.invert_yaxis()

    ax_prob.grid(True, axis="x", alpha=0.22)
    ax_se3.grid(True, axis="x", alpha=0.22)

    ax_prob.legend(loc="lower right", fontsize=8)
    fig.suptitle("Candidate selection ranking", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def create_candidate_selection_plots(
    csv_path: str | Path,
    output_dir: str | Path = "output/candidate_plots",
    fps: int = 24,
    num_frames: int = 180,
    dpi: int = 160,
) -> list[Path]:
    """Create both candidate-selection plots."""
    output_dir = Path(output_dir)
    mp4_paths = create_candidate_morphology_3d_mp4s(
        csv_path=csv_path,
        output_dir=output_dir,
        fps=fps,
        num_frames=num_frames,
        dpi=dpi,
    )
    png_path = create_candidate_ranking_chart(
        csv_path=csv_path,
        output_dir=output_dir,
        dpi=max(dpi, 200),
    )
    return [*mp4_paths, png_path]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create candidate-selection plots from optimization CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/candidate_plots"),
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-frames", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = create_candidate_selection_plots(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        fps=args.fps,
        num_frames=args.num_frames,
        dpi=args.dpi,
    )
    for path in paths:
        print(f"[candidate-selection plot] saved: {path}")


if __name__ == "__main__":
    main()
