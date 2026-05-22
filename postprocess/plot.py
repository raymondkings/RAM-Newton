# -----------------------------------------------------------------------------
# work Copyright (c) 2026 Julian Arkenau
# -----------------------------------------------------------------------------
# Postprocess plotting utilities for CSV logs created by optim.nrm.
# -----------------------------------------------------------------------------
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from util.csv_log_reader import read_optimization_csv, tensor_from_json_cell


def plot_ik_fk_trajectory(
    steps: list[int],
    pos_errs: list[float],
    rot_errs: list[float],
    se3_dists: list[float],
    save_path: str | Path,
    title: str = "IK→FK metrics over optimization",
) -> None:
    """Line plot of IK/FK validation errors during optimization."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(title, fontsize=11)

    for ax, values, ylabel, color in [
        (axes[0], pos_errs, "pos err [m]", "steelblue"),
        (axes[1], rot_errs, "rot err [rad]", "darkorange"),
        (axes[2], se3_dists, "SE3 distance", "mediumpurple"),
    ]:
        ax.plot(steps, values, color=color, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(linestyle="--", alpha=0.4)

    axes[2].set_xlabel("optimization step")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_link_lengths(
    history: torch.Tensor,
    save_path: str | Path,
    title: str = "",
) -> None:
    """Plot raw a and d link parameters over optimization iterations.

    Args:
        history:
            Tensor [n_steps, n_links, 2], usually raw morphology columns a,d.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    iters = range(len(history))
    n_links = history.shape[1]

    fig, (ax_a, ax_d) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=11)

    for j in range(n_links):
        ax_a.plot(iters, history[:, j, 0].numpy(), label=f"link {j}")
        ax_d.plot(iters, history[:, j, 1].numpy(), label=f"link {j}")

    ax_a.set_title("a — DH link length")
    ax_a.set_xlabel("iteration")
    ax_a.set_ylabel("a [m]")

    ax_d.set_title("d — DH link offset")
    ax_d.set_xlabel("iteration")
    ax_d.set_ylabel("d [m]")

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="link index",
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
    )
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_link_lengths_trajectory(
    history: torch.Tensor,
    save_path: str | Path,
    title: str = "",
) -> None:
    """Plot each link's optimization path in (a, d) parameter space.

    Args:
        history:
            Tensor [n_steps, n_links, 2], usually raw morphology columns a,d.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = history.numpy()
    n_links = data.shape[1]
    link_colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(title, fontsize=11)

    for j in range(n_links):
        a = data[:, j, 0]
        d = data[:, j, 1]
        color = link_colors[j % len(link_colors)]

        if len(a) > 2:
            ax.scatter(
                a[1:-1],
                d[1:-1],
                color=color,
                s=15,
                alpha=0.5,
                zorder=3,
                label=f"link {j}",
            )
        ax.scatter(a[0], d[0], marker="^", color=color, s=80, zorder=5)
        ax.scatter(a[-1], d[-1], marker="s", color=color, s=80, zorder=5)

    ax.scatter([], [], marker="^", color="gray", s=80, label="start")
    ax.scatter([], [], marker="s", color="gray", s=80, label="end")
    ax.legend(
        title="link index", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True
    )
    ax.set_xlabel("a — DH link length [m]")
    ax.set_ylabel("d — DH link offset [m]")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_plots_from_csv(
    csv_path: str | Path,
    output_dir: str | Path = "output/figures",
) -> list[Path]:
    """Create the same basic plots that the old nrm.py created online.

    This function reads the CSV and produces:
        - raw link lengths over iterations
        - raw link length trajectory in (a, d)
        - IK/FK validation metrics over validation iterations

    Returns:
        A list of created figure paths.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_optimization_csv(csv_path)
    stem = csv_path.stem

    created: list[Path] = []

    raw_morphologies = [
        tensor_from_json_cell(row.get("raw_morphology_json"))
        for row in rows
        if row.get("raw_morphology_json") is not None
    ]

    if raw_morphologies:
        raw_history = torch.stack(raw_morphologies)[:, :, 1:3].cpu()

        link_lengths_path = output_dir / f"{stem}_link_lengths.png"
        plot_link_lengths(
            raw_history,
            save_path=link_lengths_path,
            title="Raw link lengths during optimization (before normalisation/squashing)",
        )
        created.append(link_lengths_path)

        trajectory_path = output_dir / f"{stem}_link_lengths_trajectory.png"
        plot_link_lengths_trajectory(
            raw_history,
            save_path=trajectory_path,
            title="Raw link lengths during optimization (before normalisation/squashing)",
        )
        created.append(trajectory_path)

    validation_rows = [row for row in rows if row.get("best_se3_dist_mean") is not None]

    if validation_rows:
        steps = [row["iteration"] for row in validation_rows]
        pos_errs = [row["best_pos_err_mean"] for row in validation_rows]
        rot_errs = [row["best_rot_err_mean"] for row in validation_rows]
        se3_dists = [row["best_se3_dist_mean"] for row in validation_rows]

        ik_fk_path = output_dir / f"{stem}_ik_fk_trajectory.png"
        plot_ik_fk_trajectory(
            steps=steps,
            pos_errs=pos_errs,
            rot_errs=rot_errs,
            se3_dists=se3_dists,
            save_path=ik_fk_path,
        )
        created.append(ik_fk_path)

    return created
