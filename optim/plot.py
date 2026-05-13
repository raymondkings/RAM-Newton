import torch
import matplotlib.pyplot as plt


def plot_link_lengths(
    history: torch.Tensor,
    save_path: str = "optim/figures/link_lengths.png",
    title: str = ""
) -> None:
    """Plot a and d link parameters over optimization iterations and save to file.

    Args:
        history: Tensor of shape [n_iter, n_links, 2] with raw lengths per iteration.
        save_path: File path for the saved figure.
        title: Figure-level title displayed above both subplots.
    """
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
    fig.legend(handles, labels, title="link index", loc="center left", bbox_to_anchor=(0.92, 0.5))
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_link_lengths_trajectory(
    history: torch.Tensor,
    save_path: str = "optim/figures/link_lengths_trajectory.png",
    title: str = "",
) -> None:
    """Plot each link's optimization path in (a, d) parameter space.

    Args:
        history: Tensor of shape [n_iter, n_links, 2] with raw lengths per iteration.
        save_path: File path for the saved figure.
        title: Figure-level title displayed above the plot.
    """
    data = history.numpy()  # [n_iter, n_links, 2]
    n_links = data.shape[1]
    link_colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(title, fontsize=11)

    for j in range(n_links):
        a = data[:, j, 0]
        d = data[:, j, 1]
        color = link_colors[j % len(link_colors)]
        ax.scatter(a[1:-1], d[1:-1], color=color, s=15, alpha=0.5, zorder=3, label=f"link {j}")
        ax.scatter(a[0], d[0], marker="^", color=color, s=80, zorder=5)
        ax.scatter(a[-1], d[-1], marker="s", color=color, s=80, zorder=5)

    # legend entries for start/end markers
    ax.scatter([], [], marker="^", color="gray", s=80, label="start")
    ax.scatter([], [], marker="s", color="gray", s=80, label="end")
    ax.legend(title="link index", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.set_xlabel("a — DH link length [m]")
    ax.set_ylabel("d — DH link offset [m]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
