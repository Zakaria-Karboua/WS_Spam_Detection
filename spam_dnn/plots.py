"""Figures for the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_loss_curves(histories, path, log_y=True):
    """Training loss (left) and training accuracy (right), one curve per run."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        epochs = range(1, len(h["loss"]) + 1)
        axes[0].plot(epochs, h["loss"], label=name)
        axes[1].plot(epochs, h["train_acc"], label=name)
    axes[0].set_title("Training loss (Binary Cross-Entropy)")
    axes[0].set_ylabel("loss")
    if log_y:
        axes[0].set_yscale("log")
    axes[1].set_title("Training accuracy")
    axes[1].set_ylabel("accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grad_flow(histories, path):
    """Gradient norm of each layer (input -> output) at the recorded epochs."""
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, squeeze=False)
    for ax, (name, h) in zip(axes[0], histories.items()):
        for epoch, norms in sorted(h["grad_norms"].items()):
            ax.plot(range(1, len(norms) + 1), norms, marker=".", label=f"epoch {epoch}")
        ax.set_title(name)
        ax.set_xlabel("layer (1 = first, last = output)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0][0].set_ylabel("gradient norm of the weights")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(results, path):
    """results: {name: metrics dict with tn, fp, fn, tp}."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.8), squeeze=False)
    for ax, (name, m) in zip(axes[0], results.items()):
        grid = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        ax.imshow(grid, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, grid[i][j], ha="center", va="center", fontsize=14)
        ax.set_xticks([0, 1], ["ham", "spam"])
        ax.set_yticks([0, 1], ["ham", "spam"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
