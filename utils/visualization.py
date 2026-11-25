"""Module 3/4 - Scientific figures (Matplotlib 3D, white background, 300 DPI PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 300


def _style_axes(ax, lim=1.0):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_facecolor("white")


def plot_saliency_comparison(path, coords, importance, title_left="Original point cloud",
                             title_right="Gradient saliency", suptitle=None,
                             elev=22, azim=-60, cmap="inferno"):
    """Left: raw cloud. Right: same cloud coloured by importance in [0, 1]."""
    coords = np.asarray(coords); imp = np.asarray(importance)
    imp = np.clip(imp / max(np.quantile(imp, 0.98), 1e-12), 0, 1)   # robust display scaling (heavy-tailed gradients)
    order = np.argsort(imp)                      # draw hottest points last (on top)
    c, s = coords[order], imp[order]
    lim = float(np.abs(coords).max()) * 1.02

    fig = plt.figure(figsize=(11, 5.4), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=5, c="#4a4a4a", alpha=0.9,
                linewidths=0, depthshade=True)
    ax1.set_title(title_left, fontsize=12)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax2.scatter(c[:, 0], c[:, 1], c[:, 2], s=5 + 22 * s ** 2, c=s, cmap=cmap, vmin=0,
                     vmax=1, linewidths=0, depthshade=False)
    ax2.set_title(title_right, fontsize=12)
    for ax in (ax1, ax2):
        _style_axes(ax, lim)
        ax.view_init(elev=elev, azim=azim)
    cb = fig.colorbar(sc, ax=[ax2], shrink=0.65, pad=0.02)
    cb.set_label("Geometric importance (gradient norm, 98th-pct clipped)", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(path, history):
    """history: dict with lists train_loss, test_loss, train_acc, test_acc."""
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
    a1.plot(ep, history["train_loss"], label="train", lw=2)
    a1.plot(ep, history["test_loss"], label="test", lw=2)
    a1.set_xlabel("Epoch"); a1.set_ylabel("Cross-entropy loss"); a1.set_title("Loss")
    a2.plot(ep, np.array(history["train_acc"]) * 100, label="train", lw=2)
    a2.plot(ep, np.array(history["test_acc"]) * 100, label="test", lw=2)
    a2.set_xlabel("Epoch"); a2.set_ylabel("Accuracy (%)"); a2.set_title("Overall accuracy")
    for a in (a1, a2):
        a.grid(alpha=0.3); a.legend(frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)


def plot_deletion_curve(path, res):
    f = np.array(res["fractions"]) * 100
    fig, ax = plt.subplots(figsize=(5.5, 4), facecolor="white")
    ax.plot(f, np.array(res["salient"]) * 100, "o-", lw=2, label="remove most salient points")
    ax.plot(f, np.array(res["random"]) * 100, "s--", lw=2, label="remove random points")
    ax.set_xlabel("Points removed (%)"); ax.set_ylabel("Prediction unchanged (%)")
    ax.set_title("Saliency fidelity (deletion test)")
    ax.grid(alpha=0.3); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
