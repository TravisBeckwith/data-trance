"""
data_trance.plotting
=====================
Renders the 3-panel distribution figure (histogram + normal curve overlay,
Q-Q plot, boxplot) used identically by the assess and validate steps, so
before/after images are directly comparable.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def distribution_panel(values: np.ndarray, title: str, out_path: str) -> str:
    """Render histogram+normal-overlay, Q-Q plot, and boxplot to a .jpg."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(title, fontsize=12)

    # --- histogram with normal curve overlay ---
    ax = axes[0]
    if len(v) > 1:
        ax.hist(v, bins=min(40, max(10, len(v) // 5)), density=True,
                color="#4C72B0", alpha=0.75, edgecolor="white")
        mu, sd = np.mean(v), np.std(v)
        if sd > 0:
            x = np.linspace(v.min(), v.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, sd), color="#C44E52", linewidth=2,
                    label="normal fit")
            ax.legend(fontsize=8)
    ax.set_title("Histogram", fontsize=10)
    ax.set_xlabel("value")
    ax.set_ylabel("density")

    # --- Q-Q plot ---
    ax = axes[1]
    if len(v) > 1:
        stats.probplot(v, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot", fontsize=10)
    ax.get_lines()[0].set_markerfacecolor("#4C72B0")
    ax.get_lines()[0].set_markeredgecolor("#4C72B0")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color("#C44E52")

    # --- boxplot ---
    ax = axes[2]
    if len(v) > 0:
        ax.boxplot(v, orientation="vertical", patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                    medianprops=dict(color="#C44E52", linewidth=2))
    ax.set_title("Boxplot", fontsize=10)
    ax.set_xticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110, format="jpg")
    plt.close(fig)
    return out_path
