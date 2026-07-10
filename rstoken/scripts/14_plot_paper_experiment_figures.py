"""Generate paper experiment figures for RS-Token v0.4.

Figures:
- fig_exp_l0_task_robustness: task-path h0/L0-BoW accuracy, mean +/- std.
- fig_exp_progressive_reconstruction: reconstruction-path metrics over k.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")


def plot_l0_task_robustness() -> None:
    conditions = [
        "No\nchannel",
        "AWGN\n+5 dB",
        "AWGN\n+10 dB",
        "Rayleigh\n+5 dB",
        "Rayleigh\n+10 dB",
    ]
    baseline = np.array([58.23, 55.73, 58.20, 28.67, 48.63])
    baseline_std = np.array([1.57, 0.72, 1.59, 0.65, 1.31])
    rstoken = np.array([83.33, 82.57, 83.37, 58.57, 78.80])
    rstoken_std = np.array([0.81, 0.31, 0.76, 0.47, 0.72])

    x = np.arange(len(conditions))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    ax.bar(
        x - width / 2,
        baseline,
        width,
        yerr=baseline_std,
        capsize=3,
        label="RVQ baseline",
        color="#4C78A8",
        edgecolor="#1f2937",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        rstoken,
        width,
        yerr=rstoken_std,
        capsize=3,
        label="RS-Token",
        color="#E45756",
        edgecolor="#1f2937",
        linewidth=0.5,
    )

    for idx, (base_value, rstoken_value) in enumerate(zip(baseline, rstoken)):
        ax.text(
            idx,
            max(base_value, rstoken_value) + 4.2,
            f"+{rstoken_value - base_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#7f1d1d",
        )

    ax.set_title("L0 task-path robustness under channel noise")
    ax.set_ylabel("h0 / L0 BoW accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 96)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=True)
    ax.text(
        0.99,
        0.08,
        "Task path only, k = 1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout()
    save_figure(fig, "fig_exp_l0_task_robustness")
    plt.close(fig)


def plot_progressive_reconstruction() -> None:
    k = np.array([1, 2, 3, 4])
    no_channel = {
        "PSNR (dB)": np.array([22.95, 24.77, 25.57, 25.92]),
        "LPIPS ?": np.array([0.284, 0.205, 0.183, 0.175]),
        "Recon. cls. acc. (%)": np.array([70.3, 84.3, 86.4, 86.9]),
    }
    awgn_5db = {
        "PSNR (dB)": np.array([21.97, 23.24, 23.76, 23.96]),
        "LPIPS ?": np.array([0.312, 0.244, 0.223, 0.217]),
        "Recon. cls. acc. (%)": np.array([67.3, 81.6, 84.1, 84.3]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharex=True)
    for ax, metric in zip(axes, no_channel):
        ax.plot(
            k,
            no_channel[metric],
            "-o",
            color="#333333",
            label="No channel",
            linewidth=2.0,
            markersize=5,
        )
        ax.plot(
            k,
            awgn_5db[metric],
            "-s",
            color="#F58518",
            label="AWGN +5 dB",
            linewidth=2.0,
            markersize=5,
        )
        ax.set_title(metric)
        ax.set_xlabel("Transmitted RVQ layers k")
        ax.set_xticks(k)
        ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.7)

        values = np.concatenate([no_channel[metric], awgn_5db[metric]])
        span = values.max() - values.min()
        pad = 0.08 * span if span > 1e-9 else 0.1
        ax.set_ylim(values.min() - pad, values.max() + pad)

    axes[0].set_ylabel("Value")
    axes[0].legend(loc="lower right", frameon=True)
    fig.suptitle("Progressive reconstruction from additional RVQ layers", y=1.04, fontsize=13)
    fig.text(
        0.5,
        -0.02,
        "Reconstruction path only; h0 / L0 BoW is not used for these k = 1..4 claims.",
        ha="center",
        fontsize=9.5,
        color="#4b5563",
    )
    fig.tight_layout()
    save_figure(fig, "fig_exp_progressive_reconstruction")
    plt.close(fig)


def main() -> None:
    plot_l0_task_robustness()
    plot_progressive_reconstruction()
    print(FIG_DIR / "fig_exp_l0_task_robustness.png")
    print(FIG_DIR / "fig_exp_progressive_reconstruction.png")


if __name__ == "__main__":
    main()
