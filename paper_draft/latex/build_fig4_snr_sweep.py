# -*- coding: utf-8 -*-
"""Build the four-panel continuous-SNR figure used by RS-Token v0.6."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "rstoken" / "logs" / "paper_v05"
FIG_DIR = ROOT / "rstoken" / "figs"

PANELS = (
    ("e36_curve_h0_awgn.csv", "(a) Task accuracy under AWGN", "Accuracy (%)"),
    ("e36_curve_h0_rayleigh.csv", "(b) Task accuracy under Rayleigh fading", "Accuracy (%)"),
    ("e36_curve_psnr_k4_awgn.csv", r"(c) $k=4$ PSNR under AWGN", "PSNR (dB)"),
    ("e36_curve_psnr_k4_rayleigh.csv", r"(d) $k=4$ PSNR under Rayleigh fading", "PSNR (dB)"),
)


def build() -> tuple[Path, Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.9), constrained_layout=True)
    colors = {"baseline": "#4C566A", "distill": "#C13B32"}

    for ax, (file_name, title, ylabel) in zip(axes.flat, PANELS):
        frame = pd.read_csv(LOG_DIR / file_name)
        x = frame["snr_db"].to_numpy(dtype=float)
        for key, label in (("rvq_baseline", "RVQ baseline"),
                           ("rvq_distill", "RS-Token (distilled)")):
            mean = frame[f"{key}_mean"].to_numpy(dtype=float)
            std = frame[f"{key}_std"].to_numpy(dtype=float)
            color = colors["baseline" if key == "rvq_baseline" else "distill"]
            ax.plot(x, mean, color=color, marker="o", markersize=3.5,
                    linewidth=1.5, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16,
                            linewidth=0)
        ax.set_title(title, pad=5)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIG_DIR / "fig_v06_snr_sweep.png"
    pdf_path = FIG_DIR / "fig_v06_snr_sweep.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = build()
    print(png)
    print(pdf)
