"""Generate v0.4 paper-ready figures from finalized CSV tables.

Outputs:
  figs/fig_v04_seed_task_stats.{pdf,png}
  figs/fig_v04_reconstruction_path.{pdf,png}
  figs/fig_v04_external_baseline.{pdf,png}
  figs/fig_v04_rayleigh0_context.{pdf,png}
"""
from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = PROJECT_ROOT / "figs"
TABLE_DIR = PROJECT_ROOT / "logs" / "v04_tables"
LOGS_DIR = PROJECT_ROOT / "logs"

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 220,
    }
)

COLORS = {
    "baseline": "#6F6F6F",
    "distill": "#C44E52",
    "webp": "#4C72B0",
    "jpeg2000": "#55A868",
    "awgn": "#DD8452",
    "rayleigh": "#8172B2",
    "none": "#333333",
}


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def save(fig: plt.Figure, name: str) -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIGS_DIR / f"{name}.pdf"
    png = FIGS_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


def fig_seed_task_stats() -> None:
    rows = read_csv(TABLE_DIR / "table2_task_path_mean_std.csv")
    metric_labels = [
        ("no-channel h0/L0_bow acc", "No ch."),
        ("AWGN +5 h0 acc", "AWGN\n+5"),
        ("AWGN +10 h0 acc", "AWGN\n+10"),
        ("Rayleigh +5 h0 acc", "Rayleigh\n+5"),
        ("Rayleigh +10 h0 acc", "Rayleigh\n+10"),
    ]
    models = ["rvq_baseline", "rvq_distill"]
    x = np.arange(len(metric_labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for idx, model in enumerate(models):
        means = []
        stds = []
        for metric, _ in metric_labels:
            row = next(r for r in rows if r["model"] == model and r["metric"] == metric)
            means.append(fnum(row["mean"]))
            stds.append(fnum(row["std"]))
        offset = (idx - 0.5) * width
        label = "RVQ baseline" if model == "rvq_baseline" else "RS-Token"
        color = COLORS["baseline"] if model == "rvq_baseline" else COLORS["distill"]
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            color=color,
            label=label,
            alpha=0.92,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 2.0,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metric_labels])
    ax.set_ylabel("h0 / L0_bow task accuracy (%)")
    ax.set_ylim(0, 92)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(framealpha=0.95, loc="upper left")
    ax.set_title("Task-path robustness across 3 model seeds")
    ax.text(
        0.0,
        -0.20,
        "Aggregation: channel seeds averaged within each model seed, then mean +/- std across model seeds.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    save(fig, "fig_v04_seed_task_stats")


def fig_reconstruction_path() -> None:
    rows = [
        r
        for r in read_csv(TABLE_DIR / "table3_reconstruction_path.csv")
        if r["model"] == "rvq_distill"
    ]
    series = [
        ("none", "inf", "No channel", COLORS["none"], "-"),
        ("awgn", "5", "AWGN +5", COLORS["awgn"], "-"),
        ("awgn", "10", "AWGN +10", "#D55E00", "--"),
        ("rayleigh", "10", "Rayleigh +10", COLORS["rayleigh"], "-"),
        ("rayleigh", "0", "Rayleigh 0", "#999999", ":"),
    ]

    fig, (ax_psnr, ax_cls) = plt.subplots(1, 2, figsize=(9.0, 3.9), sharex=True)
    for channel, snr, label, color, linestyle in series:
        subset = [
            r for r in rows if r["channel"] == channel and r["snr"] == snr
        ]
        if not subset:
            continue
        subset = sorted(subset, key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in subset]
        psnr = [fnum(r["psnr"]) for r in subset]
        cls = [fnum(r["recon_cls_acc"]) * 100.0 for r in subset]
        ax_psnr.plot(ks, psnr, marker="o", color=color, linestyle=linestyle, label=label)
        ax_cls.plot(ks, cls, marker="o", color=color, linestyle=linestyle, label=label)

    ax_psnr.set_title("Reconstruction fidelity")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_xlabel("transmitted layers k")
    ax_psnr.set_xticks([1, 2, 3, 4])
    ax_psnr.grid(alpha=0.25)

    ax_cls.set_title("Reconstruction classifier")
    ax_cls.set_ylabel("AID top-1 (%)")
    ax_cls.set_xlabel("transmitted layers k")
    ax_cls.set_xticks([1, 2, 3, 4])
    ax_cls.grid(alpha=0.25)
    ax_cls.legend(framealpha=0.95, loc="lower right")

    fig.suptitle("Reconstruction path: k=1..4 evaluated with PSNR / LPIPS / E16 classifier", y=1.02)
    save(fig, "fig_v04_reconstruction_path")


def fig_external_baseline() -> None:
    rows = read_csv(TABLE_DIR / "table4_external_baseline.csv")
    methods = ["webp", "jpeg2000"]
    budgets = [2560, 5120, 10240]
    panels = [("none", "inf", "No channel"), ("awgn", "10", "AWGN +10 dB")]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), sharey=True)
    for ax, (channel, snr, title) in zip(axes, panels):
        x = np.arange(len(budgets))
        width = 0.34
        for idx, method in enumerate(methods):
            vals = []
            fails = []
            for budget in budgets:
                row = next(
                    r
                    for r in rows
                    if r["method"] == method
                    and int(r["target_bits"]) == budget
                    and r["channel"] == channel
                    and r["snr"] == snr
                )
                vals.append(fnum(row["cls_acc_all"]) * 100.0)
                fails.append(fnum(row["decode_failure_rate"]) * 100.0)
            offset = (idx - 0.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                color=COLORS[method],
                label=method.upper() if method == "webp" else "JPEG2000",
                alpha=0.9,
            )
            for bar, val, fail in zip(bars, vals, fails):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 1.2,
                    f"{val:.1f}\nfail {fail:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_title(title)
        ax.set_xlabel("target bits / image")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("classifier accuracy, decode failures counted wrong (%)")
    axes[1].legend(
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
    )
    fig.suptitle("External classic baselines: unprotected compressed bitstream over BPSK", y=1.02)
    save(fig, "fig_v04_external_baseline")


def fig_rayleigh0_context() -> None:
    rows = read_csv(TABLE_DIR / "table5_rayleigh0_context.csv")
    selected = [
        ("rvq_baseline", "task", "Baseline\nh0"),
        ("rvq_distill", "task", "RS-Token\nh0"),
        ("rvq_distill", "reconstruction", "RS-Token\nrecon k=1"),
        ("rvq_distill", "reconstruction", "RS-Token\nrecon k=4"),
        ("webp_target_10240", "decode_then_classify", "WebP\n10240"),
        ("jpeg2000_target_10240", "decode_then_classify", "JPEG2000\n10240"),
    ]
    values = []
    labels = []
    colors = []
    for system, path, label in selected:
        candidates = [r for r in rows if r["system"] == system and r["path"] == path]
        if system == "rvq_distill" and path == "reconstruction":
            want_k = "1" if "k=1" in label else "4"
            candidates = [r for r in candidates if r["k"] == want_k]
        if not candidates:
            continue
        row = candidates[0]
        if path == "task":
            value = fnum(row["h0_acc"]) * 100.0
            color = COLORS["baseline"] if system == "rvq_baseline" else COLORS["distill"]
        elif path == "reconstruction":
            value = fnum(row["recon_cls_acc"]) * 100.0
            color = COLORS["distill"]
        else:
            value = fnum(row["cls_acc_all"]) * 100.0
            color = COLORS["webp"] if system.startswith("webp") else COLORS["jpeg2000"]
        values.append(value)
        labels.append(label)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy at Rayleigh 0 dB (%)")
    ax.set_ylim(0, 22)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Rayleigh 0 dB is a breakdown / stress boundary, not an operating claim")
    ax.text(
        0.0,
        -0.23,
        "Classic compressed bitstreams are unprotected; decode failures are counted as wrong.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    save(fig, "fig_v04_rayleigh0_context")


def main() -> None:
    fig_seed_task_stats()
    fig_reconstruction_path()
    fig_external_baseline()
    fig_rayleigh0_context()


if __name__ == "__main__":
    main()
