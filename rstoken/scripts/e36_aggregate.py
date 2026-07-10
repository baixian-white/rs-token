"""E36 aggregate — continuous SNR sweep across 3 seeds × 2 model families.

Inputs (12 csvs):
  logs/paper_v05/e36_task_rvq_baseline_s{41,42,43}.csv
  logs/paper_v05/e36_task_rvq_distill_s{41,42,43}.csv
  logs/paper_v05/e36_recon_rvq_baseline_s{41,42,43}.csv
  logs/paper_v05/e36_recon_rvq_distill_s{41,42,43}.csv

Outputs:
  logs/paper_v05/e36_continuous_snr_raw.csv          long-form per-seed
  logs/paper_v05/e36_continuous_snr_mean_std.csv     long-form mean±std
  logs/paper_v05/e36_curve_h0_awgn.csv               wide-form: snr | baseline_mean,std | distill_mean,std
  logs/paper_v05/e36_curve_h0_rayleigh.csv
  logs/paper_v05/e36_curve_psnr_k4_awgn.csv
  logs/paper_v05/e36_curve_psnr_k4_rayleigh.csv
  logs/paper_v05/final_table_e36_snr_curve.md        human-readable summary
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "paper_v05"

SEEDS = (41, 42, 43)
FAMILIES = ("rvq_baseline", "rvq_distill")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ms(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------
# 1. Build long-form raw rows
# ---------------------------------------------------------------------------
raw_rows: list[dict] = []

for fam in FAMILIES:
    for seed in SEEDS:
        # task
        task_path = LOG / f"e36_task_{fam}_s{seed}.csv"
        for r in read_csv(task_path):
            raw_rows.append({
                "family": fam, "seed": seed,
                "metric": "h0_acc_pct",
                "channel": r["channel"], "snr": r["snr"],
                "k": int(r["k"]),
                "value": float(r["h0_acc"]) * 100.0,
            })
        # recon
        recon_path = LOG / f"e36_recon_{fam}_s{seed}.csv"
        for r in read_csv(recon_path):
            raw_rows.append({
                "family": fam, "seed": seed,
                "metric": "psnr_db",
                "channel": r["channel"], "snr": r["snr"],
                "k": int(r["k"]),
                "value": float(r["psnr"]),
            })
            raw_rows.append({
                "family": fam, "seed": seed,
                "metric": "lpips",
                "channel": r["channel"], "snr": r["snr"],
                "k": int(r["k"]),
                "value": float(r["lpips"]),
            })
            raw_rows.append({
                "family": fam, "seed": seed,
                "metric": "recon_cls_pct",
                "channel": r["channel"], "snr": r["snr"],
                "k": int(r["k"]),
                "value": float(r["recon_cls_acc"]) * 100.0,
            })

raw_out = LOG / "e36_continuous_snr_raw.csv"
with raw_out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["family", "seed", "metric",
                                       "channel", "snr", "k", "value"])
    w.writeheader()
    for r in raw_rows:
        w.writerow({**r, "value": f"{r['value']:.4f}"})

# ---------------------------------------------------------------------------
# 2. mean ± std long-form
# ---------------------------------------------------------------------------
mean_rows: list[dict] = []
keys = sorted({(r["family"], r["metric"], r["channel"], r["snr"], r["k"])
               for r in raw_rows})
for fam, metric, ch, snr, k in keys:
    vals = [r["value"] for r in raw_rows
            if r["family"] == fam and r["metric"] == metric
            and r["channel"] == ch and r["snr"] == snr and r["k"] == k]
    seeds_used = sorted({r["seed"] for r in raw_rows
                         if r["family"] == fam and r["metric"] == metric
                         and r["channel"] == ch and r["snr"] == snr
                         and r["k"] == k})
    m, s = ms(vals)
    mean_rows.append({
        "family": fam, "metric": metric, "channel": ch, "snr": snr,
        "k": k, "n_seeds": len(vals),
        "seeds": ",".join(str(x) for x in seeds_used),
        "mean": f"{m:.4f}", "std": f"{s:.4f}",
        "values": ",".join(f"{v:.4f}" for v in vals),
    })

mean_out = LOG / "e36_continuous_snr_mean_std.csv"
with mean_out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["family", "metric", "channel", "snr",
                                       "k", "n_seeds", "seeds",
                                       "mean", "std", "values"])
    w.writeheader()
    w.writerows(mean_rows)

# ---------------------------------------------------------------------------
# 3. Wide-form curves for plotting (one CSV per metric × channel)
# ---------------------------------------------------------------------------
def write_curve(metric: str, k: int, channel: str, file_name: str) -> None:
    rows = [r for r in mean_rows if r["metric"] == metric
            and r["k"] == k and r["channel"] == channel]
    # Sort by SNR; "inf" to the end (we only call this for awgn/rayleigh).
    def snr_key(r):
        try:
            return float(r["snr"])
        except ValueError:
            return float("inf")
    rows.sort(key=snr_key)

    # group by snr -> {family: (mean,std)}
    by_snr: dict[str, dict[str, tuple[str, str]]] = {}
    for r in rows:
        by_snr.setdefault(r["snr"], {})[r["family"]] = (r["mean"], r["std"])

    out = LOG / file_name
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snr_db",
                    "rvq_baseline_mean", "rvq_baseline_std",
                    "rvq_distill_mean",  "rvq_distill_std",
                    "delta_mean"])
        for snr in sorted(by_snr.keys(), key=lambda s: float(s)):
            b = by_snr[snr].get("rvq_baseline", ("", ""))
            d = by_snr[snr].get("rvq_distill",  ("", ""))
            try:
                delta = float(d[0]) - float(b[0])
                delta_s = f"{delta:+.4f}"
            except ValueError:
                delta_s = ""
            w.writerow([snr, b[0], b[1], d[0], d[1], delta_s])
    return out

paths = []
paths.append(write_curve("h0_acc_pct",   k=1, channel="awgn",     file_name="e36_curve_h0_awgn.csv"))
paths.append(write_curve("h0_acc_pct",   k=1, channel="rayleigh", file_name="e36_curve_h0_rayleigh.csv"))
paths.append(write_curve("psnr_db",      k=4, channel="awgn",     file_name="e36_curve_psnr_k4_awgn.csv"))
paths.append(write_curve("psnr_db",      k=4, channel="rayleigh", file_name="e36_curve_psnr_k4_rayleigh.csv"))

# ---------------------------------------------------------------------------
# 4. Markdown summary — distill gain envelope
# ---------------------------------------------------------------------------
md = LOG / "final_table_e36_snr_curve.md"
lines: list[str] = []
lines.append("# E36 — Continuous SNR sweep, distill vs. baseline (3 seeds)")
lines.append("")
lines.append("Seeds: 41, 42, 43 for both `rvq_baseline` and `rvq_distill`. "
             "Inference-only on existing v0.4 checkpoints. "
             "Test set: AID 30-class, n=1000.")
lines.append("")
lines.append("`condition_grid()` evaluates the same SNR list under AWGN and "
             "Rayleigh i.i.d. fading, with `none/inf` always inserted at the "
             "head of the grid for the no-channel baseline. SNR grid = "
             "{−5, −2, 0, 2, 5, 7, 10, 12, 15, 20} dB.")
lines.append("")

def render_curve(title: str, metric: str, k: int, channel: str, decimals: int):
    fmt = f"{{:.{decimals}f}}"
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| SNR (dB) | rvq_baseline (mean ± std) | rvq_distill (mean ± std) | Δ (distill − baseline) |")
    lines.append("|---|---|---|---|")
    rows = [r for r in mean_rows if r["metric"] == metric
            and r["k"] == k and r["channel"] == channel]
    by_snr: dict[str, dict[str, tuple[float, float]]] = {}
    for r in rows:
        by_snr.setdefault(r["snr"], {})[r["family"]] = (float(r["mean"]),
                                                         float(r["std"]))
    for snr in sorted(by_snr.keys(), key=lambda s: float(s)):
        b = by_snr[snr].get("rvq_baseline")
        d = by_snr[snr].get("rvq_distill")
        delta = d[0] - b[0]
        lines.append(
            f"| {snr} | {fmt.format(b[0])} ± {fmt.format(b[1])} | "
            f"{fmt.format(d[0])} ± {fmt.format(d[1])} | "
            f"{delta:+.{decimals}f} |"
        )
    lines.append("")

render_curve("Task path: h₀ accuracy at k=1 vs SNR — AWGN",
             "h0_acc_pct", 1, "awgn", 2)
render_curve("Task path: h₀ accuracy at k=1 vs SNR — Rayleigh",
             "h0_acc_pct", 1, "rayleigh", 2)
render_curve("Reconstruction: PSNR at k=4 vs SNR — AWGN",
             "psnr_db", 4, "awgn", 2)
render_curve("Reconstruction: PSNR at k=4 vs SNR — Rayleigh",
             "psnr_db", 4, "rayleigh", 2)

# Distill-gain envelope — where is the gap largest?
def envelope(metric: str, k: int, channel: str) -> tuple[str, float, float, float]:
    by_snr: dict[str, tuple[float, float]] = {}
    for r in mean_rows:
        if r["metric"] != metric or r["k"] != k or r["channel"] != channel:
            continue
        by_snr.setdefault(r["snr"], (None, None))
        m = float(r["mean"])
        if r["family"] == "rvq_baseline":
            by_snr[r["snr"]] = (m, by_snr[r["snr"]][1])
        else:
            by_snr[r["snr"]] = (by_snr[r["snr"]][0], m)
    best_snr, best_delta, b_at, d_at = "", float("-inf"), 0.0, 0.0
    for snr, (b, d) in by_snr.items():
        if b is None or d is None:
            continue
        delta = d - b
        if delta > best_delta:
            best_delta, best_snr, b_at, d_at = delta, snr, b, d
    return best_snr, best_delta, b_at, d_at

lines.append("## Distill-gain envelope")
lines.append("")
lines.append("| Metric | Channel | SNR @ max gain | Δ at peak | baseline | distill |")
lines.append("|---|---|---|---|---|---|")
for label, metric, k, ch, decimals in [
    ("h₀ acc (%)", "h0_acc_pct", 1, "awgn",     2),
    ("h₀ acc (%)", "h0_acc_pct", 1, "rayleigh", 2),
    ("PSNR k=4 (dB)", "psnr_db", 4, "awgn",     2),
    ("PSNR k=4 (dB)", "psnr_db", 4, "rayleigh", 2),
]:
    snr, dlt, b, d = envelope(metric, k, ch)
    fmt = f"{{:.{decimals}f}}"
    lines.append(f"| {label} | {ch} | {snr} dB | {dlt:+.{decimals}f} | "
                 f"{fmt.format(b)} | {fmt.format(d)} |")
lines.append("")

lines.append("## Curve CSVs (wide-form, ready for matplotlib)")
lines.append("")
for p in paths:
    lines.append(f"- `{p.relative_to(ROOT).as_posix()}`")
lines.append("")
lines.append("Source files:")
lines.append("- raw long-form: `logs/paper_v05/e36_continuous_snr_raw.csv`")
lines.append("- mean±std long-form: `logs/paper_v05/e36_continuous_snr_mean_std.csv`")
lines.append("- per-checkpoint task / recon CSVs: `logs/paper_v05/e36_*_s{41,42,43}.csv`")

md.write_text("\n".join(lines), encoding="utf-8")

print(f"raw  -> {raw_out}  ({len(raw_rows)} rows)")
print(f"mean -> {mean_out}  ({len(mean_rows)} rows)")
print(f"md   -> {md}")
for p in paths:
    print(f"curve -> {p}")
