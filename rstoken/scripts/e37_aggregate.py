"""E37 aggregate — placement counterfactual: All-layers 3-seed mean±std.

Inputs:
  logs/paper_v05/e37_task_s41_s43.csv          (seeds 41, 43, all-layers)
  logs/paper_v05/e37_recon_s41_s43.csv         (seeds 41, 43, all-layers)
  logs/paper_v05/e37_layered_probe_s41.csv
  logs/paper_v05/e37_layered_probe_s43.csv
  logs/paper_p0/e25_task_l0_vs_all.csv         (seed 42 main: rvq_distill_l0 + rvq_distill_all)
  logs/paper_p0/e25_recon_l0_vs_all.csv        (seed 42 main, both)
  logs/paper_p0/e25_layered_probe_all.csv      (seed 42 all-layers probe)

Outputs:
  logs/paper_v05/e37_placement_3seed_raw.csv
  logs/paper_v05/e37_placement_3seed_mean_std.csv
  logs/paper_v05/final_table_placement_3seed.md
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_V05 = ROOT / "logs" / "paper_v05"
LOG_P0 = ROOT / "logs" / "paper_p0"


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


def f3(x: float) -> str:
    return f"{x:.3f}"


def channel_key(row: dict) -> str:
    ch, snr = row["channel"], row["snr"]
    if ch == "none":
        return "none"
    return f"{ch}_{snr}"


# --- 1. Load all-layers data across 3 seeds ---------------------------------

# seed 42 came from e25 (rvq_distill_all = all-layers seed 42)
seed42_task = [r for r in read_csv(LOG_P0 / "e25_task_l0_vs_all.csv")
               if r["model"] == "rvq_distill_all"]
seed42_recon = [r for r in read_csv(LOG_P0 / "e25_recon_l0_vs_all.csv")
                if r["model"] == "rvq_distill_all"]
seed42_probe = read_csv(LOG_P0 / "e25_layered_probe_all.csv")

# seed 41 / 43 came from e37
new_task = read_csv(LOG_V05 / "e37_task_s41_s43.csv")
new_recon = read_csv(LOG_V05 / "e37_recon_s41_s43.csv")
probe_s41 = read_csv(LOG_V05 / "e37_layered_probe_s41.csv")
probe_s43 = read_csv(LOG_V05 / "e37_layered_probe_s43.csv")

raw_rows: list[dict] = []
# unify model name -> "all_layers" + add seed column
def tag(rows: list[dict], seed: int) -> list[dict]:
    out = []
    for r in rows:
        r2 = dict(r)
        r2["seed"] = str(seed)
        r2["model"] = "all_layers"
        out.append(r2)
    return out

raw_task = tag(seed42_task, 42) + [
    {**r, "seed": "41" if r["model"].endswith("s41") else "43", "model": "all_layers"}
    for r in new_task
]
raw_recon = tag(seed42_recon, 42) + [
    {**r, "seed": "41" if r["model"].endswith("s41") else "43", "model": "all_layers"}
    for r in new_recon
]

# --- 2. Aggregate task path: h0 acc per channel x seeds ---------------------

task_keys = sorted({channel_key(r) for r in raw_task})
task_summary: list[dict] = []
for ck in task_keys:
    vals = [float(r["h0_acc"]) * 100.0 for r in raw_task if channel_key(r) == ck]
    seeds = sorted({r["seed"] for r in raw_task if channel_key(r) == ck})
    m, s = ms(vals)
    task_summary.append({
        "metric": "h0_acc_pct",
        "channel": ck,
        "n_seeds": len(vals),
        "seeds": ",".join(seeds),
        "mean": f3(m),
        "std": f3(s),
        "values": ",".join(f3(v) for v in vals),
    })

# --- 3. Aggregate recon path: PSNR / LPIPS / recon_cls at k=4 ---------------

recon_summary: list[dict] = []
for ck in task_keys:  # same channel keys
    rows_k4 = [r for r in raw_recon if channel_key(r) == ck and int(r["k"]) == 4]
    seeds = sorted({r["seed"] for r in rows_k4})
    psnr_vals = [float(r["psnr"]) for r in rows_k4]
    lpips_vals = [float(r["lpips"]) for r in rows_k4]
    cls_vals = [float(r["recon_cls_acc"]) * 100.0 for r in rows_k4]
    for label, vals in [("psnr_db_k4", psnr_vals),
                         ("lpips_k4", lpips_vals),
                         ("recon_cls_pct_k4", cls_vals)]:
        m, s = ms(vals)
        recon_summary.append({
            "metric": label,
            "channel": ck,
            "n_seeds": len(vals),
            "seeds": ",".join(seeds),
            "mean": f3(m),
            "std": f3(s),
            "values": ",".join(f3(v) for v in vals),
        })

# --- 4. Aggregate layered probe (cumulative codeword test_acc) --------------

def probe_pairs(rows: list[dict]) -> dict[int, float]:
    return {int(r["k"]): float(r["test_acc"]) for r in rows}

probe_by_seed = {41: probe_pairs(probe_s41),
                 42: probe_pairs(seed42_probe),
                 43: probe_pairs(probe_s43)}

probe_summary: list[dict] = []
for k in (1, 2, 3, 4):
    vals = [probe_by_seed[s][k] for s in (41, 42, 43)]
    m, s = ms(vals)
    probe_summary.append({
        "metric": f"layered_probe_k{k}_pct",
        "channel": "none",
        "n_seeds": len(vals),
        "seeds": "41,42,43",
        "mean": f3(m),
        "std": f3(s),
        "values": ",".join(f3(v) for v in vals),
    })

# --- 5. Write CSVs ----------------------------------------------------------

raw_out = LOG_V05 / "e37_placement_3seed_raw.csv"
mean_out = LOG_V05 / "e37_placement_3seed_mean_std.csv"

# raw: union of task + recon + probe at row-grain
raw_rows_out: list[dict] = []
for r in raw_task:
    raw_rows_out.append({
        "seed": r["seed"], "metric": "h0_acc_pct", "channel": channel_key(r),
        "k": "1", "value": f3(float(r["h0_acc"]) * 100.0),
    })
for r in raw_recon:
    raw_rows_out.append({
        "seed": r["seed"], "metric": "psnr_db", "channel": channel_key(r),
        "k": r["k"], "value": f3(float(r["psnr"])),
    })
    raw_rows_out.append({
        "seed": r["seed"], "metric": "lpips", "channel": channel_key(r),
        "k": r["k"], "value": f3(float(r["lpips"])),
    })
    raw_rows_out.append({
        "seed": r["seed"], "metric": "recon_cls_pct", "channel": channel_key(r),
        "k": r["k"], "value": f3(float(r["recon_cls_acc"]) * 100.0),
    })
for sd, pp in probe_by_seed.items():
    for k, v in pp.items():
        raw_rows_out.append({
            "seed": str(sd), "metric": "layered_probe_pct", "channel": "none",
            "k": str(k), "value": f3(v),
        })

with raw_out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["seed", "metric", "channel", "k", "value"])
    w.writeheader()
    w.writerows(raw_rows_out)

mean_rows = task_summary + recon_summary + probe_summary
with mean_out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["metric", "channel", "n_seeds", "seeds",
                                       "mean", "std", "values"])
    w.writeheader()
    w.writerows(mean_rows)

# --- 6. Markdown table comparing to paper Table II "L0 only (main)" --------

# Paper Table II L0-only column (rs_token_v0.5.tex/PDF page 6, single seed s42).
# Hard-coded for narrative comparison only; raw numbers in e37 raw CSV above.
TABLE2_L0 = {
    ("h0_acc_pct", "none"):          82.6,
    ("h0_acc_pct", "awgn_5"):        82.1,
    ("h0_acc_pct", "awgn_10"):       82.6,
    ("h0_acc_pct", "rayleigh_5"):    58.6,
    ("h0_acc_pct", "rayleigh_10"):   77.7,
    ("psnr_db_k4", "none"):          25.92,
    ("psnr_db_k4", "awgn_5"):        23.94,
    ("psnr_db_k4", "awgn_10"):       25.92,
    ("psnr_db_k4", "rayleigh_5"):    16.93,
    ("psnr_db_k4", "rayleigh_10"):   20.63,
    ("recon_cls_pct_k4", "none"):    86.9,
    ("recon_cls_pct_k4", "awgn_5"):  84.6,
    ("recon_cls_pct_k4", "awgn_10"): 86.9,
    ("recon_cls_pct_k4", "rayleigh_5"):  22.6,
    ("recon_cls_pct_k4", "rayleigh_10"): 66.8,
    ("layered_probe_k1_pct", "none"): 86.0,
    ("layered_probe_k2_pct", "none"): 87.8,
    ("layered_probe_k3_pct", "none"): 87.9,
    ("layered_probe_k4_pct", "none"): 87.9,
}

def lookup(metric: str, ch: str) -> tuple[str, str]:
    for r in mean_rows:
        if r["metric"] == metric and r["channel"] == ch:
            return r["mean"], r["std"]
    return "-", "-"

md_lines: list[str] = []
md_lines.append("# E37 — Placement counterfactual: All-layers 3-seed mean±std")
md_lines.append("")
md_lines.append("Seeds: 41 (E37), 42 (E25 main), 43 (E37). Same training recipe; "
                "only `seed`, `run_name`, `logging.*` differ across seeds.")
md_lines.append("")
md_lines.append("Comparison column: paper Table II `L0 only (main)` (single seed 42, "
                "values copied from rs_token_v0.5.pdf p.6).")
md_lines.append("")
md_lines.append("## Task path: h0 / L0 bag-of-words accuracy (k=1, %)")
md_lines.append("")
md_lines.append("| Channel | L0-only (Table II, %) | All-layers 3-seed mean ± std (%) | Δ vs L0-only (pp) |")
md_lines.append("|---|---|---|---|")
for ch in ("none", "awgn_5", "awgn_10", "rayleigh_5", "rayleigh_10"):
    m, s = lookup("h0_acc_pct", ch)
    l0 = TABLE2_L0[("h0_acc_pct", ch)]
    delta = float(m) - l0 if m != "-" else float("nan")
    md_lines.append(f"| {ch} | {l0:.1f} | {m} ± {s} | {delta:+.2f} |")

md_lines.append("")
md_lines.append("## Reconstruction path at k=4")
md_lines.append("")
md_lines.append("| Channel | Metric | L0-only (Table II) | All-layers 3-seed mean ± std | Δ vs L0-only |")
md_lines.append("|---|---|---|---|---|")
for ch in ("none", "awgn_5", "awgn_10", "rayleigh_5", "rayleigh_10"):
    for metric, label, fmt in (
        ("psnr_db_k4", "PSNR (dB)", "{:.2f}"),
        ("recon_cls_pct_k4", "recon-cls (%)", "{:.1f}"),
    ):
        m, s = lookup(metric, ch)
        l0 = TABLE2_L0[(metric, ch)]
        delta = float(m) - l0 if m != "-" else float("nan")
        md_lines.append(f"| {ch} | {label} | {fmt.format(l0)} | {m} ± {s} | {delta:+.2f} |")

md_lines.append("")
md_lines.append("## Layered linear probe on cumulative codewords (no-channel, %)")
md_lines.append("")
md_lines.append("| k | L0-only (Table II) | All-layers 3-seed mean ± std | Δ vs L0-only (pp) |")
md_lines.append("|---|---|---|---|")
for k in (1, 2, 3, 4):
    metric = f"layered_probe_k{k}_pct"
    m, s = lookup(metric, "none")
    l0 = TABLE2_L0[(metric, "none")]
    delta = float(m) - l0 if m != "-" else float("nan")
    md_lines.append(f"| {k} | {l0:.1f} | {m} ± {s} | {delta:+.2f} |")

md_lines.append("")
md_lines.append("## Per-seed raw values used")
md_lines.append("")
md_lines.append("Each row in `e37_placement_3seed_raw.csv` is one (seed, metric, "
                "channel, k) cell; this markdown only summarises mean±std.")
md_lines.append("")
md_lines.append("Source files:")
md_lines.append("- seed 41/43 task: `logs/paper_v05/e37_task_s41_s43.csv`")
md_lines.append("- seed 41/43 recon: `logs/paper_v05/e37_recon_s41_s43.csv`")
md_lines.append("- seed 41/43 probe: `logs/paper_v05/e37_layered_probe_s41.csv`, "
                "`logs/paper_v05/e37_layered_probe_s43.csv`")
md_lines.append("- seed 42 task/recon: `logs/paper_p0/e25_*.csv` (rvq_distill_all)")
md_lines.append("- seed 42 probe: `logs/paper_p0/e25_layered_probe_all.csv`")

(LOG_V05 / "final_table_placement_3seed.md").write_text(
    "\n".join(md_lines), encoding="utf-8")

print(f"raw  -> {raw_out}  ({len(raw_rows_out)} rows)")
print(f"mean -> {mean_out}  ({len(mean_rows)} rows)")
print(f"md   -> {LOG_V05 / 'final_table_placement_3seed.md'}")
