"""E40 — Init-seed paired control aggregation.

Inputs:
  logs/paper_v05/e40_task_paired.csv       (6 models × 5 channels)
  logs/paper_v05/e40_recon_paired.csv      (3 distill × 5 channels × k=1..4)
  logs/paper_v05/e40_layered_probe_paired_s{41,42,43}.csv (optional)

Reference (paper Table I, unpaired 3-seed mean ± std):
  logs/v04_tables/table2_task_path_mean_std.csv

Outputs:
  logs/paper_v05/e40_paired_h0_3seed.csv
  logs/paper_v05/e40_paired_recon_3seed.csv
  logs/paper_v05/final_table_e40_paired.md
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_V05 = ROOT / "logs" / "paper_v05"
LOG_V04 = ROOT / "logs" / "v04_tables"


def read_csv(path: Path) -> list[dict]:
    # utf-8-sig handles PowerShell-written BOM (lesson 14 in experiment log).
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ms(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def f2(x: float) -> str:
    return f"{x:.2f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def channel_key(row: dict) -> str:
    ch, snr = row["channel"], row["snr"]
    if ch == "none":
        return "none"
    return f"{ch}_{snr}"


CHANNELS_ORDER = ("none", "awgn_5", "awgn_10", "rayleigh_5", "rayleigh_10")
CHANNEL_LABELS = {
    "none":        "no-channel",
    "awgn_5":      "AWGN +5 dB",
    "awgn_10":     "AWGN +10 dB",
    "rayleigh_5":  "Rayleigh +5 dB",
    "rayleigh_10": "Rayleigh +10 dB",
}


# --- 1. Unpaired reference (paper Table I, 3-seed mean±std) -----------------

# Mapping from `metric` column in table2_task_path_mean_std.csv to channel_key.
TABLE1_KEY = {
    "no-channel h0/L0_bow acc":  "none",
    "AWGN +5 h0 acc":            "awgn_5",
    "AWGN +10 h0 acc":           "awgn_10",
    "Rayleigh +5 h0 acc":        "rayleigh_5",
    "Rayleigh +10 h0 acc":       "rayleigh_10",
}


def load_unpaired_ref() -> dict[tuple[str, str], tuple[float, float]]:
    """Returns {(family, channel_key): (mean, std)} from Table I."""
    ref = {}
    rows = read_csv(LOG_V04 / "table2_task_path_mean_std.csv")
    for r in rows:
        if r["metric"] not in TABLE1_KEY:
            continue
        ch = TABLE1_KEY[r["metric"]]
        ref[(r["model"], ch)] = (float(r["mean"]), float(r["std"]))
    return ref


# --- 2. Load paired task path csv ------------------------------------------

def parse_seed(model_name: str) -> int:
    # rvq_baseline_paired_s41 -> 41
    return int(model_name.rsplit("_s", 1)[1])


def parse_family(model_name: str) -> str:
    if "baseline" in model_name:
        return "rvq_baseline"
    if "distill" in model_name:
        return "rvq_distill"
    raise ValueError(model_name)


def load_paired_task() -> list[dict]:
    rows = read_csv(LOG_V05 / "e40_task_paired.csv")
    out = []
    for r in rows:
        family = parse_family(r["model"])
        seed = parse_seed(r["model"])
        out.append({
            "family": family,
            "seed": seed,
            "channel": channel_key(r),
            "h0_acc_pct": float(r["h0_acc"]) * 100.0,
        })
    return out


def load_paired_recon() -> list[dict]:
    rows = read_csv(LOG_V05 / "e40_recon_paired.csv")
    out = []
    for r in rows:
        family = parse_family(r["model"])
        seed = parse_seed(r["model"])
        out.append({
            "family": family,
            "seed": seed,
            "channel": channel_key(r),
            "k": int(r["k"]),
            "psnr": float(r["psnr"]),
            "lpips": float(r["lpips"]),
            "recon_cls_pct": float(r["recon_cls_acc"]) * 100.0,
        })
    return out


# --- 3. Aggregate paired Δ per channel (paired-difference statistics) -------

def aggregate_paired(task_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Per-channel: mean±std of (distill_h0 - baseline_h0) over the 3 seeds.

    Returns (per-cell rows, paired-Δ rows).
    """
    # cell rows: one per (family, seed, channel)
    cells = []
    for r in task_rows:
        cells.append({
            "family": r["family"], "seed": r["seed"],
            "channel": r["channel"], "metric": "h0_acc_pct",
            "value": r["h0_acc_pct"],
        })

    # delta per (seed, channel)
    by_key = {}  # (family, seed, channel) -> value
    for r in task_rows:
        by_key[(r["family"], r["seed"], r["channel"])] = r["h0_acc_pct"]

    delta_rows = []
    for ch in CHANNELS_ORDER:
        pair_deltas = []
        seeds_used = []
        baseline_vals = []
        distill_vals = []
        for sd in (41, 42, 43):
            b = by_key.get(("rvq_baseline", sd, ch))
            d = by_key.get(("rvq_distill", sd, ch))
            if b is None or d is None:
                continue
            pair_deltas.append(d - b)
            baseline_vals.append(b)
            distill_vals.append(d)
            seeds_used.append(sd)
        m, s = ms(pair_deltas)
        bm, bs = ms(baseline_vals)
        dm, ds = ms(distill_vals)
        delta_rows.append({
            "channel": ch,
            "n_seeds": len(pair_deltas),
            "seeds": ",".join(str(x) for x in seeds_used),
            "baseline_mean": f3(bm), "baseline_std": f3(bs),
            "distill_mean":  f3(dm), "distill_std":  f3(ds),
            "paired_delta_mean": f3(m), "paired_delta_std": f3(s),
            "paired_delta_values": ",".join(f3(v) for v in pair_deltas),
        })
    return cells, delta_rows


# --- 4. Aggregate distill recon path (k=4) ---------------------------------

def aggregate_recon(recon_rows: list[dict]) -> list[dict]:
    out = []
    by_key = {}
    for r in recon_rows:
        if r["family"] != "rvq_distill":
            continue
        by_key.setdefault((r["channel"], r["k"]), []).append(r)

    for ch in CHANNELS_ORDER:
        for k in (1, 2, 3, 4):
            rows = by_key.get((ch, k), [])
            seeds = sorted({r["seed"] for r in rows})
            psnr_vals = [r["psnr"] for r in rows]
            lpips_vals = [r["lpips"] for r in rows]
            cls_vals = [r["recon_cls_pct"] for r in rows]
            for label, vals, fmt in (("psnr_db", psnr_vals, f3),
                                      ("lpips", lpips_vals, f3),
                                      ("recon_cls_pct", cls_vals, f3)):
                m, s = ms(vals)
                out.append({
                    "metric": f"{label}_k{k}",
                    "channel": ch,
                    "k": k,
                    "n_seeds": len(vals),
                    "seeds": ",".join(str(x) for x in seeds),
                    "mean": fmt(m), "std": fmt(s),
                    "values": ",".join(fmt(v) for v in vals),
                })
    return out


# --- 5. Build final markdown report ---------------------------------------

def build_markdown(unpaired_ref, delta_rows, recon_rows, cells) -> str:
    lines: list[str] = []
    lines.append("# E40 — Init-Seed Paired Control: paired Δ vs unpaired Δ")
    lines.append("")
    lines.append("Each (seed, channel) pair shares **bit-identical step-0 weights**; "
                 "only `loss.distill_weight` differs across the pair. Paired Δ is "
                 "computed per seed as `distill_h0 - baseline_h0`, then mean ± std "
                 "is taken across 3 seeds.")
    lines.append("")
    lines.append("Unpaired reference: paper Table I `table2_task_path_mean_std.csv` "
                 "(3 model seeds, channel seeds averaged within each model seed).")
    lines.append("")
    lines.append("## Paired vs unpaired Δ (h₀ accuracy, %)")
    lines.append("")
    lines.append("| Channel | Unpaired baseline | Unpaired distill | Unpaired Δ "
                 "| Paired baseline | Paired distill | Paired Δ | Δ_paired − Δ_unpaired |")
    lines.append("|---|---|---|---|---|---|---|---|")

    paired_by_ch = {r["channel"]: r for r in delta_rows}

    for ch in CHANNELS_ORDER:
        ub = unpaired_ref.get(("rvq_baseline", ch))
        ud = unpaired_ref.get(("rvq_distill", ch))
        if ub is None or ud is None:
            continue
        u_delta_mean = ud[0] - ub[0]
        u_delta_std = math.sqrt(ub[1] ** 2 + ud[1] ** 2)  # treat as independent
        p = paired_by_ch[ch]
        p_delta_m = float(p["paired_delta_mean"])
        p_delta_s = float(p["paired_delta_std"])
        diff = p_delta_m - u_delta_mean
        # combined std for the diff (treat the two Δs as independent)
        diff_std = math.sqrt(p_delta_s ** 2 + u_delta_std ** 2)
        in_range = abs(diff) <= 2 * diff_std

        lines.append(
            f"| {CHANNEL_LABELS[ch]} "
            f"| {ub[0]:.2f} ± {ub[1]:.2f} "
            f"| {ud[0]:.2f} ± {ud[1]:.2f} "
            f"| **+{u_delta_mean:.2f}** "
            f"| {float(p['baseline_mean']):.2f} ± {float(p['baseline_std']):.2f} "
            f"| {float(p['distill_mean']):.2f} ± {float(p['distill_std']):.2f} "
            f"| **+{p_delta_m:.2f} ± {p_delta_s:.2f}** "
            f"| {diff:+.2f} "
            f"({'within 2σ' if in_range else 'OUTSIDE 2σ'}) |"
        )

    lines.append("")
    lines.append("**Interpretation.** A paired Δ within 2σ of the unpaired Δ "
                 "supports the claim that the 25.10 pp distillation gap is driven "
                 "by `λ_distill`, not by init-noise. A paired Δ significantly "
                 "below the unpaired Δ would indicate that init noise inflated "
                 "the unpaired number; a paired Δ significantly above would mean "
                 "the unpaired number understated the effect.")
    lines.append("")

    # Recon path
    lines.append("## Paired distill reconstruction path (3-seed mean ± std)")
    lines.append("")
    lines.append("| Channel | k=4 PSNR (dB) | k=4 LPIPS | k=4 recon-cls (%) |")
    lines.append("|---|---|---|---|")
    by_metric = {(r["channel"], r["metric"]): r for r in recon_rows}
    for ch in CHANNELS_ORDER:
        psnr_r = by_metric.get((ch, "psnr_db_k4"))
        lp_r   = by_metric.get((ch, "lpips_k4"))
        cls_r  = by_metric.get((ch, "recon_cls_pct_k4"))
        if not psnr_r or not lp_r or not cls_r:
            continue
        lines.append(
            f"| {CHANNEL_LABELS[ch]} "
            f"| {float(psnr_r['mean']):.2f} ± {float(psnr_r['std']):.2f} "
            f"| {float(lp_r['mean']):.3f} ± {float(lp_r['std']):.3f} "
            f"| {float(cls_r['mean']):.2f} ± {float(cls_r['std']):.2f} |"
        )

    lines.append("")
    lines.append("## Per-seed paired h₀ values")
    lines.append("")
    lines.append("| Seed | Channel | Baseline h₀ (%) | Distill h₀ (%) | Δ (pp) |")
    lines.append("|---|---|---|---|---|")
    by_seed_ch = {(c["seed"], c["channel"], c["family"]): c["value"] for c in cells}
    for sd in (41, 42, 43):
        for ch in CHANNELS_ORDER:
            b = by_seed_ch.get((sd, ch, "rvq_baseline"))
            d = by_seed_ch.get((sd, ch, "rvq_distill"))
            if b is None or d is None:
                continue
            lines.append(
                f"| {sd} | {CHANNEL_LABELS[ch]} | {b:.2f} | {d:.2f} | "
                f"{d - b:+.2f} |"
            )

    lines.append("")
    lines.append("## Source files")
    lines.append("")
    lines.append("- Paired task: `logs/paper_v05/e40_task_paired.csv` "
                 "(6 ckpts × 5 channels)")
    lines.append("- Paired recon: `logs/paper_v05/e40_recon_paired.csv` "
                 "(3 paired distill ckpts × 5 channels × k=1..4)")
    lines.append("- Init dumps: `checkpoints/paper_v05/rvq_init_s{41,42,43}/step0.pt`")
    lines.append("- Paired ckpts: "
                 "`checkpoints/paper_v05/rvq_{baseline,distill}_paired_s{41,42,43}/best.pt`")
    lines.append("- Unpaired ref: `logs/v04_tables/table2_task_path_mean_std.csv`")

    return "\n".join(lines) + "\n"


# --- main ------------------------------------------------------------------

def main():
    print("[e40_aggregate] loading inputs...", flush=True)
    unpaired_ref = load_unpaired_ref()
    task_rows = load_paired_task()
    recon_rows = load_paired_recon()

    print(f"  unpaired ref entries: {len(unpaired_ref)}")
    print(f"  paired task rows:     {len(task_rows)}")
    print(f"  paired recon rows:    {len(recon_rows)}")

    cells, delta_rows = aggregate_paired(task_rows)
    recon_summary = aggregate_recon(recon_rows)

    h0_csv = LOG_V05 / "e40_paired_h0_3seed.csv"
    recon_csv = LOG_V05 / "e40_paired_recon_3seed.csv"
    md_path = LOG_V05 / "final_table_e40_paired.md"

    # write h0 csv (per-channel paired-Δ summary)
    with h0_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "channel", "n_seeds", "seeds",
            "baseline_mean", "baseline_std",
            "distill_mean", "distill_std",
            "paired_delta_mean", "paired_delta_std",
            "paired_delta_values",
        ])
        w.writeheader()
        w.writerows(delta_rows)

    # write recon csv (per-channel × k mean±std for distill family)
    with recon_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "metric", "channel", "k",
            "n_seeds", "seeds", "mean", "std", "values",
        ])
        w.writeheader()
        w.writerows(recon_summary)

    # markdown
    md = build_markdown(unpaired_ref, delta_rows, recon_summary, cells)
    md_path.write_text(md, encoding="utf-8")

    print(f"h0 csv    -> {h0_csv}")
    print(f"recon csv -> {recon_csv}")
    print(f"md        -> {md_path}")


if __name__ == "__main__":
    main()
