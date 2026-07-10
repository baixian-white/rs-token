"""E41 — Direct-quantize-RemoteCLIP (DCRC) aggregation.

Compares DCRC 3-seed (s41/s42/s43) vs RS-Token rvq_distill 3-seed.

Inputs:
  logs/paper_v05/e41_task.csv               (3 DCRC ckpts × 5 channels)
  logs/paper_v05/e41_recon.csv              (3 DCRC ckpts × 5 channels × k=1..4)
  logs/paper_p0/e24_task_s{41,42,43}.csv    (RS-Token reference)
  logs/paper_p0/e24_recon_3seed_mean_std.csv (RS-Token reference)

Outputs:
  logs/paper_v05/e41_dcrc_3seed.csv         (DCRC mean±std per metric)
  logs/paper_v05/final_table_e41_direct_clip.md
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


def parse_seed(model_name: str) -> int:
    return int(model_name.rsplit("_s", 1)[1])


# --- 1. Load DCRC 3-seed task path -----------------------------------------

def load_dcrc_task() -> dict[str, list[float]]:
    """{channel_key: [h0 across seeds]}."""
    rows = read_csv(LOG_V05 / "e41_task.csv")
    by_ch: dict[str, list[float]] = {ch: [] for ch in CHANNELS_ORDER}
    for r in rows:
        ck = channel_key(r)
        if ck in by_ch:
            by_ch[ck].append(float(r["h0_acc"]) * 100.0)
    return by_ch


def load_dcrc_recon() -> dict[tuple[str, int], list[dict]]:
    rows = read_csv(LOG_V05 / "e41_recon.csv")
    out: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        ck = channel_key(r)
        k = int(r["k"])
        out.setdefault((ck, k), []).append({
            "psnr": float(r["psnr"]),
            "lpips": float(r["lpips"]),
            "recon_cls_pct": (
                float(r["recon_cls_acc"]) * 100.0
                if r["recon_cls_acc"] not in ("", "nan", None) else float("nan")
            ),
        })
    return out


# --- 2. Load RS-Token rvq_distill 3-seed reference -------------------------

def load_rstoken_task() -> dict[str, tuple[float, float]]:
    """{channel_key: (mean_pct, std_pct)} for rvq_distill 3 seeds."""
    by_ch: dict[str, list[float]] = {ch: [] for ch in CHANNELS_ORDER}
    for s in (41, 42, 43):
        for r in read_csv(LOG_P0 / f"e24_task_s{s}.csv"):
            ck = channel_key(r)
            if ck in by_ch:
                by_ch[ck].append(float(r["h0_acc"]) * 100.0)
    return {ch: ms(by_ch[ch]) for ch in CHANNELS_ORDER}


def load_rstoken_recon_mean_std() -> dict[tuple[str, int], dict]:
    """rvq_distill 3-seed mean/std at every (channel, k)."""
    out: dict[tuple[str, int], dict] = {}
    for r in read_csv(LOG_P0 / "e24_recon_3seed_mean_std.csv"):
        if r["model"] != "rvq_distill":
            continue
        ch = r["channel"]
        snr = r["snr"]
        ck = "none" if ch == "none" else f"{ch}_{snr}"
        out[(ck, int(r["k"]))] = {
            "psnr_mean": float(r["psnr_mean"]),
            "psnr_std":  float(r["psnr_std"]),
            "lpips_mean": float(r["lpips_mean"]),
            "lpips_std":  float(r["lpips_std"]),
            "recon_cls_mean": float(r["recon_cls_acc_mean"]) * 100.0,
            "recon_cls_std":  float(r["recon_cls_acc_std"]) * 100.0,
        }
    return out


# --- 3. Aggregate DCRC into mean±std ---------------------------------------

def main():
    print("[e41_aggregate] loading inputs...", flush=True)
    dcrc_task = load_dcrc_task()
    dcrc_recon = load_dcrc_recon()
    rs_task = load_rstoken_task()
    rs_recon = load_rstoken_recon_mean_std()

    print(f"  DCRC task channels: {sum(1 for v in dcrc_task.values() if v)}")
    print(f"  DCRC recon cells:   {len(dcrc_recon)}")

    # 3.1 wide-form summary CSV
    summary_rows = []
    for ch in CHANNELS_ORDER:
        h0_vals = dcrc_task[ch]
        m, s = ms(h0_vals)
        summary_rows.append({
            "channel": ch, "metric": "h0_acc_pct_k1",
            "n_seeds": len(h0_vals),
            "mean": f"{m:.3f}", "std": f"{s:.3f}",
            "values": ",".join(f"{v:.3f}" for v in h0_vals),
        })
        for k in (1, 2, 3, 4):
            cells = dcrc_recon.get((ch, k), [])
            for label, key in (("psnr_db", "psnr"),
                                ("lpips", "lpips"),
                                ("recon_cls_pct", "recon_cls_pct")):
                vals = [c[key] for c in cells]
                m, s = ms(vals)
                summary_rows.append({
                    "channel": ch, "metric": f"{label}_k{k}",
                    "n_seeds": len(vals),
                    "mean": f"{m:.3f}", "std": f"{s:.3f}",
                    "values": ",".join(f"{v:.3f}" for v in vals),
                })

    out_csv = LOG_V05 / "e41_dcrc_3seed.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "channel", "metric", "n_seeds", "mean", "std", "values",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    # 3.2 markdown report
    lines = []
    lines.append("# E41 — Direct-Quantize-RemoteCLIP (DCRC) vs RS-Token")
    lines.append("")
    lines.append("DCRC = RemoteCLIP visual encoder (frozen) → up-project to "
                 "256 tokens × 8-d → RVQ(K=4, codebook=1024) → decoder. "
                 "Same bit budget as RS-Token rvq_distill (2,560/5,120/7,680/10,240 "
                 "at k=1..4). 3 seeds (41, 42, 43).")
    lines.append("")
    lines.append("**Architectural difference vs RS-Token:** DCRC skips the "
                 "convolutional encoder entirely; the input to RVQ is a global "
                 "RemoteCLIP CLS feature reshaped into 256 spatial tokens, not a "
                 "per-region CNN feature map.")
    lines.append("")
    lines.append("Trainable params: DCRC **6.71 M** (up_proj 1.05M + RVQ 0.0M + "
                 "decoder 5.65M) vs RS-Token rvq_distill **10.87 M** (encoder "
                 "5.16M + decoder 5.71M; quantizer + distill_head are tiny).")
    lines.append("")

    # Task path table
    lines.append("## Task path: h₀ (k=1) accuracy (3-seed mean ± std)")
    lines.append("")
    lines.append("| Channel | DCRC h₀ (%) | RS-Token h₀ (%) | Δ DCRC − RS (pp) |")
    lines.append("|---|---|---|---|")
    for ch in CHANNELS_ORDER:
        d_vals = dcrc_task[ch]
        d_m, d_s = ms(d_vals)
        rs_m, rs_s = rs_task[ch]
        delta = d_m - rs_m
        lines.append(
            f"| {CHANNEL_LABELS[ch]} "
            f"| {d_m:.2f} ± {d_s:.2f} "
            f"| {rs_m:.2f} ± {rs_s:.2f} "
            f"| {delta:+.2f} |"
        )
    lines.append("")

    # Recon path table at k=4 (matched-bits headline)
    lines.append("## Reconstruction path at k=4 (10,240 bits/img, 3-seed mean ± std)")
    lines.append("")
    lines.append("| Channel | DCRC PSNR (dB) | RS-Token PSNR (dB) | Δ PSNR (dB) "
                 "| DCRC LPIPS | RS-Token LPIPS | Δ LPIPS "
                 "| DCRC recon-cls (%) | RS-Token recon-cls (%) | Δ recon-cls (pp) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for ch in CHANNELS_ORDER:
        d_cells = dcrc_recon.get((ch, 4), [])
        d_psnr = ms([c["psnr"] for c in d_cells])
        d_lp = ms([c["lpips"] for c in d_cells])
        d_cls = ms([c["recon_cls_pct"] for c in d_cells if not math.isnan(c["recon_cls_pct"])])
        rs = rs_recon.get((ch, 4))
        if not d_cells or rs is None:
            continue
        lines.append(
            f"| {CHANNEL_LABELS[ch]} "
            f"| {d_psnr[0]:.2f} ± {d_psnr[1]:.2f} "
            f"| {rs['psnr_mean']:.2f} ± {rs['psnr_std']:.2f} "
            f"| {d_psnr[0] - rs['psnr_mean']:+.2f} "
            f"| {d_lp[0]:.3f} ± {d_lp[1]:.3f} "
            f"| {rs['lpips_mean']:.3f} ± {rs['lpips_std']:.3f} "
            f"| {d_lp[0] - rs['lpips_mean']:+.3f} "
            f"| {d_cls[0]:.2f} ± {d_cls[1]:.2f} "
            f"| {rs['recon_cls_mean']:.2f} ± {rs['recon_cls_std']:.2f} "
            f"| {d_cls[0] - rs['recon_cls_mean']:+.2f} |"
        )
    lines.append("")

    # Recon path full sweep (no-channel)
    lines.append("## Reconstruction sweep at k=1..4 (no-channel)")
    lines.append("")
    lines.append("| k | Bits/img | DCRC PSNR | RS-Token PSNR | DCRC recon-cls (%) "
                 "| RS-Token recon-cls (%) |")
    lines.append("|---|---|---|---|---|---|")
    for k in (1, 2, 3, 4):
        d_cells = dcrc_recon.get(("none", k), [])
        d_psnr = ms([c["psnr"] for c in d_cells])
        d_cls = ms([c["recon_cls_pct"] for c in d_cells if not math.isnan(c["recon_cls_pct"])])
        rs = rs_recon.get(("none", k))
        if not d_cells or rs is None:
            continue
        lines.append(
            f"| {k} | {2560 * k:,} "
            f"| {d_psnr[0]:.2f} ± {d_psnr[1]:.2f} "
            f"| {rs['psnr_mean']:.2f} ± {rs['psnr_std']:.2f} "
            f"| {d_cls[0]:.2f} ± {d_cls[1]:.2f} "
            f"| {rs['recon_cls_mean']:.2f} ± {rs['recon_cls_std']:.2f} |"
        )
    lines.append("")

    # Per-seed task path (so reviewers can see raw values)
    lines.append("## Per-seed DCRC h₀ raw values")
    lines.append("")
    lines.append("| Channel | s41 | s42 | s43 | mean ± std |")
    lines.append("|---|---|---|---|---|")
    for ch in CHANNELS_ORDER:
        vals = dcrc_task[ch]
        if len(vals) != 3:
            continue
        m, s = ms(vals)
        lines.append(
            f"| {CHANNEL_LABELS[ch]} "
            f"| {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} "
            f"| {m:.2f} ± {s:.2f} |"
        )
    lines.append("")

    lines.append("## Source files")
    lines.append("")
    lines.append("- DCRC eval CSVs: `logs/paper_v05/e41_{task,recon}.csv`")
    lines.append("- RS-Token reference: `logs/paper_p0/e24_task_s{41,42,43}.csv`, "
                 "`logs/paper_p0/e24_recon_3seed_mean_std.csv`")
    lines.append("- DCRC ckpts: `checkpoints/paper_v05/e41_direct_clip_s{41,42,43}/best.pt`")

    md_path = LOG_V05 / "final_table_e41_direct_clip.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"summary csv -> {out_csv}")
    print(f"md          -> {md_path}")


if __name__ == "__main__":
    main()
