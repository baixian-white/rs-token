"""E42 — RVQ K-ablation aggregation: K∈{2,3,4,6} × 5 channels × k=1..K.

Inputs:
  logs/paper_v05/e42_task_K{2,3,6}.csv      (one per K, single-seed s42)
  logs/paper_v05/e42_recon_K{2,3,6}.csv     (one per K, single-seed s42)
  logs/paper_p0/e24_task_s42.csv            (K=4 reference, distill s42)
  logs/paper_p0/e24_recon_s42.csv           (K=4 reference)

Outputs:
  logs/paper_v05/e42_k_ablation_summary.csv
  logs/paper_v05/final_table_e42_k_ablation.md

Notes:
- The "headline" comparison is at full K (each row uses its own k = K_total).
- Per-prefix bit cost = 256 * log2(1024) * k = 2,560 * k bits/image.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_V05 = ROOT / "logs" / "paper_v05"
LOG_P0 = ROOT / "logs" / "paper_p0"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
K_VALUES = (2, 3, 4, 6)


def load_task_for_k(K: int) -> dict[str, float]:
    """{channel: h0_acc_pct} — task path is k=1 always."""
    if K == 4:
        rows = read_csv(LOG_P0 / "e24_task_s42.csv")
        rows = [r for r in rows if r["model"] == "rvq_distill_s42"]
    else:
        rows = read_csv(LOG_V05 / f"e42_task_K{K}.csv")
    out = {}
    for r in rows:
        out[channel_key(r)] = float(r["h0_acc"]) * 100.0
    return out


def load_recon_for_k(K: int) -> dict[tuple[str, int], dict]:
    """{(channel, k): {psnr, lpips, recon_cls_pct}}."""
    if K == 4:
        rows = read_csv(LOG_P0 / "e24_recon_s42.csv")
        rows = [r for r in rows if r["model"] == "rvq_distill_s42"]
    else:
        rows = read_csv(LOG_V05 / f"e42_recon_K{K}.csv")
    out = {}
    for r in rows:
        out[(channel_key(r), int(r["k"]))] = {
            "psnr": float(r["psnr"]),
            "lpips": float(r["lpips"]),
            "recon_cls_pct": float(r["recon_cls_acc"]) * 100.0,
        }
    return out


def main():
    print("[e42_aggregate] loading inputs...", flush=True)
    task_by_K = {K: load_task_for_k(K) for K in K_VALUES}
    recon_by_K = {K: load_recon_for_k(K) for K in K_VALUES}

    for K in K_VALUES:
        print(f"  K={K}: task channels={len(task_by_K[K])}, "
              f"recon cells={len(recon_by_K[K])}")

    # --- 1. Wide-form summary CSV ----------------------------------------
    summary_rows = []
    for K in K_VALUES:
        bits_full = 2560 * K
        for ch in CHANNELS_ORDER:
            h0 = task_by_K[K].get(ch)
            full_k = recon_by_K[K].get((ch, K))
            row = {
                "K_total": K,
                "channel": ch,
                "bits_at_full_K": bits_full,
                "h0_acc_pct_k1": "" if h0 is None else f"{h0:.3f}",
            }
            if full_k is not None:
                row["psnr_db_at_full_K"] = f"{full_k['psnr']:.3f}"
                row["lpips_at_full_K"] = f"{full_k['lpips']:.4f}"
                row["recon_cls_pct_at_full_K"] = f"{full_k['recon_cls_pct']:.3f}"
            else:
                row["psnr_db_at_full_K"] = ""
                row["lpips_at_full_K"] = ""
                row["recon_cls_pct_at_full_K"] = ""
            summary_rows.append(row)

    out_csv = LOG_V05 / "e42_k_ablation_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "K_total", "channel", "bits_at_full_K",
            "h0_acc_pct_k1",
            "psnr_db_at_full_K", "lpips_at_full_K", "recon_cls_pct_at_full_K",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    # --- 2. Markdown report ----------------------------------------------
    lines: list[str] = []
    lines.append("# E42 — RVQ K-ablation: K∈{2, 3, 4, 6} (seed 42)")
    lines.append("")
    lines.append("All four K values use the same recipe (50 epoch, λ_distill=0.5, "
                 "L0-only, RemoteCLIP teacher). K=4 is the main paper choice; K∈{2,3,6} "
                 "are E42's three new trainings. All single-seed (42).")
    lines.append("")
    lines.append("**Per-layer bit cost** = 256 patches × log₂(1024) bits = 2,560 bits/image. "
                 "**Total bits at full K** = 2,560 × K bits/image.")
    lines.append("")

    # Headline: task acc at k=1 vs full-K PSNR
    lines.append("## Headline: task path (h₀, k=1) and reconstruction at full K")
    lines.append("")
    lines.append("| K | Bits at full K | h₀ no-channel (%) | h₀ AWGN +5 (%) | h₀ AWGN +10 (%) "
                 "| h₀ Ray +5 (%) | h₀ Ray +10 (%) | PSNR no-ch (dB) | PSNR Ray +10 (dB) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for K in K_VALUES:
        bits = 2560 * K
        marker = " **(main)**" if K == 4 else ""
        h0 = task_by_K[K]
        rec = recon_by_K[K]
        psnr_none = rec.get(("none", K), {}).get("psnr")
        psnr_ray10 = rec.get(("rayleigh_10", K), {}).get("psnr")
        lines.append(
            f"| {K}{marker} | {bits:,} "
            f"| {h0.get('none', float('nan')):.2f} "
            f"| {h0.get('awgn_5', float('nan')):.2f} "
            f"| {h0.get('awgn_10', float('nan')):.2f} "
            f"| {h0.get('rayleigh_5', float('nan')):.2f} "
            f"| {h0.get('rayleigh_10', float('nan')):.2f} "
            f"| {psnr_none:.2f} " if psnr_none is not None else "| — "
        )
        # rebuild the line cleanly (above had conditional close issue):
        lines[-1] = (
            f"| {K}{marker} | {bits:,} "
            f"| {h0.get('none', float('nan')):.2f} "
            f"| {h0.get('awgn_5', float('nan')):.2f} "
            f"| {h0.get('awgn_10', float('nan')):.2f} "
            f"| {h0.get('rayleigh_5', float('nan')):.2f} "
            f"| {h0.get('rayleigh_10', float('nan')):.2f} "
            f"| {psnr_none:.2f} | {psnr_ray10:.2f} |"
            if psnr_none is not None and psnr_ray10 is not None else
            f"| {K}{marker} | {bits:,} | — | — | — | — | — | — | — |"
        )
    lines.append("")
    lines.append("## Full reconstruction path (k=1..K, no-channel)")
    lines.append("")
    lines.append("| K | k | Bits/img | PSNR (dB) | LPIPS | recon-cls (%) |")
    lines.append("|---|---|---|---|---|---|")
    for K in K_VALUES:
        for k in range(1, K + 1):
            cell = recon_by_K[K].get(("none", k))
            if cell is None:
                continue
            bits = 2560 * k
            lines.append(
                f"| {K} | {k} | {bits:,} "
                f"| {cell['psnr']:.2f} | {cell['lpips']:.3f} | {cell['recon_cls_pct']:.2f} |"
            )

    lines.append("")
    lines.append("## Headline reconstruction Δ vs K=4 (no-channel, full-K)")
    lines.append("")
    lines.append("| K | h₀ no-ch (%) | Δ h₀ vs K=4 (pp) | PSNR full-K (dB) | "
                 "Δ PSNR vs K=4 (dB) | Bits/img | Bits Δ vs K=4 |")
    lines.append("|---|---|---|---|---|---|---|")
    h0_K4 = task_by_K[4]["none"]
    psnr_K4_full = recon_by_K[4][("none", 4)]["psnr"]
    bits_K4 = 2560 * 4
    for K in K_VALUES:
        bits = 2560 * K
        h0 = task_by_K[K]["none"]
        psnr = recon_by_K[K][("none", K)]["psnr"]
        marker = " **(main)**" if K == 4 else ""
        lines.append(
            f"| {K}{marker} | {h0:.2f} | {h0 - h0_K4:+.2f} "
            f"| {psnr:.2f} | {psnr - psnr_K4_full:+.2f} "
            f"| {bits:,} | {bits - bits_K4:+,} |"
        )

    lines.append("")
    lines.append("## Acceptance check")
    lines.append("")
    lines.append("Paper claim: K=4 is a sweet spot — small K loses recon fidelity "
                 "at full bit budget, large K wastes bits without proportional gain. "
                 "Acceptance criteria (from review §2.0):")
    lines.append("")
    lines.append("- [ ] **K=4 ≥ K=2 on no-ch h₀**: K=4 task path ≥ K=2 (deeper RVQ "
                 "should not hurt task at k=1).")
    lines.append("- [ ] **K=4 ≥ K=3 on no-ch full-K PSNR by a meaningful margin** "
                 "(else paper should re-run main results at K=3).")
    lines.append("- [ ] **K=6 PSNR gain over K=4 is sub-linear in bits** (else "
                 "K=6 dominates K=4 and paper needs to justify K=4 differently).")
    lines.append("")

    lines.append("## Source files")
    lines.append("")
    lines.append("- K=2,3,6 task: `logs/paper_v05/e42_task_K{2,3,6}.csv`")
    lines.append("- K=2,3,6 recon: `logs/paper_v05/e42_recon_K{2,3,6}.csv`")
    lines.append("- K=4 reference (s42 main): `logs/paper_p0/e24_{task,recon}_s42.csv` "
                 "(rvq_distill_s42)")
    lines.append("- ckpts: `checkpoints/paper_v05/rvq_distill_K{2,3,6}_s42/best.pt`, "
                 "`checkpoints/rvq_distill/best.pt` (K=4 main)")

    md_path = LOG_V05 / "final_table_e42_k_ablation.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"summary csv -> {out_csv}")
    print(f"md          -> {md_path}")


if __name__ == "__main__":
    main()
