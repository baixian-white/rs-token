"""E39 aggregate — NWPU-RESISC45 zero-shot reconstruction transfer.

Inputs:
  logs/paper_v05/e39_task_nwpu.csv          h0 task path on NWPU
  logs/paper_v05/e39_recon_nwpu.csv         PSNR/LPIPS/recon_cls_acc
  logs/paper_v05/e36_continuous_snr_mean_std.csv  AID 3-seed mean±std
                                                   (used for cross-domain Δ)

Outputs:
  logs/paper_v05/e39_nwpu_summary.csv       wide-form, per (model, channel, k)
  logs/paper_v05/final_table_e39_nwpu.md    human-readable comparison
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "paper_v05"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. Load NWPU task + recon
# ---------------------------------------------------------------------------
task_rows = read_csv(LOG / "e39_task_nwpu.csv")
recon_rows = read_csv(LOG / "e39_recon_nwpu.csv")

CHANNELS = ("none", "awgn_5", "awgn_10", "rayleigh_5", "rayleigh_10")


def channel_key(channel: str, snr: str) -> str:
    if channel == "none":
        return "none"
    return f"{channel}_{snr}"


def task_lookup(model: str, ch_key: str) -> float:
    for r in task_rows:
        if r["model"] == model and channel_key(r["channel"], r["snr"]) == ch_key:
            return float(r["h0_acc"]) * 100.0
    return float("nan")


def recon_lookup(model: str, ch_key: str, k: int):
    for r in recon_rows:
        if r["model"] == model and int(r["k"]) == k \
                and channel_key(r["channel"], r["snr"]) == ch_key:
            return (float(r["psnr"]),
                    float(r["lpips"]),
                    float(r["recon_cls_acc"]) * 100.0)
    return (float("nan"), float("nan"), float("nan"))


# ---------------------------------------------------------------------------
# 2. Wide-form summary CSV
# ---------------------------------------------------------------------------
summary_rows: list[dict] = []
for model in ("rvq_distill_s42", "rvq_baseline_s42"):
    for ch in CHANNELS:
        h0 = task_lookup(model, ch)
        for k in (1, 2, 3, 4):
            psnr, lpips, cls = recon_lookup(model, ch, k)
            summary_rows.append({
                "model": model, "channel": ch, "k": k,
                "h0_acc_pct": f"{h0:.2f}",
                "psnr_db": f"{psnr:.3f}",
                "lpips": f"{lpips:.4f}",
                "recon_cls_pct": f"{cls:.2f}",
                "n_samples": 6300,
            })

with (LOG / "e39_nwpu_summary.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)


# ---------------------------------------------------------------------------
# 3. Markdown report — extends Table V with PSNR/LPIPS/recon-cls columns
# ---------------------------------------------------------------------------
# AID 3-seed clean k=4 reference values from E36 mean±std (single seed s42 here).
# These are point estimates from rvq_*_s42 only, lifted from
# logs/paper_v05/e36_continuous_snr_mean_std.csv for "none" channel.
def aid_clean_k4(family: str) -> tuple[float, float]:
    """Return (PSNR_dB, h0_acc_pct) for AID clean channel, k=4 / k=1."""
    for r in read_csv(LOG / "e36_continuous_snr_mean_std.csv"):
        if r["family"] != family:
            continue
        if r["channel"] != "none":
            continue
        if r["metric"] == "psnr_db" and int(r["k"]) == 4:
            psnr = float(r["mean"])
        if r["metric"] == "h0_acc_pct" and int(r["k"]) == 1:
            h0 = float(r["mean"])
    return psnr, h0


distill_aid_psnr, distill_aid_h0 = aid_clean_k4("rvq_distill")
baseline_aid_psnr, baseline_aid_h0 = aid_clean_k4("rvq_baseline")

lines: list[str] = []
lines.append("# E39 — NWPU-RESISC45 zero-shot reconstruction transfer")
lines.append("")
lines.append("Tokenizer (encoder + RVQ + decoder) frozen at AID-trained "
             "weights; only the linear h₀ probe and the recon-cls "
             "classifier are NWPU-specific. Single seed (42), n=6300 NWPU "
             "test images per cell. Channel seeds match AID main config.")
lines.append("")
lines.append("Recon-cls evaluator: NWPU 45-class ResNet34 fine-tuned in E39, "
             "test acc 95.17% / macro-F1 0.9518 / worst-class 83.57%.")
lines.append("")

# --- Table 1: extended Table V ---------------------------------------------
lines.append("## Extended Table V — clean and noisy NWPU transfer")
lines.append("")
lines.append("| Channel | SNR | rvq_distill h₀ (%) | rvq_baseline h₀ (%) | "
             "Δh₀ (pp) | rvq_distill k=4 PSNR (dB) | rvq_distill k=4 "
             "recon-cls (%) | rvq_baseline k=4 PSNR (dB) | rvq_baseline k=4 "
             "recon-cls (%) |")
lines.append("|---|---|---|---|---|---|---|---|---|")
SNR_NAMES = {"none": "—", "awgn_5": "+5 dB", "awgn_10": "+10 dB",
             "rayleigh_5": "+5 dB", "rayleigh_10": "+10 dB"}
CH_LABEL = {"none": "None", "awgn_5": "AWGN", "awgn_10": "AWGN",
            "rayleigh_5": "Rayleigh", "rayleigh_10": "Rayleigh"}
for ch in CHANNELS:
    h0_d = task_lookup("rvq_distill_s42", ch)
    h0_b = task_lookup("rvq_baseline_s42", ch)
    psnr_d, _, cls_d = recon_lookup("rvq_distill_s42", ch, 4)
    psnr_b, _, cls_b = recon_lookup("rvq_baseline_s42", ch, 4)
    delta = h0_d - h0_b
    lines.append(
        f"| {CH_LABEL[ch]} | {SNR_NAMES[ch]} | "
        f"{h0_d:.2f} | {h0_b:.2f} | {delta:+.2f} | "
        f"{psnr_d:.2f} | {cls_d:.2f} | {psnr_b:.2f} | {cls_b:.2f} |"
    )
lines.append("")

# --- Table 2: cross-domain comparison (NWPU vs AID, clean channel) ---------
lines.append("## Cross-domain comparison (no-channel, k=4)")
lines.append("")
lines.append("| Family | AID clean k=4 PSNR (dB) | NWPU clean k=4 PSNR (dB) "
             "| Δ PSNR | AID clean k=1 h₀ (%) | NWPU clean k=1 h₀ (%) | Δ h₀ (pp) |")
lines.append("|---|---|---|---|---|---|---|")
nwpu_d_psnr, _, _ = recon_lookup("rvq_distill_s42", "none", 4)
nwpu_b_psnr, _, _ = recon_lookup("rvq_baseline_s42", "none", 4)
nwpu_d_h0 = task_lookup("rvq_distill_s42", "none")
nwpu_b_h0 = task_lookup("rvq_baseline_s42", "none")
lines.append(
    f"| rvq_distill | {distill_aid_psnr:.2f} | {nwpu_d_psnr:.2f} | "
    f"{nwpu_d_psnr - distill_aid_psnr:+.2f} | "
    f"{distill_aid_h0:.2f} | {nwpu_d_h0:.2f} | "
    f"{nwpu_d_h0 - distill_aid_h0:+.2f} |"
)
lines.append(
    f"| rvq_baseline | {baseline_aid_psnr:.2f} | {nwpu_b_psnr:.2f} | "
    f"{nwpu_b_psnr - baseline_aid_psnr:+.2f} | "
    f"{baseline_aid_h0:.2f} | {nwpu_b_h0:.2f} | "
    f"{nwpu_b_h0 - baseline_aid_h0:+.2f} |"
)
lines.append("")
lines.append("Note: AID column is single-seed (s42) drawn from E36 mean±std "
             "to match the seed used for NWPU here. PSNR Δ is positive when "
             "NWPU reconstructs better than AID at the same checkpoint.")
lines.append("")

# --- Table 3: per-k progression on NWPU ------------------------------------
lines.append("## NWPU progressive reconstruction (rvq_distill, no-channel)")
lines.append("")
lines.append("| k | Bits/image | PSNR (dB) | LPIPS | recon-cls (%) |")
lines.append("|---|---|---|---|---|")
for k in (1, 2, 3, 4):
    psnr, lpips, cls = recon_lookup("rvq_distill_s42", "none", k)
    lines.append(f"| {k} | {2560 * k} | {psnr:.2f} | {lpips:.4f} | {cls:.2f} |")
lines.append("")

lines.append("## Headline numbers")
lines.append("")
lines.append(f"- NWPU clean h₀: distill {nwpu_d_h0:.2f}% vs baseline "
             f"{nwpu_b_h0:.2f}% — gap **{nwpu_d_h0 - nwpu_b_h0:+.2f} pp**.")
lines.append(f"- AID clean h₀ (s42 only): distill {distill_aid_h0:.2f}% vs "
             f"baseline {baseline_aid_h0:.2f}% — gap "
             f"**{distill_aid_h0 - baseline_aid_h0:+.2f} pp** "
             "(≈ paper Table I 25.1 pp).")
lines.append(f"- NWPU clean k=4 PSNR distill: **{nwpu_d_psnr:.2f} dB** vs "
             f"AID clean k=4 PSNR distill {distill_aid_psnr:.2f} dB "
             f"(Δ {nwpu_d_psnr - distill_aid_psnr:+.2f} dB).")
lines.append(f"- NWPU clean k=4 recon-cls distill: "
             f"**{recon_lookup('rvq_distill_s42', 'none', 4)[2]:.2f}%** "
             "(45-class).")
lines.append("")
lines.append("## Source files")
lines.append("")
lines.append("- `logs/paper_v05/e39_task_nwpu.csv` (10 task rows)")
lines.append("- `logs/paper_v05/e39_recon_nwpu.csv` (40 recon rows)")
lines.append("- `logs/paper_v05/e39_nwpu_summary.csv` (40 wide rows)")
lines.append("- AID classifier evaluator: `checkpoints/aid_classifier_resnet34/best.pt`")
lines.append("- NWPU classifier evaluator: `checkpoints/paper_v05/nwpu_classifier_resnet34/best.pt` (test acc 95.17%)")

(LOG / "final_table_e39_nwpu.md").write_text("\n".join(lines), encoding="utf-8")

print(f"summary -> {LOG / 'e39_nwpu_summary.csv'}  ({len(summary_rows)} rows)")
print(f"md      -> {LOG / 'final_table_e39_nwpu.md'}")
