"""E35 aggregate — ADJSCC mixed-channel 3-seed mean±std vs RS-Token.

Inputs:
  logs/paper_v05/e35_adjscc_recon.csv          ADJSCC raw (45 rows)
  logs/paper_p0/e24_recon_3seed_mean_std.csv   RS-Token rvq_distill (3-seed)

Outputs:
  logs/paper_v05/e35_adjscc_mean_std.csv       ADJSCC mean±std (15 rows)
  logs/paper_v05/e35_vs_rs_token.csv           side-by-side comparison
  logs/paper_v05/final_table_e35_adjscc.md     human-readable comparison
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "paper_v05"
LOG_P0 = ROOT / "logs" / "paper_p0"

RATES = [(2560, 1), (5120, 2), (10240, 4)]   # (bits/image, RS-Token k)
SEEDS = (41, 42, 43)
CHANNELS = (("none", "inf"), ("awgn", "5"), ("awgn", "10"),
            ("rayleigh", "5"), ("rayleigh", "10"))


def read_csv(path: Path) -> list[dict]:
    # utf-8-sig strips a BOM if present (PowerShell-written CSVs have one).
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


# ---------------------------------------------------------------------------
# 1. ADJSCC mean±std across 3 seeds for each (rate, channel, snr)
# ---------------------------------------------------------------------------
adj_raw = read_csv(LOG / "e35_adjscc_recon.csv")
adj_mean: list[dict] = []
for bits, _k in RATES:
    for ch, snr in CHANNELS:
        psnr_vals, lpips_vals, cls_vals = [], [], []
        for s in SEEDS:
            for r in adj_raw:
                if (int(r["bits_per_image"]) == bits
                        and r["channel"] == ch and r["snr"] == snr
                        and r["model"] == f"adjscc_b{bits}_s{s}"):
                    psnr_vals.append(float(r["psnr"]))
                    lpips_vals.append(float(r["lpips"]))
                    cls_vals.append(float(r["recon_cls_acc"]) * 100.0)
                    break
        m_psnr, s_psnr = ms(psnr_vals)
        m_lp, s_lp = ms(lpips_vals)
        m_cls, s_cls = ms(cls_vals)
        adj_mean.append({
            "bits_per_image": bits, "channel": ch, "snr": snr,
            "n_seeds": len(psnr_vals),
            "psnr_mean": f"{m_psnr:.3f}", "psnr_std": f"{s_psnr:.3f}",
            "lpips_mean": f"{m_lp:.4f}", "lpips_std": f"{s_lp:.4f}",
            "recon_cls_pct_mean": f"{m_cls:.2f}",
            "recon_cls_pct_std": f"{s_cls:.2f}",
        })

with (LOG / "e35_adjscc_mean_std.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(adj_mean[0].keys()))
    w.writeheader()
    w.writerows(adj_mean)

# ---------------------------------------------------------------------------
# 2. RS-Token rvq_distill 3-seed mean±std (already aggregated in E24)
# ---------------------------------------------------------------------------
rs_raw = read_csv(LOG_P0 / "e24_recon_3seed_mean_std.csv")


def rs_lookup(bits: int, ch: str, snr: str, k: int) -> dict | None:
    for r in rs_raw:
        if (int(r["bits_per_img"]) == bits and r["channel"] == ch
                and r["snr"] == snr and int(r["k"]) == k
                and r["model"] == "rvq_distill"):
            return r
    return None


# ---------------------------------------------------------------------------
# 3. Side-by-side comparison CSV
# ---------------------------------------------------------------------------
cmp_rows: list[dict] = []
for bits, k in RATES:
    for ch, snr in CHANNELS:
        adj = next((a for a in adj_mean
                    if a["bits_per_image"] == bits
                    and a["channel"] == ch and a["snr"] == snr), None)
        rs = rs_lookup(bits, ch, snr, k)
        cmp_rows.append({
            "bits_per_image": bits, "channel": ch, "snr": snr,
            "rs_token_k": k,
            "adjscc_psnr_mean": adj["psnr_mean"] if adj else "",
            "adjscc_psnr_std":  adj["psnr_std"]  if adj else "",
            "rstoken_psnr_mean": f"{float(rs['psnr_mean']):.3f}" if rs else "",
            "rstoken_psnr_std":  f"{float(rs['psnr_std']):.3f}"  if rs else "",
            "adjscc_lpips_mean": adj["lpips_mean"] if adj else "",
            "rstoken_lpips_mean": f"{float(rs['lpips_mean']):.4f}" if rs else "",
            "adjscc_recon_cls_pct_mean": adj["recon_cls_pct_mean"] if adj else "",
            "rstoken_recon_cls_pct_mean": f"{float(rs['recon_cls_acc_mean']) * 100:.2f}" if rs else "",
        })

with (LOG / "e35_vs_rs_token.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
    w.writeheader()
    w.writerows(cmp_rows)


# ---------------------------------------------------------------------------
# 4. Markdown comparison
# ---------------------------------------------------------------------------
def fmt_psnr(m: dict, prefix: str) -> str:
    if not m or m.get(f"{prefix}_psnr_mean", "") == "":
        return "—"
    return f"{m[f'{prefix}_psnr_mean']} ± {m[f'{prefix}_psnr_std']}"


def channel_label(ch: str, snr: str) -> str:
    if ch == "none":
        return "None"
    return f"{ch.capitalize()} {snr} dB"


lines: list[str] = []
lines.append("# E35 — ADJSCC vs RS-Token at matched transmitted bits")
lines.append("")
lines.append("ADJSCC trained with mixed AWGN + complex Rayleigh (per-sample "
             "random channel choice; SNR ∈ [-2, 12] dB uniform). Each cell "
             "is mean ± std over 3 model seeds (41/42/43). Single channel "
             "seed per (model, channel, snr) for both ADJSCC and RS-Token.")
lines.append("")
lines.append("ADJSCC sends N real symbols per image (BPSK-equivalent bit "
             "rate); for Rayleigh we pair adjacent reals into complex "
             "symbols, apply CN(0,1) flat fading, AWGN, and coherent "
             "ZF equalisation at the receiver. Bit-equivalent SNR matches "
             "RS-Token's index-domain BPSK convention exactly.")
lines.append("")

# Table 1: PSNR
lines.append("## PSNR (dB) — mean ± std, 3 seeds")
lines.append("")
lines.append("| Bits/image | k (RS-Token) | Channel | ADJSCC | RS-Token (rvq_distill) | Δ (RS − ADJSCC) |")
lines.append("|---|---|---|---|---|---|")
for bits, k in RATES:
    for ch, snr in CHANNELS:
        c = next(c for c in cmp_rows if c["bits_per_image"] == bits
                 and c["channel"] == ch and c["snr"] == snr)
        adj = (f"{c['adjscc_psnr_mean']} ± {c['adjscc_psnr_std']}"
               if c["adjscc_psnr_mean"] else "—")
        rs = (f"{c['rstoken_psnr_mean']} ± {c['rstoken_psnr_std']}"
              if c["rstoken_psnr_mean"] else "—")
        if c["adjscc_psnr_mean"] and c["rstoken_psnr_mean"]:
            delta = float(c["rstoken_psnr_mean"]) - float(c["adjscc_psnr_mean"])
            d = f"{delta:+.2f}"
        else:
            d = "—"
        lines.append(f"| {bits} | {k} | {channel_label(ch, snr)} | {adj} | {rs} | {d} |")
lines.append("")

# Table 2: recon-cls
lines.append("## Reconstructed-image classifier accuracy (clean AID ResNet34, %)")
lines.append("")
lines.append("| Bits/image | k | Channel | ADJSCC | RS-Token | Δ (RS − ADJSCC, pp) |")
lines.append("|---|---|---|---|---|---|")
for bits, k in RATES:
    for ch, snr in CHANNELS:
        c = next(c for c in cmp_rows if c["bits_per_image"] == bits
                 and c["channel"] == ch and c["snr"] == snr)
        adj = c["adjscc_recon_cls_pct_mean"] or "—"
        rs = c["rstoken_recon_cls_pct_mean"] or "—"
        if adj != "—" and rs != "—":
            delta = float(rs) - float(adj)
            d = f"{delta:+.2f}"
        else:
            d = "—"
        lines.append(f"| {bits} | {k} | {channel_label(ch, snr)} | {adj} | {rs} | {d} |")
lines.append("")

# Table 3: LPIPS (lower is better)
lines.append("## LPIPS (AlexNet, lower is better)")
lines.append("")
lines.append("| Bits/image | k | Channel | ADJSCC | RS-Token | Δ (ADJSCC − RS) |")
lines.append("|---|---|---|---|---|---|")
for bits, k in RATES:
    for ch, snr in CHANNELS:
        c = next(c for c in cmp_rows if c["bits_per_image"] == bits
                 and c["channel"] == ch and c["snr"] == snr)
        adj = c["adjscc_lpips_mean"] or "—"
        rs = c["rstoken_lpips_mean"] or "—"
        if adj != "—" and rs != "—":
            delta = float(adj) - float(rs)
            d = f"{delta:+.4f}"
        else:
            d = "—"
        lines.append(f"| {bits} | {k} | {channel_label(ch, snr)} | {adj} | {rs} | {d} |")
lines.append("")

# Headline summary
lines.append("## Headline takeaways")
lines.append("")


def headline_psnr(bits: int, k: int):
    by_ch = {}
    for ch, snr in CHANNELS:
        c = next(c for c in cmp_rows if c["bits_per_image"] == bits
                 and c["channel"] == ch and c["snr"] == snr)
        if c["adjscc_psnr_mean"] and c["rstoken_psnr_mean"]:
            adj = float(c["adjscc_psnr_mean"])
            rs = float(c["rstoken_psnr_mean"])
            by_ch[(ch, snr)] = (adj, rs, rs - adj)
    return by_ch


for bits, k in RATES:
    bc = headline_psnr(bits, k)
    none_adj, none_rs, none_d = bc[("none", "inf")]
    awgn_adj, awgn_rs, awgn_d = bc[("awgn", "10")]
    ray_adj, ray_rs, ray_d = bc[("rayleigh", "10")]
    lines.append(f"- **{bits} bits / k={k}**:")
    lines.append(f"  - clean PSNR: ADJSCC **{none_adj:.2f}** vs RS-Token "
                 f"{none_rs:.2f} (Δ {none_d:+.2f} dB)")
    lines.append(f"  - Rayleigh +10 dB PSNR: ADJSCC **{ray_adj:.2f}** vs "
                 f"RS-Token {ray_rs:.2f} (Δ {ray_d:+.2f} dB)")
    # recon-cls clean
    c_clean = next(c for c in cmp_rows if c["bits_per_image"] == bits
                    and c["channel"] == "none")
    if c_clean["adjscc_recon_cls_pct_mean"] and c_clean["rstoken_recon_cls_pct_mean"]:
        adj_c = float(c_clean["adjscc_recon_cls_pct_mean"])
        rs_c = float(c_clean["rstoken_recon_cls_pct_mean"])
        lines.append(f"  - clean recon-cls: ADJSCC **{adj_c:.2f}%** vs "
                     f"RS-Token {rs_c:.2f}% (Δ {rs_c - adj_c:+.2f} pp)")
lines.append("")
lines.append("## Source files")
lines.append("")
lines.append("- ADJSCC raw: `logs/paper_v05/e35_adjscc_recon.csv` (45 rows, "
             "9 ckpts × 5 channels)")
lines.append("- ADJSCC mean±std: `logs/paper_v05/e35_adjscc_mean_std.csv`")
lines.append("- Comparison CSV: `logs/paper_v05/e35_vs_rs_token.csv`")
lines.append("- RS-Token reference: `logs/paper_p0/e24_recon_3seed_mean_std.csv` "
             "(rvq_distill 3-seed)")
lines.append("- ADJSCC checkpoints: `checkpoints/paper_v05/adjscc_b{2560,5120,10240}_s{41,42,43}/best.pt`")

(LOG / "final_table_e35_adjscc.md").write_text(
    "\n".join(lines), encoding="utf-8")

print(f"adjscc_mean -> {LOG / 'e35_adjscc_mean_std.csv'}  ({len(adj_mean)} rows)")
print(f"comparison  -> {LOG / 'e35_vs_rs_token.csv'}  ({len(cmp_rows)} rows)")
print(f"md          -> {LOG / 'final_table_e35_adjscc.md'}")
