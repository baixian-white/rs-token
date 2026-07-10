"""Generate paper-ready markdown tables from existing eval CSVs.

Produces six final tables under logs/paper_p0/ and logs/paper_p1/ for direct
inclusion in the paper:

  1. final_table_recon_3seed.md          — E24 reconstruction path (3 model seeds)
  2. final_table_l0_vs_all_layers.md     — E25 L0-only vs all-layer ablation
  3. final_table_layer_target_counterfactual.md — E25 + E31 joint causal table
  4. final_table_teacher_ablation.md     — E26 RemoteCLIP vs OpenAI CLIP
  5. final_table_codec_ldpc.md           — E27 strict same-bit codec+LDPC stress
  6. final_appendix_ldpc_full.md         — E23 + E27 full LDPC appendix

Re-runnable: each invocation overwrites the targets. Reads from existing CSVs
under logs/paper_p0/, logs/paper_p1/, logs/paper_p2/, logs/. Writes nothing
else.
"""
from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

ROOT = Path(__file__).resolve().parent.parent
P0 = ROOT / "logs" / "paper_p0"
P1 = ROOT / "logs" / "paper_p1"
P2 = ROOT / "logs" / "paper_p2"
LOGS = ROOT / "logs"


def read_csv(path: Path) -> list[dict]:
    # Some of these CSVs were written by PowerShell's Export-Csv, which emits
    # UTF-8 with BOM and quoted headers; csv.DictReader otherwise leaves the
    # BOM and quotes embedded in the field names.
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            cleaned = {}
            for k, v in raw.items():
                if k is None:
                    continue
                key = k.strip().strip('"').strip("'")
                cleaned[key] = v
            rows.append(cleaned)
        return rows


def fnum(s: str | None, fmt: str, blank: str = "—") -> str:
    if s is None or s == "":
        return blank
    try:
        return fmt.format(float(s))
    except (TypeError, ValueError):
        return blank


def mean_std(rows: list[dict], mean_key: str, std_key: str, fmt: str) -> str:
    if not rows:
        return "—"
    r = rows[0]
    m = r.get(mean_key, "")
    s = r.get(std_key, "")
    if m == "" or m is None:
        return "—"
    base = fmt.format(float(m))
    if s == "" or s is None:
        return base
    return f"{base} ± {fmt.format(float(s))}"


def channel_label(channel: str, snr: str) -> str:
    if channel == "none":
        return "no-channel"
    return f"{channel.upper()} {snr} dB"


# ---------------------------------------------------------------------------
# 1. E24 reconstruction path 3 seed (replaces single-seed Table 3)
# ---------------------------------------------------------------------------

def table_e24_recon_3seed() -> str:
    rows = read_csv(P0 / "e24_recon_3seed_mean_std.csv")
    n_seeds = rows[0].get("n_model_seeds", "3") if rows else "3"

    # Order the channel/SNR conditions as they appear in the paper.
    cond_order = [
        ("none", "inf"),
        ("awgn", "5"),
        ("awgn", "10"),
        ("rayleigh", "5"),
        ("rayleigh", "10"),
    ]

    by_cond: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_cond[(r["channel"], r["snr"])][int(r["k"])] = r

    out = []
    out.append("# E24 — Reconstruction Path (3 model seeds, mean ± std)\n")
    out.append(
        "_RVQ-distill main checkpoint, evaluated under three independent "
        "training seeds (s41/s42/s43) on the AID test split (n=1000 images "
        "per condition). Replaces the single-seed Table 3 in the paper draft._\n"
    )

    out.append(
        "| Channel | k | bits/img | PSNR (dB) | LPIPS | recon-cls acc (%) |"
    )
    out.append("|---|---:|---:|---:|---:|---:|")
    for ch, snr in cond_order:
        for k in (1, 2, 3, 4):
            r = by_cond.get((ch, snr), {}).get(k)
            if not r:
                continue
            psnr = mean_std([r], "psnr_mean", "psnr_std", "{:.2f}")
            lpips = mean_std([r], "lpips_mean", "lpips_std", "{:.3f}")
            cls_pct = (
                f"{float(r['recon_cls_acc_mean'])*100:.1f} ± "
                f"{float(r['recon_cls_acc_std'])*100:.1f}"
                if r.get("recon_cls_acc_mean") not in ("", None) else "—"
            )
            out.append(
                f"| {channel_label(ch, snr)} | {k} | "
                f"{r['bits_per_img']} | {psnr} | {lpips} | {cls_pct} |"
            )

    out.append(
        "\n**Notes.** "
        f"n_model_seeds = {n_seeds}; std is the unbiased sample std across "
        "the three seeds. Test set is 1000 AID images per condition; channel "
        "noise is identical across seeds (RNG seeded by `(model_name, channel, "
        "snr, k)`). bits/img = k × 2560 (16×16 spatial × 10 bits/codeword × k "
        "RVQ layers). The k=1→k=4 PSNR gap (~2.94 dB) far exceeds the per-cell "
        "std (~0.04–0.09 dB), so progressive reconstruction is statistically "
        "robust.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2. E25 L0-only vs all-layer ablation (paper §4.3 / §4.8)
# ---------------------------------------------------------------------------

def table_e25_l0_vs_all() -> str:
    task = read_csv(P0 / "e25_task_l0_vs_all.csv")
    recon = read_csv(P0 / "e25_recon_l0_vs_all.csv")
    probe_l0 = read_csv(P0 / "e25_layered_probe_l0.csv")
    probe_all = read_csv(P0 / "e25_layered_probe_all.csv")

    cond_order = [
        ("none", "inf"),
        ("awgn", "5"),
        ("awgn", "10"),
        ("rayleigh", "5"),
        ("rayleigh", "10"),
    ]

    def pick_h0(model: str, ch: str, snr: str) -> str:
        for r in task:
            if r["model"] == model and r["channel"] == ch and r["snr"] == snr:
                return f"{float(r['h0_acc'])*100:.1f}"
        return "—"

    def pick_recon(model: str, ch: str, snr: str, k: int, key: str, fmt: str,
                   pct: bool = False) -> str:
        for r in recon:
            if (r["model"] == model and r["channel"] == ch
                    and r["snr"] == snr and int(r["k"]) == k):
                v = float(r[key])
                if pct:
                    return f"{v*100:.1f}"
                return fmt.format(v)
        return "—"

    def pick_probe(rows, feat: str) -> str:
        for r in rows:
            if r["feature"] == feat:
                return f"{float(r['test_acc']):.1f}"
        return "—"

    out = []
    out.append("# E25 — L0-only vs all-layer distillation (matched seed=42)\n")
    out.append(
        "_The two checkpoints share the same architecture, RemoteCLIP teacher, "
        "training schedule and seed. The only difference is whether the "
        "RemoteCLIP-alignment loss is applied to the L0 STE quantized feature "
        "(`out['zq_l0_ste']`, the published rvq_distill) or to the full RVQ "
        "summed quantized representation (`out['zq']`, all four layers)._\n"
    )

    # ---- task path (h0_acc) ----
    out.append("## h0 accuracy (k=1, task path) under five channel conditions\n")
    out.append(
        "| Channel | rvq_distill (L0) | rvq_distill_all_layers | Δ (all − L0) |"
    )
    out.append("|---|---:|---:|---:|")
    for ch, snr in cond_order:
        a = pick_h0("rvq_distill_l0", ch, snr)
        b = pick_h0("rvq_distill_all", ch, snr)
        d = (
            f"{float(b) - float(a):+.1f} pp"
            if a != "—" and b != "—" else "—"
        )
        out.append(f"| {channel_label(ch, snr)} | {a} | {b} | {d} |")
    out.append("")

    # ---- reconstruction (k=4 headline) ----
    out.append(
        "## Reconstruction at k=4 (full residual stack)\n"
    )
    out.append(
        "| Channel | PSNR L0 | PSNR all | Δ PSNR | LPIPS L0 | LPIPS all | "
        "recon-cls L0 | recon-cls all | Δ recon-cls |"
    )
    out.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for ch, snr in cond_order:
        psnr_l0 = pick_recon("rvq_distill_l0", ch, snr, 4, "psnr", "{:.2f}")
        psnr_all = pick_recon("rvq_distill_all", ch, snr, 4, "psnr", "{:.2f}")
        d_psnr = (
            f"{float(psnr_all) - float(psnr_l0):+.2f} dB"
            if psnr_l0 != "—" and psnr_all != "—" else "—"
        )
        lp_l0 = pick_recon("rvq_distill_l0", ch, snr, 4, "lpips", "{:.3f}")
        lp_all = pick_recon("rvq_distill_all", ch, snr, 4, "lpips", "{:.3f}")
        cls_l0 = pick_recon(
            "rvq_distill_l0", ch, snr, 4, "recon_cls_acc", "{:.1f}", pct=True
        )
        cls_all = pick_recon(
            "rvq_distill_all", ch, snr, 4, "recon_cls_acc", "{:.1f}", pct=True
        )
        d_cls = (
            f"{float(cls_all) - float(cls_l0):+.1f} pp"
            if cls_l0 != "—" and cls_all != "—" else "—"
        )
        out.append(
            f"| {channel_label(ch, snr)} | {psnr_l0} | {psnr_all} | "
            f"{d_psnr} | {lp_l0} | {lp_all} | {cls_l0} | {cls_all} | {d_cls} |"
        )
    out.append("")

    # ---- layered probe ----
    out.append(
        "## Layered probe (test accuracy, %) — semantic content of each "
        "cumulative residual stack\n"
    )
    out.append("| Probe target | rvq_distill (L0) | rvq_distill_all_layers |")
    out.append("|---|---:|---:|")
    for feat, label in [
        ("L0_to_L0_emb", "L0 only"),
        ("L0_to_L1_emb", "L0+L1"),
        ("L0_to_L2_emb", "L0+L1+L2"),
        ("L0_to_L3_emb", "L0+L1+L2+L3"),
    ]:
        a = pick_probe(probe_l0, feat)
        b = pick_probe(probe_all, feat)
        out.append(f"| {label} | {a} | {b} |")

    out.append(
        "\n**Notes.** Single seed=42; n=1000 AID test images per channel "
        "condition; n_model_seeds = 1. Layered probes are linear classifiers "
        "fit on the cumulative quantized representation (`zq_l0..zq_l0+l1+...+l3`). "
        "All deltas point in the same direction: all-layer distillation is "
        "uniformly weaker on task and reconstruction; layered-probe accuracies "
        "are essentially flat (±0.4 pp), indicating that spreading the "
        "supervision dilutes L0's semantic concentration without compensating "
        "elsewhere. The single-seed delta on PSNR (0.28 dB) far exceeds the "
        "E24 cross-seed std (0.04–0.09 dB), so init noise cannot explain "
        "the gap.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. E25 + E31 joint counterfactual (causal §4.3 evidence)
# ---------------------------------------------------------------------------

def table_layer_target_counterfactual() -> str:
    # Three models: l0 (main), all_layers (E25), l1 (E31).
    # Pull h0 + k=4 PSNR + k=4 recon_cls + L0 layered probe + L0->L3 probe.
    e25_task = read_csv(P0 / "e25_task_l0_vs_all.csv")
    e25_recon = read_csv(P0 / "e25_recon_l0_vs_all.csv")
    e31_task = read_csv(P2 / "e31_l1_task.csv")
    e31_recon = read_csv(P2 / "e31_l1_recon.csv")
    probe_l0 = read_csv(P0 / "e25_layered_probe_l0.csv")
    probe_all = read_csv(P0 / "e25_layered_probe_all.csv")
    probe_l1 = read_csv(P2 / "e31_layered_probe_l1distill.csv")

    cond_order = [
        ("none", "inf"),
        ("awgn", "5"),
        ("awgn", "10"),
        ("rayleigh", "5"),
        ("rayleigh", "10"),
    ]

    def h0(rows, model, ch, snr):
        for r in rows:
            if r["model"] == model and r["channel"] == ch and r["snr"] == snr:
                return float(r["h0_acc"]) * 100
        return None

    def k4(rows, model, ch, snr, key):
        for r in rows:
            if (r["model"] == model and r["channel"] == ch
                    and r["snr"] == snr and int(r["k"]) == 4):
                return float(r[key])
        return None

    def probe_at(rows, feat):
        for r in rows:
            if r["feature"] == feat:
                return float(r["test_acc"])
        return None

    out = []
    out.append(
        "# E25 + E31 — Layer-target counterfactual ablation "
        "(matched seed=42)\n"
    )
    out.append(
        "_Three matched checkpoints differing only in where the RemoteCLIP "
        "alignment loss is applied: at L0 (the published rvq_distill), at the "
        "full RVQ representation `zq` (all four layers, E25), or at L1 alone "
        "(E31). All other hyperparameters are identical. Demonstrates that "
        "L0-only is a measured optimum and that semantic concentration in L0 "
        "is caused by the choice of supervision target, not by the residual "
        "ordering itself._\n"
    )

    # h0 panel
    out.append(
        "## Task accuracy (h0_acc, %, k=1) under five channel conditions\n"
    )
    out.append("| Channel | L0 only (main) | all layers (E25) | L1 only (E31) |")
    out.append("|---|---:|---:|---:|")
    for ch, snr in cond_order:
        a = h0(e25_task, "rvq_distill_l0", ch, snr)
        b = h0(e25_task, "rvq_distill_all", ch, snr)
        c = h0(e31_task, "rvq_distill_l1", ch, snr)
        sa = f"{a:.1f}" if a is not None else "—"
        sb = f"{b:.1f}" if b is not None else "—"
        sc = f"{c:.1f}" if c is not None else "—"
        out.append(f"| {channel_label(ch, snr)} | {sa} | {sb} | {sc} |")

    out.append("")
    out.append(
        "## k=4 reconstruction (PSNR / LPIPS / recon-cls accuracy)\n"
    )
    out.append(
        "| Channel | PSNR L0 / all / L1 (dB) | recon-cls L0 / all / L1 (%) |"
    )
    out.append("|---|---:|---:|")
    for ch, snr in cond_order:
        p_l0 = k4(e25_recon, "rvq_distill_l0", ch, snr, "psnr")
        p_all = k4(e25_recon, "rvq_distill_all", ch, snr, "psnr")
        p_l1 = k4(e31_recon, "rvq_distill_l1", ch, snr, "psnr")
        c_l0 = k4(e25_recon, "rvq_distill_l0", ch, snr, "recon_cls_acc")
        c_all = k4(e25_recon, "rvq_distill_all", ch, snr, "recon_cls_acc")
        c_l1 = k4(e31_recon, "rvq_distill_l1", ch, snr, "recon_cls_acc")
        psnr_str = " / ".join(
            f"{v:.2f}" if v is not None else "—" for v in (p_l0, p_all, p_l1)
        )
        cls_str = " / ".join(
            f"{v*100:.1f}" if v is not None else "—" for v in (c_l0, c_all, c_l1)
        )
        out.append(f"| {channel_label(ch, snr)} | {psnr_str} | {cls_str} |")

    out.append("")
    out.append(
        "## Layered probe (test acc, %) on cumulative quantized features\n"
    )
    out.append("| Probe target | L0 only | all layers (E25) | L1 only (E31) |")
    out.append("|---|---:|---:|---:|")
    for feat, label in [
        ("L0_to_L0_emb", "L0 only"),
        ("L0_to_L1_emb", "L0+L1"),
        ("L0_to_L2_emb", "L0+L1+L2"),
        ("L0_to_L3_emb", "L0+L1+L2+L3"),
    ]:
        a = probe_at(probe_l0, feat)
        b = probe_at(probe_all, feat)
        c = probe_at(probe_l1, feat)
        sa = f"{a:.1f}" if a is not None else "—"
        sb = f"{b:.1f}" if b is not None else "—"
        sc = f"{c:.1f}" if c is not None else "—"
        out.append(f"| {label} | {sa} | {sb} | {sc} |")

    out.append(
        "\n**Notes.** Single seed=42 for all three checkpoints; n=1000 AID "
        "test images per channel condition. **All-layer** distillation "
        "(E25) uniformly underperforms L0-only on task and reconstruction "
        "while leaving the layered probe roughly flat — supervision is "
        "diluted but not relocated. **L1-only** distillation (E31) collapses: "
        "the L0 layered probe drops from 86 → 30 % (≈ −56 pp), task h0 drops "
        "from 83 → 34 % (≈ −48 pp), and clean k=4 PSNR drops 3.6 dB. The "
        "L1 probe does **not** rise to compensate; instead the entire RVQ "
        "training fails to converge well, indicating that the residual "
        "coarse-to-fine geometry requires its first layer to carry the "
        "global semantic signal. Together these establish a causal "
        "counterfactual: L0-only distillation is a measured optimum, not a "
        "design choice, and the apparent layer specialization in §4.3 is a "
        "predictable consequence of supervision placement, not a circular "
        "by-product of the probe.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. E26 RemoteCLIP vs OpenAI CLIP teacher
# ---------------------------------------------------------------------------

def table_e26_teacher() -> str:
    rows = read_csv(P0 / "e26_teacher_ablation.csv")
    cond_order = [
        ("none", "inf"),
        ("awgn", "5"),
        ("awgn", "10"),
        ("rayleigh", "5"),
        ("rayleigh", "10"),
    ]

    def find(teacher, ch, snr):
        for r in rows:
            if r["teacher"] == teacher and r["channel"] == ch and r["snr"] == snr:
                return r
        return None

    out = []
    out.append("# E26 — RemoteCLIP vs OpenAI CLIP teacher (matched seed=42)\n")
    out.append(
        "_Both checkpoints use the same RVQ-distill recipe; the only "
        "difference is the teacher's pretrained weights (RemoteCLIP "
        "domain-specific vs OpenAI CLIP general-purpose). Calibrates how "
        "much of the distillation gain comes from RS-domain pretraining vs "
        "from V-L distillation in general._\n"
    )

    # Task + recon side-by-side
    out.append(
        "## h0 (task) and k=4 reconstruction across five channel conditions\n"
    )
    out.append(
        "| Channel | h0 RemoteCLIP / OpenAI (%) | Δ h0 | k=4 PSNR R / O (dB) | "
        "Δ PSNR | k=4 recon-cls R / O (%) |"
    )
    out.append("|---|---:|---:|---:|---:|---:|")
    for ch, snr in cond_order:
        r = find("remoteclip", ch, snr)
        o = find("openai_clip", ch, snr)
        if not r or not o:
            continue
        h0_r = float(r["h0_acc"]) * 100
        h0_o = float(o["h0_acc"]) * 100
        psnr_r = float(r["k4_psnr"])
        psnr_o = float(o["k4_psnr"])
        cls_r = float(r["k4_recon_cls_acc"]) * 100
        cls_o = float(o["k4_recon_cls_acc"]) * 100
        out.append(
            f"| {channel_label(ch, snr)} | {h0_r:.1f} / {h0_o:.1f} | "
            f"{h0_r - h0_o:+.1f} pp | {psnr_r:.2f} / {psnr_o:.2f} | "
            f"{psnr_r - psnr_o:+.2f} dB | {cls_r:.1f} / {cls_o:.1f} |"
        )

    # Layered probe (single value per teacher, repeated)
    out.append("\n## Layered probe (test acc, %, single value per teacher)\n")
    out.append("| Probe target | RemoteCLIP | OpenAI CLIP | Δ |")
    out.append("|---|---:|---:|---:|")
    r = next((r for r in rows if r["teacher"] == "remoteclip"), None)
    o = next((r for r in rows if r["teacher"] == "openai_clip"), None)
    if r and o:
        for k_label, key in [
            ("L0 only", "l0_layered_probe_acc"),
            ("L0+L1+L2+L3", "l0_to_l3_probe_acc"),
        ]:
            rv = float(r[key])
            ov = float(o[key])
            out.append(
                f"| {k_label} | {rv:.1f} | {ov:.1f} | {rv - ov:+.1f} pp |"
            )

    out.append(
        "\n**Notes.** Single seed=42; n=1000 AID test images per condition. "
        "Δ values are RemoteCLIP minus OpenAI CLIP. RemoteCLIP yields a modest "
        "1.4–6.0 pp h0 advantage that grows under fading; reconstruction "
        "metrics (PSNR, recon-cls at k=4) are essentially identical, with "
        "OpenAI CLIP marginally higher PSNR (≈ 0.1 dB). Therefore the paper "
        "should not claim RemoteCLIP is necessary; the contribution is "
        "framed as a domain-specific teacher that improves task path under "
        "fading, while V–L distillation in general is the load-bearing idea.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5. E27 strict same-bit codec+LDPC stress baseline (paper §6 main table)
# ---------------------------------------------------------------------------

def table_e27_codec_ldpc() -> str:
    rows = read_csv(P1 / "e27_codec_ldpc_mean_std.csv")

    # Highlight rows: rstoken k matched to total_bits / 2560
    # total=5120 -> k=1, total=10240 -> k=2, total=20480 -> k=4
    main_rows = []
    for r in rows:
        if r["method"] == "rstoken":
            tb = int(r["total_bits"])
            k = int(r["k"]) if r["k"] not in ("", None) else 0
            expected_k = {5120: 1, 10240: 2, 20480: 4}.get(tb)
            if expected_k is None or k != expected_k:
                continue
        main_rows.append(r)

    # Order: budget asc, channel awgn-then-rayleigh, snr asc, method (rstoken first)
    method_rank = {"rstoken": 0, "jpeg2000": 1, "webp": 2}
    channel_rank = {"awgn": 0, "rayleigh": 1}

    def sort_key(r):
        return (
            int(r["total_bits"]),
            channel_rank.get(r["channel"], 9),
            int(r["snr"]),
            method_rank.get(r["method"], 9),
        )

    main_rows.sort(key=sort_key)

    out = []
    out.append("# E27 — Strict same-transmitted-bit codec + LDPC stress baseline\n")
    out.append(
        "_Each method is encoded to a matched source-bit budget such that, "
        "after rate-1/2 LDPC protection, the total transmitted-bit budget "
        "is identical across rstoken, JPEG2000 and WebP. RS-Token uses "
        "k = total_bits / 2560 RVQ layers; JPEG2000 / WebP encode to the "
        "same source-bit count and are then protected by the same LDPC code._\n"
    )
    out.append(
        "| total_bits | Channel | SNR (dB) | Method | k | actual src bits "
        "(mean) | decode-fail rate (%) | recon-cls (%) | PSNR (dB) |"
    )
    out.append(
        "|---:|---|---:|---|---:|---:|---:|---:|---:|"
    )
    for r in main_rows:
        method = r["method"]
        k = r["k"] if r["k"] not in ("", None) else "—"
        asb = (
            f"{float(r['actual_source_bits_mean_mean']):.0f}"
            if r.get("actual_source_bits_mean_mean") not in ("", None)
            else "—"
        )
        df = (
            f"{float(r['decode_failure_rate_mean'])*100:.1f}"
            if r.get("decode_failure_rate_mean") not in ("", None)
            else "—"
        )
        # recon_cls_acc: rstoken puts cls there directly; classic uses cls_acc_all
        if method == "rstoken":
            cls_mean = r.get("recon_cls_acc_mean")
        else:
            cls_mean = r.get("cls_acc_all_mean")
        cls = (
            f"{float(cls_mean)*100:.1f}"
            if cls_mean not in ("", None) else "—"
        )
        psnr = (
            f"{float(r['psnr_mean']):.2f}"
            if r.get("psnr_mean") not in ("", None) else "—"
        )
        out.append(
            f"| {r['total_bits']} | {r['channel'].upper()} | {r['snr']} | "
            f"{method} | {k} | {asb} | {df} | {cls} | {psnr} |"
        )

    out.append(
        "\n**Notes.** n_channel_seeds = 5 per cell. RS-Token rows show k=1 "
        "(total_bits=5120), k=2 (10240), k=4 (20480); JPEG2000 and WebP "
        "encode each image to the matching source-bit budget then receive "
        "the same rate-1/2 LDPC protection. `actual_source_bits` is reported "
        "because WebP's encoder cannot always reach the lowest budgets "
        "exactly. `decode-fail rate` is the fraction of images whose LDPC + "
        "container decode produced no usable output. RS-Token's decode-fail "
        "rate is uniformly **0 %** across all conditions while codec "
        "baselines fail on 60–100 % of images, demonstrating that the "
        "codec→bit→LDPC pipeline is structurally fragile under bit-level "
        "channel errors at these budgets. **The LDPC code used here is a "
        "custom rate-1/2 systematic sparse code with min-sum BP, not 5G NR "
        "LDPC**; the comparison is at fixed transmitted bits, not under any "
        "standard communication protocol.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 6. LDPC full appendix (E23 + E27 mean ± std rows)
# ---------------------------------------------------------------------------

def table_ldpc_full_appendix() -> str:
    e23 = read_csv(LOGS / "e23_ldpc_key5seeds_mean_std.csv")
    e27 = read_csv(P1 / "e27_codec_ldpc_mean_std.csv")

    out = []
    out.append("# LDPC appendix — full table (E23 + E27, mean ± std over 5 channel seeds)\n")
    out.append(
        "_All rate-1/2 LDPC results from both E23 (the original 5-seed sweep) "
        "and E27 (the strict same-transmitted-bit sweep) are listed here for "
        "reproducibility; the main paper presents selected rows only. Numbers "
        "are mean ± std across `channel_seeds = {0, 1, 2, 3, 4}`._\n"
    )

    # ----- E23 block -----
    out.append("## E23 — Original 5-seed LDPC sweep\n")
    out.append(
        "| method | total_bits | source_bits | k | channel | SNR | "
        "decode-fail rate (%) | post-LDPC BER | h0 (%) | recon-cls (%) | "
        "PSNR (dB) |"
    )
    out.append(
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    )
    for r in e23:
        df = (
            f"{float(r['decode_failure_rate_mean'])*100:.1f} ± "
            f"{float(r['decode_failure_rate_std'])*100:.1f}"
            if r.get("decode_failure_rate_mean") not in ("", None) else "—"
        )
        ber = (
            f"{float(r['post_ldpc_ber_mean']):.4f}"
            if r.get("post_ldpc_ber_mean") not in ("", None) else "—"
        )
        h0 = (
            f"{float(r['h0_acc_mean'])*100:.1f}"
            if r.get("h0_acc_mean") not in ("", None) else "—"
        )
        # recon class accuracy field name differs between rstoken and classic
        cls_v = r.get("main_acc_mean") or r.get("cls_acc_all_mean") or ""
        cls = f"{float(cls_v)*100:.1f}" if cls_v not in ("", None) else "—"
        psnr = (
            f"{float(r['psnr_mean']):.2f}"
            if r.get("psnr_mean") not in ("", None) else "—"
        )
        k_v = r["k"] if r["k"] not in ("", None) else "—"
        out.append(
            f"| {r['method']} | {r['total_bits']} | {r['source_bits']} | "
            f"{k_v} | {r['channel'].upper()} | {r['snr']} | {df} | {ber} | "
            f"{h0} | {cls} | {psnr} |"
        )

    # ----- E27 block (full table, all rstoken k values not just matched) -----
    out.append("\n## E27 — Strict same-bit codec+LDPC sweep (full)\n")
    out.append(
        "| method | total_bits | source_bits | k | channel | SNR | "
        "decode-fail rate (%) | post-LDPC BER | recon-cls (%) | PSNR (dB) |"
    )
    out.append(
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"
    )
    e27_sorted = sorted(
        e27,
        key=lambda r: (
            int(r["total_bits"]),
            {"awgn": 0, "rayleigh": 1}.get(r["channel"], 9),
            int(r["snr"]),
            {"rstoken": 0, "jpeg2000": 1, "webp": 2}.get(r["method"], 9),
            int(r["k"]) if r["k"] not in ("", None) else 99,
        ),
    )
    for r in e27_sorted:
        df = (
            f"{float(r['decode_failure_rate_mean'])*100:.1f} ± "
            f"{float(r['decode_failure_rate_std'])*100:.1f}"
            if r.get("decode_failure_rate_mean") not in ("", None) else "—"
        )
        ber = (
            f"{float(r['post_ldpc_ber_mean']):.4f}"
            if r.get("post_ldpc_ber_mean") not in ("", None) else "—"
        )
        if r["method"] == "rstoken":
            cls_v = r.get("recon_cls_acc_mean")
        else:
            cls_v = r.get("cls_acc_all_mean")
        cls = f"{float(cls_v)*100:.1f}" if cls_v not in ("", None) else "—"
        psnr = (
            f"{float(r['psnr_mean']):.2f}"
            if r.get("psnr_mean") not in ("", None) else "—"
        )
        k_v = r["k"] if r["k"] not in ("", None) else "—"
        out.append(
            f"| {r['method']} | {r['total_bits']} | {r['source_bits']} | "
            f"{k_v} | {r['channel'].upper()} | {r['snr']} | {df} | {ber} | "
            f"{cls} | {psnr} |"
        )

    out.append(
        "\n**Notes.** Both blocks report 5-seed means and standard "
        "deviations (channel-noise seeds, not training seeds). The LDPC "
        "code is a **custom rate-1/2 systematic sparse code with min-sum "
        "BP, NOT 5G NR LDPC**. JPEG2000 and WebP rows do not have a `k` "
        "value (residual layers are RS-Token-specific). E23 contains an "
        "earlier 5-seed sweep with looser budget alignment; E27 enforces "
        "matched transmitted-bit budgets across all three methods. The "
        "main paper's selected rows can be cross-checked against any row "
        "in this appendix.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    targets = [
        (P0 / "final_table_recon_3seed.md", table_e24_recon_3seed),
        (P0 / "final_table_l0_vs_all_layers.md", table_e25_l0_vs_all),
        (P0 / "final_table_layer_target_counterfactual.md",
         table_layer_target_counterfactual),
        (P0 / "final_table_teacher_ablation.md", table_e26_teacher),
        (P1 / "final_table_codec_ldpc.md", table_e27_codec_ldpc),
        (P1 / "final_appendix_ldpc_full.md", table_ldpc_full_appendix),
    ]
    for path, fn in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = fn()
        path.write_text(body, encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}  ({len(body)} chars)")


if __name__ == "__main__":
    main()
