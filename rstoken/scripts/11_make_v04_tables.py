"""Aggregate v0.4 pre-submission tables and claim audit notes.

The E18 aggregation rule is explicit:
  1. Average channel seeds within the same model seed.
  2. Then report mean/std across model seeds.

This script follows that rule even when only one channel seed is available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunSpec:
    model: str
    seed: int
    run_name: str
    ckpt: str


RUNS = [
    RunSpec("rvq_baseline", 41, "rvq_baseline_s41", "checkpoints/rvq_baseline_s41/best.pt"),
    RunSpec("rvq_baseline", 42, "rvq_baseline", "checkpoints/rvq_baseline/best.pt"),
    RunSpec("rvq_baseline", 43, "rvq_baseline_s43", "checkpoints/rvq_baseline_s43/best.pt"),
    RunSpec("rvq_distill", 41, "rvq_distill_s41", "checkpoints/rvq_distill_s41/best.pt"),
    RunSpec("rvq_distill", 42, "rvq_distill", "checkpoints/rvq_distill/best.pt"),
    RunSpec("rvq_distill", 43, "rvq_distill_s43", "checkpoints/rvq_distill_s43/best.pt"),
]

LEGACY_CHANNEL_FILES = {
    ("rvq_baseline", "awgn"): "logs/stage4_rvq_baseline.csv",
    ("rvq_baseline", "rayleigh"): "logs/stage4_baseline_rayleigh.csv",
    ("rvq_distill", "awgn"): "logs/stage4_distill.csv",
    ("rvq_distill", "rayleigh"): "logs/stage4_distill_rayleigh.csv",
}


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def norm_snr(value: str) -> str:
    value = str(value).strip()
    if value.lower() in {"inf", "none"}:
        return "inf"
    value = value.replace("dB", "").replace("db", "")
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isclose(number, round(number)):
        return str(int(round(number)))
    return str(number)


def ckpt_metrics(path: Path) -> tuple[str, float | None, float | None]:
    if not path.exists():
        return "missing", None, None
    ckpt = torch_load(path)
    metrics = ckpt.get("metrics", {})
    psnr = metrics.get("val/psnr")
    lpips = metrics.get("val/lpips")
    return "complete", float(psnr) if psnr is not None else None, float(lpips) if lpips is not None else None


def read_channel_csv(path: Path) -> dict[tuple[str, str], float]:
    rows: dict[tuple[str, str], float] = {}
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("k", "")).strip() != "1":
                continue
            snr = norm_snr(row.get("snr", ""))
            acc = safe_float(row.get("L0_bow_acc"))
            if acc is None:
                continue
            rows[(snr, "L0_bow_acc")] = acc
    return rows


def channel_files_for_run(run: RunSpec, channel: str, eval_dir: Path) -> list[Path]:
    files = sorted(eval_dir.glob(f"{run.run_name}_{channel}_cs*.csv"))
    if files:
        return files
    legacy = LEGACY_CHANNEL_FILES.get((run.model, channel))
    if run.seed == 42 and legacy:
        path = project_path(legacy)
        return [path] if path.exists() else []
    return []


def channel_metric_for_run(
    run: RunSpec,
    channel: str,
    snr: str,
    eval_dir: Path,
) -> tuple[float | None, int]:
    values: list[float] = []
    for path in channel_files_for_run(run, channel, eval_dir):
        data = read_channel_csv(path)
        value = data.get((snr, "L0_bow_acc"))
        if value is not None:
            values.append(value)
    if not values:
        return None, 0
    return statistics.mean(values), len(values)


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_seed_stats(eval_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for run in RUNS:
        status, best_psnr, best_lpips = ckpt_metrics(project_path(run.ckpt))
        no_channel, no_count = channel_metric_for_run(run, "awgn", "inf", eval_dir)
        awgn5, awgn5_count = channel_metric_for_run(run, "awgn", "5", eval_dir)
        awgn10, awgn10_count = channel_metric_for_run(run, "awgn", "10", eval_dir)
        ray5, ray5_count = channel_metric_for_run(run, "rayleigh", "5", eval_dir)
        ray10, ray10_count = channel_metric_for_run(run, "rayleigh", "10", eval_dir)
        rows.append(
            {
                "model": run.model,
                "seed": run.seed,
                "run_name": run.run_name,
                "checkpoint": run.ckpt,
                "status": status,
                "best_psnr": fmt(best_psnr),
                "best_lpips": fmt(best_lpips),
                "no_channel_h0_acc": fmt(no_channel),
                "awgn_5_h0_acc": fmt(awgn5),
                "awgn_10_h0_acc": fmt(awgn10),
                "rayleigh_5_h0_acc": fmt(ray5),
                "rayleigh_10_h0_acc": fmt(ray10),
                "channel_seed_count_no_channel": no_count,
                "channel_seed_count_awgn_5": awgn5_count,
                "channel_seed_count_awgn_10": awgn10_count,
                "channel_seed_count_rayleigh_5": ray5_count,
                "channel_seed_count_rayleigh_10": ray10_count,
            }
        )
    return rows


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def make_task_mean_std(seed_rows: list[dict]) -> list[dict]:
    metric_map = [
        ("best_psnr", "best PSNR"),
        ("best_lpips", "best LPIPS"),
        ("no_channel_h0_acc", "no-channel h0/L0_bow acc"),
        ("awgn_5_h0_acc", "AWGN +5 h0 acc"),
        ("awgn_10_h0_acc", "AWGN +10 h0 acc"),
        ("rayleigh_5_h0_acc", "Rayleigh +5 h0 acc"),
        ("rayleigh_10_h0_acc", "Rayleigh +10 h0 acc"),
    ]
    rows: list[dict] = []
    for model in ["rvq_baseline", "rvq_distill"]:
        model_rows = [row for row in seed_rows if row["model"] == model and row["status"] == "complete"]
        for key, label in metric_map:
            values = [float(row[key]) for row in model_rows if row.get(key)]
            mean, std = mean_std(values)
            rows.append(
                {
                    "model": model,
                    "metric": label,
                    "mean": fmt(mean),
                    "std": fmt(std),
                    "n_model_seeds": len(values),
                    "per_seed_values": ";".join(f"{v:.6f}" for v in values),
                    "aggregation_rule": "channel seeds averaged within each model seed, then mean/std across model seeds",
                }
            )
    return rows


def make_reconstruction_path_table() -> list[dict]:
    rows: list[dict] = []
    for row in read_csv_rows(project_path("logs/e17_recon_path.csv")):
        rows.append(
            {
                "model": row.get("model", ""),
                "model_seed": "42",
                "path": "reconstruction",
                "channel": row.get("channel", ""),
                "snr": norm_snr(row.get("snr", "")),
                "ber": row.get("ber", ""),
                "k": row.get("k", ""),
                "bits_per_img": row.get("bits_per_img", ""),
                "psnr": row.get("psnr", ""),
                "lpips": row.get("lpips", ""),
                "recon_cls_acc": row.get("recon_cls_acc", ""),
                "evaluator": "aid_classifier_resnet34",
                "scope": "single model seed; reconstruction path supports k=1..4 claims",
            }
        )
    return rows


def make_external_baseline_table() -> list[dict]:
    rows: list[dict] = []
    for row in read_csv_rows(project_path("logs/e19_classic_baselines.csv")):
        rows.append(
            {
                "method": row.get("method", ""),
                "target_bits": row.get("target_bits", ""),
                "actual_bits_mean": row.get("actual_bits_mean", ""),
                "actual_bits_std": row.get("actual_bits_std", ""),
                "channel": row.get("channel", ""),
                "snr": norm_snr(row.get("snr", "")),
                "ber": row.get("ber", ""),
                "decode_failure_rate": row.get("decode_failure_rate", ""),
                "cls_acc_all": row.get("cls_acc_all", ""),
                "cls_acc_decoded": row.get("cls_acc_decoded", ""),
                "psnr_valid": row.get("psnr_valid", ""),
                "lpips_valid": row.get("lpips_valid", ""),
                "num_samples": row.get("num_samples", ""),
                "evaluator": "aid_classifier_resnet34",
                "allowed_wording": "unprotected compressed bitstream over BPSK channel; no LDPC",
            }
        )
    return rows


def _find_row(rows: list[dict], **criteria: str) -> dict | None:
    for row in rows:
        if all(str(row.get(key, "")) == str(value) for key, value in criteria.items()):
            return row
    return None


def make_rayleigh0_context_table() -> list[dict]:
    task_rows = read_csv_rows(project_path("logs/e17_task_path.csv"))
    recon_rows = read_csv_rows(project_path("logs/e17_recon_path.csv"))
    classic_rows = read_csv_rows(project_path("logs/e19_classic_baselines.csv"))
    rows: list[dict] = []

    for model in ["rvq_baseline", "rvq_distill"]:
        row = _find_row(task_rows, model=model, channel="rayleigh", snr="0", k="1")
        if row:
            rows.append(
                {
                    "system": model,
                    "family": "RS-Token internal tokenizer",
                    "path": "task",
                    "k": "1",
                    "target_or_bits_per_img": row.get("bits_per_img", ""),
                    "actual_bits_mean": "",
                    "channel": "rayleigh",
                    "snr": "0",
                    "ber": row.get("ber", ""),
                    "h0_acc": row.get("h0_acc", ""),
                    "psnr": "",
                    "lpips": "",
                    "recon_cls_acc": "",
                    "decode_failure_rate": "0",
                    "cls_acc_all": "",
                    "interpretation": "breakdown boundary; task path only supports k=1 h0/L0_bow",
                }
            )

    for row in recon_rows:
        if row.get("model") != "rvq_distill" or row.get("channel") != "rayleigh" or norm_snr(row.get("snr", "")) != "0":
            continue
        rows.append(
            {
                "system": "rvq_distill",
                "family": "RS-Token reconstruction",
                "path": "reconstruction",
                "k": row.get("k", ""),
                "target_or_bits_per_img": row.get("bits_per_img", ""),
                "actual_bits_mean": "",
                "channel": "rayleigh",
                "snr": "0",
                "ber": row.get("ber", ""),
                "h0_acc": "",
                "psnr": row.get("psnr", ""),
                "lpips": row.get("lpips", ""),
                "recon_cls_acc": row.get("recon_cls_acc", ""),
                "decode_failure_rate": "0",
                "cls_acc_all": "",
                "interpretation": "breakdown boundary; reconstruction path supports k=1..4",
            }
        )

    for row in classic_rows:
        if row.get("channel") != "rayleigh" or norm_snr(row.get("snr", "")) != "0":
            continue
        rows.append(
            {
                "system": f"{row.get('method', '')}_target_{row.get('target_bits', '')}",
                "family": "external classic compressed bitstream",
                "path": "decode_then_classify",
                "k": "",
                "target_or_bits_per_img": row.get("target_bits", ""),
                "actual_bits_mean": row.get("actual_bits_mean", ""),
                "channel": "rayleigh",
                "snr": "0",
                "ber": row.get("ber", ""),
                "h0_acc": "",
                "psnr": row.get("psnr_valid", ""),
                "lpips": row.get("lpips_valid", ""),
                "recon_cls_acc": "",
                "decode_failure_rate": row.get("decode_failure_rate", ""),
                "cls_acc_all": row.get("cls_acc_all", ""),
                "interpretation": "unprotected bitstream stress boundary; no LDPC",
            }
        )
    return rows


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_claim_audit(out_path: Path, seed_rows: list[dict], task_rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    e16 = read_json(project_path("logs/aid_classifier_resnet34/test_metrics.json"))
    e17_task = read_csv_rows(project_path("logs/e17_task_path.csv"))
    e17_recon = read_csv_rows(project_path("logs/e17_recon_path.csv"))
    e19 = read_csv_rows(project_path("logs/e19_classic_baselines.csv"))
    table3 = read_csv_rows(out_path.parent / "table3_reconstruction_path.csv")
    table4 = read_csv_rows(out_path.parent / "table4_external_baseline.csv")
    table5 = read_csv_rows(out_path.parent / "table5_rayleigh0_context.csv")
    webp_rows = [row for row in e19 if row.get("method") == "webp"]
    jp2_rows = [row for row in e19 if row.get("method") == "jpeg2000"]
    complete_seed_models = {
        model: sum(1 for row in seed_rows if row["model"] == model and row["status"] == "complete")
        for model in ["rvq_baseline", "rvq_distill"]
    }

    e16_acc = None if e16 is None else e16.get("test_acc")
    e16_status = "missing"
    if isinstance(e16_acc, (float, int)):
        if e16_acc >= 0.90:
            e16_status = "main evaluator allowed"
        elif e16_acc >= 0.85:
            e16_status = "usable with evaluator ceiling caveat"
        else:
            e16_status = "risk: evaluator below 85%"

    e18_complete = all(count >= 3 for count in complete_seed_models.values())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# v0.4 Claim Audit\n\n")
        f.write("This audit separates supported claims from internal-only or blocked claims.\n\n")
        f.write("## Status Snapshot\n\n")
        f.write(f"- E16 clean classifier: {e16_status}")
        if isinstance(e16_acc, (float, int)):
            f.write(f" (test top-1={e16_acc * 100:.2f}%)")
        f.write("\n")
        f.write(f"- E17 task rows: {len(e17_task)}, reconstruction rows: {len(e17_recon)}\n")
        f.write(
            f"- E18 complete model seeds: baseline={complete_seed_models['rvq_baseline']}/3, "
            f"distill={complete_seed_models['rvq_distill']}/3\n"
        )
        f.write(f"- E19 WebP rows: {len(webp_rows)}, JPEG2000 rows: {len(jp2_rows)}\n\n")
        f.write(
            f"- v0.4 derived tables: reconstruction={len(table3)} rows, "
            f"external={len(table4)} rows, rayleigh0={len(table5)} rows\n\n"
        )

        f.write("## Claims That Can Be Written\n\n")
        if isinstance(e16_acc, (float, int)) and e16_acc >= 0.90:
            f.write("- The ResNet34 clean AID classifier can be used as the reconstruction-path evaluator.\n")
        if e17_task and e17_recon:
            f.write("- Task path and reconstruction path are now separated: h0/L0_bow is reported only for k=1, while PSNR/LPIPS/recon classifier are reported for k=1..4.\n")
        if e18_complete:
            f.write("- E18 main seed statistics can be reported as mean +/- std across model seeds, after averaging channel seeds within each model seed.\n")
        if webp_rows:
            f.write("- WebP can be reported as an external unprotected compressed-bitstream baseline at matched bit budgets.\n")
        if jp2_rows:
            f.write("- JPEG2000 can be reported as an external unprotected compressed-bitstream baseline at matched bit budgets.\n")
        f.write("\n")

        f.write("## Claims Limited To Internal Baselines\n\n")
        f.write("- RS-Token vs. rvq_baseline is an internal tokenizer baseline unless compared against WebP/JPEG2000 rows from E19.\n")
        if not e18_complete:
            f.write("- E18 mean +/- std remains preliminary until both rvq_baseline and rvq_distill have seeds 41, 42, and 43 completed.\n")
        f.write("- h0/L0_bow task-path accuracy must not be used to support k=2..4 reconstruction claims.\n\n")

        f.write("## External Baseline Boundary\n\n")
        if webp_rows or jp2_rows:
            f.write("- External baseline wording allowed: unprotected compressed bitstream over BPSK channel.\n")
        else:
            f.write("- External baseline is not yet supported because no E19 rows are present.\n")
        if not jp2_rows:
            f.write("- JPEG2000 is not supported unless E19 contains JPEG2000 rows; if skipped, cite encoder unavailability.\n")
        f.write("- Do not write JPEG2000+LDPC or WebP+LDPC. No LDPC is implemented in these experiments.\n\n")

        f.write("## Rayleigh 0 dB Boundary\n\n")
        f.write("- Rayleigh 0 dB should still be framed as a breakdown/stress boundary, not as a strong operating-point claim.\n")
        f.write("- Strong deployment claims should use milder Rayleigh points such as +5/+10 dB unless E19/E20 shows otherwise.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--table",
        choices=["seed", "task", "recon", "external", "rayleigh0", "audit"],
        help="Generate one table/audit. Omit with --all to generate all outputs.",
    )
    parser.add_argument("--eval_dir", default="logs/seed_sweep")
    parser.add_argument("--out_dir", default="logs/v04_tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = project_path(args.eval_dir)
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = make_seed_stats(eval_dir)
    task_rows = make_task_mean_std(seed_rows)
    recon_rows = make_reconstruction_path_table()
    external_rows = make_external_baseline_table()
    rayleigh0_rows = make_rayleigh0_context_table()

    generate_all = args.all or args.table is None

    if generate_all or args.table == "seed":
        write_csv(out_dir / "table1_seed_stats.csv", seed_rows)
        print(f"Wrote {out_dir / 'table1_seed_stats.csv'}")
    if generate_all or args.table == "task":
        write_csv(out_dir / "table2_task_path_mean_std.csv", task_rows)
        print(f"Wrote {out_dir / 'table2_task_path_mean_std.csv'}")
    if generate_all or args.table == "recon":
        write_csv(out_dir / "table3_reconstruction_path.csv", recon_rows)
        print(f"Wrote {out_dir / 'table3_reconstruction_path.csv'}")
    if generate_all or args.table == "external":
        write_csv(out_dir / "table4_external_baseline.csv", external_rows)
        print(f"Wrote {out_dir / 'table4_external_baseline.csv'}")
    if generate_all or args.table == "rayleigh0":
        write_csv(out_dir / "table5_rayleigh0_context.csv", rayleigh0_rows)
        print(f"Wrote {out_dir / 'table5_rayleigh0_context.csv'}")
    if generate_all or args.table == "audit":
        make_claim_audit(out_dir / "claim_audit.md", seed_rows, task_rows)
        print(f"Wrote {out_dir / 'claim_audit.md'}")


if __name__ == "__main__":
    main()
