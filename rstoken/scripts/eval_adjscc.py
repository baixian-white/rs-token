"""E35 — Evaluate ADJSCC checkpoint(s) on the same 5-channel grid as E36.

For each checkpoint and each (channel, snr) cell, compute reconstruction
PSNR / LPIPS / clean AID ResNet34 recon-cls accuracy on the AID test split.

ADJSCC has no L0/h0 task path; this script is reconstruction-only.

Usage:
    python scripts/eval_adjscc.py \
        --models "adjscc_b2560_s41=checkpoints/paper_v05/adjscc_b2560_s41/best.pt,..." \
        --out logs/paper_v05/e35_adjscc_recon.csv \
        --snrs 5,10 \
        --device cuda \
        --batch_size 64
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import rearrange  # noqa: F401  (kept for parity)
from torchvision import models as tv_models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDConfig, build_loaders   # noqa: E402
from models.adjscc import ADJSCC, ADJSCCConfig          # noqa: E402
from models.perceptual import LPIPSLoss                 # noqa: E402

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def project_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def torch_load(path: Path, device: str | torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_specs(spec: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for it in spec.split(","):
        it = it.strip()
        if not it:
            continue
        if "=" not in it:
            raise ValueError(f"--models entry must be name=path: {it}")
        n, p = it.split("=", 1)
        out[n.strip()] = project_path(p.strip())
    return out


def condition_grid(snrs: list[str]) -> list[tuple[str, str]]:
    rows = [("none", "inf")]
    for ch in ("awgn", "rayleigh"):
        for s in snrs:
            rows.append((ch, s))
    return rows


def cell_seed(base_seed: int, model_name: str, channel: str, snr: str) -> int:
    text = f"{model_name}|{channel}|{snr}"
    acc = base_seed & 0x7FFFFFFF
    for ch in text:
        acc = (acc * 131 + ord(ch)) & 0x7FFFFFFF
    return acc


def load_classifier(ckpt_path: Path, device: str
                    ) -> tuple[nn.Module | None, str | None]:
    if not ckpt_path.exists():
        return None, f"classifier not found: {ckpt_path}"
    try:
        ckpt = torch_load(ckpt_path, device)
        cfg = ckpt.get("config", {})
        m = cfg.get("model", {})
        backbone = str(m.get("backbone", "resnet34")).lower()
        n_cls = int(m.get("num_classes", 30))
        if backbone == "resnet34":
            net = tv_models.resnet34(weights=None)
        elif backbone == "resnet50":
            net = tv_models.resnet50(weights=None)
        else:
            return None, f"unsupported backbone: {backbone}"
        net.fc = nn.Linear(net.fc.in_features, n_cls)
        net.load_state_dict(ckpt["model"])
        return net.to(device).eval(), None
    except Exception as exc:
        return None, f"classifier load failed: {exc}"


def classifier_input_from_neg11(x: torch.Tensor, crop: int = 224) -> torch.Tensor:
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    _, _, h, w = x01.shape
    if h < crop or w < crop:
        x01 = F.interpolate(x01, size=(crop, crop), mode="bilinear",
                            align_corners=False)
    else:
        top = (h - crop) // 2
        left = (w - crop) // 2
        x01 = x01[:, :, top:top + crop, left:left + crop]
    mean = IMAGENET_MEAN.to(x01.device)
    std = IMAGENET_STD.to(x01.device)
    return (x01 - mean) / std


def load_adjscc(path: Path, device: str) -> tuple[ADJSCC, dict, ADJSCCConfig]:
    ckpt = torch_load(path, device)
    cfg = ckpt["config"]
    mc = ADJSCCConfig(**cfg["model"])
    model = ADJSCC(mc).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, mc


@torch.no_grad()
def evaluate_cell(
    model: ADJSCC,
    loader,
    channel: str,
    snr_db: float,
    cell_seed_int: int,
    device: str,
    classifier: nn.Module | None,
    lpips_fn: LPIPSLoss,
) -> tuple[float, float, float | None, int]:
    psnr_sum = 0.0
    lpips_sum = 0.0
    cls_correct = 0
    n = 0
    gen = torch.Generator(device=device).manual_seed(int(cell_seed_int))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        b = x.shape[0]
        snr = torch.full((b,), float(snr_db), device=device,
                         dtype=torch.float32)
        z = model.encoder(x, snr)
        from models.adjscc import normalise_power, channel_layer
        z = normalise_power(z)
        z_hat = channel_layer(z, snr, channel=channel, generator=gen)
        recon = model.decoder(z_hat, snr)

        mse = F.mse_loss(recon, x, reduction="none").flatten(1).mean(dim=1)
        psnr_sum += float((10 * torch.log10(4.0 / (mse + 1e-12))).sum().item())
        lpips_sum += float(lpips_fn(recon, x).item()) * b
        if classifier is not None:
            logits = classifier(classifier_input_from_neg11(recon))
            pred = logits.argmax(dim=1).cpu()
            cls_correct += int((pred == y).sum().item())
        n += b
    cls_acc = None if classifier is None else cls_correct / n
    return psnr_sum / n, lpips_sum / n, cls_acc, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True,
                        help="comma-separated name=path entries")
    parser.add_argument("--out", required=True)
    parser.add_argument("--snrs", default="5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--classifier_ckpt",
                        default="checkpoints/aid_classifier_resnet34/best.pt")
    parser.add_argument("--data_yaml", default=None,
                        help="Optional override for data block (cross-dataset).")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.")
        device = "cpu"

    models_specs = parse_specs(args.models)
    classifier, note = load_classifier(project_path(args.classifier_ckpt), device)
    if note:
        print(f"[classifier] {note}; recon_cls left blank.")
    else:
        print(f"[classifier] loaded {args.classifier_ckpt}")
    lpips_fn = LPIPSLoss(net="alex").to(device).eval()

    snrs = [s.strip() for s in args.snrs.split(",") if s.strip()]
    rows: list[dict] = []

    for model_name, ckpt_path in models_specs.items():
        print("\n" + "=" * 72)
        print(f"[model] {model_name}: {ckpt_path}")
        print("=" * 72)
        t0 = time.time()
        model, ckpt_cfg, model_cfg = load_adjscc(ckpt_path, device)

        if args.data_yaml is not None:
            with open(args.data_yaml, "r", encoding="utf-8") as f:
                ov = yaml.safe_load(f) or {}
            data_override = ov.get("data", {}) or {}
            ckpt_cfg["data"] = {**ckpt_cfg["data"], **data_override}
            print(f"[data_yaml] applied {args.data_yaml}: {data_override}")

        data_cfg = AIDConfig(**ckpt_cfg["data"])
        data_cfg.batch_size = args.batch_size
        loaders = build_loaders(data_cfg)
        test_loader = loaders["test"]

        n_uses = model.n_channel_uses_per_image()
        for ch, snr in condition_grid(snrs):
            seed = cell_seed(args.seed, model_name, ch, snr)
            # ADJSCC's AF block was trained with SNR ∈ [train_snr_min,
            # train_snr_max]; feeding the no-channel cell with snr=inf would
            # push the AF gate far out of distribution. Conditioning on
            # train_snr_max (12 dB) is the closest in-distribution proxy
            # for "no channel noise", and the channel layer itself bypasses
            # noise injection when channel=='none'.
            if snr == "inf":
                snr_val = float(model_cfg.train_snr_max)
            else:
                snr_val = float(snr)
            psnr, lpips, cls, n = evaluate_cell(
                model, test_loader, ch, snr_val, seed, device,
                classifier, lpips_fn,
            )
            cls_str = "" if cls is None else f"{cls:.6f}"
            row = {
                "model": model_name,
                "n_channel_uses": n_uses,
                "bits_per_image": n_uses,
                "channel": ch,
                "snr": snr,
                "psnr": f"{psnr:.6f}",
                "lpips": f"{lpips:.6f}",
                "recon_cls_acc": cls_str,
                "num_samples": n,
            }
            rows.append(row)
            cls_msg = "NA" if cls is None else f"{cls * 100:.2f}%"
            print(f"[recon] {model_name} {ch:8s} snr={snr:>3s} "
                  f"psnr={psnr:.2f} lpips={lpips:.4f} cls={cls_msg}")

        print(f"[model] done in {(time.time() - t0) / 60:.1f} min")

    out = project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
