"""E41 — Evaluate DirectClipQuantizer ckpts (task + reconstruction).

Mirrors `09_eval_rvqs_recon_task_split.py` exactly (same channel BER model,
same h₀/L0_bow probe, same LPIPS/recon-cls/PSNR backends), but loads the
`DirectClipQuantizer` model class instead of `VQVAE`.

Usage:
    python scripts/eval_direct_clip.py \
        --models "e41_s41=checkpoints/paper_v05/e41_direct_clip_s41/best.pt,..." \
        --recon_models "e41_s41,e41_s42,e41_s43" \
        --task_out logs/paper_v05/e41_task.csv \
        --recon_out logs/paper_v05/e41_recon.csv \
        --task_snrs=5,10 --recon_snrs=5,10 --ks 1,2,3,4 \
        --classifier_ckpt checkpoints/aid_classifier_resnet34/best.pt \
        --seed 42 --device cuda --batch_size 64
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDConfig, AIDDataset, build_loaders, build_transforms
from models.perceptual import LPIPSLoss
from models.direct_clip_quantizer import (
    DirectClipQuantizer, DirectClipQuantizerConfig,
)

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def project_path(p): return Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# ---- Channel model (identical to 09_eval) ---------------------------------

def ber_from_snr(snr_db: float, channel: str) -> float:
    snr_lin = 10 ** (snr_db / 10.0)
    if channel == "awgn":
        return 0.5 * math.erfc(math.sqrt(snr_lin))
    if channel == "rayleigh":
        return 0.5 * (1.0 - math.sqrt(snr_lin / (1.0 + snr_lin)))
    raise ValueError(channel)


def condition_seed(base_seed: int, model_name: str, channel: str, snr: str, k: int) -> int:
    text = f"{model_name}|{channel}|{snr}|{k}"
    acc = base_seed & 0x7FFFFFFF
    for ch in text:
        acc = (acc * 131 + ord(ch)) & 0x7FFFFFFF
    return acc


def corrupt_indices(indices, codebook_size, ber, seed):
    if ber <= 0:
        return indices.clone()
    bits_per = int(math.ceil(math.log2(codebook_size)))
    flat = indices.reshape(-1)
    bit_pos = torch.arange(bits_per, device=flat.device)
    bit_mat = (flat.unsqueeze(1) >> bit_pos) & 1
    rng = torch.Generator(device=flat.device).manual_seed(seed)
    flip = torch.rand(bit_mat.shape, device=flat.device, generator=rng) < ber
    bit_mat = bit_mat ^ flip.long()
    weights = (1 << bit_pos).long()
    new_idx = (bit_mat * weights).sum(dim=1).clamp(0, codebook_size - 1)
    return new_idx.reshape(indices.shape)


def maybe_corrupt(indices, codebook_size, channel, snr, seed):
    if channel == "none":
        return indices.clone(), 0.0
    ber = ber_from_snr(float(snr), channel)
    return corrupt_indices(indices, codebook_size, ber, seed), ber


def condition_grid(snrs):
    yield ("none", "inf")
    for snr in snrs:
        yield ("awgn", str(snr))
    for snr in snrs:
        yield ("rayleigh", str(snr))


# ---- Model loader ---------------------------------------------------------

def load_direct_clip(ckpt_path: Path, device: str):
    ckpt = torch_load(ckpt_path, device)
    cfg = ckpt["config"]
    model_cfg = DirectClipQuantizerConfig(**cfg["model"])
    model = DirectClipQuantizer(model_cfg).to(device)
    # Trainable state was saved without "teacher.*" keys — use strict=False
    # so the freshly-loaded teacher weights are kept and only the trainable
    # tensors are overwritten.
    state = ckpt.get("model_trainable") or ckpt.get("model")
    if state is None:
        raise KeyError(f"ckpt missing model state: keys={list(ckpt.keys())}")
    missing_unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing_unexpected.missing_keys
               if not k.startswith("teacher.")]
    if missing:
        raise RuntimeError(f"missing non-teacher keys after load: {missing}")
    model.eval()
    return model, cfg, model_cfg


# ---- Encode / probe (mirror of 09_eval) -----------------------------------

@torch.no_grad()
def encode_indices(model, loader, device, keep_images):
    model.eval()
    all_indices, all_labels, all_images = [], [], []
    spatial = (model.cfg.grid_size, model.cfg.grid_size)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # Same path the model uses: features → quantizer
        z_seq = model.encode_features(x)
        _, indices, _ = model.quantizer(z_seq)
        all_indices.append(indices.cpu())
        all_labels.append(y.cpu())
        if keep_images:
            all_images.append(x.cpu())
    images = torch.cat(all_images) if keep_images else None
    return torch.cat(all_indices), torch.cat(all_labels), images, spatial


def l0_bow(indices_k1, codebook_size):
    l0_idx = indices_k1[..., 0]
    bow = torch.zeros(l0_idx.shape[0], codebook_size, dtype=torch.float32)
    bow.scatter_add_(1, l0_idx, torch.ones_like(l0_idx, dtype=torch.float32))
    bow = bow / l0_idx.shape[1]
    return bow.numpy()


def fit_h0_probe(train_idx, train_labels, codebook_size):
    x_train = l0_bow(train_idx[..., :1], codebook_size)
    scaler = StandardScaler(with_mean=False)
    x_train_s = scaler.fit_transform(x_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1)
    clf.fit(x_train_s, train_labels.numpy())
    return scaler, clf


def score_h0_probe(scaler, clf, test_idx_k1, test_labels, codebook_size):
    x_test = l0_bow(test_idx_k1, codebook_size)
    return float(clf.score(scaler.transform(x_test), test_labels.numpy()))


# ---- Recon-cls ------------------------------------------------------------

def build_classifier(backbone, num_classes):
    if backbone.lower() == "resnet34":
        m = models.resnet34(weights=None)
    elif backbone.lower() == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError(backbone)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classifier(ckpt_path, device):
    if not ckpt_path.exists():
        return None, f"missing: {ckpt_path}"
    ckpt = torch_load(ckpt_path, device)
    cfg = ckpt.get("config", {})
    mc = cfg.get("model", {})
    model = build_classifier(
        str(mc.get("backbone", "resnet34")),
        int(mc.get("num_classes", 30)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, None


def classifier_input_from_neg11(x, crop=224):
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    _, _, h, w = x01.shape
    if h < crop or w < crop:
        x01 = F.interpolate(x01, size=(crop, crop), mode="bilinear", align_corners=False)
    else:
        top = (h - crop) // 2
        left = (w - crop) // 2
        x01 = x01[:, :, top:top+crop, left:left+crop]
    mean = IMAGENET_MEAN.to(x01.device)
    std = IMAGENET_STD.to(x01.device)
    return (x01 - mean) / std


@torch.no_grad()
def reconstruction_metrics(model, indices, x_ref, labels, lpips_fn,
                           classifier, batch_size, device):
    n_total = int(indices.shape[0])
    psnr_sum = 0.0
    lpips_sum = 0.0
    cls_correct = 0
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        idx = indices[start:end].to(device)
        recon = model.decode_from_indices(idx)
        ref = x_ref[start:end].to(device)
        mse = F.mse_loss(recon, ref, reduction="none").flatten(1).mean(dim=1)
        psnr_sum += float((10 * torch.log10(4.0 / (mse + 1e-12))).sum().item())
        lpips_sum += float(lpips_fn(recon, ref).item()) * (end - start)
        if classifier is not None:
            logits = classifier(classifier_input_from_neg11(recon))
            pred = logits.argmax(dim=1).cpu()
            cls_correct += int((pred == labels[start:end]).sum().item())
    cls_acc = None if classifier is None else cls_correct / n_total
    return psnr_sum / n_total, lpips_sum / n_total, cls_acc


# ---- Loaders --------------------------------------------------------------

def train_eval_loader(data_cfg: AIDConfig) -> DataLoader:
    ds = AIDDataset(
        f"{data_cfg.splits_dir}/train.csv",
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    return DataLoader(
        ds, batch_size=data_cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory,
        drop_last=False,
        persistent_workers=data_cfg.num_workers > 0,
    )


def parse_model_specs(spec: str) -> dict[str, Path]:
    out = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        name, p = part.split("=", 1)
        out[name.strip()] = project_path(p.strip())
    return out


# ---- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True,
                        help="name=path,name=path,... (paths relative to project root)")
    parser.add_argument("--recon_models", default="",
                        help="comma list of model names that should also report reconstruction")
    parser.add_argument("--task_out", required=True)
    parser.add_argument("--recon_out", required=True)
    parser.add_argument("--task_snrs", default="5,10")
    parser.add_argument("--recon_snrs", default="5,10")
    parser.add_argument("--ks", default="1,2,3,4")
    parser.add_argument("--classifier_ckpt",
                        default="checkpoints/aid_classifier_resnet34/best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = args.device

    model_specs = parse_model_specs(args.models)
    recon_model_names = {
        s.strip() for s in args.recon_models.split(",") if s.strip()
    }
    task_snrs = [s.strip() for s in args.task_snrs.split(",") if s.strip()]
    recon_snrs = [s.strip() for s in args.recon_snrs.split(",") if s.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    classifier, cls_msg = load_classifier(project_path(args.classifier_ckpt), device)
    if cls_msg:
        print(f"[classifier] {cls_msg}; recon-cls will be NaN")

    task_rows = []
    recon_rows = []
    lpips_fn = None

    for model_name, ckpt_path in model_specs.items():
        print("\n" + "=" * 72)
        print(f"[model] {model_name}: {ckpt_path}")
        print("=" * 72)
        t_model = time.time()
        model, cfg, model_cfg = load_direct_clip(ckpt_path, device)

        data_cfg = AIDConfig(**cfg["data"])
        loaders = build_loaders(data_cfg)
        tr_loader = train_eval_loader(data_cfg)
        if lpips_fn is None:
            lpips_fn = LPIPSLoss(net=cfg["loss"]["lpips_net"]).to(device).eval()

        print("[encode] train split, no augmentation")
        tr_idx, y_tr, _, spatial = encode_indices(model, tr_loader, device, keep_images=False)
        print("[encode] test split")
        te_idx, y_te, x_te, _ = encode_indices(model, loaders["test"], device, keep_images=True)
        print(f"[encode] train_idx={tuple(tr_idx.shape)} test_idx={tuple(te_idx.shape)}")

        print("[task] fitting h0/L0_bow probe on clean train indices")
        scaler, clf = fit_h0_probe(tr_idx[..., :1], y_tr, model_cfg.codebook_size)

        for channel, snr in condition_grid(task_snrs):
            seed = condition_seed(args.seed, model_name, channel, snr, 1)
            te_k, ber = maybe_corrupt(
                te_idx[..., :1], model_cfg.codebook_size, channel, snr, seed,
            )
            acc = score_h0_probe(scaler, clf, te_k, y_te, model_cfg.codebook_size)
            task_rows.append({
                "model": model_name, "path": "task", "metric": "h0_acc",
                "channel": channel, "snr": snr,
                "ber": f"{ber:.8f}", "k": 1, "bits_per_img": 2560,
                "h0_acc": f"{acc:.6f}",
                "num_samples": int(y_te.numel()),
            })
            print(f"[task] {model_name} {channel:8s} snr={snr:>3s} acc={acc*100:.2f}%")

        if model_name in recon_model_names:
            print("[recon] running reconstruction path")
            for channel, snr in condition_grid(recon_snrs):
                for k in ks:
                    if k > model_cfg.rvq_num_quantizers:
                        continue
                    seed = condition_seed(args.seed, model_name, channel, snr, k)
                    te_k, ber = maybe_corrupt(
                        te_idx[..., :k], model_cfg.codebook_size, channel, snr, seed,
                    )
                    psnr_v, lpips_v, cls_acc = reconstruction_metrics(
                        model, te_k, x_te, y_te, lpips_fn,
                        classifier, args.batch_size, device,
                    )
                    recon_rows.append({
                        "model": model_name, "path": "reconstruction",
                        "channel": channel, "snr": snr,
                        "ber": f"{ber:.8f}", "k": k,
                        "bits_per_img": 2560 * k,
                        "psnr": f"{psnr_v:.6f}",
                        "lpips": f"{lpips_v:.6f}",
                        "recon_cls_acc": "" if cls_acc is None else f"{cls_acc:.6f}",
                        "num_samples": int(y_te.numel()),
                    })
                    print(f"[recon] {model_name} {channel:8s} snr={snr:>3s} k={k} "
                          f"psnr={psnr_v:.2f} lpips={lpips_v:.4f} "
                          f"cls={'NaN' if cls_acc is None else f'{cls_acc*100:.2f}%'}")

        print(f"[model] done in {(time.time() - t_model) / 60:.1f} min")

    # Write CSVs
    Path(args.task_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.recon_out).parent.mkdir(parents=True, exist_ok=True)

    task_keys = ["model", "path", "metric", "channel", "snr", "ber",
                 "k", "bits_per_img", "h0_acc", "num_samples"]
    with open(args.task_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=task_keys)
        w.writeheader()
        w.writerows(task_rows)

    recon_keys = ["model", "path", "channel", "snr", "ber", "k",
                  "bits_per_img", "psnr", "lpips", "recon_cls_acc",
                  "num_samples"]
    with open(args.recon_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=recon_keys)
        w.writeheader()
        w.writerows(recon_rows)

    print(f"\nWrote task table: {Path(args.task_out).resolve()}")
    print(f"Wrote reconstruction table: {Path(args.recon_out).resolve()}")


if __name__ == "__main__":
    main()
