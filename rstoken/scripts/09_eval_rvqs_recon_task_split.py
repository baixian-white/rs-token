"""E17 split evaluation for RS-Token task path and reconstruction path.

Task path:
  - k is fixed to 1.
  - Uses only h0/L0_bow features.
  - Reports h0_acc at 2560 bits/image.

Reconstruction path:
  - Uses k=1..4.
  - Reports PSNR, LPIPS, and optional reconstructed-image classifier accuracy.

The script intentionally keeps these two paths in separate CSV files so h0
metrics are not used to explain k=2..4 reconstruction conclusions.
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
from models.vqvae import VQVAE, VQVAEConfig

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def torch_load(path: Path, device: str | torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def ber_from_snr(snr_db: float, channel: str) -> float:
    snr_lin = 10 ** (snr_db / 10.0)
    if channel == "awgn":
        return 0.5 * math.erfc(math.sqrt(snr_lin))
    if channel == "rayleigh":
        return 0.5 * (1.0 - math.sqrt(snr_lin / (1.0 + snr_lin)))
    raise ValueError(f"unknown channel: {channel}")


def condition_seed(base_seed: int, model_name: str, channel: str, snr: str, k: int) -> int:
    text = f"{model_name}|{channel}|{snr}|{k}"
    acc = base_seed & 0x7FFFFFFF
    for ch in text:
        acc = (acc * 131 + ord(ch)) & 0x7FFFFFFF
    return acc


def corrupt_indices(
    indices: torch.Tensor,
    codebook_size: int,
    ber: float,
    seed: int,
) -> torch.Tensor:
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


def maybe_corrupt(
    indices: torch.Tensor,
    codebook_size: int,
    channel: str,
    snr: str,
    seed: int,
) -> tuple[torch.Tensor, float]:
    if channel == "none":
        return indices.clone(), 0.0
    ber = ber_from_snr(float(snr), channel)
    return corrupt_indices(indices, codebook_size, ber, seed), ber


@torch.no_grad()
def encode_indices(
    model: VQVAE,
    loader,
    device: str,
    keep_images: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    model.eval()
    all_indices: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_images: list[torch.Tensor] = []
    spatial = (16, 16)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model.encoder(x)
        _, _, h, w = z.shape
        spatial = (h, w)
        z_seq = rearrange(z, "b c h w -> b (h w) c")
        _, indices, _ = model.quantizer(z_seq)
        all_indices.append(indices.cpu())
        all_labels.append(y.cpu())
        if keep_images:
            all_images.append(x.cpu())

    images = torch.cat(all_images) if keep_images else None
    return torch.cat(all_indices), torch.cat(all_labels), images, spatial


def l0_bow(indices_k1: torch.Tensor, codebook_size: int) -> np.ndarray:
    l0_idx = indices_k1[..., 0]
    bow = torch.zeros(l0_idx.shape[0], codebook_size, dtype=torch.float32)
    bow.scatter_add_(1, l0_idx, torch.ones_like(l0_idx, dtype=torch.float32))
    bow = bow / l0_idx.shape[1]
    return bow.numpy()


def fit_h0_probe(
    train_idx: torch.Tensor,
    train_labels: torch.Tensor,
    codebook_size: int,
) -> tuple[StandardScaler, LogisticRegression]:
    x_train = l0_bow(train_idx[..., :1], codebook_size)
    scaler = StandardScaler(with_mean=False)
    x_train_s = scaler.fit_transform(x_train)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=-1)
    clf.fit(x_train_s, train_labels.numpy())
    return scaler, clf


def score_h0_probe(
    scaler: StandardScaler,
    clf: LogisticRegression,
    test_idx_k1: torch.Tensor,
    test_labels: torch.Tensor,
    codebook_size: int,
) -> float:
    x_test = l0_bow(test_idx_k1, codebook_size)
    return float(clf.score(scaler.transform(x_test), test_labels.numpy()))


def load_rvq_model(ckpt_path: Path, device: str) -> tuple[VQVAE, dict, VQVAEConfig]:
    ckpt = torch_load(ckpt_path, device)
    cfg = ckpt["config"]
    model_cfg = VQVAEConfig(**cfg["model"])
    if model_cfg.quantizer != "rvq":
        raise ValueError(f"{ckpt_path} is not an RVQ checkpoint")
    model = VQVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, model_cfg


def train_eval_loader(data_cfg: AIDConfig) -> DataLoader:
    dataset = AIDDataset(
        f"{data_cfg.splits_dir}/train.csv",
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
        persistent_workers=data_cfg.num_workers > 0,
    )


def build_classifier(backbone: str, num_classes: int) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet34":
        model = models.resnet34(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"unsupported classifier backbone: {backbone}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(
    ckpt_path: Path,
    device: str,
) -> tuple[nn.Module | None, str | None]:
    if not ckpt_path.exists():
        return None, f"classifier checkpoint not found: {ckpt_path}"
    try:
        ckpt = torch_load(ckpt_path, device)
        cfg = ckpt.get("config", {})
        model_cfg = cfg.get("model", {})
        model = build_classifier(
            str(model_cfg.get("backbone", "resnet34")),
            int(model_cfg.get("num_classes", 30)),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, None
    except Exception as exc:  # noqa: BLE001
        return None, f"classifier load failed: {exc}"


def classifier_input_from_neg11(x: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    _, _, h, w = x01.shape
    if h < crop_size or w < crop_size:
        x01 = F.interpolate(x01, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        x01 = x01[:, :, top : top + crop_size, left : left + crop_size]
    mean = IMAGENET_MEAN.to(x01.device)
    std = IMAGENET_STD.to(x01.device)
    return (x01 - mean) / std


@torch.no_grad()
def reconstruction_metrics(
    model: VQVAE,
    indices: torch.Tensor,
    x_ref: torch.Tensor,
    labels: torch.Tensor,
    spatial: tuple[int, int],
    lpips_fn: LPIPSLoss,
    classifier: nn.Module | None,
    batch_size: int,
    device: str,
) -> tuple[float, float, float | None]:
    n_total = int(indices.shape[0])
    psnr_sum = 0.0
    lpips_sum = 0.0
    cls_correct = 0
    h, w = spatial

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        idx = indices[start:end].to(device)
        zq_seq = model.quantizer.get_output_from_indices(idx)
        zq = rearrange(zq_seq, "b (h w) c -> b c h w", h=h, w=w)
        recon = model.decoder(zq)
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


def parse_model_specs(spec: str) -> dict[str, Path]:
    models_out: dict[str, Path] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"model spec must be name=path, got: {item}")
        name, path = item.split("=", 1)
        models_out[name.strip()] = project_path(path.strip())
    return models_out


def condition_grid(snrs: list[str]) -> list[tuple[str, str]]:
    rows = [("none", "inf")]
    for channel in ["awgn", "rayleigh"]:
        for snr in snrs:
            rows.append((channel, snr))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=(
            "rvq_baseline=checkpoints/rvq_baseline/best.pt,"
            "rvq_distill=checkpoints/rvq_distill/best.pt"
        ),
    )
    parser.add_argument("--recon_models", default="rvq_distill")
    parser.add_argument("--classifier_ckpt", default="checkpoints/aid_classifier_resnet34/best.pt")
    parser.add_argument("--task_out", default="logs/e17_task_path.csv")
    parser.add_argument("--recon_out", default="logs/e17_recon_path.csv")
    parser.add_argument("--task_snrs", default="-5,0,5,10")
    parser.add_argument("--recon_snrs", default="0,5,10")
    parser.add_argument("--ks", default="1,2,3,4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--data_yaml",
        default=None,
        help="Optional yaml whose top-level 'data' block overrides the data "
             "config baked into the checkpoint (e.g. for cross-dataset "
             "evaluation on NWPU-RESISC45). The yaml may also include "
             "'num_classes' under a 'model' block to adjust the linear "
             "probe head; the encoder/quantizer are not re-fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device = "cpu"

    model_specs = parse_model_specs(args.models)
    recon_model_names = {name.strip() for name in args.recon_models.split(",") if name.strip()}
    task_snrs = [s.strip() for s in args.task_snrs.split(",") if s.strip()]
    recon_snrs = [s.strip() for s in args.recon_snrs.split(",") if s.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    classifier, classifier_note = load_classifier(project_path(args.classifier_ckpt), device)
    if classifier_note:
        print(f"[classifier] {classifier_note}")
        print("[classifier] recon_cls_acc will be left blank.")
    else:
        print(f"[classifier] loaded {args.classifier_ckpt}")

    lpips_fn: LPIPSLoss | None = None
    task_rows: list[dict] = []
    recon_rows: list[dict] = []

    for model_name, ckpt_path in model_specs.items():
        print("\n" + "=" * 72)
        print(f"[model] {model_name}: {ckpt_path}")
        print("=" * 72)
        t_model = time.time()
        model, cfg, model_cfg = load_rvq_model(ckpt_path, device)
        # Optional override: cross-dataset evaluation via external yaml.
        # Encoder/quantizer/decoder weights are not retrained — only the data
        # source is swapped. The h0/L0_bow probe is re-fit on the new
        # dataset's train split (see fit_h0_probe call below).
        if args.data_yaml is not None:
            import yaml as _yaml
            with open(args.data_yaml, "r", encoding="utf-8") as f:
                override = _yaml.safe_load(f) or {}
            data_override = override.get("data", {}) or {}
            cfg["data"] = {**cfg["data"], **data_override}
            print(f"[data_yaml] applied {args.data_yaml}: {data_override}")
        data_cfg = AIDConfig(**cfg["data"])
        loaders = build_loaders(data_cfg)
        tr_loader = train_eval_loader(data_cfg)
        if lpips_fn is None:
            lpips_fn = LPIPSLoss(net=cfg["loss"]["lpips_net"]).to(device).eval()

        print("[encode] train split, no augmentation")
        tr_idx, y_tr, _, spatial = encode_indices(model, tr_loader, device, keep_images=False)
        print("[encode] test split")
        te_idx, y_te, x_te, _ = encode_indices(model, loaders["test"], device, keep_images=True)
        assert x_te is not None
        print(f"[encode] train_idx={tuple(tr_idx.shape)} test_idx={tuple(te_idx.shape)}")

        print("[task] fitting h0/L0_bow probe on clean train indices")
        scaler, clf = fit_h0_probe(tr_idx[..., :1], y_tr, model_cfg.codebook_size)

        for channel, snr in condition_grid(task_snrs):
            seed = condition_seed(args.seed, model_name, channel, snr, 1)
            te_k, ber = maybe_corrupt(
                te_idx[..., :1],
                model_cfg.codebook_size,
                channel,
                snr,
                seed,
            )
            acc = score_h0_probe(scaler, clf, te_k, y_te, model_cfg.codebook_size)
            row = {
                "model": model_name,
                "path": "task",
                "metric": "h0_acc",
                "channel": channel,
                "snr": snr,
                "ber": f"{ber:.8f}",
                "k": 1,
                "bits_per_img": 2560,
                "h0_acc": f"{acc:.6f}",
                "num_samples": int(y_te.numel()),
            }
            task_rows.append(row)
            print(f"[task] {model_name} {channel:8s} snr={snr:>3s} acc={acc*100:.2f}%")

        if model_name in recon_model_names:
            print("[recon] running reconstruction path")
            for channel, snr in condition_grid(recon_snrs):
                for k in ks:
                    seed = condition_seed(args.seed, model_name, channel, snr, k)
                    te_k, ber = maybe_corrupt(
                        te_idx[..., :k],
                        model_cfg.codebook_size,
                        channel,
                        snr,
                        seed,
                    )
                    psnr, lpips, cls_acc = reconstruction_metrics(
                        model=model,
                        indices=te_k,
                        x_ref=x_te,
                        labels=y_te,
                        spatial=spatial,
                        lpips_fn=lpips_fn,
                        classifier=classifier,
                        batch_size=args.batch_size,
                        device=device,
                    )
                    row = {
                        "model": model_name,
                        "path": "reconstruction",
                        "channel": channel,
                        "snr": snr,
                        "ber": f"{ber:.8f}",
                        "k": k,
                        "bits_per_img": 2560 * k,
                        "psnr": f"{psnr:.6f}",
                        "lpips": f"{lpips:.6f}",
                        "recon_cls_acc": "" if cls_acc is None else f"{cls_acc:.6f}",
                        "num_samples": int(y_te.numel()),
                    }
                    recon_rows.append(row)
                    cls_msg = "NA" if cls_acc is None else f"{cls_acc*100:.2f}%"
                    print(
                        f"[recon] {model_name} {channel:8s} snr={snr:>3s} "
                        f"k={k} psnr={psnr:.2f} lpips={lpips:.4f} cls={cls_msg}"
                    )

        print(f"[model] done in {(time.time() - t_model) / 60:.1f} min")

    write_csv(project_path(args.task_out), task_rows)
    if recon_rows:
        write_csv(project_path(args.recon_out), recon_rows)
        print(f"\nWrote task table: {project_path(args.task_out)}")
        print(f"Wrote reconstruction table: {project_path(args.recon_out)}")
    else:
        print(f"\nWrote task table: {project_path(args.task_out)}")
        print(f"[recon] no reconstruction rows (no models in --recon_models matched); skipping {args.recon_out}")
    print("Reminder: task path is k=1 h0/L0_bow only; reconstruction path is k=1..4.")


if __name__ == "__main__":
    main()
