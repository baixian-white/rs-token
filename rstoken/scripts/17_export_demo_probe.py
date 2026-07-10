"""Export the frozen L0 task probe used by the interactive demo.

The paper evaluation fits a fresh logistic-regression probe for each batch
experiment. The demo needs the exact same probe as a versioned, load-once
artifact so that a single image can be classified without touching training
data at request time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDDataset, build_transforms
from models.vqvae import VQVAE, VQVAEConfig


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_loader(csv_path: Path, batch_size: int, num_workers: int) -> DataLoader:
    dataset = AIDDataset(csv_path, transform=build_transforms(256, train=False))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def extract_bow(model: VQVAE, loader: DataLoader, device: torch.device, codebook_size: int):
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_idx, (images, target) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        z = model.encoder(images)
        z_seq = rearrange(z, "b c h w -> b (h w) c")
        _, indices, _ = model.quantizer(z_seq)
        l0 = indices[..., 0].cpu()
        bow = torch.zeros(l0.shape[0], codebook_size, dtype=torch.float32)
        bow.scatter_add_(1, l0, torch.ones_like(l0, dtype=torch.float32))
        bow /= l0.shape[1]
        features.append(bow)
        labels.append(target.cpu())
        if batch_idx % 25 == 0:
            print(f"[encode] {batch_idx * loader.batch_size:5d}/{len(loader.dataset)}")
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/rvq_distill/best.pt")
    parser.add_argument("--splits", default="data/AID_splits_local")
    parser.add_argument("--output", default="demo/artifacts/h0_probe.joblib")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device_name = args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    splits = PROJECT_ROOT / args.splits
    output = PROJECT_ROOT / args.output

    ckpt = torch_load(checkpoint_path, device)
    model_cfg = VQVAEConfig(**ckpt["config"]["model"])
    model = VQVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[probe] device={device} checkpoint={checkpoint_path}")
    train_loader = build_loader(splits / "train.csv", args.batch_size, args.num_workers)
    test_loader = build_loader(splits / "test.csv", args.batch_size, args.num_workers)
    x_train, y_train = extract_bow(model, train_loader, device, model_cfg.codebook_size)
    x_test, y_test = extract_bow(model, test_loader, device, model_cfg.codebook_size)

    scaler = StandardScaler(with_mean=False)
    x_train_scaled = scaler.fit_transform(x_train)
    classifier = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", n_jobs=1)
    classifier.fit(x_train_scaled, y_train)
    accuracy = float(classifier.score(scaler.transform(x_test), y_test))

    class_names = load_classes(splits / "classes.txt")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    artifact = {
        "scaler": scaler,
        "classifier": classifier,
        "class_names": class_names,
        "metadata": {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": digest,
            "test_accuracy": accuracy,
            "train_samples": int(y_train.size),
            "test_samples": int(y_test.size),
            "feature": "L0_bow",
            "codebook_size": model_cfg.codebook_size,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    print(f"[probe] saved={output} test_accuracy={accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

