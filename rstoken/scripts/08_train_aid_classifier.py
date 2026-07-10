"""E16 AID classifier evaluator using ImageNet-pretrained ResNet.

This is an external low-bitrate baseline evaluator, not the main RS-Token
channel-reconstruction metric. It fine-tunes a ResNet on the AID train split,
selects the best checkpoint by val accuracy, then evaluates test once.

Smoke:
    python -X utf8 scripts/08_train_aid_classifier.py \
        --config configs/aid_classifier_resnet34.yaml --smoke --no_pretrained

Formal E16:
    python -X utf8 scripts/08_train_aid_classifier.py \
        --config configs/aid_classifier_resnet34.yaml
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDDataset  # noqa: E402

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

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LimitedLoader:
    """Yield only the first max_batches from an existing loader."""

    def __init__(self, loader: DataLoader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for idx, batch in enumerate(self.loader):
            if idx >= self.max_batches:
                break
            yield batch

    def __len__(self) -> int:
        return min(len(self.loader), self.max_batches)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_imagenet_transforms(image_size: int, train: bool):
    """Classifier transforms with ImageNet normalization."""
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.95, 1.05)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(image_size + 32),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_loaders(cfg: dict, smoke: bool) -> dict[str, DataLoader]:
    data_cfg = dict(cfg["data"])
    splits_dir = resolve_project_path(data_cfg["splits_dir"])
    image_size = int(data_cfg.get("image_size", 224))
    num_workers = int(data_cfg.get("num_workers", 4))
    if smoke:
        num_workers = 0

    loaders = {}
    for split, shuffle, train_aug in [
        ("train", True, True),
        ("val", False, False),
        ("test", False, False),
    ]:
        dataset = AIDDataset(
            splits_dir / f"{split}.csv",
            transform=build_imagenet_transforms(image_size, train=train_aug),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(data_cfg.get("pin_memory", True)),
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

    if smoke:
        loaders["train"] = LimitedLoader(loaders["train"], 2)
        loaders["val"] = LimitedLoader(loaders["val"], 1)
        loaders["test"] = LimitedLoader(loaders["test"], 1)

    return loaders


def build_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    backbone = backbone.lower()
    registry = {
        "resnet34": (
            models.resnet34,
            getattr(models, "ResNet34_Weights", None),
        ),
        "resnet50": (
            models.resnet50,
            getattr(models, "ResNet50_Weights", None),
        ),
    }
    if backbone not in registry:
        raise ValueError(f"Unsupported backbone: {backbone}. Use resnet34 or resnet50.")

    builder, weights_enum = registry[backbone]
    if pretrained and weights_enum is not None:
        model = builder(weights=weights_enum.IMAGENET1K_V1)
    else:
        try:
            model = builder(weights=None)
        except TypeError:
            model = builder(pretrained=bool(pretrained))

    if pretrained and weights_enum is None:
        try:
            model = builder(pretrained=True)
        except TypeError as exc:
            raise RuntimeError(
                "This torchvision build does not expose pretrained ResNet weights."
            ) from exc

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    name = str(optim_cfg.get("name", "adamw")).lower()
    lr = float(optim_cfg["lr"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(optim_cfg.get("betas", [0.9, 0.999])),
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optim_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(optim_cfg.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, epochs: int):
    scheduler_cfg = cfg.get("scheduler", {"name": "none"})
    name = str(scheduler_cfg.get("name", "none")).lower()
    if name in {"none", "off"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def amp_settings(cfg: dict, device: torch.device) -> tuple[bool, torch.dtype]:
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    dtype_name = str(cfg.get("amp_dtype", "bf16")).lower()
    if dtype_name == "fp16":
        return use_amp, torch.float16
    return use_amp, torch.bfloat16


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return int(correct), int(targets.numel())


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler,
    grad_clip: float | None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    t0 = time.time()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = labels.numel()
        correct, count = accuracy(logits.detach(), labels)
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += correct
        total_count += count

        if batch_idx == len(loader):
            imgs_per_sec = total_count / max(time.time() - t0, 1e-6)
            print(
                f"  train batches={batch_idx} "
                f"loss={total_loss / total_count:.4f} "
                f"acc={total_correct / total_count:.4f} "
                f"({imgs_per_sec:.1f} img/s)"
            )

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    collect_predictions: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_count += int(batch_size)

        if collect_predictions:
            all_targets.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    loss_avg = total_loss / max(total_count, 1)
    acc_avg = total_correct / max(total_count, 1)
    if collect_predictions:
        return loss_avg, acc_avg, all_targets, all_preds
    return loss_avg, acc_avg


def load_class_names(splits_dir: Path, num_classes: int) -> list[str]:
    classes_path = splits_dir / "classes.txt"
    if not classes_path.exists():
        return [f"class_{idx}" for idx in range(num_classes)]
    with open(classes_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) != num_classes:
        return [f"class_{idx}" for idx in range(num_classes)]
    return names


def classification_metrics(
    targets: list[int],
    preds: list[int],
    num_classes: int,
    class_names: list[str],
) -> dict:
    total = len(targets)
    correct_total = sum(int(y == p) for y, p in zip(targets, preds))

    per_class_acc: dict[str, float | None] = {}
    f1_scores: list[float] = []
    observed_accs: list[float] = []

    for cls_idx in range(num_classes):
        support = sum(int(y == cls_idx) for y in targets)
        cls_correct = sum(
            int(y == cls_idx and p == cls_idx) for y, p in zip(targets, preds)
        )
        acc = None if support == 0 else cls_correct / support
        per_class_acc[class_names[cls_idx]] = acc
        if acc is not None:
            observed_accs.append(acc)

        tp = cls_correct
        fp = sum(int(y != cls_idx and p == cls_idx) for y, p in zip(targets, preds))
        fn = sum(int(y == cls_idx and p != cls_idx) for y, p in zip(targets, preds))
        denom = 2 * tp + fp + fn
        f1_scores.append(0.0 if denom == 0 else (2 * tp) / denom)

    return {
        "test_acc": correct_total / max(total, 1),
        "macro_f1": sum(f1_scores) / max(num_classes, 1),
        "worst_class_acc": min(observed_accs) if observed_accs else None,
        "per_class_acc": per_class_acc,
        "num_test_samples": total,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: dict,
    epoch: int,
    best_val_acc: float,
    metrics: dict,
) -> None:
    obj = {
        "model": model.state_dict(),
        "config": cfg,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "metrics": metrics,
    }
    torch.save(obj, path)


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Disable ImageNet weights, avoiding any weight download.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 2 train batches, 1 val batch, and 1 test batch.",
    )
    parser.add_argument(
        "--run_suffix",
        default=None,
        help=(
            "Append a suffix subdirectory under ckpt_dir/log_dir. "
            "Smoke runs use 'smoke' by default."
        ),
    )
    parser.add_argument("--device", default=None, help="Override config device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False

    set_seed(int(cfg.get("seed", 42)))

    device_name = args.device or cfg.get("device", "cuda")
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    cfg["device"] = str(device)

    total_epochs = 1 if args.smoke else int(cfg["optim"]["total_epochs"])
    ckpt_dir = resolve_project_path(cfg["logging"]["ckpt_dir"])
    log_dir = resolve_project_path(cfg["logging"]["log_dir"])
    run_suffix = args.run_suffix
    if run_suffix is None and args.smoke:
        run_suffix = "smoke"
    if run_suffix:
        ckpt_dir = ckpt_dir / run_suffix
        log_dir = log_dir / run_suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("E16 AID classifier evaluator")
    print(f"Run: {cfg['run_name']}")
    print(f"Config: {args.config}")
    print(f"Backbone: {cfg['model']['backbone']}")
    print(f"Pretrained: {cfg['model']['pretrained']}")
    print(f"Smoke: {args.smoke}")
    print(f"Run suffix: {run_suffix or '<none>'}")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print(f"Log dir: {log_dir}")
    print("=" * 60)

    loaders = build_loaders(cfg, smoke=args.smoke)
    splits_dir = resolve_project_path(cfg["data"]["splits_dir"])
    class_names = load_class_names(splits_dir, int(cfg["model"]["num_classes"]))
    print(
        f"  batches: train={len(loaders['train'])} "
        f"val={len(loaders['val'])} test={len(loaders['test'])}"
    )

    model = build_model(
        backbone=str(cfg["model"]["backbone"]),
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=bool(cfg["model"].get("pretrained", True)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_epochs)
    use_amp, amp_dtype = amp_settings(cfg, device)
    scaler = make_grad_scaler(enabled=use_amp and amp_dtype == torch.float16)
    grad_clip = cfg["optim"].get("grad_clip", None)
    grad_clip = None if grad_clip is None else float(grad_clip)

    metrics_path = log_dir / "metrics.csv"
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    test_metrics_path = log_dir / "test_metrics.json"
    best_val_acc = -1.0

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(total_epochs):
            print(f"\n[epoch {epoch + 1}/{total_epochs}]")
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scaler=scaler,
                grad_clip=grad_clip,
            )
            val_loss, val_acc = evaluate(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                collect_predictions=False,
            )
            row = {
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
            }
            writer.writerow(row)
            f.flush()
            print(
                f"  val loss={val_loss:.4f} acc={val_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(
                    best_path,
                    model,
                    cfg,
                    epoch,
                    best_val_acc,
                    metrics,
                )
                print(f"  new best val_acc={best_val_acc:.4f} -> {best_path}")

            save_checkpoint(
                last_path,
                model,
                cfg,
                epoch,
                best_val_acc,
                metrics,
            )
            if scheduler is not None:
                scheduler.step()

    print("\n[test] loading best checkpoint and evaluating once")
    best_ckpt = torch_load(best_path, device)
    model.load_state_dict(best_ckpt["model"])
    test_loss, _, targets, preds = evaluate(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        collect_predictions=True,
    )
    test_metrics = classification_metrics(
        targets=targets,
        preds=preds,
        num_classes=int(cfg["model"]["num_classes"]),
        class_names=class_names,
    )
    test_metrics["test_loss"] = test_loss
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"  test_acc={test_metrics['test_acc']:.4f} "
        f"macro_f1={test_metrics['macro_f1']:.4f} "
        f"worst_class_acc={test_metrics['worst_class_acc']}"
    )
    print(f"  metrics: {metrics_path}")
    print(f"  test metrics: {test_metrics_path}")
    print(f"  checkpoints: {best_path}, {last_path}")
    if args.smoke:
        print("\n[smoke OK]")


if __name__ == "__main__":
    main()
