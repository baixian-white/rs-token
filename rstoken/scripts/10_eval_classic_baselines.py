"""E19 classic compressed bitstream baselines over an unprotected BPSK channel.

This script evaluates WebP and, if available, JPEG2000 at matched bit budgets.
It does not implement channel coding. Results should be described only as
"unprotected compressed bitstream over BPSK channel".
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.perceptual import LPIPSLoss

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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Sample:
    path: str
    label: int
    image: Image.Image
    tensor_neg11: torch.Tensor


@dataclass
class EncodedSample:
    label: int
    tensor_neg11: torch.Tensor
    payload: bytes
    actual_bits: int


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


def read_split(csv_path: Path, image_size: int, max_samples: int | None) -> list[Sample]:
    rows: list[Sample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image = Image.open(row["path"]).convert("RGB")
            image = preprocess_pil_for_codec(image, image_size)
            tensor = pil_to_neg11(image)
            rows.append(
                Sample(
                    path=row["path"],
                    label=int(row["class_id"]),
                    image=image,
                    tensor_neg11=tensor,
                )
            )
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def preprocess_pil_for_codec(image: Image.Image, image_size: int) -> Image.Image:
    image = TF.resize(image, image_size + 32, interpolation=InterpolationMode.BICUBIC)
    image = TF.center_crop(image, [image_size, image_size])
    return image


def pil_to_neg11(image: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    return tensor * 2.0 - 1.0


def ensure_image_size(image: Image.Image, image_size: int) -> Image.Image:
    if image.size == (image_size, image_size):
        return image
    return image.resize((image_size, image_size), Image.Resampling.BICUBIC)


def neg11_to_classifier_input(batch: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    x01 = (batch.clamp(-1, 1) + 1.0) * 0.5
    _, _, h, w = x01.shape
    if h < crop_size or w < crop_size:
        x01 = F.interpolate(x01, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    else:
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        x01 = x01[:, :, top : top + crop_size, left : left + crop_size]
    return (x01 - IMAGENET_MEAN.to(x01.device)) / IMAGENET_STD.to(x01.device)


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


def load_classifier(ckpt_path: Path, device: str) -> tuple[nn.Module | None, str | None]:
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


def save_to_bytes(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def encode_webp(image: Image.Image, target_bits: int) -> bytes:
    best_payload: bytes | None = None
    best_delta: int | None = None
    coarse = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    scored: list[tuple[int, int, bytes]] = []
    for quality in coarse:
        payload = save_to_bytes(image, "WEBP", quality=quality, method=6)
        delta = abs(len(payload) * 8 - target_bits)
        scored.append((delta, quality, payload))
        if best_delta is None or delta < best_delta:
            best_payload = payload
            best_delta = delta
    best_quality = min(scored, key=lambda item: item[0])[1]
    local = range(max(1, best_quality - 3), min(100, best_quality + 3) + 1)
    for quality in local:
        if quality in coarse:
            continue
        payload = save_to_bytes(image, "WEBP", quality=quality, method=6)
        delta = abs(len(payload) * 8 - target_bits)
        if best_delta is None or delta < best_delta:
            best_payload = payload
            best_delta = delta
    assert best_payload is not None
    return best_payload


def jpeg2000_candidates(target_bits: int, image_size: int) -> list[float]:
    uncompressed_bits = image_size * image_size * 3 * 8
    center = max(uncompressed_bits / max(target_bits, 1), 1.0)
    factors = [0.25, 0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0]
    values = sorted({max(center * f, 1.0) for f in factors})
    return values


def encode_jpeg2000_pillow(image: Image.Image, target_bits: int, image_size: int) -> bytes:
    best_payload: bytes | None = None
    best_delta: int | None = None
    last_error: Exception | None = None
    for rate in jpeg2000_candidates(target_bits, image_size):
        try:
            payload = save_to_bytes(
                image,
                "JPEG2000",
                quality_mode="rates",
                quality_layers=[rate],
                irreversible=True,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
        delta = abs(len(payload) * 8 - target_bits)
        if best_delta is None or delta < best_delta:
            best_payload = payload
            best_delta = delta
    if best_payload is None:
        raise RuntimeError(f"Pillow JPEG2000 encode failed: {last_error}")
    return best_payload


def encode_samples(
    method: str,
    samples: list[Sample],
    target_bits: int,
    image_size: int,
) -> tuple[list[EncodedSample], str | None]:
    encoded: list[EncodedSample] = []
    for idx, sample in enumerate(samples, start=1):
        try:
            if method == "webp":
                payload = encode_webp(sample.image, target_bits)
            elif method == "jpeg2000":
                payload = encode_jpeg2000_pillow(sample.image, target_bits, image_size)
            else:
                raise ValueError(f"unknown method: {method}")
        except Exception as exc:  # noqa: BLE001
            if method == "jpeg2000":
                return [], f"JPEG2000 skipped: encoder unavailable ({exc})"
            return [], f"{method} encode failed on sample {idx}: {exc}"
        encoded.append(
            EncodedSample(
                label=sample.label,
                tensor_neg11=sample.tensor_neg11,
                payload=payload,
                actual_bits=len(payload) * 8,
            )
        )
    return encoded, None


def corrupt_payload(payload: bytes, ber: float, rng: np.random.Generator) -> bytes:
    if ber <= 0:
        return payload
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    flips = rng.random((arr.size, 8)) < ber
    powers = (1 << np.arange(8, dtype=np.uint8)).reshape(1, 8)
    masks = (flips.astype(np.uint8) * powers).sum(axis=1).astype(np.uint8)
    arr ^= masks
    return arr.tobytes()


def decode_payload(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        return image.convert("RGB").copy()


def evaluate_decoded(
    decoded: list[torch.Tensor],
    labels: list[int],
    originals: list[torch.Tensor],
    classifier: nn.Module | None,
    lpips_fn: LPIPSLoss,
    batch_size: int,
    device: str,
) -> tuple[int | None, float, float]:
    if not decoded:
        return None, float("nan"), float("nan")

    correct = 0 if classifier is not None else None
    psnr_sum = 0.0
    lpips_sum = 0.0
    n_total = len(decoded)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        rec = torch.stack(decoded[start:end]).to(device)
        ref = torch.stack(originals[start:end]).to(device)
        mse = F.mse_loss(rec, ref, reduction="none").flatten(1).mean(dim=1)
        psnr_sum += float((10 * torch.log10(4.0 / (mse + 1e-12))).sum().item())
        lpips_sum += float(lpips_fn(rec, ref).item()) * (end - start)

        if classifier is not None:
            logits = classifier(neg11_to_classifier_input(rec))
            pred = logits.argmax(dim=1).cpu().tolist()
            correct += sum(int(p == y) for p, y in zip(pred, labels[start:end]))

    return correct, psnr_sum / n_total, lpips_sum / n_total


def evaluate_condition(
    method: str,
    target_bits: int,
    encoded: list[EncodedSample],
    channel: str,
    snr: str,
    seed: int,
    classifier: nn.Module | None,
    lpips_fn: LPIPSLoss,
    batch_size: int,
    image_size: int,
    device: str,
) -> dict:
    if channel == "none":
        ber = 0.0
    else:
        ber = ber_from_snr(float(snr), channel)

    rng = np.random.default_rng(seed)
    decoded: list[torch.Tensor] = []
    labels: list[int] = []
    originals: list[torch.Tensor] = []
    failures = 0

    for sample in encoded:
        payload = corrupt_payload(sample.payload, ber, rng)
        try:
            image = decode_payload(payload)
            image = ensure_image_size(image, image_size)
            tensor = pil_to_neg11(image)
            decoded.append(tensor)
            labels.append(sample.label)
            originals.append(sample.tensor_neg11)
        except Exception:  # noqa: BLE001
            failures += 1

    correct_decoded, psnr_valid, lpips_valid = evaluate_decoded(
        decoded=decoded,
        labels=labels,
        originals=originals,
        classifier=classifier,
        lpips_fn=lpips_fn,
        batch_size=batch_size,
        device=device,
    )

    n_total = len(encoded)
    n_decoded = len(decoded)
    if classifier is None:
        cls_acc_all = ""
        cls_acc_decoded = ""
    else:
        correct = 0 if correct_decoded is None else correct_decoded
        cls_acc_all = f"{correct / n_total:.6f}"
        cls_acc_decoded = "" if n_decoded == 0 else f"{correct / n_decoded:.6f}"

    actual_bits = [sample.actual_bits for sample in encoded]
    return {
        "method": method,
        "target_bits": target_bits,
        "actual_bits_mean": f"{statistics.mean(actual_bits):.3f}",
        "actual_bits_std": f"{statistics.pstdev(actual_bits):.3f}",
        "channel": channel,
        "snr": snr,
        "ber": f"{ber:.8f}",
        "decode_failure_rate": f"{failures / n_total:.6f}",
        "cls_acc_all": cls_acc_all,
        "cls_acc_decoded": cls_acc_decoded,
        "psnr_valid": "" if math.isnan(psnr_valid) else f"{psnr_valid:.6f}",
        "lpips_valid": "" if math.isnan(lpips_valid) else f"{lpips_valid:.6f}",
        "num_samples": n_total,
    }


def condition_seed(base_seed: int, method: str, target_bits: int, channel: str, snr: str) -> int:
    text = f"{method}|{target_bits}|{channel}|{snr}"
    acc = base_seed & 0x7FFFFFFF
    for ch in text:
        acc = (acc * 131 + ord(ch)) & 0x7FFFFFFF
    return acc


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict], notes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# E19 Classic Baselines Summary\n\n")
        f.write("Scope: unprotected compressed bitstream over BPSK channel. No LDPC is implemented.\n\n")
        if notes:
            f.write("## Notes\n\n")
            for note in notes:
                f.write(f"- {note}\n")
            f.write("\n")
        f.write("## Rows\n\n")
        for row in rows:
            f.write(
                f"- {row['method']} target={row['target_bits']} channel={row['channel']} "
                f"snr={row['snr']} fail={row['decode_failure_rate']} "
                f"cls_all={row['cls_acc_all'] or 'NA'} psnr={row['psnr_valid'] or 'NA'}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", default="data/AID_splits")
    parser.add_argument("--classifier_ckpt", default="checkpoints/aid_classifier_resnet34/best.pt")
    parser.add_argument("--methods", default="webp,jpeg2000")
    parser.add_argument("--budgets", default="2560,5120,10240")
    parser.add_argument("--snrs", default="0,5,10")
    parser.add_argument("--channels", default="awgn,rayleigh")
    parser.add_argument("--out_csv", default="logs/e19_classic_baselines.csv")
    parser.add_argument("--summary_md", default="logs/e19_classic_baselines_summary.md")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device = "cpu"

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    snrs = [s.strip() for s in args.snrs.split(",") if s.strip()]
    channels = [c.strip().lower() for c in args.channels.split(",") if c.strip()]
    conditions = [("none", "inf")] + [(channel, snr) for channel in channels for snr in snrs]

    classifier, classifier_note = load_classifier(project_path(args.classifier_ckpt), device)
    notes: list[str] = []
    if classifier_note:
        notes.append(classifier_note)
        print(f"[classifier] {classifier_note}")
    else:
        print(f"[classifier] loaded {args.classifier_ckpt}")

    print("[data] loading test split")
    samples = read_split(
        project_path(args.splits_dir) / "test.csv",
        image_size=args.image_size,
        max_samples=args.max_samples,
    )
    print(f"[data] loaded {len(samples)} samples at {args.image_size}x{args.image_size}")

    lpips_fn = LPIPSLoss(net="alex").to(device).eval()
    rows: list[dict] = []

    for method in methods:
        for budget in budgets:
            print("\n" + "=" * 72)
            print(f"[encode] method={method} target_bits={budget}")
            print("=" * 72)
            t0 = time.time()
            encoded, note = encode_samples(method, samples, budget, args.image_size)
            if note:
                notes.append(note)
                print(f"[skip] {note}")
                continue
            bits = [item.actual_bits for item in encoded]
            print(
                f"[encode] actual_bits mean={statistics.mean(bits):.1f} "
                f"std={statistics.pstdev(bits):.1f} "
                f"elapsed={(time.time() - t0) / 60:.1f} min"
            )

            for channel, snr in conditions:
                seed = condition_seed(args.seed, method, budget, channel, snr)
                row = evaluate_condition(
                    method=method,
                    target_bits=budget,
                    encoded=encoded,
                    channel=channel,
                    snr=snr,
                    seed=seed,
                    classifier=classifier,
                    lpips_fn=lpips_fn,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    device=device,
                )
                rows.append(row)
                print(
                    f"[eval] {method:8s} target={budget:5d} {channel:8s} snr={snr:>3s} "
                    f"fail={row['decode_failure_rate']} "
                    f"cls_all={row['cls_acc_all'] or 'NA'} psnr={row['psnr_valid'] or 'NA'}"
                )

    if not rows:
        raise RuntimeError("no baseline rows were produced")
    write_csv(project_path(args.out_csv), rows)
    write_summary(project_path(args.summary_md), rows, notes)
    print(f"\nWrote CSV: {project_path(args.out_csv)}")
    print(f"Wrote summary: {project_path(args.summary_md)}")
    print("Do not describe these rows as JPEG2000+LDPC; no LDPC is implemented.")


if __name__ == "__main__":
    main()
