from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import math
import sys
import time
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import models
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.perceptual import LPIPSLoss
from models.vqvae import VQVAE, VQVAEConfig

from .policy import PolicyDecision, recommend_k


CLASS_ZH = {
    "Airport": "机场", "BareLand": "裸地", "BaseballField": "棒球场", "Beach": "海滩",
    "Bridge": "桥梁", "Center": "中心区", "Church": "教堂", "Commercial": "商业区",
    "DenseResidential": "密集住宅区", "Desert": "沙漠", "Farmland": "农田", "Forest": "森林",
    "Industrial": "工业区", "Meadow": "草地", "MediumResidential": "中密度住宅区",
    "Mountain": "山地", "Park": "公园", "Parking": "停车场", "Playground": "运动场",
    "Pond": "池塘", "Port": "港口", "RailwayStation": "火车站", "Resort": "度假区",
    "River": "河流", "School": "学校", "SparseResidential": "稀疏住宅区", "Square": "广场",
    "Stadium": "体育场", "StorageTanks": "储罐区", "Viaduct": "高架桥",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load_ldpc_module():
    path = PROJECT_ROOT / "scripts" / "13_eval_ldpc_protected.py"
    spec = importlib.util.spec_from_file_location("rstoken_demo_ldpc", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load LDPC implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _image_data_url(image: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = ((x.detach().cpu().clamp(-1, 1) + 1.0) * 0.5).squeeze(0)
    return TF.to_pil_image(x)


def _build_classifier(ckpt: dict, device: torch.device) -> nn.Module:
    model_cfg = ckpt.get("config", {}).get("model", {})
    backbone = str(model_cfg.get("backbone", "resnet34")).lower()
    if backbone != "resnet34":
        raise ValueError(f"unsupported demo classifier: {backbone}")
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, int(model_cfg.get("num_classes", 30)))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


class DemoEngine:
    def __init__(self, device: str | None = None):
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested)
        self.lock = Lock()
        self.ldpc = _load_ldpc_module()
        self.ldpc_cache: dict[int, object] = {}

        model_path = PROJECT_ROOT / "checkpoints" / "rvq_distill" / "best.pt"
        ckpt = _torch_load(model_path, self.device)
        self.config = ckpt["config"]
        self.model_cfg = VQVAEConfig(**self.config["model"])
        self.model = VQVAE(self.model_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        cls_path = PROJECT_ROOT / "checkpoints" / "aid_classifier_resnet34" / "best.pt"
        self.classifier = _build_classifier(_torch_load(cls_path, self.device), self.device)

        probe_path = DEMO_ROOT / "artifacts" / "h0_probe.joblib"
        if not probe_path.exists():
            raise FileNotFoundError(
                f"missing {probe_path}; run scripts/17_export_demo_probe.py before starting the demo"
            )
        probe = joblib.load(probe_path)
        self.scaler = probe["scaler"]
        self.probe = probe["classifier"]
        self.class_names = list(probe["class_names"])
        self.probe_meta = dict(probe.get("metadata", {}))

        try:
            self.lpips = LPIPSLoss(net="alex").to(self.device).eval()
        except Exception:
            self.lpips = None

        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict]:
        csv_path = PROJECT_ROOT / "data" / "AID_splits_local" / "test.csv"
        selected_names = [
            "Airport", "Bridge", "DenseResidential", "Farmland", "Forest",
            "Industrial", "Port", "River", "StorageTanks", "Viaduct",
        ]
        found: dict[str, dict] = {}
        with open(csv_path, "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                name = row["class_name"]
                if name in selected_names and name not in found:
                    path = Path(row["path"])
                    if path.exists():
                        found[name] = {
                            "id": name.lower(),
                            "name": name,
                            "name_zh": CLASS_ZH.get(name, name),
                            "class_id": int(row["class_id"]),
                            "path": str(path),
                        }
        return [found[name] for name in selected_names if name in found]

    def sample_manifest(self) -> list[dict]:
        return [{k: v for k, v in sample.items() if k != "path"} for sample in self.samples]

    def sample_image(self, sample_id: str) -> tuple[bytes, str]:
        for sample in self.samples:
            if sample["id"] == sample_id:
                return Path(sample["path"]).read_bytes(), Path(sample["path"]).name
        raise KeyError(sample_id)

    def health(self) -> dict:
        return {
            "status": "ready",
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "model": "RS-Token / RVQ-4 / RemoteCLIP L0 distillation",
            "classes": len(self.class_names),
            "probe_test_accuracy": self.probe_meta.get("test_accuracy"),
            "lpips_available": self.lpips is not None,
        }

    @staticmethod
    def _prepare_image(image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        width, height = image.size
        scale = 288 / min(width, height)
        resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
        left = (resized.width - 256) // 2
        top = (resized.height - 256) // 2
        cropped = resized.crop((left, top, left + 256, top + 256))
        tensor = TF.to_tensor(cropped)
        return ((tensor - 0.5) / 0.5).unsqueeze(0)

    @staticmethod
    def _classifier_input(x: torch.Tensor) -> torch.Tensor:
        x01 = (x.clamp(-1, 1) + 1.0) * 0.5
        x01 = TF.center_crop(x01, [224, 224])
        return (x01 - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)

    def _decode(self, indices: torch.Tensor) -> torch.Tensor:
        zq_seq = self.model.quantizer.get_output_from_indices(indices.to(self.device))
        zq = rearrange(zq_seq, "b (h w) c -> b c h w", h=16, w=16)
        return self.model.decoder(zq)

    def _probe_predictions(self, indices: torch.Tensor) -> list[dict]:
        l0 = indices[..., 0].cpu()
        bow = torch.zeros(1, self.model_cfg.codebook_size, dtype=torch.float32)
        bow.scatter_add_(1, l0, torch.ones_like(l0, dtype=torch.float32))
        bow /= l0.shape[1]
        probabilities = self.probe.predict_proba(self.scaler.transform(bow.numpy()))[0]
        order = np.argsort(probabilities)[::-1][:3]
        return [
            {
                "class_id": int(idx),
                "name": self.class_names[int(idx)],
                "name_zh": CLASS_ZH.get(self.class_names[int(idx)], self.class_names[int(idx)]),
                "score": float(probabilities[int(idx)]),
            }
            for idx in order
        ]

    def _recon_predictions(self, recon: torch.Tensor) -> list[dict]:
        logits = self.classifier(self._classifier_input(recon))
        probabilities = logits.softmax(dim=1)[0]
        values, indices = probabilities.topk(3)
        return [
            {
                "class_id": int(idx),
                "name": self.class_names[int(idx)],
                "name_zh": CLASS_ZH.get(self.class_names[int(idx)], self.class_names[int(idx)]),
                "score": float(value),
            }
            for value, idx in zip(values.cpu(), indices.cpu())
        ]

    def _transmit(self, clean_indices: torch.Tensor, decision: PolicyDecision, channel: str, snr_db: float, protection: str, seed: int):
        k = decision.k
        source_bits = self.ldpc.indices_to_bits(clean_indices, k)
        raw_ber = 0.0 if channel == "none" else self.ldpc.ber_from_snr(float(snr_db), channel)
        if protection == "ldpc":
            if k not in self.ldpc_cache:
                code = self.ldpc.SystematicLDPC(source_bits=source_bits.size, rate_num=1, rate_den=2, seed=2026)
                self.ldpc_cache[k] = self.ldpc.TorchLDPCBP(code, str(self.device))
            decoded, meta = self.ldpc_cache[k].transmit_decode(
                source_bits[None, :], channel, "inf" if channel == "none" else str(snr_db), seed, 30
            )
            received_bits = decoded[0]
            actual_raw_ber = float(meta["raw_ber"])
            post_ber = float(meta["post_ldpc_ber"][0])
            ldpc_success = bool(meta["ldpc_success"][0])
            ldpc_iterations = int(meta["ldpc_iterations"][0])
        else:
            rng = np.random.default_rng(seed)
            flips = (rng.random(source_bits.shape) < raw_ber).astype(np.uint8)
            received_bits = source_bits ^ flips
            actual_raw_ber = float(flips.mean()) if flips.size else 0.0
            post_ber = actual_raw_ber
            ldpc_success = None
            ldpc_iterations = None

        received = self.ldpc.bits_to_indices(received_bits, tuple(clean_indices.shape), k)
        clean_k = clean_indices[..., :k].cpu()
        bit_errors = (source_bits != received_bits).reshape(256, k, 10).sum(axis=(1, 2))
        index_errors = (clean_k != received).sum(dim=2).squeeze(0).numpy()
        return received, {
            "theoretical_ber": raw_ber,
            "raw_ber": actual_raw_ber,
            "post_ber": post_ber,
            "bit_errors": int(np.sum(source_bits != received_bits)),
            "index_errors": int((clean_k != received).sum().item()),
            "token_error_rate": float(np.mean(index_errors > 0)),
            "error_grid": bit_errors.astype(int).tolist(),
            "ldpc_success": ldpc_success,
            "ldpc_iterations": ldpc_iterations,
        }

    @torch.inference_mode()
    def infer(
        self,
        image_bytes: bytes,
        *,
        filename: str,
        channel: str,
        snr_db: float,
        protection: str,
        priority: str,
        max_transmitted_bits: int,
        auto_k: bool,
        manual_k: int,
        seed: int,
        previous_k: int | None = None,
    ) -> dict:
        with self.lock:
            total_start = time.perf_counter()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = image.size
            x = self._prepare_image(image).to(self.device)

            decision = recommend_k(
                channel=channel,
                snr_db=snr_db,
                protection=protection,
                priority=priority,
                max_transmitted_bits=max_transmitted_bits,
                manual_k=None if auto_k else manual_k,
                previous_k=previous_k,
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            encode_start = time.perf_counter()
            z = self.model.encoder(x)
            z_seq = rearrange(z, "b c h w -> b (h w) c")
            _, clean_indices, _ = self.model.quantizer(z_seq)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            encode_ms = (time.perf_counter() - encode_start) * 1000

            channel_start = time.perf_counter()
            received, channel_meta = self._transmit(
                clean_indices, decision, channel, snr_db, protection, seed
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            channel_ms = (time.perf_counter() - channel_start) * 1000

            decode_start = time.perf_counter()
            progressive = []
            final_recon = None
            for layer_count in range(1, decision.k + 1):
                recon = self._decode(received[..., :layer_count])
                final_recon = recon
                progressive.append({
                    "k": layer_count,
                    "bits": 2560 * layer_count,
                    "image": _image_data_url(_tensor_to_pil(recon)),
                })
            assert final_recon is not None
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - decode_start) * 1000

            mse = F.mse_loss(final_recon, x).item()
            psnr = 10 * math.log10(4.0 / max(mse, 1e-12))
            lpips = float(self.lpips(final_recon, x).item()) if self.lpips is not None else None
            task_predictions = self._probe_predictions(received)
            recon_predictions = self._recon_predictions(final_recon)
            total_ms = (time.perf_counter() - total_start) * 1000

            input_preview = _tensor_to_pil(x)
            return {
                "request": {
                    "filename": filename,
                    "original_width": original_size[0],
                    "original_height": original_size[1],
                    "channel": channel,
                    "snr_db": snr_db,
                    "protection": protection,
                    "priority": priority,
                    "auto_k": auto_k,
                    "seed": seed,
                },
                "decision": decision.to_dict(),
                "channel": channel_meta,
                "metrics": {
                    "psnr_db": psnr,
                    "lpips": lpips,
                    "bandwidth_saving_pct": 100.0 * (1.0 - decision.k / 4.0),
                    "encode_ms": encode_ms,
                    "channel_ms": channel_ms,
                    "decode_ms": decode_ms,
                    "total_ms": total_ms,
                },
                "task_predictions": task_predictions,
                "recon_predictions": recon_predictions,
                "images": {
                    "input": _image_data_url(input_preview),
                    "reconstruction": progressive[-1]["image"],
                    "progressive": progressive,
                },
                "model": {
                    "name": "RS-Token",
                    "token_grid": "16 x 16",
                    "codebook_size": self.model_cfg.codebook_size,
                    "max_layers": self.model_cfg.rvq_num_quantizers,
                },
            }

    @staticmethod
    def content_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:12]
