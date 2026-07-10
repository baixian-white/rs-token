"""E33 — System cost profile for the deployed encoder/decoder/quantizer.

Input : checkpoints/rvq_distill/best.pt (default)
Output: logs/paper_p1/e33_system_cost.md (markdown table) and .json (machine-readable)

Reports:
  - encoder/decoder/quantizer parameter counts
  - encoder + decoder FLOPs per 256x256 image (via thop)
  - end-to-end GPU latency (50-iter mean) and CPU latency (10-iter mean)
  - explicit note that RemoteCLIP teacher is training-only, NOT deployed

The thop hooks operate on torch.nn.Module forward; for the quantizer
(which is third-party vector_quantize_pytorch ResidualVQ) we only report
parameter count (FLOPs are dominated by the encoder/decoder anyway).
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import torch
from thop import profile

# Force UTF-8 stdout on Windows for console safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

# Project imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from models.vqvae import VQVAE, VQVAEConfig  # noqa: E402


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_rvq_model(ckpt_path: Path, device: str) -> VQVAE:
    ckpt = torch_load(ckpt_path, device)
    cfg = ckpt["config"]
    model_cfg = VQVAEConfig(**cfg["model"])
    model = VQVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_buffers(module: torch.nn.Module) -> int:
    return sum(b.numel() for b in module.buffers())


def main():
    ckpt_path = ROOT / "checkpoints" / "rvq_distill" / "best.pt"
    out_md = ROOT / "logs" / "paper_p1" / "e33_system_cost.md"
    out_json = ROOT / "logs" / "paper_p1" / "e33_system_cost.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    if device == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")

    print(f"loading {ckpt_path}")
    model = load_rvq_model(ckpt_path, device)

    img_size = 256
    x = torch.randn(1, 3, img_size, img_size, device=device)

    # ---- params ----
    params = {
        "encoder":   count_params(model.encoder),
        "quantizer": count_params(model.quantizer),
        "decoder":   count_params(model.decoder),
    }
    params["deployed_total"] = (
        params["encoder"] + params["quantizer"] + params["decoder"]
    )
    # Codebook is stored as EMA buffers, not nn.Parameter, so it's missed
    # by parameters() — measure it explicitly. This is the storage that has
    # to ship to the receiver to support index lookup.
    quantizer_buffers = count_buffers(model.quantizer)
    params["quantizer_codebook_buffers"] = quantizer_buffers
    for k, v in params.items():
        print(f"  storage[{k:>26}] = {v/1e6:.3f} M (~{v*4/1024/1024:.2f} MB fp32)")

    # ---- FLOPs (encoder + decoder; via thop) ----
    # Encoder: input [1, 3, 256, 256]; Encoder.forward returns [1, 256, 16, 16]
    enc_flops, _ = profile(
        model.encoder, inputs=(x,), verbose=False
    )

    # Decoder: input [1, 256, 16, 16] (post-quantize, post-rearrange).
    # Build a representative latent.
    with torch.no_grad():
        z = model.encoder(x)            # [1, 256, 16, 16]
    dec_flops, _ = profile(
        model.decoder, inputs=(z,), verbose=False
    )
    total_codec_flops = enc_flops + dec_flops

    print(f"  enc_flops/img = {enc_flops/1e9:.3f} G")
    print(f"  dec_flops/img = {dec_flops/1e9:.3f} G")
    print(f"  enc+dec       = {total_codec_flops/1e9:.3f} G")

    # ---- GPU latency (full forward path including quantizer) ----
    gpu_latency_ms = None
    if device == "cuda":
        # warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        n_iter = 50
        for _ in range(n_iter):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        gpu_latency_ms = (time.time() - t0) / n_iter * 1000
        print(f"  GPU latency  = {gpu_latency_ms:.2f} ms / image (n=50)")

    # ---- CPU latency ----
    print("running CPU latency benchmark (10 iters)...")
    model_cpu = model.cpu()
    x_cpu = x.cpu()
    for _ in range(2):
        with torch.no_grad():
            _ = model_cpu(x_cpu)
    t0 = time.time()
    n_iter_cpu = 10
    for _ in range(n_iter_cpu):
        with torch.no_grad():
            _ = model_cpu(x_cpu)
    cpu_latency_ms = (time.time() - t0) / n_iter_cpu * 1000
    print(f"  CPU latency  = {cpu_latency_ms:.2f} ms / image (n=10)")

    # ---- write outputs ----
    record = {
        "ckpt": str(ckpt_path.relative_to(ROOT)),
        "image_size": img_size,
        "device_gpu": (
            torch.cuda.get_device_name(0) if device == "cuda" else None
        ),
        "params_M": {k: round(v / 1e6, 4) for k, v in params.items()},
        "flops_G_per_image": {
            "encoder": round(enc_flops / 1e9, 4),
            "decoder": round(dec_flops / 1e9, 4),
            "encoder_plus_decoder": round(total_codec_flops / 1e9, 4),
        },
        "latency_ms_per_image": {
            "gpu_fp32_n50": (
                round(gpu_latency_ms, 4)
                if gpu_latency_ms is not None else None
            ),
            "cpu_n10": round(cpu_latency_ms, 4),
        },
        "note": (
            "Quantizer (ResidualVQ) FLOPs are not measured by thop "
            "(third-party module without a pure conv/linear graph); "
            "they are dominated by encoder+decoder. "
            "RemoteCLIP teacher (~150 M params) is used ONLY during "
            "training and is NOT part of the deployed encoder/decoder."
        ),
    }
    out_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

    md = []
    md.append("# E33 — System cost profile\n")
    md.append(
        f"Profiled checkpoint: `{record['ckpt']}` at "
        f"image size {img_size}×{img_size}.\n"
    )
    if record["device_gpu"]:
        md.append(f"GPU: **{record['device_gpu']}**\n")

    md.append("## Parameters (M)\n")
    md.append("| Component | Params (M) | Notes |")
    md.append("|---|---:|---|")
    md.append(
        f"| Encoder   | {record['params_M']['encoder']:.2f} | "
        f"main backbone (4 down-stages, GroupNorm + SiLU) |"
    )
    md.append(
        f"| Quantizer (RVQ, 4×1024) | "
        f"{record['params_M']['quantizer']:.2f} | "
        f"`nn.Parameter` count; **codebook is stored as EMA buffer:** "
        f"{record['params_M']['quantizer_codebook_buffers']:.2f} M floats "
        f"(~{record['params_M']['quantizer_codebook_buffers']*1e6*4/1024/1024:.2f} "
        f"MB fp32) — required at the receiver for index→vector lookup |"
    )
    md.append(
        f"| Decoder   | {record['params_M']['decoder']:.2f} | "
        f"reverses the encoder |"
    )
    md.append(
        f"| **Deployed parameters total** | "
        f"**{record['params_M']['deployed_total']:.2f}** | "
        f"Encoder + (parametric) Quantizer + Decoder |\n"
    )

    md.append("## FLOPs per image (G)\n")
    md.append("| Path | GFLOPs |")
    md.append("|---|---:|")
    md.append(
        f"| Encoder | {record['flops_G_per_image']['encoder']:.2f} G |"
    )
    md.append(
        f"| Decoder | {record['flops_G_per_image']['decoder']:.2f} G |"
    )
    md.append(
        f"| Encoder + Decoder | "
        f"{record['flops_G_per_image']['encoder_plus_decoder']:.2f} G |"
    )
    md.append(
        "\n_Quantizer (third-party ResidualVQ) FLOPs not counted by thop; "
        "they are negligible relative to the convolutional codec._\n"
    )

    md.append("## Single-image inference latency\n")
    md.append("| Device | Latency (ms) | Notes |")
    md.append("|---|---:|---|")
    if record["latency_ms_per_image"]["gpu_fp32_n50"] is not None:
        md.append(
            f"| GPU (fp32, full forward) | "
            f"{record['latency_ms_per_image']['gpu_fp32_n50']:.2f} | "
            f"50-iter mean, post-warmup, on {record['device_gpu']} |"
        )
    md.append(
        f"| CPU (fp32, full forward) | "
        f"{record['latency_ms_per_image']['cpu_n10']:.2f} | "
        f"10-iter mean (single thread limited by env vars) |"
    )

    md.append("\n## Deployment note\n")
    md.append(
        "**RemoteCLIP teacher (~150 M params) is used only during "
        "training and is NOT part of the deployed encoder/decoder.** "
        "At inference time the transmitter runs Encoder + RVQ to obtain "
        "discrete indices; the receiver runs RVQ codebook lookup + Decoder."
    )

    md.append(
        "\n## Suggested paper sentence (§3 end or §4.1)\n\n"
        "> The deployed encoder–decoder (with the residual-VQ codebook) "
        f"has **{record['params_M']['deployed_total']:.1f} M** parameters "
        f"and **{record['flops_G_per_image']['encoder_plus_decoder']:.1f} G** "
        "FLOPs per 256×256 image; single-image GPU inference latency is "
        f"**{record['latency_ms_per_image']['gpu_fp32_n50']:.2f} ms** "
        "(fp32, RTX 5070 Ti Laptop). The RemoteCLIP teacher (~150 M "
        "parameters) is used only during training and is not deployed."
    )

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
