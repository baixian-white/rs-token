"""Train ADJSCC for E35 — single rate point per run, 50 epochs, 3 seeds.

Usage:
    python scripts/train_adjscc.py --config configs/paper_v05/adjscc_b2560_s42.yaml
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDConfig, build_loaders, denormalize  # noqa: E402
from models.adjscc import ADJSCC, ADJSCCConfig                    # noqa: E402

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def psnr_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Inputs in [-1, 1]. Peak = 2."""
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
    return (10 * torch.log10(4.0 / (mse + 1e-12))).mean()


def sample_train_snrs(batch: int, cfg: ADJSCCConfig, device: str) -> torch.Tensor:
    """Uniform random SNR per sample in [train_snr_min, train_snr_max] dB."""
    return (cfg.train_snr_min + (cfg.train_snr_max - cfg.train_snr_min) *
            torch.rand(batch, device=device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="Run 2 train + 1 val batch only.")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    print("=" * 72)
    print(f"Run: {cfg_dict['run_name']}")
    print(f"Config: {args.config}")
    print(f"Smoke: {args.smoke}")
    print("=" * 72)

    set_seed(int(cfg_dict.get("seed", 42)))
    device = cfg_dict.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    # ---- Data ---------------------------------------------------------------
    data_cfg = AIDConfig(**cfg_dict["data"])
    if args.smoke:
        data_cfg.num_workers = 0
    loaders = build_loaders(data_cfg)

    # ---- Model --------------------------------------------------------------
    model_cfg = ADJSCCConfig(**cfg_dict["model"])
    model = ADJSCC(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_uses = model.n_channel_uses_per_image()
    bits_per_image = n_uses  # 1 real symbol = 1 BPSK bit equivalent
    print(f"  params: {n_params / 1e6:.2f}M")
    print(f"  N (real symbols / image) = {n_uses}  (=> {bits_per_image} bits "
          f"under BPSK match)")

    # ---- Optimiser + scheduler ---------------------------------------------
    optim_cfg = cfg_dict["optim"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        betas=tuple(optim_cfg.get("betas", [0.9, 0.95])),
    )
    total_epochs = int(optim_cfg["total_epochs"])
    warmup_steps = int(optim_cfg.get("warmup_steps", 500))
    base_lr = float(optim_cfg["lr"])
    grad_clip = float(optim_cfg.get("grad_clip", 0.0))

    # ---- AMP ----------------------------------------------------------------
    use_amp = bool(cfg_dict.get("amp", True))
    amp_dtype_name = cfg_dict.get("amp_dtype", "bf16").lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16

    # ---- Logging ------------------------------------------------------------
    log_cfg = cfg_dict["logging"]
    ckpt_dir = Path(log_cfg["ckpt_dir"])
    log_dir = Path(log_cfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cfg_dict['run_name']}.log"
    log_file = log_path.open("a", encoding="utf-8")

    def log_print(s: str):
        print(s, flush=True)
        log_file.write(s + "\n")
        log_file.flush()

    log_print(f"\n=== Run start at epoch 0 ===")

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    if args.smoke:
        n_train_batches = 2
        n_val_batches = 1
    print(f"  train batches: {n_train_batches}")
    print(f"  val   batches: {n_val_batches}")

    total_steps = n_train_batches * total_epochs
    best_val_psnr = -1.0
    step = 0
    save_every = int(log_cfg.get("save_every_epoch", 5))
    log_every = int(log_cfg.get("log_every", 50))

    for ep in range(total_epochs):
        model.train()
        t_ep = time.time()
        ema_psnr = None
        for bi, (x, _) in enumerate(train_loader):
            if args.smoke and bi >= n_train_batches:
                break
            x = x.to(device, non_blocking=True)
            snr_db = sample_train_snrs(x.shape[0], model_cfg, device)
            lr = cosine_warmup_lr(step, total_steps, warmup_steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                recon = model(x, snr_db, channel=model_cfg.train_channel)
                loss = F.mse_loss(recon, x)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            with torch.no_grad():
                p = float(psnr_batch(recon.float(), x.float()).item())
            ema_psnr = p if ema_psnr is None else 0.9 * ema_psnr + 0.1 * p

            if (bi + 1) % log_every == 0 or bi == 0:
                log_print(
                    f"  ep {ep:3d}  step {step:6d}  lr {lr:.2e}  "
                    f"mse={loss.item():.4f}  psnr={p:.2f}dB"
                )
            step += 1

        # ---- Validation ---------------------------------------------------
        model.eval()
        val_psnr = 0.0
        val_count = 0
        with torch.no_grad():
            for vi, (x, _) in enumerate(val_loader):
                if args.smoke and vi >= n_val_batches:
                    break
                x = x.to(device, non_blocking=True)
                # Validation SNR sweep: average over a small grid for stability.
                psnr_vals = []
                for s in (0, 5, 10):
                    snr_db = torch.full((x.shape[0],), float(s),
                                        device=device, dtype=torch.float32)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        recon = model(x, snr_db, channel=model_cfg.train_channel)
                    psnr_vals.append(float(psnr_batch(recon.float(), x.float()).item()))
                val_psnr += sum(psnr_vals) / len(psnr_vals) * x.shape[0]
                val_count += x.shape[0]
        val_psnr /= max(val_count, 1)
        log_print(f"  val | psnr={val_psnr:.4f}dB  (avg over snr ∈ {{0,5,10}})")

        # ---- Save ckpt ----------------------------------------------------
        ckpt_payload = {
            "model": model.state_dict(),
            "config": cfg_dict,
            "epoch": ep + 1,
            "val_psnr": val_psnr,
            "n_channel_uses": n_uses,
        }
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        if (ep + 1) % save_every == 0:
            torch.save(ckpt_payload, ckpt_dir / f"epoch_{ep + 1:03d}.pt")

        if args.smoke:
            break

    log_file.close()
    print(f"  best val_psnr={best_val_psnr:.4f}dB -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
