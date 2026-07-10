"""E41 — Train DirectClipQuantizer (RemoteCLIP → RVQ → decoder).

Training signal: L1 + LPIPS + vq_loss only. There's no separate distillation
loss because the input is RemoteCLIP itself — distillation is implicit.

Mirrors `03_train_vqvae.py` (same optim/LR schedule/AMP/log/ckpt pattern)
to keep the comparison apples-to-apples.

Usage:
    python scripts/train_direct_clip.py --config configs/paper_v05/e41_direct_clip_s42.yaml
    python scripts/train_direct_clip.py --config <yaml> --smoke
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDConfig, build_loaders, denormalize  # noqa: E402
from models.perceptual import LPIPSLoss                             # noqa: E402
from models.direct_clip_quantizer import (                          # noqa: E402
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


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
    return (10 * torch.log10(4.0 / (mse + 1e-12))).mean()


def codebook_usage(indices: torch.Tensor, codebook_size: int) -> float:
    if indices.dim() == 2:
        used = torch.unique(indices.flatten()).numel()
        return used / codebook_size
    L = indices.shape[-1]
    return sum(
        torch.unique(indices[..., l].flatten()).numel() / codebook_size
        for l in range(L)
    ) / L


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_val_psnr: float = -1.0


def train_one_epoch(model, loader, optimizer, lpips_fn, cfg, state,
                    total_steps, log_writer, tb_writer=None,
                    log_every_override=None):
    model.train()
    log = cfg["logging"]
    loss_cfg = cfg["loss"]
    optim_cfg = cfg["optim"]
    cb_size = cfg["model"]["codebook_size"]
    log_every = log_every_override if log_every_override is not None else log["log_every"]

    use_amp = cfg.get("amp", False)
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16

    t0 = time.time()
    running = {"l1": 0.0, "lpips": 0.0, "vq": 0.0, "psnr": 0.0, "use": 0.0, "n": 0}

    for x, _ in loader:
        x = x.to(cfg["device"], non_blocking=True)

        lr = cosine_warmup_lr(
            state.step, total_steps, optim_cfg["warmup_steps"], optim_cfg["lr"]
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                l1 = F.l1_loss(out["recon"], x)
                lp = lpips_fn(out["recon"], x) if lpips_fn else torch.tensor(0.0)
                loss = (
                    loss_cfg["l1_weight"] * l1
                    + loss_cfg["lpips_weight"] * lp
                    + loss_cfg["vq_weight"] * out["vq_loss"]
                )
        else:
            out = model(x)
            l1 = F.l1_loss(out["recon"], x)
            lp = lpips_fn(out["recon"], x) if lpips_fn else torch.tensor(0.0)
            loss = (
                loss_cfg["l1_weight"] * l1
                + loss_cfg["lpips_weight"] * lp
                + loss_cfg["vq_weight"] * out["vq_loss"]
            )

        loss.backward()
        if optim_cfg.get("grad_clip"):
            # Only clip TRAINABLE params (teacher is frozen, no grads).
            params_to_clip = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, optim_cfg["grad_clip"])
        optimizer.step()

        with torch.no_grad():
            running["l1"]    += l1.item()
            running["lpips"] += lp.item()
            running["vq"]    += out["vq_loss"].item()
            running["psnr"]  += psnr(out["recon"], x).item()
            running["use"]   += codebook_usage(out["indices"], cb_size)
            running["n"]     += 1

        state.step += 1

        if state.step % log_every == 0:
            n = running["n"]
            msg = (
                f"  ep {state.epoch:>3d}  step {state.step:>6d}  "
                f"lr {lr:.2e}  "
                f"l1={running['l1']/n:.4f}  "
                f"lpips={running['lpips']/n:.4f}  "
                f"vq={running['vq']/n:.4f}  "
                f"psnr={running['psnr']/n:.2f}dB  "
                f"cb_use={running['use']/n*100:.1f}%  "
                f"({n*x.shape[0]/(time.time()-t0):.1f} img/s)"
            )
            print(msg)
            log_writer.write(msg + "\n")
            log_writer.flush()
            if tb_writer is not None:
                tb_writer.add_scalar("train/lr", lr, state.step)
                tb_writer.add_scalar("train/l1", running["l1"]/n, state.step)
                tb_writer.add_scalar("train/lpips", running["lpips"]/n, state.step)
                tb_writer.add_scalar("train/vq", running["vq"]/n, state.step)
                tb_writer.add_scalar("train/psnr", running["psnr"]/n, state.step)
                tb_writer.add_scalar("train/cb_use_batch", running["use"]/n, state.step)
            running = {k: 0.0 for k in running}; running["n"] = 0
            t0 = time.time()


@torch.no_grad()
def validate(model, loader, lpips_fn, cfg, epoch):
    model.eval()
    cb_size = cfg["model"]["codebook_size"]
    sums = {"l1": 0.0, "lpips": 0.0, "psnr": 0.0, "use_per_batch": 0.0}
    n_layers = cfg["model"].get("rvq_num_quantizers", 4)
    layer_sets: list[set[int]] = [set() for _ in range(n_layers)]
    n_batches = 0

    use_amp = cfg.get("amp", False)
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" else torch.float16

    vis_x, vis_recon = None, None

    for i, (x, _) in enumerate(loader):
        x = x.to(cfg["device"], non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
        else:
            out = model(x)

        sums["l1"]    += F.l1_loss(out["recon"], x).item()
        if lpips_fn:
            sums["lpips"] += lpips_fn(out["recon"], x).item()
        sums["psnr"] += psnr(out["recon"], x).item()
        sums["use_per_batch"] += codebook_usage(out["indices"], cb_size)
        idx = out["indices"]
        for l in range(n_layers):
            layer_sets[l].update(idx[..., l].flatten().tolist())
        n_batches += 1

        if i == 0:
            vis_x = x[: cfg["logging"]["num_vis_samples"]].cpu()
            vis_recon = out["recon"][: cfg["logging"]["num_vis_samples"]].cpu()

    metrics = {
        "val/l1":    sums["l1"]    / n_batches,
        "val/lpips": sums["lpips"] / n_batches,
        "val/psnr":  sums["psnr"]  / n_batches,
        "val/cb_use_per_batch": sums["use_per_batch"] / n_batches,
    }
    per_layer_use = [len(s) / cb_size for s in layer_sets]
    for l, u in enumerate(per_layer_use):
        metrics[f"val/cb_use_global_L{l}"] = u
    metrics["val/cb_use_global"] = sum(per_layer_use) / len(per_layer_use)
    return metrics, vis_x, vis_recon


def save_vis(vis_x, vis_recon, out_path, epoch):
    from torchvision.utils import make_grid, save_image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = torch.cat([denormalize(vis_x), denormalize(vis_recon)], dim=0)
    grid = make_grid(pairs, nrow=vis_x.shape[0])
    save_image(grid, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = cfg["device"]
    log_cfg = cfg["logging"]
    ckpt_dir = Path(log_cfg["ckpt_dir"])
    log_dir = Path(log_cfg["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"E41 Run: {cfg['run_name']}")
    print(f"Config:  {args.config}")
    print(f"Smoke:   {args.smoke}")
    print("=" * 60)

    data_cfg = AIDConfig(**cfg["data"])
    loaders = build_loaders(data_cfg)
    print(f"  train batches: {len(loaders['train'])}")
    print(f"  val   batches: {len(loaders['val'])}")

    model_cfg = DirectClipQuantizerConfig(**cfg["model"])
    model = DirectClipQuantizer(model_cfg).to(device)
    np_info = model.num_parameters()
    print(f"  trainable params: {np_info['trainable']/1e6:.2f}M  "
          f"(up_proj {np_info['up_proj']/1e6:.2f}M / "
          f"q {np_info['quantizer']/1e6:.3f}M / "
          f"dec {np_info['decoder']/1e6:.2f}M)")
    print(f"  teacher (frozen): {np_info['teacher_frozen']/1e6:.2f}M")

    lpips_fn = None
    if cfg["loss"]["lpips_weight"] > 0:
        lpips_fn = LPIPSLoss(net=cfg["loss"]["lpips_net"]).to(device)

    # Optimizer over trainable params only (teacher excluded).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    state = TrainState()
    total_epochs = cfg["optim"]["total_epochs"] if not args.smoke else 1
    total_steps = len(loaders["train"]) * total_epochs

    log_path = log_dir / f"{cfg['run_name']}.log"
    log_writer = open(log_path, "a", encoding="utf-8")
    log_writer.write(f"\n=== Run start at epoch {state.epoch} ===\n")
    tb_writer = SummaryWriter(log_dir=str(log_dir / "tb"))

    for epoch in range(total_epochs):
        state.epoch = epoch
        print(f"\n[epoch {epoch+1}/{total_epochs}]")
        if args.smoke:
            iter_loader = iter(loaders["train"])
            mini = [next(iter_loader) for _ in range(2)]
            class _MockLoader:
                def __init__(self, items): self.items = items
                def __iter__(self): return iter(self.items)
                def __len__(self): return len(self.items)
            train_loader = _MockLoader(mini)
        else:
            train_loader = loaders["train"]

        train_one_epoch(
            model, train_loader, optimizer, lpips_fn, cfg, state,
            total_steps, log_writer, tb_writer=tb_writer,
            log_every_override=1 if args.smoke else None,
        )

        if (epoch + 1) % log_cfg["val_every_epoch"] == 0 or args.smoke:
            val_loader = loaders["val"]
            if args.smoke:
                vbatch = next(iter(val_loader))
                class _MockLoader:
                    def __init__(self, items): self.items = items
                    def __iter__(self): return iter(self.items)
                    def __len__(self): return 1
                val_loader = _MockLoader([vbatch])

            metrics, vis_x, vis_recon = validate(
                model, val_loader, lpips_fn, cfg, epoch
            )
            line = "  val | " + "  ".join(
                f"{k.split('/',1)[1]}={v:.4f}" for k, v in metrics.items()
            )
            print(line)
            log_writer.write(line + "\n")
            log_writer.flush()
            for k, v in metrics.items():
                tb_writer.add_scalar(k, v, state.step)
            tb_writer.add_scalar("val/epoch", epoch + 1, state.step)
            if vis_x is not None:
                save_vis(
                    vis_x, vis_recon,
                    log_dir / f"vis_epoch_{epoch+1:03d}.png", epoch
                )

            if metrics["val/psnr"] > state.best_val_psnr:
                state.best_val_psnr = metrics["val/psnr"]
                # Save trainable parts only — re-loading is by reconstructing
                # the teacher anew (which is deterministic given ckpt_path).
                ckpt_obj = {
                    "model_trainable": {
                        k: v for k, v in model.state_dict().items()
                        if not k.startswith("teacher.")
                    },
                    "config": cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                }
                torch.save(ckpt_obj, ckpt_dir / "best.pt")
                print(f"  ★ new best val PSNR: {metrics['val/psnr']:.2f} dB "
                      f"-> saved best.pt")

        if (epoch + 1) % log_cfg["save_every_epoch"] == 0:
            torch.save({
                "model_trainable": {
                    k: v for k, v in model.state_dict().items()
                    if not k.startswith("teacher.")
                },
                "config": cfg,
                "epoch": epoch,
            }, ckpt_dir / f"epoch_{epoch+1:03d}.pt")

        torch.save({
            "model_trainable": {
                k: v for k, v in model.state_dict().items()
                if not k.startswith("teacher.")
            },
            "config": cfg,
            "epoch": epoch,
            "best_val_psnr": state.best_val_psnr,
        }, ckpt_dir / "last.pt")

        if args.smoke:
            print("\n[smoke OK]")
            break

    log_writer.close()
    tb_writer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
