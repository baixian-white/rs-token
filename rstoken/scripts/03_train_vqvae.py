"""Stage 1 训练入口 · 单层 VQ-VAE baseline.

特点:
  - 从 yaml 读全部超参, 修改不需改代码
  - bf16 AMP (5070 Ti Blackwell 支持原生 bf16, 不用 GradScaler)
  - Cosine 学习率 + warmup
  - 每 epoch 跑 val, 记录 PSNR/SSIM/codebook 利用率 + 8 张可视化对比图
  - ckpt 自动保留最佳 val PSNR + 最近 N 个 epoch
  - Windows 兼容: utf-8 stdout, num_workers 默认 4 不阻塞

启动:
    conda activate rstoken
    python scripts/03_train_vqvae.py --config configs/vqvae_baseline.yaml

Resume (待实现, 暂用 ckpt 手动加载):
    python scripts/03_train_vqvae.py --config configs/vqvae_baseline.yaml \
        --resume checkpoints/vqvae_baseline/last.pt
"""
from __future__ import annotations

import argparse
import io
import json
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

# 让脚本可以从 rstoken/ 根直接 import models 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDConfig, build_loaders, denormalize  # noqa: E402
from models.vqvae import VQVAE, VQVAEConfig                         # noqa: E402
from models.perceptual import LPIPSLoss                             # noqa: E402
from models.distillation import (                                   # noqa: E402
    RemoteCLIPTeacher, DistillHead, distill_loss as distill_loss_fn,
)

# Windows GBK -> UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Cosine schedule with linear warmup."""
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x, y in [-1, 1]. 返回 batch-mean PSNR (dB)."""
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
    # 范围 [-1, 1] => 峰值 2
    return (10 * torch.log10(4.0 / (mse + 1e-12))).mean()


def codebook_usage(indices: torch.Tensor, codebook_size: int) -> float:
    """统计 codebook 利用率.

    VQ:  indices shape [B, T]      —— 直接 unique / codebook_size
    RVQ: indices shape [B, T, L]   —— 每层独立 codebook, 按层算后取均值
    """
    if indices.dim() == 2:
        used = torch.unique(indices.flatten()).numel()
        return used / codebook_size
    # RVQ: [B, T, L]
    L = indices.shape[-1]
    per_layer = []
    for l in range(L):
        per_layer.append(
            torch.unique(indices[..., l].flatten()).numel() / codebook_size
        )
    return sum(per_layer) / L


def codebook_usage_per_layer(indices: torch.Tensor, codebook_size: int) -> list[float]:
    """RVQ 专用: 返回每层利用率列表. VQ 时返回单元素列表."""
    if indices.dim() == 2:
        return [torch.unique(indices.flatten()).numel() / codebook_size]
    L = indices.shape[-1]
    return [
        torch.unique(indices[..., l].flatten()).numel() / codebook_size
        for l in range(L)
    ]


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------

@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_val_psnr: float = -1.0


def train_one_epoch(
    model: VQVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lpips_fn: LPIPSLoss | None,
    cfg: dict,
    state: TrainState,
    total_steps: int,
    log_writer,
    tb_writer: SummaryWriter | None = None,
    log_every_override: int | None = None,
    teacher: "RemoteCLIPTeacher | None" = None,
    distill_head: "DistillHead | None" = None,
):
    model.train()
    if distill_head is not None:
        distill_head.train()
    log = cfg["logging"]
    loss_cfg = cfg["loss"]
    optim_cfg = cfg["optim"]
    cb_size = cfg["model"]["codebook_size"]
    log_every = log_every_override if log_every_override is not None \
                else log["log_every"]

    use_amp = cfg.get("amp", False)
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" \
                else torch.float16

    distill_w = float(loss_cfg.get("distill_weight", 0.0))
    use_distill = distill_w > 0 and teacher is not None and distill_head is not None
    distill_target = (cfg.get("distill") or {}).get("target", "l0")
    if distill_target not in ("l0", "all_layers", "l1"):
        raise ValueError(
            f"distill.target must be 'l0', 'l1', or 'all_layers', got {distill_target!r}"
        )

    t0 = time.time()
    running = {"l1": 0.0, "lpips": 0.0, "vq": 0.0, "psnr": 0.0,
               "use": 0.0, "distill": 0.0, "n": 0}

    for x, _ in loader:
        x = x.to(cfg["device"], non_blocking=True)

        # LR schedule
        lr = cosine_warmup_lr(
            state.step, total_steps,
            optim_cfg["warmup_steps"], optim_cfg["lr"]
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # 教师 embedding (在 AMP 外算, fp32 / no grad)
        teacher_emb = None
        if use_distill:
            with torch.no_grad():
                teacher_emb = teacher.encode_image(x)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
                l1 = F.l1_loss(out["recon"], x)
                lp = lpips_fn(out["recon"], x) if lpips_fn else torch.tensor(0.0)
                if use_distill:
                    if distill_target == "l0":
                        distill_feat = out["zq_l0_ste"]
                    elif distill_target == "l1":
                        distill_feat = out["zq_l1_ste"]
                    else:  # all_layers
                        distill_feat = out["zq"]
                    s_emb = distill_head(distill_feat)
                    d_loss = distill_loss_fn(s_emb, teacher_emb)
                else:
                    d_loss = torch.tensor(0.0, device=x.device)
                loss = (
                    loss_cfg["l1_weight"] * l1
                    + loss_cfg["lpips_weight"] * lp
                    + loss_cfg["vq_weight"] * out["vq_loss"]
                    + distill_w * d_loss
                )
        else:
            out = model(x)
            l1 = F.l1_loss(out["recon"], x)
            lp = lpips_fn(out["recon"], x) if lpips_fn else torch.tensor(0.0)
            if use_distill:
                if distill_target == "l0":
                    distill_feat = out["zq_l0_ste"]
                elif distill_target == "l1":
                    distill_feat = out["zq_l1_ste"]
                else:  # all_layers
                    distill_feat = out["zq"]
                s_emb = distill_head(distill_feat)
                d_loss = distill_loss_fn(s_emb, teacher_emb)
            else:
                d_loss = torch.tensor(0.0, device=x.device)
            loss = (
                loss_cfg["l1_weight"] * l1
                + loss_cfg["lpips_weight"] * lp
                + loss_cfg["vq_weight"] * out["vq_loss"]
                + distill_w * d_loss
            )

        loss.backward()
        if optim_cfg.get("grad_clip"):
            params_to_clip = list(model.parameters())
            if distill_head is not None:
                params_to_clip += list(distill_head.parameters())
            torch.nn.utils.clip_grad_norm_(
                params_to_clip, optim_cfg["grad_clip"]
            )
        optimizer.step()

        with torch.no_grad():
            running["l1"]    += l1.item()
            running["lpips"] += lp.item()
            running["vq"]    += out["vq_loss"].item()
            running["psnr"]  += psnr(out["recon"], x).item()
            running["use"]   += codebook_usage(out["indices"], cb_size)
            running["distill"] += d_loss.item() if use_distill else 0.0
            running["n"]     += 1

        state.step += 1

        if state.step % log_every == 0:
            n = running["n"]
            distill_str = f"  distill={running['distill']/n:.4f}" if use_distill else ""
            msg = (
                f"  ep {state.epoch:>3d}  step {state.step:>6d}  "
                f"lr {lr:.2e}  "
                f"l1={running['l1']/n:.4f}  "
                f"lpips={running['lpips']/n:.4f}  "
                f"vq={running['vq']/n:.4f}{distill_str}  "
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
                tb_writer.add_scalar("train/img_per_sec",
                                     n*x.shape[0]/(time.time()-t0), state.step)
                if use_distill:
                    tb_writer.add_scalar("train/distill",
                                         running["distill"]/n, state.step)
            running = {k: 0.0 for k in running}; running["n"] = 0
            t0 = time.time()


@torch.no_grad()
def validate(
    model: VQVAE,
    loader: DataLoader,
    lpips_fn: LPIPSLoss | None,
    cfg: dict,
    epoch: int,
) -> dict:
    model.eval()
    cb_size = cfg["model"]["codebook_size"]
    sums = {"l1": 0.0, "lpips": 0.0, "psnr": 0.0, "use_per_batch": 0.0}
    # 跨整个 val 集统计 codebook 真实使用率
    # VQ: 单层 set; RVQ: 每层一个 set
    is_rvq = cfg["model"].get("quantizer", "vq") == "rvq"
    n_layers = cfg["model"].get("rvq_num_quantizers", 1) if is_rvq else 1
    layer_sets: list[set[int]] = [set() for _ in range(n_layers)]
    n_batches = 0

    use_amp = cfg.get("amp", False)
    amp_dtype = torch.bfloat16 if cfg.get("amp_dtype", "bf16") == "bf16" \
                else torch.float16

    vis_x, vis_recon = None, None  # 记录前几张用于可视化

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
        if is_rvq:
            # idx: [B, T, L]
            for l in range(n_layers):
                layer_sets[l].update(idx[..., l].flatten().tolist())
        else:
            layer_sets[0].update(idx.flatten().tolist())
        n_batches += 1

        if i == 0:
            vis_x = x[: cfg["logging"]["num_vis_samples"]].cpu()
            vis_recon = out["recon"][: cfg["logging"]["num_vis_samples"]].cpu()

    metrics = {
        "val/l1":    sums["l1"]    / n_batches,
        "val/lpips": sums["lpips"] / n_batches,
        "val/psnr":  sums["psnr"]  / n_batches,
        "val/cb_use_per_batch":  sums["use_per_batch"] / n_batches,
    }
    if is_rvq:
        per_layer_use = [len(s) / cb_size for s in layer_sets]
        for l, u in enumerate(per_layer_use):
            metrics[f"val/cb_use_global_L{l}"] = u
        # 平均利用率, 兼容旧的 cb_use_global key
        metrics["val/cb_use_global"] = sum(per_layer_use) / len(per_layer_use)
    else:
        metrics["val/cb_use_global"] = len(layer_sets[0]) / cb_size
    return metrics, vis_x, vis_recon


def save_vis(vis_x: torch.Tensor, vis_recon: torch.Tensor,
             out_path: Path, epoch: int):
    """把 [N,3,H,W] in [-1,1] 拼成上下两行 grid 写出 PNG."""
    from torchvision.utils import make_grid, save_image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = torch.cat([
        denormalize(vis_x),
        denormalize(vis_recon),
    ], dim=0)
    grid = make_grid(pairs, nrow=vis_x.shape[0])
    save_image(grid, out_path)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="只跑 2 个 train batch + 1 个 val batch, 验证逻辑")
    parser.add_argument("--init_from_ckpt", default=None,
                        help="E40: load model state_dict from this ckpt right "
                             "after random init (no optimizer / distill_head / "
                             "teacher reuse). Skipped when omitted.")
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
    print(f"Run: {cfg['run_name']}")
    print(f"Config: {args.config}")
    print(f"Smoke: {args.smoke}")
    print("=" * 60)

    # --- DataLoader
    data_cfg = AIDConfig(**cfg["data"])
    loaders = build_loaders(data_cfg)
    print(f"  train batches: {len(loaders['train'])}")
    print(f"  val   batches: {len(loaders['val'])}")

    # --- Model
    model_cfg = VQVAEConfig(**cfg["model"])
    model = VQVAE(model_cfg).to(device)
    np_info = model.num_parameters()
    print(f"  params: total {np_info['total']/1e6:.2f}M  "
          f"(enc {np_info['encoder']/1e6:.2f}M / "
          f"dec {np_info['decoder']/1e6:.2f}M / "
          f"q {np_info['quantizer']/1e6:.3f}M)")

    # --- Optional: load model weights from a step-0 / paired init ckpt (E40).
    # Only the model's state_dict is reused; optimizer / distill_head / teacher
    # are rebuilt from the current config below.
    if args.init_from_ckpt is not None:
        init_path = Path(args.init_from_ckpt)
        if not init_path.is_absolute():
            init_path = (PROJECT_ROOT / init_path).resolve()
        ck = torch.load(str(init_path), map_location=device, weights_only=False)
        if "model" not in ck:
            raise KeyError(
                f"init ckpt {init_path} missing 'model' key; got keys={list(ck.keys())}"
            )
        missing_unexpected = model.load_state_dict(ck["model"], strict=True)
        missing = list(missing_unexpected.missing_keys)
        unexpected = list(missing_unexpected.unexpected_keys)
        print(f"  loaded init weights from {init_path}")
        if missing:
            print(f"    [warn] missing keys: {missing}")
        if unexpected:
            print(f"    [warn] unexpected keys: {unexpected}")

    # --- Loss
    lpips_fn = None
    if cfg["loss"]["lpips_weight"] > 0:
        lpips_fn = LPIPSLoss(net=cfg["loss"]["lpips_net"]).to(device)

    # --- Distillation (Stage 3): 仅当 cfg.distill 存在时启用
    teacher = None
    distill_head = None
    distill_cfg = cfg.get("distill", None)
    if distill_cfg is not None and float(cfg["loss"].get("distill_weight", 0)) > 0:
        assert model_cfg.quantizer == "rvq", "蒸馏仅支持 RVQ"
        print(f"  loading teacher from {distill_cfg['teacher_ckpt']}")
        teacher = RemoteCLIPTeacher(
            ckpt_path=distill_cfg["teacher_ckpt"],
            model_name=distill_cfg.get("teacher_name", "ViT-B-32"),
        ).to(device)
        distill_head = DistillHead(
            latent_dim=model_cfg.latent_dim,
            teacher_dim=teacher.embed_dim,
            hidden=distill_cfg.get("head_hidden", 512),
        ).to(device)
        n_head = sum(p.numel() for p in distill_head.parameters()) / 1e6
        print(f"  distill head: {n_head:.2f}M params, "
              f"weight={cfg['loss']['distill_weight']}")

    # --- Optim
    params = list(model.parameters())
    if distill_head is not None:
        params += list(distill_head.parameters())
    optimizer = torch.optim.AdamW(
        params,
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

    tb_dir = log_dir / "tb"
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"  tensorboard: {tb_dir}")

    for epoch in range(total_epochs):
        state.epoch = epoch
        print(f"\n[epoch {epoch+1}/{total_epochs}]")

        if args.smoke:
            # smoke 模式: 取 2 batch
            iter_loader = iter(loaders["train"])
            mini = []
            for _ in range(2):
                mini.append(next(iter_loader))
            from torch.utils.data import DataLoader as _DL
            class _MockLoader:
                def __init__(self, items): self.items = items
                def __iter__(self): return iter(self.items)
                def __len__(self): return len(self.items)
            train_loader = _MockLoader(mini)
        else:
            train_loader = loaders["train"]

        train_one_epoch(
            model, train_loader, optimizer, lpips_fn, cfg, state,
            total_steps, log_writer,
            tb_writer=tb_writer,
            log_every_override=1 if args.smoke else None,
            teacher=teacher,
            distill_head=distill_head,
        )

        # Validate
        if (epoch + 1) % log_cfg["val_every_epoch"] == 0 or args.smoke:
            val_loader = loaders["val"]
            if args.smoke:
                # 只取 1 batch 验证
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
                    log_dir / f"vis_epoch_{epoch+1:03d}.png",
                    epoch
                )
                from torchvision.utils import make_grid
                pairs = torch.cat([denormalize(vis_x), denormalize(vis_recon)], dim=0)
                grid = make_grid(pairs, nrow=vis_x.shape[0])
                tb_writer.add_image("val/recon_top_orig_bot", grid, state.step)

            # Save best
            if metrics["val/psnr"] > state.best_val_psnr:
                state.best_val_psnr = metrics["val/psnr"]
                ckpt_obj = {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "metrics": metrics,
                }
                if distill_head is not None:
                    ckpt_obj["distill_head"] = distill_head.state_dict()
                torch.save(ckpt_obj, ckpt_dir / "best.pt")
                print(f"  ★ new best val PSNR: {metrics['val/psnr']:.2f} dB "
                      f"-> saved best.pt")

        # Save periodic
        if (epoch + 1) % log_cfg["save_every_epoch"] == 0:
            ckpt_obj = {
                "model": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
            }
            if distill_head is not None:
                ckpt_obj["distill_head"] = distill_head.state_dict()
            torch.save(ckpt_obj, ckpt_dir / f"epoch_{epoch+1:03d}.pt")

        # Always save last
        ckpt_obj = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "best_val_psnr": state.best_val_psnr,
        }
        if distill_head is not None:
            ckpt_obj["distill_head"] = distill_head.state_dict()
        torch.save(ckpt_obj, ckpt_dir / "last.pt")

        if args.smoke:
            print("\n[smoke OK]")
            break

    log_writer.close()
    tb_writer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
