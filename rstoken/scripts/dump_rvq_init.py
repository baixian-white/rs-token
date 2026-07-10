"""E40 — Dump a step-0 RVQ-VAE checkpoint for paired-init experiments.

Reads a yaml (same model schema as the rvq_distill / rvq_baseline configs),
seeds torch with cfg["seed"], constructs the VQVAE on CPU, and saves
{"model": state_dict(), "config": cfg, "epoch": 0} to --out.

The resulting ckpt is loaded by 03_train_vqvae.py via --init_from_ckpt to
guarantee both the baseline and distill paired runs start from bit-identical
random weights.

Usage:
    python scripts/dump_rvq_init.py \
        --config configs/paper_v05/rvq_init_s41.yaml \
        --out checkpoints/paper_v05/rvq_init_s41/step0.pt
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqvae import VQVAE, VQVAEConfig  # noqa: E402

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    set_seed(seed)

    model_cfg = VQVAEConfig(**cfg["model"])
    # Force CPU construction so the saved state_dict is device-agnostic and
    # the dump is independent of GPU availability.
    model = VQVAE(model_cfg)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (PROJECT_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "config": cfg,
        "epoch": 0,
        "note": "step-0 random-init checkpoint dumped by dump_rvq_init.py "
                "for E40 paired init-seed control",
    }
    torch.save(ckpt, str(out_path))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[dump_rvq_init] seed={seed}  params={n_params/1e6:.2f}M  "
          f"out={out_path}  size={out_path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
