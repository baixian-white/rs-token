"""Stage 2 评测: 仅用 L0 索引做线性分类 AID, 给 Stage 3 蒸馏建立对照锚点.

四种特征 (从严到宽):
  L0_bow   - L0 索引在 1024 词表上的词频直方图. 代表"只传 L0 索引的下游能力"
  L0_emb   - L0 索引查表回 embedding 后空间均池, 256 维. 同上但保留 codebook 几何
  zq_pool  - 全 4 层量化输出空间均池, 256 维. "传完整 RVQ" 上限
  zpre_pool- encoder 输出空间均池(无量化), 256 维. 无量化天花板

每种特征跑一个 LinearSVC (one-vs-rest), 报告 train/test top-1 acc.

启动:
    conda activate rstoken
    python scripts/04_eval_l0_linear.py \
        --ckpt checkpoints/rvq_baseline/best.pt
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

# Windows + sklearn/BLAS can otherwise terminate the process in native code
# while fitting the small multinomial probe.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import AIDConfig, build_loaders  # noqa: E402
from models.vqvae import VQVAE, VQVAEConfig            # noqa: E402

LOGREG_MAX_ITER = int(os.environ.get("RSTOKEN_LOGREG_MAX_ITER", "300"))
LOGREG_TOL = float(os.environ.get("RSTOKEN_LOGREG_TOL", "1e-3"))
PROBE_FEATURES = [
    item.strip()
    for item in os.environ.get("RSTOKEN_PROBE_FEATURES", "L0_bow").split(",")
    if item.strip()
]

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )


@torch.no_grad()
def extract_features(model: VQVAE, loader, device: str, codebook_size: int):
    """对一个 split 提取四种特征.

    返回 dict:
      L0_bow    : [N, codebook_size]
      L0_emb    : [N, latent_dim]
      zq_pool   : [N, latent_dim]
      zpre_pool : [N, latent_dim]
      labels    : [N]
    """
    model.eval()
    is_rvq = model.cfg.quantizer == "rvq"
    assert is_rvq, "本脚本面向 Stage 2 RVQ ckpt"

    L0_bow_all, L0_emb_all = [], []
    zq_pool_all, zpre_pool_all = [], []
    labels_all = []

    # 取 L0 codebook 用于 indices -> embedding
    # ResidualVQ.layers[0] 是 VectorQuantize, 其 codebook 在 _codebook.embed
    l0_layer = model.quantizer.layers[0]
    # vector_quantize_pytorch 的 codebook tensor: [num_codebooks=1, codebook_size, dim]
    l0_codebook = l0_layer._codebook.embed.detach()
    if l0_codebook.dim() == 3:
        l0_codebook = l0_codebook[0]  # [codebook_size, dim]

    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # encoder
        z = model.encoder(x)                                  # [B, C, H, W]
        b, c, h, w = z.shape
        z_seq = rearrange(z, "b c h w -> b (h w) c")          # [B, T, C]

        # quantize (得到 zq, indices)
        zq_seq, indices, _ = model.quantizer(z_seq)           # indices: [B, T, L]
        # 第 0 层索引
        l0_idx = indices[..., 0]                              # [B, T]

        # L0 BoW: 词频直方图
        bow = torch.zeros(b, codebook_size, device=device)
        bow.scatter_add_(
            1, l0_idx,
            torch.ones_like(l0_idx, dtype=torch.float)
        )
        bow = bow / l0_idx.shape[1]                           # 归一化为频率

        # L0 embedding pool: 索引查表 -> 池化
        l0_emb = l0_codebook[l0_idx]                          # [B, T, dim]
        l0_emb_pool = l0_emb.mean(dim=1)                      # [B, dim]

        # zq pool: 全 4 层重建 zq 池化
        zq_pool = zq_seq.mean(dim=1)                          # [B, dim]

        # z_pre pool: 无量化 encoder 输出池化
        zpre_pool = z_seq.mean(dim=1)                         # [B, dim]

        L0_bow_all.append(bow.cpu())
        L0_emb_all.append(l0_emb_pool.cpu())
        zq_pool_all.append(zq_pool.cpu())
        zpre_pool_all.append(zpre_pool.cpu())
        labels_all.append(y)

    feats = {
        "L0_bow":   torch.cat(L0_bow_all).numpy(),
        "L0_emb":   torch.cat(L0_emb_all).numpy(),
        "zq_pool":  torch.cat(zq_pool_all).numpy(),
        "zpre_pool":torch.cat(zpre_pool_all).numpy(),
    }
    labels = torch.cat(labels_all).numpy()
    print(f"  feature extraction took {time.time()-t0:.1f}s, "
          f"N={len(labels)}")
    return feats, labels


def fit_linear(name, X_tr, y_tr, X_te, y_te):
    """在 train 上拟合 logistic regression, 报 train/test acc."""
    print(f"\n[{name}]  X_tr.shape={X_tr.shape}")
    scaler = StandardScaler(with_mean=name != "L0_bow")  # bow 是稀疏频率, 不去均值
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    t0 = time.time()
    clf = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        tol=LOGREG_TOL,
        C=1.0,
        n_jobs=1,
        solver="lbfgs",
    )
    clf.fit(X_tr_s, y_tr)
    fit_t = time.time() - t0

    acc_tr = clf.score(X_tr_s, y_tr)
    acc_te = clf.score(X_te_s, y_te)
    print(f"  fit {fit_t:.1f}s  train={acc_tr*100:.2f}%  test={acc_te*100:.2f}%")
    return acc_tr, acc_te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Stage 2 RVQ best.pt 路径")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    print("=" * 60)
    print(f"L0 linear probe eval")
    print(f"ckpt: {args.ckpt}")
    print("=" * 60)

    # 加载 ckpt
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    model_cfg = VQVAEConfig(**cfg_dict["model"])
    model = VQVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  model: quantizer={model_cfg.quantizer} "
          f"x{model_cfg.rvq_num_quantizers}  cb={model_cfg.codebook_size}")

    # DataLoader: 评测用 train + test (val 不参与)
    data_cfg = AIDConfig(**cfg_dict["data"])
    # 评测时不要 train aug, 用 val transform 跑 train.csv
    # 简单做法: 直接复用 build_loaders, 但把 train 的 transform 换掉
    loaders = build_loaders(data_cfg)
    # 注: build_loaders 给 train 用的是 RandomResizedCrop, 评测用要换 center crop
    # 直接构造一个用 val transform 的 train loader
    from models.datasets import AIDDataset, build_transforms
    from torch.utils.data import DataLoader
    train_eval_ds = AIDDataset(
        f"{data_cfg.splits_dir}/train.csv",
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=data_cfg.batch_size,
        shuffle=False, num_workers=data_cfg.num_workers,
        pin_memory=True, drop_last=False,
    )

    print("\n>>> extracting features on TRAIN split (no aug)")
    feats_tr, y_tr = extract_features(
        model, train_eval_loader, device, model_cfg.codebook_size
    )

    print("\n>>> extracting features on TEST split")
    feats_te, y_te = extract_features(
        model, loaders["test"], device, model_cfg.codebook_size
    )

    print("\n" + "=" * 60)
    print(f"AID linear probe results (n_train={len(y_tr)}, n_test={len(y_te)})")
    print("=" * 60)

    valid_features = {"L0_bow", "L0_emb", "zq_pool", "zpre_pool"}
    unknown = sorted(set(PROBE_FEATURES) - valid_features)
    if unknown:
        raise ValueError(f"unknown probe features: {unknown}")

    results = {}
    for name in PROBE_FEATURES:
        results[name] = fit_linear(
            name, feats_tr[name], y_tr, feats_te[name], y_te
        )

    print("\n" + "=" * 60)
    print("Summary (test top-1 acc)")
    print("=" * 60)
    for name in PROBE_FEATURES:
        print(f"  {name:12s}  {results[name][1]*100:.2f}%")


if __name__ == "__main__":
    main()
