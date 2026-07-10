"""E7 分层 linear probe — 验证 RVQ 层级分工干净.

motivation §3.2 第 146 行明确承诺: 通过分层 linear probe 量化各层信息类型,
验证"L0 是语义层 / L1-L3 是细节层"的分工是否干净.

E5 (04_eval_l0_linear.py) 只测了 L0 与 zq_full 两个端点, 中间档 (L0+L1, L0+L1+L2) 缺位.
本脚本扩展为 4 个累加配置, 直接画出"再加一层带来多少分类增益"的曲线.

四种累加特征 (k = 1..4):
  L0_to_L{k-1}_emb   - 取前 k 层 codebook embedding 求和后空间均池, 256 维
                       k=1 时等价于 E5 的 L0_emb, k=4 时等价于 E5 的 zq_pool

判据 (motivation §3.2 配套):
  - 蒸馏版 L0 → L0+L1 提升 < 2 pp → 分层解耦干净, motivation §3.2 站住
  - ∈ [2, 5) pp                   → 部分耦合
  - ≥ 5 pp                         → 严重耦合, k=1 承诺要改成 k ∈ {1, 2}

启动:
    & $py -X utf8 scripts/04b_eval_layered_probe.py \
        --ckpt checkpoints/rvq_baseline/best.pt --out logs/layered_probe_baseline.csv
    & $py -X utf8 scripts/04b_eval_layered_probe.py \
        --ckpt checkpoints/rvq_distill/best.pt  --out logs/layered_probe_distill.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.datasets import (                            # noqa: E402
    AIDConfig, AIDDataset, build_loaders, build_transforms,
)
from models.vqvae import VQVAE, VQVAEConfig              # noqa: E402
from torch.utils.data import DataLoader                  # noqa: E402

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )


@torch.no_grad()
def extract_layered_features(model: VQVAE, loader, device: str):
    """对一个 split 提取 4 个累加配置的 embedding 特征.

    取所有 RVQ 层的 codebook, 按 k=1..4 累加 codebook embedding(L0..L_{k-1})
    然后在 patch 维度均池, 得到 [N, latent_dim] 特征.

    返回:
      feats[k] : [N, latent_dim]  for k in 1..4
      labels   : [N]
    """
    model.eval()
    is_rvq = model.cfg.quantizer == "rvq"
    assert is_rvq, "本脚本面向 RVQ ckpt"

    L = len(model.quantizer.layers)
    # 取所有层的 codebook -> list of [K, D]
    codebooks = []
    for lyr in model.quantizer.layers:
        cb = lyr._codebook.embed.detach()
        if cb.dim() == 3:
            cb = cb[0]
        codebooks.append(cb.to(device))   # [K, D]

    feats = {k: [] for k in range(1, L + 1)}
    labels_all = []

    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model.encoder(x)
        z_seq = rearrange(z, "b c h w -> b (h w) c")     # [B, T, D]

        _, indices, _ = model.quantizer(z_seq)            # [B, T, L]

        # 累加 codebook embedding: zq_k = sum_{i<k} codebook[i][indices[..., i]]
        zq_acc = torch.zeros_like(z_seq)                  # [B, T, D]
        for k in range(1, L + 1):
            cb_i = codebooks[k - 1]                       # [K, D]
            idx_i = indices[..., k - 1]                   # [B, T]
            zq_acc = zq_acc + cb_i[idx_i]                 # [B, T, D]
            pooled = zq_acc.mean(dim=1)                   # [B, D]
            feats[k].append(pooled.cpu())

        labels_all.append(y)

    feats_np = {
        f"L0_to_L{k-1}_emb": torch.cat(feats[k]).numpy()
        for k in range(1, L + 1)
    }
    labels = torch.cat(labels_all).numpy()
    print(f"  feature extraction took {time.time()-t0:.1f}s, N={len(labels)}")
    return feats_np, labels


def fit_linear(name, X_tr, y_tr, X_te, y_te):
    print(f"\n[{name}]  X_tr.shape={X_tr.shape}")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, solver="lbfgs")
    clf.fit(X_tr_s, y_tr)
    fit_t = time.time() - t0
    acc_tr = clf.score(X_tr_s, y_tr)
    acc_te = clf.score(X_te_s, y_te)
    print(f"  fit {fit_t:.1f}s  train={acc_tr*100:.2f}%  test={acc_te*100:.2f}%")
    return acc_tr, acc_te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="RVQ best.pt 路径")
    parser.add_argument("--out", required=True, help="输出 csv 路径")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    print("=" * 60)
    print(f"E7 layered linear probe")
    print(f"ckpt: {args.ckpt}")
    print("=" * 60)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    model_cfg = VQVAEConfig(**cfg_dict["model"])
    model = VQVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    L = model_cfg.rvq_num_quantizers
    print(f"  model: rvq L={L} cb={model_cfg.codebook_size} "
          f"latent={model_cfg.latent_dim}")

    data_cfg = AIDConfig(**cfg_dict["data"])
    loaders = build_loaders(data_cfg)
    train_eval_ds = AIDDataset(
        f"{data_cfg.splits_dir}/train.csv",
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=data_cfg.batch_size, shuffle=False,
        num_workers=data_cfg.num_workers, pin_memory=True, drop_last=False,
    )

    print("\n>>> extracting layered features on TRAIN split (no aug)")
    feats_tr, y_tr = extract_layered_features(model, train_eval_loader, device)
    print("\n>>> extracting layered features on TEST split")
    feats_te, y_te = extract_layered_features(model, loaders["test"], device)

    print("\n" + "=" * 60)
    print(f"Layered linear probe results "
          f"(n_train={len(y_tr)}, n_test={len(y_te)})")
    print("=" * 60)

    rows = []
    feat_names = [f"L0_to_L{k}_emb" for k in range(L)]
    for name in feat_names:
        acc_tr, acc_te = fit_linear(name, feats_tr[name], y_tr,
                                    feats_te[name], y_te)
        rows.append({
            "feature": name,
            "k": int(name.split("L")[2].split("_")[0]) + 1,
            "dim": feats_tr[name].shape[1],
            "train_acc": round(acc_tr * 100, 2),
            "test_acc":  round(acc_te * 100, 2),
        })

    print("\n" + "=" * 60)
    print("Summary (test top-1 acc, accumulating layers)")
    print("=" * 60)
    for r in rows:
        print(f"  {r['feature']:18s}  k={r['k']}  test={r['test_acc']:.2f}%")

    # 给出"加一层增益"
    print("\n  Layer increment gains:")
    for i in range(1, len(rows)):
        delta = rows[i]["test_acc"] - rows[i-1]["test_acc"]
        print(f"    +L{i}:  {rows[i-1]['test_acc']:.2f}% -> "
              f"{rows[i]['test_acc']:.2f}%   (delta = {delta:+.2f} pp)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
