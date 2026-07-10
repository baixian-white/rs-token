"""B6 几何分析: 在两种教师 (RemoteCLIP / OpenAI CLIP) 蒸馏 ckpt 上,
比较 L0 codebook 在类标签条件下的几何分离度.

motivation: 论文 §5.1 主张 "RemoteCLIP 蒸馏让 30 个 AID 类簇在特征空间中保持
更大的簇间距与更紧的簇内分布", 这条目前为合理猜想; 本脚本基于 train 集 L0
量化后特征 (zq_L0) 直接量化该几何性质, 给出三种距离比指标 + t-SNE 可视化.

指标:
  ratio  = mean_inter / mean_intra
           其中 mean_inter = ||mu_c - mu_{c'}||_2 跨类对均值
                mean_intra = ||x - mu_{label(x)}||_2 类内距离均值
  silhouette: sklearn silhouette_score (越大越好)
  CH:        Calinski-Harabasz score (越大越好)
  DB:        Davies-Bouldin score (越小越好)

启动:
    python scripts/07_eval_codebook_geometry.py \
        --ckpts checkpoints/rvq_distill/best.pt checkpoints/rvq_distill_openai/best.pt \
        --tags RemoteCLIP OpenAI_CLIP \
        --out_csv logs/codebook_geometry.csv \
        --tsne_fig figs/fig_codebook_tsne.png
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score, davies_bouldin_score, silhouette_score,
)

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
def extract_l0_features(model, loader, device):
    """对一个 split 提取每张图的 L0 量化后特征 (zq_L0 在 patch 维度均池).

    返回:
      feats : [N, D]  L0 codebook embedding pool, 每张图一行
      labels: [N]     类标签 (0..29)
    """
    model.eval()
    L = len(model.quantizer.layers)
    cb_l0 = model.quantizer.layers[0]._codebook.embed.detach()
    if cb_l0.dim() == 3:
        cb_l0 = cb_l0[0]
    cb_l0 = cb_l0.to(device)   # [K, D]

    feats_all, labels_all = [], []
    t0 = time.time()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model.encoder(x)
        z_seq = rearrange(z, "b c h w -> b (h w) c")
        _, indices, _ = model.quantizer(z_seq)            # [B, T, L]
        l0_idx = indices[..., 0]                          # [B, T]
        l0_emb = cb_l0[l0_idx]                            # [B, T, D]
        l0_pool = l0_emb.mean(dim=1)                      # [B, D]
        feats_all.append(l0_pool.cpu())
        labels_all.append(y)
    feats = torch.cat(feats_all).numpy()
    labels = torch.cat(labels_all).numpy()
    print(f"  feature extraction took {time.time()-t0:.1f}s, "
          f"N={len(labels)}, D={feats.shape[1]}")
    return feats, labels


def compute_geometry(feats, labels):
    """对 [N, D] 特征 + [N] 标签算三种几何分离度指标.

    返回 dict:
      ratio       : mean_inter / mean_intra
      mean_inter  : 跨类簇心两两距离均值
      mean_intra  : 类内点到簇心距离均值
      silhouette  : sklearn silhouette_score
      ch          : Calinski-Harabasz
      db          : Davies-Bouldin
    """
    classes = np.unique(labels)
    C = len(classes)

    # 类心 mu_c
    centroids = np.stack(
        [feats[labels == c].mean(axis=0) for c in classes]
    )  # [C, D]

    # mean_inter: 跨类对距离均值
    diff = centroids[:, None, :] - centroids[None, :, :]   # [C, C, D]
    dist_inter = np.linalg.norm(diff, axis=-1)             # [C, C]
    mask = ~np.eye(C, dtype=bool)
    mean_inter = dist_inter[mask].mean()

    # mean_intra: 每点到自己簇心距离均值
    centroids_per_sample = centroids[labels]              # [N, D]
    intra = np.linalg.norm(feats - centroids_per_sample, axis=-1)
    mean_intra = intra.mean()

    ratio = mean_inter / (mean_intra + 1e-12)

    # sklearn 指标 (sample 较多时 silhouette 慢, 抽样到 2000)
    n = len(feats)
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2000, replace=False)
        f_sub, l_sub = feats[idx], labels[idx]
    else:
        f_sub, l_sub = feats, labels

    sil = silhouette_score(f_sub, l_sub)
    ch = calinski_harabasz_score(feats, labels)
    db = davies_bouldin_score(feats, labels)

    return dict(
        ratio=float(ratio),
        mean_inter=float(mean_inter),
        mean_intra=float(mean_intra),
        silhouette=float(sil),
        ch=float(ch),
        db=float(db),
    )


def make_tsne(feats_dict, labels, out_path, perplexity=30, max_show=30):
    """对每个 ckpt 各画一个 t-SNE 子图. 共享同一 labels."""
    n_ckpt = len(feats_dict)
    fig, axes = plt.subplots(1, n_ckpt, figsize=(6.2 * n_ckpt, 5.5))
    if n_ckpt == 1:
        axes = [axes]

    classes = np.unique(labels)
    cmap = plt.get_cmap("tab20", len(classes))

    # 抽样 (t-SNE 大样本慢)
    n = len(labels)
    if n > 2500:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2500, replace=False)
    else:
        idx = np.arange(n)

    for ax, (tag, feats) in zip(axes, feats_dict.items()):
        f_sub = feats[idx]
        l_sub = labels[idx]
        print(f"  running t-SNE on {tag} ({len(f_sub)} samples)...")
        t0 = time.time()
        tsne = TSNE(
            n_components=2, perplexity=perplexity,
            random_state=42, init="pca",
        )
        emb = tsne.fit_transform(f_sub)
        print(f"    done in {time.time()-t0:.1f}s")

        for c in classes[:max_show]:
            mask = l_sub == c
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                color=cmap(c), s=8, alpha=0.7,
                edgecolors="none",
            )
        ax.set_title(f"{tag} L0 codebook embedding\n"
                     f"(t-SNE on AID train)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Geometric separation of L0 codebook embeddings under different teachers",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path} + .pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True,
                        help="一个或多个 RVQ best.pt 路径")
    parser.add_argument("--tags", nargs="+", required=True,
                        help="对应每个 ckpt 的标签 (用于表头与图例)")
    parser.add_argument("--out_csv", default="logs/codebook_geometry.csv")
    parser.add_argument("--tsne_fig", default="figs/fig_codebook_tsne.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert len(args.ckpts) == len(args.tags), \
        "ckpts 数量必须与 tags 一致"

    device = args.device
    rows = []
    feats_dict = {}
    labels_ref = None

    for ckpt_path, tag in zip(args.ckpts, args.tags):
        print("=" * 60)
        print(f"[{tag}] {ckpt_path}")
        print("=" * 60)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]
        model_cfg = VQVAEConfig(**cfg_dict["model"])
        model = VQVAE(model_cfg).to(device)
        model.load_state_dict(ckpt["model"])

        # train split (8000 imgs, 30 classes), 用 eval transform 不做 aug
        data_cfg = AIDConfig(**cfg_dict["data"])
        train_eval_ds = AIDDataset(
            f"{data_cfg.splits_dir}/train.csv",
            transform=build_transforms(data_cfg.image_size, train=False),
        )
        loader = DataLoader(
            train_eval_ds, batch_size=data_cfg.batch_size, shuffle=False,
            num_workers=data_cfg.num_workers, pin_memory=True, drop_last=False,
        )

        feats, labels = extract_l0_features(model, loader, device)
        if labels_ref is None:
            labels_ref = labels
        else:
            assert np.array_equal(labels, labels_ref), \
                "两个 ckpt 必须使用同一 train split 顺序"

        feats_dict[tag] = feats

        metrics = compute_geometry(feats, labels)
        print(f"  ratio (inter/intra) = {metrics['ratio']:.4f}")
        print(f"  mean_inter          = {metrics['mean_inter']:.4f}")
        print(f"  mean_intra          = {metrics['mean_intra']:.4f}")
        print(f"  silhouette          = {metrics['silhouette']:.4f}")
        print(f"  CH                  = {metrics['ch']:.2f}")
        print(f"  DB                  = {metrics['db']:.4f}")

        rows.append(dict(tag=tag, ckpt=str(ckpt_path), **metrics))

    # 写 csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {out_csv}")

    # 总结对照
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'tag':<16s}  {'ratio':>8s}  {'silhouette':>11s}  "
          f"{'CH':>9s}  {'DB':>7s}")
    for r in rows:
        print(f"{r['tag']:<16s}  {r['ratio']:>8.4f}  "
              f"{r['silhouette']:>11.4f}  {r['ch']:>9.2f}  {r['db']:>7.4f}")

    # t-SNE
    print("\nGenerating t-SNE visualization...")
    make_tsne(feats_dict, labels_ref, args.tsne_fig)


if __name__ == "__main__":
    main()
