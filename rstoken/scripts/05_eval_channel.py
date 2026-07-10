"""Stage 4 信道仿真 + 优雅降级验证.

方案:
  - 每个 codebook index 占 10 bit (log2 1024). RVQ 4 层 -> 每空间位置 40 bit.
  - 数字调制: BPSK (一比特一符号), AWGN 信道.
    SNR_dB 决定每比特误码率 BER = Q(sqrt(2 * 10^(SNR/10))).
  - 比特错乱后, 索引随之错码 -> 接收端用错的 codebook 项重建.
  - 传输策略 k=1..4: 只发前 k 层索引, 后 4-k 层不传.
    ResidualVQ 自身支持 indices 列数 < num_quantizers, 余下层数被自动 skip.

输出:
  table[snr_db][k] -> {'psnr', 'lpips', 'l0_emb_acc', 'l0_bow_acc', 'zq_acc'}
  写入 csv + 控制台.

启动:
    conda activate rstoken
    python scripts/05_eval_channel.py --ckpt checkpoints/rvq_distill/best.pt
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
import time
from pathlib import Path

# Keep sklearn/BLAS single-threaded on Windows; previous seed-sweep runs
# sometimes exited in native code while fitting the L0_bow probe.
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

from models.datasets import (                            # noqa: E402
    AIDConfig, AIDDataset, build_loaders, build_transforms, denormalize,
)
from models.vqvae import VQVAE, VQVAEConfig              # noqa: E402
from models.perceptual import LPIPSLoss                  # noqa: E402
from torch.utils.data import DataLoader                  # noqa: E402

LOGREG_MAX_ITER = int(os.environ.get("RSTOKEN_LOGREG_MAX_ITER", "300"))
LOGREG_TOL = float(os.environ.get("RSTOKEN_LOGREG_TOL", "1e-3"))
CHANNEL_PROBE_FEATURES = [
    item.strip()
    for item in os.environ.get("RSTOKEN_CHANNEL_PROBE_FEATURES", "L0_bow").split(",")
    if item.strip()
]
LOGREG_CACHE: dict[tuple, tuple[StandardScaler, LogisticRegression]] = {}

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )


# ---------------------------------------------------------------------------
# 信道模拟 (BPSK over AWGN / Rayleigh, hard decision, 闭式 BER)
# ---------------------------------------------------------------------------

def ber_from_snr(snr_db: float, channel_type: str = "awgn") -> float:
    """BPSK 硬判决在不同信道下的闭式平均 BER.

    awgn:     BER = Q(sqrt(2 * SNR_lin)) = 0.5 * erfc(sqrt(SNR_lin))
    rayleigh: |h|^2 ~ Exp(1), 对瞬时 BER 在 |h| 上求期望得到
              BER = 0.5 * (1 - sqrt(SNR_lin / (1 + SNR_lin)))
              (Proakis & Salehi, 经典结论)
    """
    snr_lin = 10 ** (snr_db / 10.0)
    if channel_type == "awgn":
        return 0.5 * math.erfc(math.sqrt(snr_lin))
    elif channel_type == "rayleigh":
        return 0.5 * (1.0 - math.sqrt(snr_lin / (1.0 + snr_lin)))
    else:
        raise ValueError(f"unknown channel_type: {channel_type}")


def corrupt_indices(
    indices: torch.Tensor, codebook_size: int, ber: float,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """对每个索引按 BER 翻 bit. indices 任意形状, dtype long.

    1. 拆 indices -> bits 长度 ceil(log2 cb_size) (=10 for 1024).
    2. 每个 bit 独立按 BER 概率翻转.
    3. 重新组装 -> indices, clamp 到合法区间 (越界视为最后一个码字).
    """
    bits_per = int(math.ceil(math.log2(codebook_size)))
    shape = indices.shape
    flat = indices.reshape(-1)
    n = flat.numel()

    # 拆 bits: [n, bits_per]
    bit_pos = torch.arange(bits_per, device=flat.device)
    bit_mat = (flat.unsqueeze(1) >> bit_pos) & 1   # [n, bits_per] long {0,1}

    # 信道翻转
    if ber > 0:
        flip = torch.rand(bit_mat.shape, device=flat.device, generator=rng) < ber
        bit_mat = bit_mat ^ flip.long()

    # 重组
    weights = (1 << bit_pos).long()
    new_idx = (bit_mat * weights).sum(dim=1)        # [n]
    # clamp 到 [0, codebook_size-1] (1024 这个特殊值: 10 bit 全 1 = 1023, 不会越界)
    new_idx = new_idx.clamp(0, codebook_size - 1)

    return new_idx.reshape(shape)


# ---------------------------------------------------------------------------
# 特征 + 重建提取 (单次 encoder/quantizer forward, 缓存复用)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_clean_indices(model: VQVAE, loader, device: str):
    """对一个 split 提取干净的 RVQ indices + 标签. 不做信道扰动."""
    model.eval()
    all_indices, all_labels = [], []
    all_x = []   # 留作 PSNR/LPIPS 参考
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model.encoder(x)
        b, c, h, w = z.shape
        z_seq = rearrange(z, "b c h w -> b (h w) c")
        _, indices, _ = model.quantizer(z_seq)             # [B, T, L]
        all_indices.append(indices.cpu())
        all_labels.append(y)
        all_x.append(x.cpu())
    return (
        torch.cat(all_indices),      # [N, T, L]
        torch.cat(all_labels),       # [N]
        torch.cat(all_x),            # [N, 3, H, W]
        (h, w),
    )


@torch.no_grad()
def decode_from_corrupted(
    model: VQVAE, indices_kept: torch.Tensor, spatial: tuple[int, int],
    batch_size: int = 32, device: str = "cuda",
):
    """从 (可能被信道污染的) 前 k 层索引解码出图像 + zq + L0 embedding.

    indices_kept: [N, T, k]  (k <= L)
    返回 dict:
      recon : [N, 3, H, W]
      zq    : [N, T, latent_dim]   送入 decoder 之前的特征
      l0_emb: [N, T, latent_dim]   只查 L0 codebook 得到的 emb
    """
    model.eval()
    N = indices_kept.shape[0]
    h, w = spatial
    recon_all, zq_all, l0_all = [], [], []

    # L0 codebook 提取一次
    l0_layer = model.quantizer.layers[0]
    l0_cb = l0_layer._codebook.embed
    if l0_cb.dim() == 3:
        l0_cb = l0_cb[0]   # [K, D]
    l0_cb = l0_cb.to(device)

    for i in range(0, N, batch_size):
        idx = indices_kept[i:i+batch_size].to(device)      # [b, T, k]
        # ResidualVQ 自带 indices 列数<L 时的处理 (line 316-318 of residual_vq.py)
        zq_seq = model.quantizer.get_output_from_indices(idx)  # [b, T, dim]
        zq = rearrange(zq_seq, "b (h w) c -> b c h w", h=h, w=w)
        recon = model.decoder(zq)

        # L0 emb (只看第 0 层索引, 用 L0 codebook 查表)
        l0_idx = idx[..., 0]                                # [b, T]
        l0_emb = l0_cb[l0_idx]                              # [b, T, D]

        recon_all.append(recon.cpu())
        zq_all.append(zq_seq.cpu())
        l0_all.append(l0_emb.cpu())

    return {
        "recon": torch.cat(recon_all),
        "zq":    torch.cat(zq_all),
        "l0":    torch.cat(l0_all),
    }


# ---------------------------------------------------------------------------
# 评测指标
# ---------------------------------------------------------------------------

def psnr_full(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
    return (10 * torch.log10(4.0 / (mse + 1e-12))).mean().item()


def fit_logreg(name: str, X_tr, y_tr, X_te, y_te) -> float:
    """返回 test top-1 acc."""
    cache_key = (
        name,
        X_tr.shape,
        y_tr.shape,
        float(np.asarray(X_tr[: min(16, len(X_tr))]).sum()),
        int(np.asarray(y_tr).sum()),
    )
    if cache_key in LOGREG_CACHE:
        scaler, clf = LOGREG_CACHE[cache_key]
    else:
        scaler = StandardScaler(with_mean=name != "L0_bow")
        X_tr_s = scaler.fit_transform(X_tr)
        clf = LogisticRegression(
            max_iter=LOGREG_MAX_ITER,
            tol=LOGREG_TOL,
            C=1.0,
            solver="lbfgs",
            n_jobs=1,
        )
        clf.fit(X_tr_s, y_tr)
        LOGREG_CACHE[cache_key] = (scaler, clf)
    X_te_s = scaler.transform(X_te)
    return clf.score(X_te_s, y_te)


def indices_to_features(
    indices_kept: torch.Tensor, l0_codebook_cpu: torch.Tensor,
    codebook_size: int,
    model: VQVAE, spatial: tuple[int,int],
    batch_size: int, device: str,
):
    """从 (污染后) 索引提取三种分类特征:
      L0_bow    : [N, codebook_size]
      L0_emb    : [N, latent_dim]
      zq_pool   : [N, latent_dim]   (用前 k 层 zq pool, 与重建一致)
    """
    N, T, k = indices_kept.shape
    bow_all, l0_emb_all, zq_all = [], [], []

    for i in range(0, N, batch_size):
        idx = indices_kept[i:i+batch_size]                  # [b, T, k]
        l0_idx = idx[..., 0]                                # [b, T]
        # L0 BoW
        bow = torch.zeros(l0_idx.shape[0], codebook_size)
        bow.scatter_add_(1, l0_idx, torch.ones_like(l0_idx, dtype=torch.float))
        bow = bow / T
        bow_all.append(bow)
        # L0 emb
        l0_emb = l0_codebook_cpu[l0_idx]                    # [b, T, D]
        l0_emb_all.append(l0_emb.mean(dim=1))               # [b, D]
        # zq pool: 用 model 的 partial decode 得到 zq
        idx_dev = idx.to(device)
        with torch.no_grad():
            zq_seq = model.quantizer.get_output_from_indices(idx_dev)
        zq_all.append(zq_seq.mean(dim=1).cpu())            # [b, D]

    return {
        "L0_bow":  torch.cat(bow_all).numpy(),
        "L0_emb":  torch.cat(l0_emb_all).numpy(),
        "zq_pool": torch.cat(zq_all).numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--snrs", default="-10,-5,0,5,10,inf",
                        help="逗号分隔, inf 表示无信道")
    parser.add_argument("--ks", default="1,2,3,4",
                        help="只传前 k 层")
    parser.add_argument("--out_csv", default="logs/stage4_channel_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channel_seed", type=int, default=None,
                        help="信道翻 bit 用的随机种子. 不传则沿用 --seed. "
                             "E9 多种子复测时显式传 0/1/2.")
    parser.add_argument("--channel_type", choices=["awgn", "rayleigh"],
                        default="awgn",
                        help="信道模型. AWGN 用闭式 Q 函数, "
                             "Rayleigh 用 |h|^2~Exp(1) 上平均的闭式 BER.")
    args = parser.parse_args()

    device = args.device
    channel_seed = args.channel_seed if args.channel_seed is not None else args.seed
    torch.manual_seed(channel_seed)
    np.random.seed(channel_seed)
    print(f"  channel_seed = {channel_seed}")
    print(f"  channel_type = {args.channel_type}")

    print("=" * 60)
    print(f"Stage 4 channel + graceful degradation eval")
    print(f"ckpt: {args.ckpt}")
    print("=" * 60)

    # 加载 ckpt
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    model_cfg = VQVAEConfig(**cfg_dict["model"])
    assert model_cfg.quantizer == "rvq", "Stage 4 仅支持 RVQ ckpt"
    L = model_cfg.rvq_num_quantizers
    cb_size = model_cfg.codebook_size

    model = VQVAE(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  model: rvq L={L} cb={cb_size} latent={model_cfg.latent_dim}")

    # LPIPS
    lpips_fn = LPIPSLoss(net=cfg_dict["loss"]["lpips_net"]).to(device).eval()

    # DataLoaders: train 用 center crop 的非 aug 版本
    data_cfg = AIDConfig(**cfg_dict["data"])
    train_eval_ds = AIDDataset(
        f"{data_cfg.splits_dir}/train.csv",
        transform=build_transforms(data_cfg.image_size, train=False),
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=data_cfg.batch_size,
        shuffle=False, num_workers=data_cfg.num_workers,
        pin_memory=True, drop_last=False,
    )
    loaders = build_loaders(data_cfg)

    # 1) 一次性 encode 干净索引 (train + test)
    print("\n[1/3] encoding clean indices on TRAIN split (no aug)")
    t0 = time.time()
    tr_idx, y_tr, x_tr, spatial = encode_clean_indices(
        model, train_eval_loader, device
    )
    print(f"  done in {time.time()-t0:.1f}s, "
          f"indices={tuple(tr_idx.shape)}, spatial={spatial}")

    print("\n[2/3] encoding clean indices on TEST split")
    t0 = time.time()
    te_idx, y_te, x_te, _ = encode_clean_indices(
        model, loaders["test"], device
    )
    print(f"  done in {time.time()-t0:.1f}s, indices={tuple(te_idx.shape)}")

    # L0 codebook (CPU 拷贝, 节省 GPU 内存)
    l0_layer = model.quantizer.layers[0]
    l0_cb_cpu = l0_layer._codebook.embed
    if l0_cb_cpu.dim() == 3:
        l0_cb_cpu = l0_cb_cpu[0]
    l0_cb_cpu = l0_cb_cpu.detach().cpu()

    # 2) 跑组合
    snrs = [(s.strip()) for s in args.snrs.split(",")]
    ks = [int(k) for k in args.ks.split(",")]

    rows = []
    print(f"\n[3/3] sweeping {len(snrs)} SNRs × {len(ks)} k values "
          f"= {len(snrs)*len(ks)} cells")

    for snr_str in snrs:
        if snr_str == "inf":
            ber = 0.0
            snr_disp = "inf"
        else:
            snr_db = float(snr_str)
            ber = ber_from_snr(snr_db, args.channel_type)
            snr_disp = f"{snr_db:+.0f}dB"

        for k in ks:
            t_cell = time.time()

            # 截取前 k 层
            tr_idx_k = tr_idx[..., :k].clone()
            te_idx_k = te_idx[..., :k].clone()

            # 信道污染 (test 集才被信道污染; train 用于训分类器, 用干净索引)
            #   理由: 接收端在部署时, 分类器是在干净训练数据上拟合的;
            #         我们想测试的是 "测试时分布偏移" 在信道下的表现.
            if ber > 0:
                te_idx_k = corrupt_indices(te_idx_k, cb_size, ber)

            # 重建 (test) 用于 PSNR / LPIPS
            test_recon_pack = decode_from_corrupted(
                model, te_idx_k, spatial,
                batch_size=64, device=device,
            )
            recon_te = test_recon_pack["recon"]   # [N, 3, H, W]

            # PSNR / LPIPS (vs 原 test 图)
            psnr_v = psnr_full(recon_te, x_te)
            with torch.no_grad():
                # LPIPS 一批批算 (避免显存爆)
                lpips_vals = []
                for i in range(0, recon_te.shape[0], 32):
                    a = recon_te[i:i+32].to(device)
                    b = x_te[i:i+32].to(device)
                    lpips_vals.append(lpips_fn(a, b).item())
                lpips_v = float(np.mean(lpips_vals))

            # 三种分类特征 (train 干净, test 污染)
            tr_feats = indices_to_features(
                tr_idx_k, l0_cb_cpu, cb_size, model, spatial,
                batch_size=64, device=device,
            )
            te_feats = indices_to_features(
                te_idx_k, l0_cb_cpu, cb_size, model, spatial,
                batch_size=64, device=device,
            )
            valid_features = {"L0_bow", "L0_emb", "zq_pool"}
            unknown = sorted(set(CHANNEL_PROBE_FEATURES) - valid_features)
            if unknown:
                raise ValueError(f"unknown channel probe features: {unknown}")
            accs = {
                name: fit_logreg(
                    name, tr_feats[name], y_tr.numpy(),
                    te_feats[name], y_te.numpy()
                )
                for name in CHANNEL_PROBE_FEATURES
            }

            elapsed = time.time() - t_cell
            row = {
                "snr": snr_disp,
                "ber": f"{ber:.4f}",
                "k": k,
                "bits_per_token": k * int(math.ceil(math.log2(cb_size))),
                "psnr": round(psnr_v, 3),
                "lpips": round(lpips_v, 4),
                "L0_bow_acc": round(accs["L0_bow"] * 100, 2) if "L0_bow" in accs else "",
                "L0_emb_acc": round(accs["L0_emb"] * 100, 2) if "L0_emb" in accs else "",
                "zq_pool_acc": round(accs["zq_pool"] * 100, 2) if "zq_pool" in accs else "",
            }
            rows.append(row)
            l0_emb_text = f"L0_emb={accs['L0_emb']*100:5.2f}%  " if "L0_emb" in accs else ""
            zq_text = f"zq={accs['zq_pool']*100:5.2f}%  " if "zq_pool" in accs else ""
            print(f"  SNR={snr_disp:>6s}  k={k}  ber={ber:.4f}  "
                  f"psnr={psnr_v:.2f}  lpips={lpips_v:.3f}  "
                  f"L0_bow={accs['L0_bow']*100:5.2f}%  "
                  f"{l0_emb_text}"
                  f"{zq_text}"
                  f"({elapsed:.1f}s)")

    # 写 csv
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {out_path}")

    # 打印矩阵视图: L0_bow_acc by (snr, k)
    print("\n" + "=" * 60)
    print("Matrix: L0_bow test acc (%) by SNR x k")
    print("=" * 60)
    snr_set = list(dict.fromkeys(r["snr"] for r in rows))
    print(f"  {'SNR':>8s}  " + "  ".join(f"k={k:1d}" for k in ks))
    for s in snr_set:
        cells = []
        for k in ks:
            v = next(r["L0_bow_acc"] for r in rows
                     if r["snr"] == s and r["k"] == k)
            cells.append(f"{v:5.2f}")
        print(f"  {s:>8s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
