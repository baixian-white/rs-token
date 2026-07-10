"""单层 VQ-VAE 模型 (Stage 1 baseline).

架构(对应 256×256 输入):
  Encoder: 256 -> 128 -> 64 -> 32 -> 16, 通道 [3, 64, 128, 256, 256]
  量化层 : VectorQuantize(dim=256, codebook_size=1024)
  Decoder: 16 -> 32 -> 64 -> 128 -> 256, 反卷积+ResBlock

Stage 2 时把 self.quantizer 从 VectorQuantize 换成 ResidualVQ 即可,
其他模块复用. 这就是为什么 quantizer 单独抽出.

设计原则:
  - 最小可行: 不加 attention / GAN / discriminator
  - GroupNorm 而非 BatchNorm: 小 batch 更稳定
  - SiLU 激活: 比 ReLU 略好,且在 latent diffusion 等现代 VAE 实现中通用
  - Decoder 尾部 tanh: 输出范围 [-1, 1] 与归一化一致
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import VectorQuantize, ResidualVQ


@dataclass
class VQVAEConfig:
    in_channels: int = 3
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)  # 每级通道倍数
    num_res_blocks: int = 1
    latent_dim: int = 256
    codebook_size: int = 1024
    commitment_weight: float = 0.25
    # quantizer 类型: "vq" (Stage 1) 或 "rvq" (Stage 2)
    quantizer: str = "vq"
    rvq_num_quantizers: int = 4  # 仅 quantizer="rvq" 时生效


class ResBlock(nn.Module):
    """带 GroupNorm 的残差块."""
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Downsample(nn.Module):
    """2× 下采样: stride=2 conv."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """2× 上采样: 最近邻 + conv 平滑(避免反卷积棋盘效应)."""
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        ch = cfg.base_channels
        self.in_conv = nn.Conv2d(cfg.in_channels, ch, 3, padding=1)

        layers: list[nn.Module] = []
        cur_ch = ch
        for i, mult in enumerate(cfg.channel_multipliers):
            out_ch = cfg.base_channels * mult
            # 通道变换 conv
            layers.append(nn.Conv2d(cur_ch, out_ch, 3, padding=1))
            cur_ch = out_ch
            for _ in range(cfg.num_res_blocks):
                layers.append(ResBlock(cur_ch))
            # 除最后一级外都下采样, 但我们要 4 次下采样 (256->16)
            # 所以全部 4 级都下采样
            layers.append(Downsample(cur_ch))
        self.blocks = nn.Sequential(*layers)

        # 投到 latent_dim
        self.norm_out = nn.GroupNorm(
            num_groups=min(32, cur_ch), num_channels=cur_ch
        )
        self.out_conv = nn.Conv2d(cur_ch, cfg.latent_dim, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.blocks(x)
        x = F.silu(self.norm_out(x))
        x = self.out_conv(x)
        return x  # [B, latent_dim, 16, 16]


class Decoder(nn.Module):
    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        ch_mults = list(reversed(cfg.channel_multipliers))
        cur_ch = cfg.base_channels * ch_mults[0]
        self.in_conv = nn.Conv2d(cfg.latent_dim, cur_ch, 1)

        layers: list[nn.Module] = []
        for i, mult in enumerate(ch_mults):
            out_ch = cfg.base_channels * mult
            layers.append(nn.Conv2d(cur_ch, out_ch, 3, padding=1))
            cur_ch = out_ch
            for _ in range(cfg.num_res_blocks):
                layers.append(ResBlock(cur_ch))
            # 全部 4 级都上采样, 16 -> 256
            layers.append(Upsample(cur_ch))
        self.blocks = nn.Sequential(*layers)

        self.norm_out = nn.GroupNorm(
            num_groups=min(32, cur_ch), num_channels=cur_ch
        )
        self.out_conv = nn.Conv2d(cur_ch, cfg.in_channels, 3, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.blocks(x)
        x = F.silu(self.norm_out(x))
        x = self.out_conv(x)
        return torch.tanh(x)  # [-1, 1]


class VQVAE(nn.Module):
    """Encoder + Quantizer + Decoder.

    forward 返回 dict 便于训练循环灵活处理:
      'recon'   : 重建图像 [-1, 1]
      'indices' : codebook 索引, Stage 1 形状 [B, T] (T=16*16=256),
                  Stage 2 形状 [B, T, num_quantizers]
      'vq_loss' : 量化 commitment loss (标量)
      'z_pre'   : 量化前特征 [B, T, latent_dim] (用于蒸馏阶段对接)
      'zq'      : 量化后特征 [B, T, latent_dim]
    """

    def __init__(self, cfg: VQVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

        if cfg.quantizer == "vq":
            self.quantizer = VectorQuantize(
                dim=cfg.latent_dim,
                codebook_size=cfg.codebook_size,
                commitment_weight=cfg.commitment_weight,
                kmeans_init=True,
                threshold_ema_dead_code=2,
            )
        elif cfg.quantizer == "rvq":
            self.quantizer = ResidualVQ(
                dim=cfg.latent_dim,
                num_quantizers=cfg.rvq_num_quantizers,
                codebook_size=cfg.codebook_size,
                commitment_weight=cfg.commitment_weight,
                kmeans_init=True,
                threshold_ema_dead_code=2,
                quantize_dropout=True,
                quantize_dropout_cutoff_index=1,
            )
        else:
            raise ValueError(f"Unknown quantizer: {cfg.quantizer}")

    def forward(self, x: torch.Tensor) -> dict:
        # Encoder
        z = self.encoder(x)                                  # [B, C, H, W]
        b, c, h, w = z.shape
        z_seq = rearrange(z, "b c h w -> b (h w) c")         # [B, T, C]

        # Quantize
        if self.cfg.quantizer == "vq":
            zq_seq, indices, vq_loss = self.quantizer(z_seq)
            # indices: [B, T]
        else:
            # ResidualVQ 返回 (zq, indices, all_losses)
            zq_seq, indices, all_losses = self.quantizer(z_seq)
            # indices: [B, T, num_quantizers]
            # all_losses: [num_quantizers] —— 把它们求和当 vq_loss
            vq_loss = all_losses.sum() if torch.is_tensor(all_losses) \
                      else sum(all_losses)

        # Reshape back
        zq = rearrange(zq_seq, "b (h w) c -> b c h w", h=h, w=w)

        # Decode
        recon = self.decoder(zq)

        # 仅 RVQ: 额外构造带 STE 的层级量化特征供蒸馏使用
        # 量化值无梯度, 但通过 STE 把梯度透传到 z_seq, 进而到 encoder
        zq_l0_ste = None
        zq_l1_ste = None
        if self.cfg.quantizer == "rvq":
            l0_layer = self.quantizer.layers[0]
            l0_cb = l0_layer._codebook.embed              # [1, K, D] or [K, D]
            if l0_cb.dim() == 3:
                l0_cb = l0_cb[0]                          # [K, D]
            l0_idx = indices[..., 0]                      # [B, T]
            zq_l0_q = l0_cb[l0_idx]                       # [B, T, D] no grad
            zq_l0_ste = z_seq + (zq_l0_q - z_seq).detach()

            # E31: L1-only quantized representation with STE.
            # By symmetry with zq_l0_ste, this exposes the L1 codeword alone
            # (NOT cumulative L0+L1). The forward value is the L1 residual
            # codeword; STE routes the gradient back through z_seq → encoder.
            # When used as the distill target, this is the L1-only
            # counterfactual to L0-only distillation: it asks "if the same
            # supervision is moved one layer down, does the semantics also
            # move down?" — testing whether L0 is privileged or arbitrary.
            if self.cfg.rvq_num_quantizers >= 2:
                l1_layer = self.quantizer.layers[1]
                l1_cb = l1_layer._codebook.embed
                if l1_cb.dim() == 3:
                    l1_cb = l1_cb[0]
                l1_idx = indices[..., 1]
                zq_l1_q = l1_cb[l1_idx]                   # [B, T, D] no grad
                zq_l1_ste = z_seq + (zq_l1_q - z_seq).detach()

        return {
            "recon": recon,
            "indices": indices,
            "vq_loss": vq_loss,
            "z_pre": z_seq,
            "zq": zq_seq,
            "zq_l0_ste": zq_l0_ste,
            "zq_l1_ste": zq_l1_ste,
        }

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """仅返回 codebook 索引, 推理时用."""
        return self.forward(x)["indices"]

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor,
                            spatial: tuple[int, int] = (16, 16)) -> torch.Tensor:
        """从索引重建图像. 推理 / 信道仿真后端用.

        indices: VQ 时 [B, T], RVQ 时 [B, T, L].
        """
        if self.cfg.quantizer == "vq":
            # VectorQuantize 提供 get_output_from_indices
            zq_seq = self.quantizer.get_output_from_indices(indices)
        else:
            zq_seq = self.quantizer.get_output_from_indices(indices)
        h, w = spatial
        zq = rearrange(zq_seq, "b (h w) c -> b c h w", h=h, w=w)
        return self.decoder(zq)

    def num_parameters(self) -> dict[str, int]:
        def cnt(m): return sum(p.numel() for p in m.parameters())
        return {
            "encoder"  : cnt(self.encoder),
            "decoder"  : cnt(self.decoder),
            "quantizer": cnt(self.quantizer),
            "total"    : cnt(self),
        }


if __name__ == "__main__":
    # 自检: python -m models.vqvae
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    for q in ["vq", "rvq"]:
        cfg = VQVAEConfig(quantizer=q)
        model = VQVAE(cfg).cuda()
        x = torch.randn(2, 3, 256, 256, device="cuda")
        out = model(x)
        print(f"\n[{q}] params: "
              f"{model.num_parameters()['total'] / 1e6:.2f} M")
        print(f"  recon  : {tuple(out['recon'].shape)}")
        print(f"  indices: {tuple(out['indices'].shape)}")
        print(f"  vq_loss: {out['vq_loss'].item():.4f}")
        print(f"  z_pre  : {tuple(out['z_pre'].shape)}")

        # 自检 backward 通
        loss = F.l1_loss(out["recon"], x) + out["vq_loss"]
        loss.backward()
        print(f"  backward OK")
