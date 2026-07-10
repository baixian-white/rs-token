"""E41 — Direct-quantize-RemoteCLIP control model.

Architecture:
    Image (256×256) ── RemoteCLIP visual encoder ── 512-d global embedding
                                                        │
                                                        ▼
                                       Linear(512 → 2048) + LayerNorm
                                                        │
                                       reshape to 256 tokens × 8-dim
                                                        │
                                                        ▼
                            ResidualVQ(num_quantizers=4, codebook_size=1024,
                                       dim=8) ── indices [B, 256, 4]
                                                        │
                                                        ▼
                                              quantized z_q [B, 256, 8]
                                                        │
                            reshape [B, 16, 16, 8] → [B, 8, 16, 16]
                                                        │
                                                        ▼
                                Decoder (mirror RS-Token's): 16→32→64→128→256
                                                        │
                                                        ▼
                                            recon image [-1, 1]

Bit budget matching RS-Token exactly:
    256 patches × K × 10 bits/code = 2,560 K bits/image
    K=1 → 2,560 bits  /  K=4 → 10,240 bits

Notes:
- The RemoteCLIP encoder is FROZEN (eval mode, no grad). Only the
  up-project head, the RVQ quantizer, and the decoder are trained.
- The "L0 task path" for this model is defined as: re-fit a logistic-regression
  classifier on the L0 codebook indices (same h₀/L0_bow protocol as RS-Token).
- Training loss is L1 + LPIPS + vq_loss (no separate distillation loss
  because the input itself is RemoteCLIP — distillation is implicit by
  construction).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import ResidualVQ

# Re-use the existing decoder building blocks; avoids divergence in conv stack.
from models.vqvae import ResBlock, Upsample
from models.distillation import RemoteCLIPTeacher


@dataclass
class DirectClipQuantizerConfig:
    # RemoteCLIP teacher
    teacher_ckpt: str = ""
    teacher_name: str = "ViT-B-32"
    teacher_dim: int = 512                  # ViT-B-32 default; auto-checked

    # Reshape grid: 256 tokens × token_dim
    grid_size: int = 16                     # 16×16 = 256 tokens
    token_dim: int = 8                      # 256 × 8 = 2048

    # RVQ (matches RS-Token bit budget when codebook_size=1024 + K=4)
    codebook_size: int = 1024
    rvq_num_quantizers: int = 4
    commitment_weight: float = 0.25

    # Decoder mirrors RS-Token (latent_dim=8 here, channels expand back to 256)
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 1
    out_channels: int = 3


class DirectClipDecoder(nn.Module):
    """Mirror of RS-Token's Decoder, but `latent_dim=token_dim` (typically 8)."""

    def __init__(self, cfg: DirectClipQuantizerConfig):
        super().__init__()
        ch_mults = list(reversed(cfg.channel_multipliers))
        cur_ch = cfg.base_channels * ch_mults[0]
        self.in_conv = nn.Conv2d(cfg.token_dim, cur_ch, 1)

        layers: list[nn.Module] = []
        for mult in ch_mults:
            out_ch = cfg.base_channels * mult
            layers.append(nn.Conv2d(cur_ch, out_ch, 3, padding=1))
            cur_ch = out_ch
            for _ in range(cfg.num_res_blocks):
                layers.append(ResBlock(cur_ch))
            layers.append(Upsample(cur_ch))
        self.blocks = nn.Sequential(*layers)

        self.norm_out = nn.GroupNorm(
            num_groups=min(32, cur_ch), num_channels=cur_ch
        )
        self.out_conv = nn.Conv2d(cur_ch, cfg.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.blocks(x)
        x = F.silu(self.norm_out(x))
        x = self.out_conv(x)
        return torch.tanh(x)


class DirectClipQuantizer(nn.Module):
    """RemoteCLIP encoder → up-project → RVQ → decoder.

    forward returns dict with same keys as VQVAE so eval scripts can be
    near-direct (`indices`, `recon`, `zq`, `vq_loss`, `z_pre`).
    """

    def __init__(self, cfg: DirectClipQuantizerConfig):
        super().__init__()
        self.cfg = cfg

        # Teacher (frozen). Importing RemoteCLIPTeacher here means the model
        # loads its weights once at construction time.
        self.teacher = RemoteCLIPTeacher(
            ckpt_path=cfg.teacher_ckpt,
            model_name=cfg.teacher_name,
        )
        if self.teacher.embed_dim != cfg.teacher_dim:
            print(f"  [warn] teacher_dim={self.teacher.embed_dim} differs "
                  f"from config {cfg.teacher_dim}; using actual.")
        teacher_dim = self.teacher.embed_dim

        # Up-project: 512 → 16×16×8 = 2048
        target_dim = cfg.grid_size * cfg.grid_size * cfg.token_dim
        self.up_proj = nn.Sequential(
            nn.Linear(teacher_dim, target_dim),
            nn.LayerNorm(target_dim),
        )

        # RVQ on the 8-d token sequence
        self.quantizer = ResidualVQ(
            dim=cfg.token_dim,
            num_quantizers=cfg.rvq_num_quantizers,
            codebook_size=cfg.codebook_size,
            commitment_weight=cfg.commitment_weight,
            kmeans_init=True,
            threshold_ema_dead_code=2,
            quantize_dropout=True,
            quantize_dropout_cutoff_index=1,
        )

        self.decoder = DirectClipDecoder(cfg)

    @property
    def device(self) -> torch.device:
        return next(self.up_proj.parameters()).device

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] image → 256 tokens × token_dim (pre-quantization)."""
        with torch.no_grad():
            t_emb = self.teacher.encode_image(x)            # [B, teacher_dim]
        # Move to the up-proj layer's dtype (bf16 under AMP)
        z = self.up_proj(t_emb.to(next(self.up_proj.parameters()).dtype))
        # [B, 2048] → [B, 256, 8]
        z_seq = rearrange(
            z, "b (t c) -> b t c",
            t=self.cfg.grid_size * self.cfg.grid_size,
            c=self.cfg.token_dim,
        )
        return z_seq

    def decode_from_zq(self, zq_seq: torch.Tensor) -> torch.Tensor:
        """[B, 256, 8] quantized → recon image."""
        z_2d = rearrange(
            zq_seq, "b (h w) c -> b c h w",
            h=self.cfg.grid_size, w=self.cfg.grid_size,
        )
        return self.decoder(z_2d)

    def forward(self, x: torch.Tensor) -> dict:
        z_seq = self.encode_features(x)                         # [B, T, C]
        zq_seq, indices, all_losses = self.quantizer(z_seq)
        # all_losses: tensor [num_quantizers] or list-of-scalars
        vq_loss = (
            all_losses.sum() if torch.is_tensor(all_losses)
            else sum(all_losses)
        )
        recon = self.decode_from_zq(zq_seq)
        return {
            "recon": recon,
            "indices": indices,                                 # [B, T, K]
            "vq_loss": vq_loss,
            "z_pre": z_seq,
            "zq": zq_seq,
        }

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["indices"]

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor,
                            spatial: tuple[int, int] | None = None) -> torch.Tensor:
        """Mirror of VQVAE.decode_from_indices for eval-script reuse.

        `indices`: [B, T, k] (any prefix length 1..K).
        """
        zq_seq = self.quantizer.get_output_from_indices(indices)
        return self.decode_from_zq(zq_seq)

    def num_parameters(self) -> dict[str, int]:
        def cnt(m): return sum(p.numel() for p in m.parameters())

        teacher_params = cnt(self.teacher)
        up_params = cnt(self.up_proj)
        q_params = cnt(self.quantizer)
        dec_params = cnt(self.decoder)
        # "trainable" = everything except the frozen teacher
        trainable = up_params + q_params + dec_params
        return {
            "teacher_frozen": teacher_params,
            "up_proj": up_params,
            "quantizer": q_params,
            "decoder": dec_params,
            "trainable": trainable,
            "total_with_teacher": teacher_params + trainable,
        }


if __name__ == "__main__":
    # python -m models.direct_clip_quantizer
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    cfg = DirectClipQuantizerConfig(
        teacher_ckpt="checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt",
    )
    model = DirectClipQuantizer(cfg).cuda()
    np_info = model.num_parameters()
    for k, v in np_info.items():
        print(f"  {k:20s}: {v/1e6:.3f} M")

    x = torch.randn(2, 3, 256, 256, device="cuda").clamp(-1, 1)
    out = model(x)
    print(f"  recon  : {tuple(out['recon'].shape)}")
    print(f"  indices: {tuple(out['indices'].shape)}")
    print(f"  zq     : {tuple(out['zq'].shape)}")
    print(f"  vq_loss: {out['vq_loss'].item():.4f}")

    loss = F.l1_loss(out["recon"], x) + out["vq_loss"]
    loss.backward()
    print("  backward OK")
