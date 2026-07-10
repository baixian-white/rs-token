"""ADJSCC — Attention-based Deep Joint Source-Channel Coding.

Reference: Xu, Wang, Gao, Zhao, "Wireless Image Transmission Using Deep
Source Channel Coding With Attention Modules", IEEE TCSVT 2022.

Key idea: a single end-to-end CNN encoder/decoder pair that maps an image
to a fixed-length real-valued vector x ∈ R^N transmitted directly over an
AWGN/Rayleigh channel; both encoder and decoder are conditioned on the SNR
through Attention Feature (AF) blocks so the same model handles a wide
SNR range.

Strict alignment with RS-Token's transmitted-bit budget:
- ADJSCC sends N real symbols over a real BPSK-equivalent channel.
- Each real symbol is one channel use; under BPSK at the same SNR the noise
  variance σ² = 1/(2·SNR_lin) matches RS-Token's bit-channel exactly
  (RS-Token: N_bits real symbols with σ² = 1/(2·SNR); ADJSCC: same).
- We therefore pick the encoder to output N = bits_per_image real symbols
  for each rate point, so RS-Token at k bits and ADJSCC at the same N use
  the same number of channel uses under the same noise process.

Spatial layout: encoder downsamples 256×256 → 16×16 with C output channels;
N = 16·16·C = 256·C. We pick C ∈ {10, 20, 40} for bits/image
∈ {2560, 5120, 10240}.

Power constraint: after encoding, the N-dim real symbol vector is
normalised to unit average symbol energy (Es = 1) to match RS-Token's
BPSK convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ADJSCCConfig:
    image_size: int = 256
    in_channels: int = 3
    base_channels: int = 64
    n_symbol_channels: int = 10        # C; total real symbols = 256 * C
    af_hidden: int = 64
    train_snr_min: float = -2.0        # SNR range used during training (dB)
    train_snr_max: float = 12.0
    # "mixed" = randomly draw AWGN or Rayleigh per sample at training time;
    # this matches the ADJSCC paper's protocol for fading-trained models.
    train_channel: Literal["awgn", "rayleigh", "mixed"] = "mixed"


# ---------------------------------------------------------------------------
# Attention Feature (AF) block — SNR-conditioned channel reweighting.
# Original ADJSCC concatenates [global avg pool, snr] → MLP → sigmoid gate.
# ---------------------------------------------------------------------------
class AFBlock(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gap = x.mean(dim=(2, 3))                              # [B, C]
        # snr_db: [B] in dB; broadcast to [B,1]
        s = snr_db.view(b, 1).to(x.dtype)
        z = torch.cat([gap, s], dim=1)                        # [B, C+1]
        g = self.gate(z).view(b, c, 1, 1)
        return x * g


def _conv_block(in_c: int, out_c: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
        nn.PReLU(),
    )


def _deconv_block(in_c: int, out_c: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                           stride=stride, padding=1),
        nn.PReLU(),
    )


# ---------------------------------------------------------------------------
# Encoder: 256 -> 128 -> 64 -> 32 -> 16, each stage followed by an AF block.
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, cfg: ADJSCCConfig):
        super().__init__()
        c0 = cfg.base_channels
        ch = [cfg.in_channels, c0, c0 * 2, c0 * 4, c0 * 4]
        self.stages = nn.ModuleList([
            _conv_block(ch[0], ch[1], 2),
            _conv_block(ch[1], ch[2], 2),
            _conv_block(ch[2], ch[3], 2),
            _conv_block(ch[3], ch[4], 2),
        ])
        self.afs = nn.ModuleList([
            AFBlock(ch[1], cfg.af_hidden),
            AFBlock(ch[2], cfg.af_hidden),
            AFBlock(ch[3], cfg.af_hidden),
            AFBlock(ch[4], cfg.af_hidden),
        ])
        self.proj = nn.Conv2d(ch[4], cfg.n_symbol_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        for stage, af in zip(self.stages, self.afs):
            x = stage(x)
            x = af(x, snr_db)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Decoder: 16 -> 32 -> 64 -> 128 -> 256, mirror of Encoder.
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, cfg: ADJSCCConfig):
        super().__init__()
        c0 = cfg.base_channels
        ch = [cfg.n_symbol_channels, c0 * 4, c0 * 4, c0 * 2, c0]
        self.proj = nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1)
        self.stages = nn.ModuleList([
            _deconv_block(ch[1], ch[2], 2),
            _deconv_block(ch[2], ch[3], 2),
            _deconv_block(ch[3], ch[4], 2),
        ])
        self.afs = nn.ModuleList([
            AFBlock(ch[1], cfg.af_hidden),
            AFBlock(ch[2], cfg.af_hidden),
            AFBlock(ch[3], cfg.af_hidden),
            AFBlock(ch[4], cfg.af_hidden),
        ])
        self.tail = nn.Sequential(
            nn.ConvTranspose2d(ch[4], cfg.in_channels, kernel_size=4,
                               stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.afs[0](x, snr_db)
        for stage, af in zip(self.stages, self.afs[1:]):
            x = stage(x)
            x = af(x, snr_db)
        return self.tail(x)


# ---------------------------------------------------------------------------
# Power normalisation (Es = 1 per real symbol, matching BPSK convention).
# We use a global per-sample normalisation: ||x||² = N → average symbol
# power E[|x_i|²] = 1.
# ---------------------------------------------------------------------------
def normalise_power(x: torch.Tensor) -> torch.Tensor:
    n = x.flatten(1).shape[1]
    norm = x.flatten(1).pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    scale = (n ** 0.5) / norm
    return x * scale.view(-1, *([1] * (x.ndim - 1)))


# ---------------------------------------------------------------------------
# Channel layer.
#
# We send N real symbols per image (matched to RS-Token's bit budget under
# BPSK). For Rayleigh we pair adjacent real symbols into N/2 *complex*
# symbols and apply complex flat-fading + coherent equalisation at the
# receiver — the standard ADJSCC convention from Xu et al. 2022.
#
# Noise variance in both AWGN and Rayleigh modes is calibrated so that, at
# the same SNR_lin, an equivalent BPSK link sees the same per-bit SNR as
# RS-Token's index channel (σ² = 1/(2·SNR_lin) per real dim). Concretely:
#   - AWGN:     y_real = x_real + n_real,   n_real ~ N(0, 1/(2 SNR))
#   - Rayleigh: pair (x_2k, x_2k+1) into complex symbol s_k;
#                y_k = h_k s_k + n_k,  h_k ~ CN(0,1),  n_k ~ CN(0, 1/SNR)
#                receiver returns real/imag of y_k / h_k (coherent eq).
# Average symbol energy E[|x|²] = 1 is enforced upstream by normalise_power.
# ---------------------------------------------------------------------------
def _awgn(x: torch.Tensor, snr_db: torch.Tensor,
          generator: torch.Generator | None) -> torch.Tensor:
    snr_lin = (10.0 ** (snr_db.to(x.dtype) / 10.0)).view(
        -1, *([1] * (x.ndim - 1)))
    sigma = (1.0 / (2.0 * snr_lin)).sqrt()
    n = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                    generator=generator)
    return x + sigma * n


def _rayleigh_complex(x: torch.Tensor, snr_db: torch.Tensor,
                      generator: torch.Generator | None) -> torch.Tensor:
    """Pair real symbols into complex symbols, apply CN(0,1) fading + AWGN,
    coherent-equalise at the receiver, return real-valued output of same
    shape as ``x``."""
    flat = x.flatten(1)                                  # [B, N]
    b, n = flat.shape
    assert n % 2 == 0, f"need even N for complex pairing, got {n}"
    re = flat[:, 0::2]                                   # [B, N/2]
    im = flat[:, 1::2]                                   # [B, N/2]
    s = torch.complex(re.float(), im.float())            # complex symbols

    # complex CN(0,1) fading: real and imag parts ~ N(0, 1/2)
    h_re = torch.randn(s.shape, device=x.device, dtype=torch.float32,
                       generator=generator) * (0.5 ** 0.5)
    h_im = torch.randn(s.shape, device=x.device, dtype=torch.float32,
                       generator=generator) * (0.5 ** 0.5)
    h = torch.complex(h_re, h_im)

    # complex AWGN with E[|n|²] = 1/SNR
    snr_lin = (10.0 ** (snr_db.to(torch.float32) / 10.0)).view(b, 1)
    sigma_n_complex = (1.0 / snr_lin).sqrt()
    n_re = torch.randn(s.shape, device=x.device, dtype=torch.float32,
                       generator=generator) * (0.5 ** 0.5)
    n_im = torch.randn(s.shape, device=x.device, dtype=torch.float32,
                       generator=generator) * (0.5 ** 0.5)
    nz = torch.complex(n_re, n_im) * sigma_n_complex.to(torch.float32)

    y = h * s + nz
    # coherent equaliser (Zero-Forcing); receiver knows h.
    s_eq = y / h.where(h.abs() > 1e-6, torch.ones_like(h) * 1e-6)

    out = flat.clone()
    out[:, 0::2] = s_eq.real.to(x.dtype)
    out[:, 1::2] = s_eq.imag.to(x.dtype)
    return out.view_as(x)


def channel_layer(
    x: torch.Tensor,
    snr_db: torch.Tensor,
    channel: Literal["none", "awgn", "rayleigh", "mixed"] = "awgn",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if channel == "none":
        return x
    if channel == "awgn":
        return _awgn(x, snr_db, generator)
    if channel == "rayleigh":
        return _rayleigh_complex(x, snr_db, generator)
    if channel == "mixed":
        # Per-sample random choice between AWGN and Rayleigh; we run both
        # branches on the full tensor and select per-sample for simplicity.
        b = x.shape[0]
        coin_gen = generator
        coin = torch.rand(b, device=x.device, generator=coin_gen)
        # < 0.5 → AWGN, else Rayleigh.
        y_awgn = _awgn(x, snr_db, generator)
        y_ray = _rayleigh_complex(x, snr_db, generator)
        mask = (coin < 0.5).view(-1, *([1] * (x.ndim - 1))).to(x.dtype)
        return mask * y_awgn + (1.0 - mask) * y_ray
    raise ValueError(f"unknown channel: {channel}")


# ---------------------------------------------------------------------------
# Top-level model: encoder + power norm + channel + decoder.
# ---------------------------------------------------------------------------
class ADJSCC(nn.Module):
    def __init__(self, cfg: ADJSCCConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def n_channel_uses_per_image(self) -> int:
        # 16x16 spatial × C real channels.
        return (self.cfg.image_size // 16) ** 2 * self.cfg.n_symbol_channels

    def forward(
        self,
        x: torch.Tensor,
        snr_db: torch.Tensor,
        channel: Literal["none", "awgn", "rayleigh", "mixed"] = "awgn",
    ) -> torch.Tensor:
        z = self.encoder(x, snr_db)
        z = normalise_power(z)
        z_hat = channel_layer(z, snr_db, channel)
        return self.decoder(z_hat, snr_db)


def build_adjscc(cfg: ADJSCCConfig) -> ADJSCC:
    return ADJSCC(cfg)
