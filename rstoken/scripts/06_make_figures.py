"""画论文 method / experiments 章节的 figure.

跑完前期实验后一次性产出:
  fig_method_arch.pdf    - RS-Token 总体架构 (encoder + RVQ + RemoteCLIP 蒸馏 + 信道 + 双输出)
  fig_trade_off.pdf      - E11 蒸馏权重 trade-off (PSNR vs L0_bow + 信道下 marker)
  fig_layered_probe.pdf  - E7 累加层 linear probe (蒸馏 vs 无蒸馏 双曲线)
  fig_channel_snr_k.pdf  - E6 + E10 信道 SNR x k 任务保真矩阵 (AWGN + Rayleigh 双面板)
  fig_teacher_ablation.pdf - E8 OpenAI CLIP vs RemoteCLIP 在不同 SNR 下的表现

每张图独立, 无依赖, 直接读 csv 与硬编码 E5 数字。

运行:
    python scripts/06_make_figures.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGS_DIR = PROJECT_ROOT / "figs"
LOGS_DIR = PROJECT_ROOT / "logs"

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True,
    )

# 通用 matplotlib 设置
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

COLORS = {
    "baseline": "#888888",
    "w01": "#4C72B0",   # 浅蓝
    "w05": "#C44E52",   # 红 (主推)
    "w10": "#55A868",   # 绿
    "openai": "#DD8452",
    "distill": "#C44E52",
}


def fig_method_arch():
    """RS-Token 总体架构图.

    布局 (从左到右):
      input image -> Encoder E -> z (16x16x256) -> ResidualVQ x4 -> indices [4]
        -> 上分支 (训练时, 虚线): zq_L0 -> DistillHead phi -> cosine align -> RemoteCLIP teacher (frozen, 锁图标)
        -> 主分支: BPSK + AWGN/Rayleigh channel -> received indices -> codebook lookup -> zq_hat -> Decoder D -> 重建图
        -> 下分支 (任务): L0_bow histogram (1024-d) -> linear classifier -> AID class
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.set_aspect("equal")
    ax.axis("off")

    # 颜色方案 (与 paper 其它 figure 协调)
    C_DATA = "#FFFFFF"
    C_ENC = "#E8E4F3"        # encoder 浅紫
    C_RVQ = "#FCE4DC"        # RVQ 浅橙
    C_TEACHER = "#D6E4F0"    # RemoteCLIP 浅蓝
    C_CHANNEL = "#F5F5F5"    # 信道浅灰
    C_DEC = "#E8E4F3"        # decoder 浅紫
    C_TASK = "#FCE4DC"       # 任务输出浅橙
    C_HIGHLIGHT = "#C44E52"  # 红色: 蒸馏路径与论文核心
    C_ARROW = "#333333"
    C_DASH = "#888888"

    def box(x, y, w, h, label, color=C_DATA, lw=1.0, edge="#333", fontsize=9.5,
            label_color="#222", label_weight="normal"):
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            linewidth=lw, edgecolor=edge, facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                color=label_color, fontweight=label_weight, wrap=True)

    def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.2, ls="-", label=None,
              label_offset=(0, 0.7), arrow_style="->"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=arrow_style, color=color,
                            lw=lw, linestyle=ls,
                            shrinkA=2, shrinkB=2),
        )
        if label is not None:
            mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=8.5, color=color, style="italic")

    # === 主流水线 (中间行, y ≈ 22-30) ===
    # 1. 输入图像
    box(2, 22, 8, 8, "Input\n$x$\n3×256×256", color=C_DATA, fontsize=9)
    # 2. Encoder
    box(13, 22, 9, 8, "Encoder\n$E$", color=C_ENC, fontsize=10, label_weight="bold")
    # 3. z (latent)
    box(25, 23.5, 7, 5, "$z$\n16×16×256", color=C_DATA, fontsize=8.5)
    # 4. ResidualVQ
    box(35, 21, 13, 10, "ResidualVQ ×4\nK=1024, d=256",
        color=C_RVQ, fontsize=10, label_weight="bold")
    # 5. indices
    box(51, 23, 8, 6, "Indices\n$\\mathbf{I}\\in\\{0..1023\\}^{T\\times4}$",
        color=C_DATA, fontsize=8)
    # 6. Channel
    box(62, 22, 12, 8, "BPSK + Channel\n(AWGN / Rayleigh)",
        color=C_CHANNEL, fontsize=9.5, label_weight="bold")
    # 7. received indices + lookup
    box(77, 23, 9, 6, "Received\n$\\hat{\\mathbf{I}}$ (k layers)",
        color=C_DATA, fontsize=8.5)
    # 8. Decoder
    box(89, 22, 9, 8, "Decoder\n$D$", color=C_DEC, fontsize=10, label_weight="bold")

    # 主流水线箭头
    arrow(10, 26, 13, 26)
    arrow(22, 26, 25, 26)
    arrow(32, 26, 35, 26)
    arrow(48, 26, 51, 26)
    arrow(59, 26, 62, 26)
    arrow(74, 26, 77, 26)
    arrow(86, 26, 89, 26)

    # 9. 重建输出 (decoder 后)
    box(89, 8, 9, 8, "Recon\n$\\hat{x}$", color=C_DATA, fontsize=9)
    arrow(93.5, 22, 93.5, 16)

    # === 上分支: RemoteCLIP 蒸馏 (训练时, 虚线) ===
    # 教师路径: 输入图像 -> RemoteCLIP teacher
    box(13, 44, 18, 8, "RemoteCLIP\n(ViT-B/32, frozen)\nteacher $f_T$",
        color=C_TEACHER, lw=1.4, edge=C_HIGHLIGHT,
        fontsize=9.5, label_color=C_HIGHLIGHT, label_weight="bold")

    # student 投影路径: zq_L0 -> mean pool -> MLP head phi
    box(35, 44, 13, 8, "DistillHead $\\phi$\nMLP 256 → 512",
        color=C_TEACHER, lw=1.4, edge=C_HIGHLIGHT,
        fontsize=9.5, label_color=C_HIGHLIGHT)

    # cosine align
    box(51, 45, 9, 6, "cosine\nalign",
        color="#FFE6E6", lw=1.4, edge=C_HIGHLIGHT,
        fontsize=9, label_color=C_HIGHLIGHT, label_weight="bold")
    # L_distill 标签
    ax.text(55.5, 53.5, "$\\mathcal{L}_{\\mathrm{distill}} = 1 - \\cos(s, t)$",
            ha="center", fontsize=9.5, color=C_HIGHLIGHT, fontweight="bold")

    # input -> teacher (虚线, 训练专用)
    arrow(6, 30, 13, 44, color=C_HIGHLIGHT, lw=1.2, ls="--")
    # teacher -> cosine
    arrow(31, 48, 51, 48, color=C_HIGHLIGHT, lw=1.2, ls="--",
          label="$t \\in \\mathbb{R}^{512}$", label_offset=(0, 1.2))
    # zq_L0 (来自 RVQ 的 L0 输出) -> DistillHead
    # 从 RVQ 顶部画一条虚线到 DistillHead
    arrow(41.5, 31, 41.5, 44, color=C_HIGHLIGHT, lw=1.2, ls="--",
          label="$\\bar{z}_q^{(1)}$", label_offset=(2.5, 0))
    # DistillHead -> cosine
    arrow(48, 48, 51, 48, color=C_HIGHLIGHT, lw=1.2, ls="--",
          label="$s$", label_offset=(0, 1.2))
    # 训练时标记
    ax.text(20, 41, "training only", ha="center", fontsize=8.5,
            color=C_HIGHLIGHT, style="italic")

    # === 下分支: 任务保真 · L0_bow → linear classifier ===
    # L0 BoW 抽取 (从 received indices 出发)
    box(63, 8, 11, 8, "L0_bow\n1024-d histogram",
        color=C_TASK, fontsize=9, label_weight="bold")
    box(78, 8, 9, 8, "Linear\nClassifier",
        color=C_TASK, fontsize=9.5, label_weight="bold")
    box(89, -2, 9, 7, "Class\nlabel", color=C_DATA, fontsize=9)

    # 接收索引 -> L0 BoW
    arrow(81.5, 23, 81.5, 18.5, color=C_ARROW, lw=1.0)
    arrow(81.5, 18.5, 68.5, 18.5, color=C_ARROW, lw=1.0)
    arrow(68.5, 18.5, 68.5, 16, color=C_ARROW, lw=1.0)
    # L0_bow -> linear classifier
    arrow(74, 12, 78, 12)
    # classifier -> class label
    arrow(82.5, 8, 82.5, 5)

    # 任务分支标签
    ax.text(68.5, 4, "task path (deployment, $k=1$)",
            ha="center", fontsize=8.5, color="#666", style="italic")

    # === 顶部标题 ===
    ax.text(50, 55,
            "RS-Token: Hierarchical RemoteCLIP-Distilled RVQ Tokenizer "
            "for Channel-Robust Remote Sensing Communication",
            ha="center", fontsize=11, fontweight="bold", color="#222")

    # 左侧 / 右侧 group 标注
    ax.text(2, 36, "Transmitter side", fontsize=9, color="#555",
            style="italic", fontweight="bold")
    ax.text(76, 36, "Receiver side", fontsize=9, color="#555",
            style="italic", fontweight="bold")
    # 分割线 (信道前后)
    ax.plot([68, 68], [3, 41], ls=":", color="#bbb", lw=0.8)

    # 注释: layer-wise transmission (主流水线下方)
    ax.text(50, 17,
            "transmit first $k$ layers; $k\\in\\{1,2,3,4\\}$ chosen by "
            "channel + task",
            ha="center", fontsize=8.5, color="#666", style="italic")

    out = FIGS_DIR / "fig_method_arch.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  saved {out}  +  .png")


def fig_overall_framework():
    """Clean overview figure for talks/slides.

    This figure intentionally hides most equations and focuses on the system
    story: semantic L0 index for the task path, residual indices for progressive
    reconstruction, and RemoteCLIP supervision only during training.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 70)
    ax.axis("off")

    c_text = "#222222"
    c_muted = "#606060"
    c_line = "#333333"
    c_sem = "#C44E52"
    c_sem_fill = "#FCE4DC"
    c_detail = "#4C72B0"
    c_detail_fill = "#E2ECF8"
    c_model = "#E8E4F3"
    c_channel = "#F5F5F5"
    c_teacher = "#D6E4F0"
    c_eval = "#EAF4EA"

    def box(x, y, w, h, label, fc="#FFFFFF", ec="#333333", lw=1.4,
            fontsize=10, weight="normal", color=c_text, radius=1.0):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.22,rounding_size={radius}",
            linewidth=lw, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight=weight)
        return patch

    def arrow(x1, y1, x2, y2, color=c_line, lw=1.6, ls="-", label=None,
              text_y=1.8, style="->"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                            linestyle=ls, shrinkA=3, shrinkB=3),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + text_y, label,
                    ha="center", va="center", fontsize=8.7,
                    color=color, fontstyle="italic")

    # Section backgrounds.
    ax.add_patch(Rectangle((2, 8), 48, 50, facecolor="#FAFAFA",
                           edgecolor="#DDDDDD", linewidth=0.8))
    ax.add_patch(Rectangle((53, 8), 25, 50, facecolor="#FBFBFB",
                           edgecolor="#DDDDDD", linewidth=0.8))
    ax.add_patch(Rectangle((81, 8), 52, 50, facecolor="#FAFAFA",
                           edgecolor="#DDDDDD", linewidth=0.8))
    ax.text(4, 55, "Transmitter", fontsize=11, color=c_muted,
            fontweight="bold")
    ax.text(55, 55, "Index Channel", fontsize=11, color=c_muted,
            fontweight="bold")
    ax.text(83, 55, "Receiver", fontsize=11, color=c_muted,
            fontweight="bold")

    # Main transmitter path.
    box(5, 34, 11, 9, "RS image\nx", fontsize=10)
    box(20, 34, 13, 9, "Encoder\nE", fc=c_model, fontsize=11,
        weight="bold")
    box(37, 34, 10, 9, "latent\n16x16x256", fontsize=9)
    arrow(16, 38.5, 20, 38.5)
    arrow(33, 38.5, 37, 38.5)

    # RVQ stack.
    box(51, 28, 24, 20, "", fc="#FFFFFF", fontsize=11, weight="bold")
    ax.text(63, 45.7, "ResidualVQ tokenizer", ha="center", va="center",
            fontsize=11, color=c_text, fontweight="bold")
    layer_x = 54
    layers = [
        ("L0 semantic index", c_sem_fill, c_sem),
        ("L1 residual detail", c_detail_fill, c_detail),
        ("L2 residual detail", c_detail_fill, c_detail),
        ("L3 residual detail", c_detail_fill, c_detail),
    ]
    for i, (txt, fc, ec) in enumerate(layers):
        yy = 40.5 - i * 3.7
        box(layer_x, yy, 18, 2.8, txt, fc=fc, ec=ec, lw=1.0,
            fontsize=8.5, weight="bold" if i == 0 else "normal",
            color=ec if i == 0 else c_text, radius=0.45)
    ax.text(63, 26.5, "K=1024, 10 bits/index", ha="center",
            fontsize=8.4, color=c_muted,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    arrow(47, 38.5, 51, 38.5)

    # RemoteCLIP training-only branch.
    box(21, 11, 17, 8, "RemoteCLIP\nteacher", fc=c_teacher, ec=c_sem,
        lw=1.3, fontsize=9.5, color=c_sem, weight="bold")
    box(43, 11, 17, 8, "L0 projection\nhead", fc=c_teacher, ec=c_sem,
        lw=1.3, fontsize=9.5, color=c_sem)
    box(65, 11, 10, 8, "cosine\nalign", fc="#FFECEC", ec=c_sem,
        lw=1.3, fontsize=9.3, color=c_sem, weight="bold")
    arrow(10.5, 34, 21, 19, color=c_sem, lw=1.3, ls="--")
    arrow(38, 15, 43, 15, color=c_sem, lw=1.3, ls="--")
    arrow(63, 28, 51.5, 19, color=c_sem, lw=1.3, ls="--",
          label=None)
    arrow(60, 15, 65, 15, color=c_sem, lw=1.3, ls="--")
    ax.text(42, 8.8, "training only: make L0 index semantic",
            ha="center", fontsize=9, color=c_sem, fontstyle="italic")

    # Channel.
    box(83, 34, 15, 9, "Transmit\nindices", fontsize=9.5)
    box(102, 34, 18, 9, "BPSK channel\nAWGN / Rayleigh", fc=c_channel,
        fontsize=10, weight="bold")
    arrow(75, 38.5, 83, 38.5)
    ax.text(79, 43.0, "first k layers\nk=1..4",
            ha="center", va="center", fontsize=8.4,
            color=c_muted, fontstyle="italic",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    arrow(98, 38.5, 102, 38.5)
    ax.text(88.5, 30.7, "L0 only: 2,560 bits/img\nL0-L3: 10,240 bits/img",
            ha="center", va="top", fontsize=8.4, color=c_muted)

    # Receiver split.
    box(123, 34, 8, 9, "received\nindices", fontsize=8.8)
    arrow(120, 38.5, 123, 38.5)

    # Task path.
    box(92, 11, 14, 9, "L0_bow\nhistogram", fc=c_sem_fill, ec=c_sem,
        fontsize=9.5, weight="bold")
    box(110, 11, 13, 9, "Scene\nclassifier", fc=c_sem_fill, ec=c_sem,
        fontsize=9.5, weight="bold")
    box(126, 11, 7, 9, "class\nlabel", fc=c_eval, ec="#4C8C4A",
        fontsize=9)
    arrow(127, 34, 99, 20, color=c_sem, lw=1.5,
          label="poor link / task-first", text_y=-1.9)
    arrow(106, 15.5, 110, 15.5, color=c_sem)
    arrow(123, 15.5, 126, 15.5, color=c_sem)
    ax.text(109, 8.6, "Task path: use L0 only", ha="center",
            fontsize=9, color=c_sem, fontweight="bold")

    # Reconstruction path.
    box(112, 49, 12, 7, "Decoder\nD", fc=c_model, fontsize=10,
        weight="bold")
    box(127, 49, 6, 7, "Recon\n$\\hat{x}$", fontsize=8.8)
    arrow(127, 43, 118, 49, color=c_detail, lw=1.5,
          label="good link / more layers", text_y=2.0)
    arrow(124, 52.5, 127, 52.5, color=c_detail)
    ax.text(119, 58.8, "Reconstruction path: add L1-L3 for detail",
            ha="center", fontsize=9, color=c_detail, fontweight="bold")

    # Title and thesis line.
    ax.text(67.5, 68,
            "Overall Framework of RS-Token",
            ha="center", fontsize=15, fontweight="bold", color=c_text)
    ax.text(67.5, 64.5,
            "Index transmission is the carrier; RemoteCLIP distillation makes the first RVQ layer task-semantic.",
            ha="center", fontsize=10, color=c_muted)

    out = FIGS_DIR / "fig_overall_framework.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  saved {out}  +  .png")


def fig_base_rvq():
    """Base ResidualVQ tokenizer without semantic distillation.

    The figure explains the baseline mechanism: each RVQ layer quantizes the
    current residual, emits one index map, and contributes one selected codeword
    tensor to the reconstructed latent. No RemoteCLIP branch is present here.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 62)
    ax.axis("off")

    c_text = "#222222"
    c_muted = "#606060"
    c_line = "#333333"
    c_model = "#E8E4F3"
    c_layer = "#E2ECF8"
    c_layer_edge = "#4C72B0"
    c_index = "#FCE4DC"
    c_index_edge = "#C44E52"
    c_sum = "#EAF4EA"
    c_warn = "#FFF4D6"

    def box(x, y, w, h, label, fc="#FFFFFF", ec="#333333", lw=1.4,
            fontsize=9.5, weight="normal", color=c_text, radius=0.9):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.22,rounding_size={radius}",
            linewidth=lw, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight=weight)
        return patch

    def arrow(x1, y1, x2, y2, color=c_line, lw=1.5, ls="-", label=None,
              text_y=1.7, style="->"):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                            linestyle=ls, shrinkA=3, shrinkB=3),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + text_y, label,
                    ha="center", va="center", fontsize=8.4,
                    color=color, fontstyle="italic",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=0.8))

    ax.text(67.5, 59, "Base RVQ Tokenizer", ha="center",
            fontsize=15, fontweight="bold", color=c_text)
    ax.text(67.5, 55.5,
            "A reconstruction-driven ResidualVQ: quantize the residual layer by layer, then sum selected codewords.",
            ha="center", fontsize=10, color=c_muted)

    # Left encoder path.
    box(4, 32, 11, 8, "Image\nx", fontsize=10)
    box(19, 32, 13, 8, "Encoder\nE", fc=c_model, fontsize=11,
        weight="bold")
    box(36, 32, 12, 8, "latent z\nT=16x16\nD=256", fontsize=9)
    arrow(15, 36, 19, 36)
    arrow(32, 36, 36, 36)

    # RVQ panel.
    ax.add_patch(Rectangle((52, 8), 44, 40, facecolor="#FAFAFA",
                           edgecolor="#DDDDDD", linewidth=0.8))
    ax.text(54, 45.5, "Residual quantization stack", fontsize=11,
            color=c_muted, fontweight="bold")
    ax.text(73.5, 42.8, "r0 = z", ha="center", fontsize=9,
            color=c_muted,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))

    layer_specs = [
        ("Layer 0", "C0", "i0", "q0 = C0[i0]", "r1 = r0 - q0"),
        ("Layer 1", "C1", "i1", "q1 = C1[i1]", "r2 = r1 - q1"),
        ("Layer 2", "C2", "i2", "q2 = C2[i2]", "r3 = r2 - q2"),
        ("Layer 3", "C3", "i3", "q3 = C3[i3]", "final residual"),
    ]
    ys = [36, 28, 20, 12]
    q_points = []
    i_points = []
    for idx, (name, cb, ind, qtxt, rtxt) in enumerate(layer_specs):
        y = ys[idx]
        box(56, y - 2.6, 12, 5.2, f"{name}\n{cb}: K=1024",
            fc=c_layer, ec=c_layer_edge, lw=1.2, fontsize=8.6,
            weight="bold" if idx == 0 else "normal",
            color=c_layer_edge if idx == 0 else c_text)
        box(72, y - 2.6, 11, 5.2, qtxt, fc="#FFFFFF",
            ec=c_layer_edge, lw=1.0, fontsize=8.4)
        box(87, y - 2.2, 6, 4.4, ind, fc=c_index, ec=c_index_edge,
            lw=1.1, fontsize=9, weight="bold", color=c_index_edge)
        q_points.append((83, y))
        i_points.append((90, y - 2.2))
        if idx < 3:
            arrow(62, y - 2.8, 62, ys[idx + 1] + 2.8,
                  color=c_layer_edge, lw=1.2,
                  label=rtxt, text_y=0.0)
        else:
            ax.text(62, y - 5.0, rtxt, ha="center", fontsize=8,
                    color=c_muted)
        arrow(68, y, 72, y, color=c_layer_edge, lw=1.2)
        arrow(83, y, 87, y, color=c_index_edge, lw=1.0)

    arrow(48, 36, 56, 36, label="current residual")

    # Sum selected codewords.
    box(101, 25, 13, 10, "sum selected\ncodewords\nz_hat = sum ql",
        fc=c_sum, ec="#4C8C4A", fontsize=9.3, weight="bold")
    for x, y in q_points:
        arrow(x, y, 101, 30, color="#4C8C4A", lw=1.0, ls="--")

    # Index packet.
    box(101, 11, 13, 8, "index packet\n[i0,i1,i2,i3]",
        fc=c_index, ec=c_index_edge, fontsize=9.2, weight="bold",
        color=c_index_edge)
    for x, y in i_points:
        arrow(93, y + 2.2, 101, 15, color=c_index_edge, lw=1.0, ls="--")
    ax.text(107.5, 7.8, "what is transmitted", ha="center",
            fontsize=8.6, color=c_index_edge, fontweight="bold")

    # Decoder and output.
    box(119, 26, 12, 8, "Decoder\nD", fc=c_model, fontsize=11,
        weight="bold")
    box(119, 12, 12, 8, "Transmit\nindices", fc=c_index,
        ec=c_index_edge, fontsize=9.2, weight="bold", color=c_index_edge)
    box(119, 42, 12, 8, "Recon\nx_hat", fontsize=9.3)
    arrow(114, 30, 119, 30)
    arrow(125, 34, 125, 42)
    arrow(114, 15, 119, 15, color=c_index_edge)

    # Baseline objective callout.
    box(7, 11, 30, 9,
        "Baseline training objective\nL = L_recon + L_VGG\n(no RemoteCLIP, no semantic loss)",
        fc=c_warn, ec="#D4A72C", fontsize=9.2, color=c_text)
    arrow(22, 20, 64, 12, color="#D4A72C", lw=1.2, ls="--")

    # Key takeaway.
    ax.text(67.5, 3.2,
            "Implication: the first index layer is a coarse residual code, not necessarily a remote-sensing semantic code.",
            ha="center", fontsize=9.5, color=c_muted, fontweight="bold")

    out = FIGS_DIR / "fig_base_rvq.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  saved {out}  +  .png")


def fig_trade_off():
    """E11 trade-off: PSNR vs L0_bow + 信道下 marker.

    四个点 (w=0.0, 0.1, 0.5, 1.0), 双轴: 主轴 L0_bow (无信道), 副 marker AWGN 0dB / Rayleigh +5dB.
    """
    # 硬编码 E11 + E3 数据 (来自 pre_experiments_log.md)
    weights = [0.0, 0.1, 0.5, 1.0]
    psnr   = [26.10, 26.17, 25.88, 25.61]
    bow_clean = [57.7, 71.20, 82.40, 84.50]
    bow_awgn0 = [23.4, 35.50, 53.20, 37.50]
    bow_ray5  = [28.0, 42.10, 59.10, 49.80]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))

    # 左轴: L0_bow 三条曲线
    l1, = ax1.plot(weights, bow_clean, "o-", color="#222", lw=2,
                   ms=8, label="L0_bow (no channel)")
    l2, = ax1.plot(weights, bow_awgn0, "s--", color="#C44E52", lw=1.6,
                   ms=7, label="L0_bow @ AWGN 0 dB")
    l3, = ax1.plot(weights, bow_ray5, "^--", color="#4C72B0", lw=1.6,
                   ms=7, label="L0_bow @ Rayleigh +5 dB")
    ax1.set_xlabel("distillation weight  $w$")
    ax1.set_ylabel("classification acc. (%)")
    ax1.set_xticks(weights)
    ax1.set_ylim(15, 95)
    ax1.grid(True, alpha=0.3)

    # 右轴: PSNR
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    l4, = ax2.plot(weights, psnr, "D-", color="#55A868", lw=1.6,
                   ms=6, label="best PSNR (no channel)")
    ax2.set_ylabel("reconstruction PSNR (dB)", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")
    ax2.set_ylim(25.4, 26.3)

    # 标注 w=0.5 是论文主推
    idx = weights.index(0.5)
    ax1.axvline(0.5, color="#C44E52", lw=0.5, ls=":", alpha=0.5)
    ax1.annotate("paper choice",
                 xy=(0.5, 53.2), xytext=(0.62, 30),
                 fontsize=9, color="#C44E52",
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=0.8))

    # 合并 legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        framealpha=0.95,
    )

    ax1.set_title("Distillation weight trade-off:\n"
                  "$w=0.5$ is the sweet spot for channel-degraded task fidelity",
                  fontsize=10)

    out = FIGS_DIR / "fig_trade_off.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  saved {out}  +  .png")


def fig_layered_probe():
    """E7 累加层 linear probe: 蒸馏 vs 无蒸馏 双曲线."""
    # 硬编码 E7 数据
    layers = [1, 2, 3, 4]
    distill = [86.00, 87.70, 87.90, 88.00]
    baseline = [47.70, 47.20, 47.70, 48.30]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    ax.plot(layers, distill, "o-", color="#C44E52", lw=2, ms=9,
            label="with RemoteCLIP distillation")
    ax.plot(layers, baseline, "s--", color="#888", lw=1.5, ms=8,
            label="no distillation")

    # 标注每点
    for x, y in zip(layers, distill):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 7),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color="#C44E52")
    for x, y in zip(layers, baseline):
        ax.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, -14),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color="#666")

    # 高亮 L0 → L0+L1 增量
    ax.annotate("", xy=(2, 87.7), xytext=(1, 86.0),
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))
    ax.text(1.5, 88.5, "+1.7 pp", color="#C44E52", fontsize=9,
            ha="center", fontweight="bold")

    ax.set_xticks(layers)
    ax.set_xticklabels(["L0", "L0+L1", "L0+L1\n+L2", "L0+L1\n+L2+L3"])
    ax.set_xlabel("accumulated RVQ layers (codebook embedding pool)")
    ax.set_ylabel("AID linear probe acc. (%)")
    ax.set_ylim(40, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", framealpha=0.95)

    out = FIGS_DIR / "fig_layered_probe.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  saved {out}  +  .png")


# 长脚本占位, 下半部分在第二个 chunk 里写
def fig_channel_snr_k():
    """E6 + E10: AWGN + Rayleigh 双面板, SNR x k 任务保真矩阵 (蒸馏版).

    每个面板是热图 + 文本标注; 两个面板共享 colorbar.
    """
    snr_labels = ["-10", "-5", "0", "+5", "+10", "∞"]
    ks = [1, 2, 3, 4]

    # 蒸馏版 AWGN (来自 logs/stage4_distill.csv)
    awgn = np.array([
        [4.50, 3.40, 3.80, 2.70],
        [8.20, 7.10, 6.00, 5.70],
        [53.20, 50.40, 50.50, 52.80],
        [82.30, 81.90, 81.30, 82.00],
        [82.50, 82.40, 82.40, 82.40],
        [82.40, 82.40, 82.40, 82.40],
    ])
    # 蒸馏版 Rayleigh (来自 logs/stage4_distill_rayleigh.csv)
    ray = np.array([
        [3.70, 3.50, 3.30, 3.30],
        [4.80, 5.90, 4.60, 3.60],
        [16.60, 14.10, 16.20, 15.50],
        [59.10, 62.70, 59.80, 58.90],
        [79.00, 76.70, 78.00, 77.90],
        [82.40, 82.40, 82.40, 82.40],
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.6),
                                    sharey=True, constrained_layout=True)
    cmap = "viridis"
    vmin, vmax = 0, 90

    for ax, mat, title in [
        (ax1, awgn, "AWGN channel"),
        (ax2, ray, "Rayleigh fading channel"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_yticks(range(6))
        if ax is ax1:
            ax.set_yticklabels(snr_labels)
            ax.set_ylabel("SNR (dB)")
        ax.set_xlabel("transmitted layers $k$")
        ax.set_title(title, fontsize=10)

        for i in range(6):
            for j in range(4):
                v = mat[i, j]
                color = "white" if v < 40 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=8.5, color=color)

    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.85, pad=0.02)
    cbar.set_label("AID test acc. (%)  [L0_bow feature, distilled ckpt]",
                   fontsize=9)

    out = FIGS_DIR / "fig_channel_snr_k.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  saved {out}  +  .png")


def fig_teacher_ablation():
    """E8: OpenAI CLIP vs RemoteCLIP 在不同 SNR 下的 L0_bow @ k=1.

    bar chart 横向比较, 强调 RemoteCLIP 的 marginal benefit 集中在中等 SNR.
    """
    snrs = ["No channel", "AWGN +10dB", "AWGN +5dB", "AWGN 0dB",
            "Rayleigh +10dB", "Rayleigh +5dB", "Rayleigh 0dB"]
    openai_vals = [80.8, 80.8, 79.4, 46.5, 74.8, 55.3, 14.7]
    remote_vals = [82.4, 82.5, 82.3, 53.2, 79.0, 59.1, 16.6]

    x = np.arange(len(snrs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    b1 = ax.bar(x - width/2, openai_vals, width,
                label="OpenAI CLIP teacher", color="#DD8452")
    b2 = ax.bar(x + width/2, remote_vals, width,
                label="RemoteCLIP teacher", color="#C44E52")

    # 在每对 bar 顶上写 delta
    for i, (a, b) in enumerate(zip(openai_vals, remote_vals)):
        delta = b - a
        ymax = max(a, b)
        ax.text(i, ymax + 1.5, f"+{delta:.1f}",
                ha="center", fontsize=8.5, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(snrs, rotation=30, ha="right")
    ax.set_ylabel("L0_bow classification acc. (%)\n(transmit only L0, $k=1$)")
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.set_title("Teacher ablation: RemoteCLIP's marginal benefit (~+1.6 pp in-domain) "
                 "grows to +6.7 pp in mid-SNR channels", fontsize=9.5)

    out = FIGS_DIR / "fig_teacher_ablation.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  saved {out}  +  .png")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {FIGS_DIR}")

    print("\n[0/6] fig_overall_framework (clean overview)")
    fig_overall_framework()
    print("\n[1/5] fig_method_arch (RS-Token architecture)")
    fig_method_arch()
    print("\n[2/5] fig_trade_off (E11 distillation weight sweep)")
    fig_trade_off()
    print("\n[3/5] fig_layered_probe (E7 cumulative layer probe)")
    fig_layered_probe()
    print("\n[4/5] fig_channel_snr_k (E6 + E10 SNR x k matrix)")
    fig_channel_snr_k()
    print("\n[5/5] fig_teacher_ablation (E8 OpenAI vs RemoteCLIP)")
    fig_teacher_ablation()

    print("\nDone.")


if __name__ == "__main__":
    main()
