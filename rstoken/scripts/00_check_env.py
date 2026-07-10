"""Stage 0 · Sanity Check — 验证环境与基础工具链是否就绪。

只做 4 件事：
1. PyTorch + CUDA 是否能识别 GPU (RTX 5070 Ti, sm_120)
2. vector-quantize-pytorch 的 ResidualVQ 能否实例化并跑通一次 forward
3. open_clip 是否能加载 ViT-B-32 架构 (空权重也行, 验证接口)
4. 输出关键版本号给后续排查用

不下载任何数据, 不加载任何 checkpoint, 不真正训练。
预期总耗时 < 30 秒, GPU 显存占用 < 1GB。

使用方法:
    cd rstoken
    python scripts/00_check_env.py
"""
from __future__ import annotations

import io
import sys

# Windows 终端默认 GBK, 强制 UTF-8 输出, 避免 ✓ ✗ 这类 unicode 字符崩
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

OK = "[OK]"
NO = "[FAIL]"


def check_pytorch():
    print("\n[1/4] PyTorch + CUDA")
    print("-" * 60)
    try:
        import torch
    except ImportError:
        print(f"  {NO} PyTorch 未安装")
        print("    运行: pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return False

    print(f"  torch version : {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(f"  {NO} CUDA 不可用")
        return False

    print(f"  cuda version  : {torch.version.cuda}")
    print(f"  device name   : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  compute cap   : sm_{cap[0]}{cap[1]}")
    if cap[0] >= 12:
        print("    (Blackwell — 需 PyTorch ≥ 2.7 + cu128 wheel)")

    # 真正算一次, 验证 sm_120 kernel 能跑
    x = torch.randn(2, 3, device="cuda")
    y = (x @ x.T).sum()
    torch.cuda.synchronize()
    print(f"  GPU 矩阵乘 ok : sum = {y.item():.4f}")
    return True


def check_rvq():
    print("\n[2/4] vector-quantize-pytorch · ResidualVQ")
    print("-" * 60)
    try:
        import torch
        from vector_quantize_pytorch import ResidualVQ
    except ImportError as e:
        print(f"  {NO} 未安装: {e}")
        print("    运行: pip install vector-quantize-pytorch")
        return False

    rvq = ResidualVQ(
        dim=256,
        num_quantizers=4,
        codebook_size=1024,
        commitment_weight=0.25,
        kmeans_init=True,
        threshold_ema_dead_code=2,
    ).cuda()

    z = torch.randn(2, 16 * 16, 256, device="cuda")
    zq, indices, vq_loss = rvq(z)
    print(f"  input  shape  : {tuple(z.shape)}")
    print(f"  output shape  : {tuple(zq.shape)}")
    print(f"  indices shape : {tuple(indices.shape)} "
          f"(B, T, num_quantizers)")
    print(f"  vq_loss       : {vq_loss.sum().item():.4f}")

    n_layers = indices.shape[-1]
    n_codes = 1024
    bits_per_token = n_layers * 10  # log2(1024) = 10
    n_tokens = indices.shape[1]
    total_bits = bits_per_token * n_tokens
    print(f"  → 每张 16×16 特征图编码 {total_bits} bits "
          f"= {total_bits / 8:.0f} bytes")
    return True


def check_open_clip():
    print("\n[3/4] open_clip · 接口可用性")
    print("-" * 60)
    try:
        import open_clip
    except ImportError:
        print(f"  {NO} 未安装: pip install open_clip_torch")
        return False

    print(f"  open_clip version: {open_clip.__version__}")
    # 仅验证模型能被构造, 不下载预训练权重 (RemoteCLIP 需手动下载)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ViT-B-32 构造 ok : {n_params:.1f}M 参数")
    print("  (RemoteCLIP 权重需手动下载, 见 README)")
    return True


def print_summary(results):
    print("\n" + "=" * 60)
    print("Stage 0 Sanity Check Summary")
    print("=" * 60)
    names = ["PyTorch+CUDA", "ResidualVQ", "open_clip"]
    for name, ok in zip(names, results):
        mark = OK if ok else NO
        print(f"  {mark} {name}")
    if all(results):
        print("\n  全部通过. 下一步: 下载 RemoteCLIP 权重 + AID 数据集.")
        return 0
    else:
        print("\n  存在失败项, 修复后重跑.")
        return 1


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 0 · 环境与基础工具链 Sanity Check")
    print("=" * 60)

    results = [
        check_pytorch(),
        check_rvq(),
        check_open_clip(),
    ]
    sys.exit(print_summary(results))
