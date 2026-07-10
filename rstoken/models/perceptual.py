"""感知损失 (LPIPS) 包装.

LPIPS 第一次调用会下载 VGG/AlexNet 预训练权重 (~50 MB), 自动缓存.
我们冻结其参数, 不参与训练.

本模块只暴露一个 callable, 训练循环里直接 lpips_fn(x, y) 就行.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import lpips as _lpips_lib
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


class LPIPSLoss(nn.Module):
    """LPIPS perceptual loss, 冻结骨干.

    输入 x, y 范围预期 [-1, 1] (与 VQ-VAE decoder 输出一致).
    若数据是 [0, 1] 范围, 把 use_neg11=False.
    """

    def __init__(self, net: str = "alex", use_neg11: bool = True):
        super().__init__()
        if not _HAS_LPIPS:
            raise ImportError(
                "lpips 未安装. 运行: pip install lpips"
            )
        # net='alex' 比 'vgg' 小 (~10MB vs ~50MB), 速度快, 论文常用
        self.lpips = _lpips_lib.LPIPS(net=net, verbose=False)
        for p in self.lpips.parameters():
            p.requires_grad_(False)
        self.lpips.eval()
        self.use_neg11 = use_neg11

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # LPIPS 期待 [-1, 1] 输入
        if not self.use_neg11:
            x = x * 2 - 1
            y = y * 2 - 1
        return self.lpips(x, y).mean()


if __name__ == "__main__":
    # 自检
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    fn = LPIPSLoss(net="alex").cuda()
    x = torch.randn(2, 3, 256, 256, device="cuda").clamp(-1, 1)
    y = x + 0.1 * torch.randn_like(x)
    loss = fn(x, y)
    print(f"LPIPS(x, x+0.1*noise) = {loss.item():.4f}")
    loss_self = fn(x, x)
    print(f"LPIPS(x, x)            = {loss_self.item():.4f}")
    print("LPIPS OK" if loss > loss_self else "WARN: lpips ordering wrong")
