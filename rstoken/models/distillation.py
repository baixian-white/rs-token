"""Stage 3 蒸馏模块: RemoteCLIP 教师 + 蒸馏 head + cosine loss.

设计要点:
  - Teacher 完全 frozen, eval 模式, no grad.
  - Teacher 输入预处理在模块内完成 (我们 dataloader 输出是 [-1, 1] 256x256;
    RemoteCLIP 期望 [0, 1] 224x224 + CLIP mean/std).
  - DistillHead 输入是 L0 量化后特征 [B, T, latent_dim],
    在 T 维 mean pool 后通过 2-layer MLP 投到 teacher embed_dim.
  - distill_loss 用 1 - cosine, 对 student/teacher 幅度差异不敏感.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# open_clip 默认的 CLIP 预处理统计量, RemoteCLIP 沿用同一套
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class RemoteCLIPTeacher(nn.Module):
    """加载 RemoteCLIP 教师, 提供 encode_image 接口.

    输入: [-1, 1] 归一化的 [B, 3, H, W] (我们 dataloader 的输出格式)
    输出: [B, embed_dim], embed_dim=512 for ViT-B-32
    """

    def __init__(self, ckpt_path: str, model_name: str = "ViT-B-32"):
        super().__init__()
        import open_clip
        if ckpt_path == "openai":
            # OpenAI CLIP 预训练权重 (E8 教师消融对照: 验证 RemoteCLIP 是否不可替代)
            model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained="openai"
            )
            print(f"  using OpenAI CLIP-{model_name} (pretrained='openai')")
        else:
            model, _, _ = open_clip.create_model_and_transforms(model_name)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            msg = model.load_state_dict(ckpt)
            print(f"  RemoteCLIP load: missing={len(msg.missing_keys)}, "
                  f"unexpected={len(msg.unexpected_keys)}")

        # 只保留 visual encoder, 文本头不需要
        self.visual = model.visual
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std", std)

        # 探测 embed dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.embed_dim = self.visual(dummy).shape[-1]
        print(f"  teacher embed_dim = {self.embed_dim}")

    def train(self, mode: bool = True):
        # 永远保持 eval, 屏蔽外部 .train() 调用
        return super().train(False)

    @torch.no_grad()
    def encode_image(self, x_norm_neg1to1: torch.Tensor) -> torch.Tensor:
        """x: [-1, 1] 归一化的 [B, 3, H, W]. 返回 [B, embed_dim]."""
        x = (x_norm_neg1to1 + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        x = (x - self.clip_mean) / self.clip_std
        return self.visual(x)


class DistillHead(nn.Module):
    """L0 量化后特征 -> teacher embedding 空间的投影 head.

    输入: [B, T, latent_dim]
    输出: [B, teacher_dim]
    """

    def __init__(self, latent_dim: int = 256, teacher_dim: int = 512,
                 hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.fc2 = nn.Linear(hidden, teacher_dim)

    def forward(self, zq_l0_seq: torch.Tensor) -> torch.Tensor:
        # 在 token 维度 mean pool
        x = zq_l0_seq.mean(dim=1)              # [B, latent_dim]
        x = self.norm(x)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)                        # [B, teacher_dim]
        return x


def distill_loss(student_emb: torch.Tensor,
                 teacher_emb: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, batch-mean. 范围 [0, 2]."""
    s = F.normalize(student_emb.float(), dim=-1)
    t = F.normalize(teacher_emb.float(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


if __name__ == "__main__":
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
            line_buffering=True,
        )

    # 自检: 加载 RemoteCLIP 跑一次 forward
    teacher = RemoteCLIPTeacher(
        "H:/H-CODE/遥感+通信/rstoken/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt"
    ).cuda()

    head = DistillHead(latent_dim=256, teacher_dim=teacher.embed_dim).cuda()

    x = torch.randn(2, 3, 256, 256, device="cuda").clamp(-1, 1)
    t_emb = teacher.encode_image(x)
    print(f"  teacher emb : {tuple(t_emb.shape)}  "
          f"norm mean={t_emb.norm(dim=-1).mean():.3f}")

    zq_l0 = torch.randn(2, 256, 256, device="cuda")
    s_emb = head(zq_l0)
    print(f"  student emb : {tuple(s_emb.shape)}")

    loss = distill_loss(s_emb, t_emb)
    print(f"  distill loss: {loss.item():.4f}")
    print("  distillation module OK")
