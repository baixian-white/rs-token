"""AID 数据集 DataLoader.

读取 data/AID_splits/{train,val,test}.csv, 做基础预处理:
  - 训练: RandomResizedCrop + 水平翻转(无垂直,遥感图像方向有意义) + 归一化到 [-1, 1]
  - val/test: CenterCrop + 归一化

每张图像默认输出 [3, 256, 256] tensor.

设计选择:
  - 不做 RandomRotation: 遥感数据虽然旋转不变性强,但训练初期会拖慢收敛.
    后期消融时可加.
  - 归一化到 [-1, 1] 而非 ImageNet stats: 解码器输出用 tanh,
    数据范围与之匹配.
  - num_workers 默认从 cfg 读, Windows 上建议 4 起步, 太多会卡 fork.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


@dataclass
class AIDConfig:
    """AID 数据集配置, 由训练脚本从 yaml 注入."""
    splits_dir: str           # data/AID_splits 目录
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


class AIDDataset(Dataset):
    """从 csv 索引读 AID 图像 + class_id."""

    def __init__(self, csv_path: str | Path, transform=None):
        self.transform = transform
        self.items: list[tuple[str, int]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row["path"], int(row["class_id"])))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, cls_id = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, cls_id


def build_transforms(image_size: int, train: bool):
    """构造预处理 pipeline. 归一化到 [-1, 1]."""
    if train:
        return T.Compose([
            T.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), ratio=(0.95, 1.05)
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),                         # -> [0, 1]
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # -> [-1, 1]
        ])
    else:
        return T.Compose([
            T.Resize(image_size + 32),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])


def build_loaders(cfg: AIDConfig) -> dict[str, DataLoader]:
    """返回 {'train', 'val', 'test'} 三个 DataLoader."""
    splits = Path(cfg.splits_dir)
    loaders = {}
    for split, shuffle, drop_last, train_aug in [
        ("train", True,  True,  True),
        ("val",   False, False, False),
        ("test",  False, False, False),
    ]:
        ds = AIDDataset(
            splits / f"{split}.csv",
            transform=build_transforms(cfg.image_size, train=train_aug),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=drop_last,
            persistent_workers=cfg.num_workers > 0,
        )
    return loaders


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1], 用于可视化."""
    return (x.clamp(-1, 1) + 1) * 0.5


if __name__ == "__main__":
    # 自检: 直接 python -m models.datasets, 验证 DataLoader 通
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    cfg = AIDConfig(
        splits_dir="H:/H-CODE/遥感+通信/rstoken/data/AID_splits",
        batch_size=4,
        num_workers=0,  # 自检关掉 worker, 避免 Windows multiprocess 启动开销
    )
    loaders = build_loaders(cfg)
    for split, ld in loaders.items():
        x, y = next(iter(ld))
        print(f"  [{split:5s}] x={tuple(x.shape)} dtype={x.dtype} "
              f"range=[{x.min():.3f}, {x.max():.3f}]  y={y.tolist()}")
    print("  AID DataLoader OK")
