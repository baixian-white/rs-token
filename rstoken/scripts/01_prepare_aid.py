"""Stage 0 · AID 数据集准备 — 索引切分 (train/val/test = 8/1/1) 与可达性检查。

AID 原始结构: data/AID/<class_name>/*.jpg, 30 类共 ~10000 张。
本脚本不复制图像, 只生成 csv 索引文件供后续 DataLoader 使用,
按类分层抽样保证每个 split 内类别分布一致。

输出:
    data/AID_splits/train.csv  (path, class_id, class_name)
    data/AID_splits/val.csv
    data/AID_splits/test.csv
    data/AID_splits/classes.txt

使用方法:
    python scripts/01_prepare_aid.py --aid-root /path/to/AID
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Windows 终端默认 GBK, 强制 UTF-8 输出, 避免 ✓ ✗ 等字符崩溃
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

OK = "[OK]"
NO = "[FAIL]"


def collect_samples(aid_root: Path) -> list[tuple[Path, str]]:
    """扫描 AID 根目录, 返回 (image_path, class_name) 列表.

    注意: Windows 文件系统大小写不敏感, 不能分别 glob *.jpg 和 *.JPG —— 会
    匹配同一文件两次. 这里用 set + 后缀小写比较去重.
    """
    samples = []
    if not aid_root.exists():
        print(f"  {NO} AID 根目录不存在: {aid_root}")
        sys.exit(1)

    class_dirs = sorted(d for d in aid_root.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"  {NO} {aid_root} 下没有任何类别子目录")
        sys.exit(1)

    valid_exts = {".jpg", ".jpeg", ".png"}
    for cls_dir in class_dirs:
        cls = cls_dir.name
        imgs = []
        # 一次遍历, 用 suffix.lower() 判断, 避免重复匹配
        seen = set()
        for p in cls_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() in valid_exts:
                key = str(p).lower()  # 大小写不敏感去重
                if key not in seen:
                    seen.add(key)
                    imgs.append(p)
        imgs.sort()
        for p in imgs:
            samples.append((p, cls))

    return samples


def stratified_split(
    samples: list[tuple[Path, str]],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list, list, list]:
    """按类分层切分, 同类样本按相同比例分到 train/val/test."""
    import random

    by_class: dict[str, list[Path]] = defaultdict(list)
    for p, c in samples:
        by_class[c].append(p)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls, paths in sorted(by_class.items()):
        paths = sorted(paths)  # 保证可重复
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train += [(p, cls) for p in paths[:n_train]]
        val += [(p, cls) for p in paths[n_train : n_train + n_val]]
        test += [(p, cls) for p in paths[n_train + n_val :]]
    return train, val, test


def write_csv(items: list[tuple[Path, str]], out: Path, class_to_id: dict):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_id", "class_name"])
        for p, c in items:
            w.writerow([str(p).replace("\\", "/"), class_to_id[c], c])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aid-root",
        type=Path,
        required=True,
        help="AID 解压后的根目录 (其下应有各类别子目录)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "AID_splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("AID dataset preparation")
    print("=" * 60)
    print(f"  AID root : {args.aid_root}")
    print(f"  out dir  : {args.out_dir}")

    print("\n[1/3] 扫描图像...")
    samples = collect_samples(args.aid_root)
    cnt = Counter(c for _, c in samples)
    print(f"  共 {len(samples)} 张图, {len(cnt)} 类")
    print("  各类样本数:")
    for c, n in sorted(cnt.items()):
        print(f"    {c:<24s} {n:>4d}")

    print("\n[2/3] 分层切分 (8/1/1)...")
    train, val, test = stratified_split(samples, seed=args.seed)
    print(f"  train: {len(train):>5d}")
    print(f"  val  : {len(val):>5d}")
    print(f"  test : {len(test):>5d}")

    classes = sorted(cnt.keys())
    class_to_id = {c: i for i, c in enumerate(classes)}

    print("\n[3/3] 写出索引...")
    write_csv(train, args.out_dir / "train.csv", class_to_id)
    write_csv(val, args.out_dir / "val.csv", class_to_id)
    write_csv(test, args.out_dir / "test.csv", class_to_id)
    (args.out_dir / "classes.txt").write_text(
        "\n".join(classes), encoding="utf-8"
    )
    print(f"  {OK} 写入 {args.out_dir}")
    print(f"\n后续 DataLoader 直接读 train.csv / val.csv / test.csv 即可")


if __name__ == "__main__":
    main()
