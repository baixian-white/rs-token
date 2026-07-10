"""E30 — Prepare NWPU-RESISC45 splits for AIDDataset compatibility.

Input :
  - data/NWPU-RESISC45/<class>/<class>_<idx>.jpg              # 45 × 700 jpg
  - data/downloads/splits_official/resisc45-{train,val,test}.txt  # canonical 60/10/30 split

Output:
  - data/NWPU_splits/{train,val,test}.csv  with columns: path, class_id, class_name
  - data/NWPU_splits/classes.txt           sorted class names (one per line)

The canonical 60/10/30 split distributed with the dataset (Cheng et al. 2017
PoIEEE) is preferred over a re-shuffled split so that downstream numbers can
be compared to the published benchmark.

Class id assignment is sorted alphabetically across the 45 directory names.
"""
from __future__ import annotations

import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "NWPU-RESISC45"
SPLIT_LISTS = ROOT / "data" / "downloads" / "splits_official"
OUT_DIR = ROOT / "data" / "NWPU_splits"


def class_from_filename(name: str) -> str:
    """E.g. 'tennis_court_414.jpg' -> 'tennis_court'.

    The image filename always ends in `_<int>.jpg`. We strip the trailing
    underscore-int-extension suffix and keep the rest as the class name.
    """
    stem = name[:-4] if name.lower().endswith(".jpg") else name
    # Strip everything from the last underscore (which precedes the index).
    pos = stem.rfind("_")
    if pos == -1:
        raise ValueError(f"unexpected filename: {name}")
    return stem[:pos]


def main():
    if not DATA_ROOT.is_dir():
        raise SystemExit(
            f"Image root not found: {DATA_ROOT}\n"
            f"Unzip NWPU-RESISC45.zip into data/NWPU-RESISC45/ first."
        )
    if not SPLIT_LISTS.is_dir():
        raise SystemExit(f"split lists not found: {SPLIT_LISTS}")

    # 1) Build sorted class list from the directory layout.
    classes = sorted(
        p.name for p in DATA_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if len(classes) != 45:
        print(f"[warn] expected 45 classes, found {len(classes)}: {classes}")
    class_to_id = {c: i for i, c in enumerate(classes)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classes.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8"
    )
    print(f"[classes] {len(classes)} classes; first 5 = {classes[:5]}")

    # 2) Materialize each split CSV by reading the canonical .txt file and
    # mapping each filename to its on-disk path under data/NWPU-RESISC45/<cls>/.
    summary = []
    for split in ("train", "val", "test"):
        names = (SPLIT_LISTS / f"resisc45-{split}.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        names = [n.strip() for n in names if n.strip()]
        rows: list[tuple[str, int, str]] = []
        per_class: dict[str, int] = defaultdict(int)
        missing = 0
        for fname in names:
            cls = class_from_filename(fname)
            if cls not in class_to_id:
                raise ValueError(
                    f"class {cls!r} from {fname!r} not in directory listing"
                )
            jpg = DATA_ROOT / cls / fname
            if not jpg.is_file():
                missing += 1
                continue
            # Use forward-slash absolute paths for cross-platform CSV-stable
            # representation (matches the AID_splits convention used by
            # AIDDataset.__getitem__).
            rows.append((jpg.as_posix(), class_to_id[cls], cls))
            per_class[cls] += 1

        out_csv = OUT_DIR / f"{split}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "class_id", "class_name"])
            w.writerows(rows)

        cnt_min = min(per_class.values()) if per_class else 0
        cnt_max = max(per_class.values()) if per_class else 0
        cnt_mean = (
            sum(per_class.values()) / len(per_class) if per_class else 0.0
        )
        print(
            f"[{split}] {len(rows)} rows, {len(per_class)} classes, "
            f"per-class min/max/mean = {cnt_min}/{cnt_max}/{cnt_mean:.1f}, "
            f"missing files = {missing}"
        )
        summary.append((split, len(rows), len(per_class), cnt_min, cnt_max, cnt_mean))

    print("\nDone.")
    for s in summary:
        print("  ", s)


if __name__ == "__main__":
    main()
