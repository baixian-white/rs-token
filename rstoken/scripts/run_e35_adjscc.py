"""E35 — Generate 9 ADJSCC training configs (3 rates × 3 seeds) and a runner.

Usage:
    python scripts/run_e35_adjscc.py --gen-configs   # writes 9 yamls
    python scripts/run_e35_adjscc.py --train         # trains all 9 sequentially
    python scripts/run_e35_adjscc.py --eval          # evaluates all 9 across 5 channels
    python scripts/run_e35_adjscc.py --train --eval  # do both
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs" / "paper_v05"
LOG_DIR = ROOT / "logs" / "paper_v05"
RUN_LOG = LOG_DIR / "run_logs"
CKPT_DIR = ROOT / "checkpoints" / "paper_v05"

PY = r"C:\Users\Administrator\miniconda3\envs\rstoken\python.exe"

RATES = [
    # (n_symbol_channels, bits_per_image_label)
    (10, 2560),
    (20, 5120),
    (40, 10240),
]
SEEDS = (41, 42, 43)


CONFIG_TEMPLATE = """# ADJSCC training config — E35.
# Channel uses per image = 16*16*{c} = {n}; matches RS-Token at k={k} bits.

run_name: adjscc_b{bits}_s{seed}
seed: {seed}

data:
  splits_dir: D:/CODE/遥感+通信/遥感+通信/rstoken/data/AID_splits_local
  image_size: 256
  batch_size: 16
  num_workers: 4
  pin_memory: true

model:
  image_size: 256
  in_channels: 3
  base_channels: 64
  n_symbol_channels: {c}
  af_hidden: 64
  train_snr_min: -2.0
  train_snr_max: 12.0
  train_channel: mixed

optim:
  lr: 1.0e-4
  weight_decay: 0.0
  betas: [0.9, 0.95]
  warmup_steps: 500
  total_epochs: 50
  grad_clip: 1.0

logging:
  log_every: 100
  save_every_epoch: 10
  ckpt_dir: D:/CODE/遥感+通信/遥感+通信/rstoken/checkpoints/paper_v05/adjscc_b{bits}_s{seed}
  log_dir:  D:/CODE/遥感+通信/遥感+通信/rstoken/logs/paper_v05/adjscc_b{bits}_s{seed}

device: cuda
amp: true
amp_dtype: bf16
"""


def gen_configs() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for c, bits in RATES:
        for seed in SEEDS:
            k = bits // 2560
            content = CONFIG_TEMPLATE.format(c=c, n=16 * 16 * c, bits=bits,
                                             seed=seed, k=k)
            p = CONFIG_DIR / f"adjscc_b{bits}_s{seed}.yaml"
            p.write_text(content, encoding="utf-8")
            paths.append(p)
    print(f"Wrote {len(paths)} configs under {CONFIG_DIR}")
    return paths


def all_runs() -> list[tuple[str, Path, Path]]:
    runs = []
    for c, bits in RATES:
        for seed in SEEDS:
            name = f"adjscc_b{bits}_s{seed}"
            cfg = CONFIG_DIR / f"{name}.yaml"
            ck = CKPT_DIR / name
            runs.append((name, cfg, ck))
    return runs


def env_for_run() -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    return env


def train_all(skip_existing: bool = True) -> int:
    RUN_LOG.mkdir(parents=True, exist_ok=True)
    for name, cfg, ck in all_runs():
        best = ck / "best.pt"
        log_path = RUN_LOG / f"e35_train_{name}.log"
        if skip_existing and best.exists():
            print(f"[{time.strftime('%H:%M:%S')}] skip {name} (best.pt exists)",
                  flush=True)
            continue
        print(f"[{time.strftime('%H:%M:%S')}] === train {name} ===", flush=True)
        t0 = time.time()
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            logf.write(f"# E35 train {name}\n")
            logf.flush()
            rc = subprocess.run(
                [PY, str(ROOT / "scripts" / "train_adjscc.py"),
                 "--config", str(cfg.relative_to(ROOT)).replace("\\", "/")],
                cwd=ROOT, env=env_for_run(),
                stdout=logf, stderr=subprocess.STDOUT,
            ).returncode
        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {name} exit={rc}  dt={dt:.1f}s  "
              f"log={log_path}", flush=True)
        if rc != 0:
            print(f"FAILED on {name}", flush=True)
            return rc
    return 0


def eval_all() -> int:
    """Run a single eval call covering all 9 ckpts (so the AID classifier
    + LPIPS net are loaded once)."""
    RUN_LOG.mkdir(parents=True, exist_ok=True)
    specs = []
    for name, _cfg, ck in all_runs():
        best = ck / "best.pt"
        if not best.exists():
            print(f"  MISSING: {best}; abort", flush=True)
            return 2
        specs.append(f"{name}={(ck / 'best.pt').relative_to(ROOT).as_posix()}")
    out_csv = LOG_DIR / "e35_adjscc_recon.csv"
    log_path = RUN_LOG / "e35_eval_adjscc.log"
    print(f"[{time.strftime('%H:%M:%S')}] === eval all 9 ADJSCC ckpts ===",
          flush=True)
    cmd = [
        PY, str(ROOT / "scripts" / "eval_adjscc.py"),
        "--models", ",".join(specs),
        "--out", str(out_csv.relative_to(ROOT)).replace("\\", "/"),
        "--snrs", "5,10",
        "--seed", "42",
        "--device", "cuda",
        "--batch_size", "64",
        "--classifier_ckpt", "checkpoints/aid_classifier_resnet34/best.pt",
    ]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        rc = subprocess.run(cmd, cwd=ROOT, env=env_for_run(),
                             stdout=logf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] eval exit={rc}  dt={dt:.1f}s",
          flush=True)
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-configs", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-skip", action="store_true",
                        help="re-train even if best.pt already exists")
    args = parser.parse_args()
    if args.gen_configs:
        gen_configs()
    if args.train:
        rc = train_all(skip_existing=not args.no_skip)
        if rc != 0:
            sys.exit(rc)
    if args.eval:
        rc = eval_all()
        if rc != 0:
            sys.exit(rc)
    if not (args.gen_configs or args.train or args.eval):
        parser.print_help()


if __name__ == "__main__":
    main()
