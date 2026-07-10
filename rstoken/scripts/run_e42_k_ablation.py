"""E42 — RVQ K-ablation: train K∈{2, 3, 6} at seed 42.

K=4 main result already exists at `checkpoints/rvq_distill/best.pt` (s42).
This driver only trains the three new K values; aggregation pulls K=4
numbers from the existing main run.

Each ckpt is ~95 minutes (reference: E37/E40); 3 ckpts ≈ 4.7 hours.

Skip behavior: a run whose `best.pt` already exists is skipped.

Usage:
    python scripts/run_e42_k_ablation.py
"""
from __future__ import annotations

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

K_VALUES = (2, 3, 6)


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


def all_runs() -> list[tuple[str, Path, Path]]:
    runs = []
    for K in K_VALUES:
        name = f"rvq_distill_K{K}_s42"
        cfg = CONFIG_DIR / f"{name}.yaml"
        ck = CKPT_DIR / name
        runs.append((name, cfg, ck))
    return runs


def main() -> int:
    RUN_LOG.mkdir(parents=True, exist_ok=True)
    runs = all_runs()

    for name, cfg, _ck in runs:
        if not cfg.exists():
            print(f"MISSING config: {cfg}", flush=True)
            return 2

    print(f"E42 K-ablation driver — {len(runs)} runs, sequential", flush=True)
    print(f"  ROOT     = {ROOT}", flush=True)
    print(f"  PY       = {PY}", flush=True)
    print(f"  RUN_LOG  = {RUN_LOG}", flush=True)
    overall_t0 = time.time()

    for idx, (name, cfg, ck) in enumerate(runs, start=1):
        best = ck / "best.pt"
        log_path = RUN_LOG / f"e42_train_{name}.log"
        if best.exists():
            print(f"[{time.strftime('%H:%M:%S')}] ({idx}/{len(runs)}) skip "
                  f"{name} (best.pt exists)", flush=True)
            continue

        cfg_rel = cfg.relative_to(ROOT).as_posix()
        cmd = [
            PY, str(ROOT / "scripts" / "03_train_vqvae.py"),
            "--config", cfg_rel,
        ]

        print(f"[{time.strftime('%H:%M:%S')}] ({idx}/{len(runs)}) === "
              f"train {name} ===", flush=True)
        print(f"   cmd: {' '.join(cmd)}", flush=True)
        print(f"   log: {log_path}", flush=True)

        t0 = time.time()
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            logf.write(f"# E42 train {name}\n")
            logf.write(f"# config = {cfg_rel}\n")
            logf.flush()
            rc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env_for_run(),
                stdout=logf,
                stderr=subprocess.STDOUT,
            ).returncode
        dt = time.time() - t0
        # Truth check (lesson 2): rely on best.pt existence, not exit code.
        landed = best.exists()
        print(f"[{time.strftime('%H:%M:%S')}] {name} exit={rc}  "
              f"best.pt_exists={landed}  dt={dt:.1f}s", flush=True)

        if not landed:
            print(f"FAILED on {name}: best.pt missing.", flush=True)
            return rc if rc != 0 else 3

    overall_dt = time.time() - overall_t0
    print(f"[{time.strftime('%H:%M:%S')}] all 3 runs done. total dt={overall_dt:.1f}s "
          f"({overall_dt/3600:.2f} h)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
