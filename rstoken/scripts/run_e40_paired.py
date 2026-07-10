"""E40 — Drive 6 paired RVQ trainings (3 seeds × {baseline, distill}).

Each pair starts from the same step-0 init checkpoint
(`checkpoints/paper_v05/rvq_init_s{seed}/step0.pt`); the only difference
within a pair is `loss.distill_weight` (0.0 vs 0.5) plus the `distill` block.

Order: baseline before distill within each seed; seeds 41 → 42 → 43 in
sequence. Single GPU; trainings run sequentially. Each ckpt is ~95 minutes
(reference: E37); 6 ckpts ≈ 9.5 hours wall-clock.

Skip behavior: any run whose `best.pt` already exists is skipped (so a
crashed runner can be resumed safely).

Usage:
    python scripts/run_e40_paired.py
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

SEEDS = (41, 42, 43)
FAMILIES = ("baseline", "distill")  # baseline first inside each seed


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


def all_runs() -> list[tuple[str, Path, Path, Path]]:
    """Return (run_name, config_path, ckpt_dir, init_ckpt_path) tuples."""
    runs = []
    for seed in SEEDS:
        init_ck = CKPT_DIR / f"rvq_init_s{seed}" / "step0.pt"
        for fam in FAMILIES:
            name = f"rvq_{fam}_paired_s{seed}"
            cfg = CONFIG_DIR / f"{name}.yaml"
            ck = CKPT_DIR / name
            runs.append((name, cfg, ck, init_ck))
    return runs


def main() -> int:
    RUN_LOG.mkdir(parents=True, exist_ok=True)
    runs = all_runs()

    # Sanity: configs and init ckpts must all exist before we start.
    for name, cfg, _ck, init in runs:
        if not cfg.exists():
            print(f"MISSING config: {cfg}", flush=True)
            return 2
        if not init.exists():
            print(f"MISSING init ckpt: {init}", flush=True)
            return 2

    print(f"E40 paired driver — {len(runs)} runs, sequential", flush=True)
    print(f"  ROOT     = {ROOT}", flush=True)
    print(f"  PY       = {PY}", flush=True)
    print(f"  RUN_LOG  = {RUN_LOG}", flush=True)
    overall_t0 = time.time()

    for idx, (name, cfg, ck, init) in enumerate(runs, start=1):
        best = ck / "best.pt"
        log_path = RUN_LOG / f"e40_train_{name}.log"
        if best.exists():
            print(f"[{time.strftime('%H:%M:%S')}] ({idx}/{len(runs)}) skip "
                  f"{name} (best.pt exists)", flush=True)
            continue

        # Pass --config / --init_from_ckpt as relative paths to the project
        # root, matching the cwd=ROOT convention used by every other v0.5 driver.
        cfg_rel = cfg.relative_to(ROOT).as_posix()
        init_rel = init.relative_to(ROOT).as_posix()

        cmd = [
            PY, str(ROOT / "scripts" / "03_train_vqvae.py"),
            "--config", cfg_rel,
            "--init_from_ckpt", init_rel,
        ]

        print(f"[{time.strftime('%H:%M:%S')}] ({idx}/{len(runs)}) === "
              f"train {name} ===", flush=True)
        print(f"   cmd: {' '.join(cmd)}", flush=True)
        print(f"   log: {log_path}", flush=True)

        t0 = time.time()
        with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
            logf.write(f"# E40 train {name}\n")
            logf.write(f"# config = {cfg_rel}\n")
            logf.write(f"# init   = {init_rel}\n")
            logf.flush()
            rc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env_for_run(),
                stdout=logf,
                stderr=subprocess.STDOUT,
            ).returncode
        dt = time.time() - t0
        # Truth check: PowerShell-style exit codes can be flaky; rely on
        # whether best.pt landed on disk.
        landed = best.exists()
        print(f"[{time.strftime('%H:%M:%S')}] {name} exit={rc}  "
              f"best.pt_exists={landed}  dt={dt:.1f}s", flush=True)

        if not landed:
            print(f"FAILED on {name}: best.pt missing.", flush=True)
            return rc if rc != 0 else 3

    overall_dt = time.time() - overall_t0
    print(f"[{time.strftime('%H:%M:%S')}] all 6 runs done. total dt={overall_dt:.1f}s "
          f"({overall_dt/3600:.2f} h)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
