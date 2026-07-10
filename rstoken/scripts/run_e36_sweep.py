"""E36 — Continuous SNR sweep runner.

Spawns scripts/09_eval_rvqs_recon_task_split.py 6 times (one per checkpoint),
covering the union AWGN+Rayleigh SNR grid. Pure Python so it survives
PowerShell encoding issues with Chinese paths.

Usage:
    python scripts/run_e36_sweep.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # rstoken/
LOG_DIR = ROOT / "logs" / "paper_v05"
RUN_LOG = LOG_DIR / "run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG.mkdir(parents=True, exist_ok=True)

PY = r"C:\Users\Administrator\miniconda3\envs\rstoken\python.exe"
SCRIPT = ROOT / "scripts" / "09_eval_rvqs_recon_task_split.py"
DATA_YAML = ROOT / "configs" / "paper_v05" / "eval_aid_local.yaml"

# Union of AWGN (we want -5..15) and Rayleigh (-5..20) SNR sweeps.
# condition_grid() uses the same snr list for both channels; readers can
# trim the AWGN tail at +20 if they prefer the canonical waterfall window.
SNRS = "-5,-2,0,2,5,7,10,12,15,20"
KS = "1,4"

MODELS = [
    ("rvq_baseline_s41", "checkpoints/rvq_baseline_s41/best.pt"),
    ("rvq_baseline_s42", "checkpoints/rvq_baseline/best.pt"),
    ("rvq_baseline_s43", "checkpoints/rvq_baseline_s43/best.pt"),
    ("rvq_distill_s41",  "checkpoints/rvq_distill_s41/best.pt"),
    ("rvq_distill_s42",  "checkpoints/rvq_distill/best.pt"),
    ("rvq_distill_s43",  "checkpoints/rvq_distill_s43/best.pt"),
]


def run_one(name: str, ckpt: str) -> int:
    tcsv = LOG_DIR / f"e36_task_{name}.csv"
    rcsv = LOG_DIR / f"e36_recon_{name}.csv"
    log_path = RUN_LOG / f"e36_{name}.log"
    cmd = [
        PY, str(SCRIPT),
        "--models", f"{name}={ckpt}",
        "--recon_models", name,
        "--task_out", str(tcsv.relative_to(ROOT)).replace("\\", "/"),
        "--recon_out", str(rcsv.relative_to(ROOT)).replace("\\", "/"),
        # leading-dash values: use --flag=value so argparse never treats
        # "-5,-2,..." as a new option.
        f"--task_snrs={SNRS}",
        f"--recon_snrs={SNRS}",
        "--ks", KS,
        "--seed", "42",
        "--device", "cuda",
        "--batch_size", "64",
        "--data_yaml", str(DATA_YAML.relative_to(ROOT)).replace("\\", "/"),
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "RSTOKEN_LOGREG_MAX_ITER": "300",
        "RSTOKEN_LOGREG_TOL": "1e-3",
    })
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] === {name} ===", flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        logf.write(f"# E36 sweep {name}\n# cmd: {cmd}\n\n")
        logf.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env,
                              stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] {name} exit={proc.returncode}  "
          f"dt={dt:.1f}s  log={log_path}", flush=True)
    return proc.returncode


def main() -> int:
    print(f"E36 runner: {len(MODELS)} checkpoints, snrs={SNRS}, ks={KS}",
          flush=True)
    for name, ckpt in MODELS:
        ckpt_path = ROOT / ckpt
        if not ckpt_path.exists():
            print(f"  MISSING: {ckpt_path}", flush=True)
            return 2
    for name, ckpt in MODELS:
        rc = run_one(name, ckpt)
        if rc != 0:
            print(f"FAILED on {name} (exit {rc})", flush=True)
            return rc
    print(f"[{time.strftime('%H:%M:%S')}] E36 sweep done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
