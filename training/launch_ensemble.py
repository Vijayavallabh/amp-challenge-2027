"""Launch an 8-way generator ensemble, one model per H100, and wait for all to finish.

Run in the background:
    uv run --group train python training/launch_ensemble.py

Each model trains on its own GPU with a distinct seed; init and data-order differences give
ensemble diversity for sampling. Logs land in training/runs/gen{seed}/train.log.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
N_GPUS = 8
EPOCHS = 80
COMMON = ["--epochs", str(EPOCHS), "--batch-size", "512", "--lr", "3e-4",
          "--d-model", "384", "--n-layers", "6", "--n-heads", "6",
          "--eval-every", "10", "--sample-n", "3000", "--dropout", "0.1"]


def main() -> int:
    runs_dir = REPO_ROOT / "training" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for gpu in range(N_GPUS):
        seed = gpu
        out = runs_dir / f"gen{seed}"
        cmd = [sys.executable, str(REPO_ROOT / "training" / "train_generator.py"),
               "--gpu", str(gpu), "--seed", str(seed), "--out", str(out), *COMMON]
        logf = open(out.with_suffix(".stdout"), "w") if False else subprocess.DEVNULL
        print(f"launching generator seed={seed} on GPU {gpu} -> {out}", flush=True)
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        time.sleep(2)  # stagger startup so CUDA init doesn't thundering-herd

    print(f"all {N_GPUS} training jobs launched; waiting...", flush=True)
    t0 = time.time()
    codes = [p.wait() for p in procs]
    print(f"ensemble finished in {time.time()-t0:.0f}s; exit codes={codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
