"""Bulk APEX scoring, sharded across CPU cores or GPUs.

Offline tool (NOT part of the shipped, deterministic ``uv run generate`` path). It scores
large candidate pools through APEX-pathogen by fanning shards out over many CPU workers or
the 8 H100s, and caches the ``(n, 11)`` MIC matrix to an ``.npz`` so objective/selection
design and rejection-sampling fine-tuning can iterate without re-scoring.

Determinism is NOT required here (this never touches the submission bytes), so GPU scoring
is fair game. The shipped ranking still uses CPU APEX via ``src/.../oracle.py``.

Usage:
    python experiments/bulk_apex.py --in seqs.txt --out pool.npz --device cpu --workers 24
    python experiments/bulk_apex.py --in seqs.txt --out pool.npz --device cuda \
        --gpu-python /path/to/agpu/bin/python --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
APEX_DIR = REPO / "oracle" / "apex"
N_PATHOGENS = 11


def read_seqs(path: Path) -> list[str]:
    """Read sequences from a one-per-line .txt or a .fasta."""
    text = path.read_text()
    if path.suffix in (".fasta", ".fa"):
        seqs, cur = [], []
        for line in text.splitlines():
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        return seqs
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _shard_bounds(n: int, k: int) -> list[tuple[int, int]]:
    step = (n + k - 1) // k
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def _launch(
    seqs: list[str], tmp: Path, worker_id: int, *, python: str, gpu: str | None, threads: int
) -> tuple[subprocess.Popen, Path]:
    in_fa = tmp / f"in_{worker_id}.fasta"
    out_csv = tmp / f"out_{worker_id}.csv"
    with in_fa.open("w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">s{i}\n{s}\n")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    gpu_flag = "0"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
        gpu_flag = "1"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU, hide GPUs
    # Keep CPU workers from oversubscribing cores against each other.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(threads)
    cmd = [python, "APEX_predict.py", "-i", str(in_fa), "-o", str(out_csv), "-g", gpu_flag]
    proc = subprocess.Popen(
        cmd, cwd=str(APEX_DIR), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    return proc, out_csv


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with path.open() as fh:
        rows = list(csv.reader(fh))
    for row in rows[1:]:
        out[row[0]] = np.array([float(x) for x in row[1:]], dtype=np.float32)
    return out


CPU_PYTHON = str(APEX_DIR / ".venv" / "bin" / "python")  # APEX's own CPU torch env


def score(
    seqs: list[str], *, device: str, workers: int, gpu_python: str, gpus: list[str]
) -> tuple[list[str], np.ndarray]:
    uniq = list(dict.fromkeys(seqs))
    n = len(uniq)
    if device == "cuda":
        python = gpu_python
        assign = gpus
        nshards = len(gpus)
        threads = 8
    else:
        python = CPU_PYTHON
        assign = [None] * workers
        nshards = workers
        threads = max(1, os.cpu_count() // max(1, workers) - 1)
    bounds = _shard_bounds(n, nshards)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        procs = []
        for wid, (lo, hi) in enumerate(bounds):
            gpu = assign[wid % len(assign)]
            procs.append((_launch(uniq[lo:hi], tmp, wid, python=python, gpu=gpu, threads=threads), (lo, hi)))
        mic_by_seq: dict[str, np.ndarray] = {}
        for (proc, out_csv), (lo, hi) in procs:
            _, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"APEX worker failed (shard {lo}:{hi}): {err[-400:]}")
            mic_by_seq.update(_read_csv(out_csv))

    missing = [s for s in uniq if s not in mic_by_seq]
    if missing:
        raise RuntimeError(f"{len(missing)} seqs unscored, e.g. {missing[0]!r}")
    mic = np.array([mic_by_seq[s] for s in uniq], dtype=np.float32)
    return uniq, mic


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--workers", type=int, default=24, help="CPU shards (device=cpu)")
    ap.add_argument("--gpu-python", type=str, default="", help="python with CUDA torch 2.5.1")
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    args = ap.parse_args()

    seqs = read_seqs(args.inp)
    gpus = [g for g in args.gpus.split(",") if g != ""]
    t0 = time.time()
    uniq, mic = score(seqs, device=args.device, workers=args.workers,
                      gpu_python=args.gpu_python, gpus=gpus)
    dt = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, sequences=np.array(uniq, dtype=object), mic=mic)
    print(f"scored {len(uniq)} unique seqs in {dt:.1f}s ({len(uniq)/dt:.0f}/s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
