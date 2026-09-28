"""Bulk PER-SUB-MODEL APEX scoring, sharded across GPUs -> the full ``(n, 8, 11)`` MIC tensor
(every APEX ensemble member kept separate, not averaged).

Why keep the submodels separate: directed evolution (``experiments/directed_evolution.py``)
OPTIMIZES a reward on a *train split* of the 8 submodels and SELECTS / VALIDATES on a *held-out
split*. A genuine potency gain generalizes across submodels; a sequence that merely games APEX's
shared weights improves on the train split but not the held-out one -- so the split is the core
anti-Goodhart guard for optimizing against an oracle.

Offline only -- this never touches the byte-deterministic ``uv run generate`` path. Reuses
``experiments/apex_permodel.py`` unchanged as the per-shard worker (it already emits (n, 8, 11)).

    from bulk_permodel import score_permodel
    uniq, mic8 = score_permodel(seqs, gpus=["1","2","3","4","5","6","7"], gpu_python="/.../agpu/bin/python")
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "experiments" / "apex_permodel.py"
N_SUB, N_PATH = 8, 11


def _shard_bounds(n: int, k: int) -> list[tuple[int, int]]:
    step = (n + k - 1) // k
    return [(i, min(i + step, n)) for i in range(0, n, step)]


def score_permodel(
    seqs: list[str], *, gpus: list[str], gpu_python: str
) -> tuple[list[str], np.ndarray]:
    """Score ``seqs`` through every APEX submodel, sharded one shard per GPU.

    Returns ``(uniq_seqs, mic8)`` with ``mic8`` of shape ``(len(uniq_seqs), 8, 11)`` in uM.
    APEX drops sequences longer than 50 residues; those get NaN rows (keep peptides <= MAX_LENGTH).
    """
    uniq = list(dict.fromkeys(seqs))
    n = len(uniq)
    if n == 0:
        return uniq, np.zeros((0, N_SUB, N_PATH), dtype=np.float64)
    bounds = _shard_bounds(n, min(len(gpus), n))
    env0 = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    by_seq: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        procs = []
        for wid, (lo, hi) in enumerate(bounds):
            in_fa = tmp / f"in_{wid}.fasta"
            out_npz = tmp / f"out_{wid}.npz"
            with in_fa.open("w") as fh:
                for i, s in enumerate(uniq[lo:hi]):
                    fh.write(f">s{i}\n{s}\n")
            env = dict(env0)
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[wid % len(gpus)])
            proc = subprocess.Popen(
                [gpu_python, str(WORKER), "-i", str(in_fa), "-o", str(out_npz), "-g", "1"],
                cwd=str(REPO), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            procs.append((proc, out_npz))
        for proc, out_npz in procs:
            _, err = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"permodel worker failed (rc={proc.returncode}):\n{(err or '')[-2000:]}")
            d = np.load(out_npz, allow_pickle=True)
            for s, row in zip(d["sequences"].tolist(), d["permodel"]):
                by_seq[str(s)] = np.asarray(row, dtype=np.float64)  # (8, 11)
    mic8 = np.full((n, N_SUB, N_PATH), np.nan, dtype=np.float64)
    for i, s in enumerate(uniq):
        row = by_seq.get(s)
        if row is not None:
            mic8[i] = row
    return uniq, mic8
