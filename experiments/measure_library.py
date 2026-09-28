"""Measure a generated library on the Phase-2 advancement axes, using the organizers' own
framework (``seqme``).

Per docs/COMPETITION.md, Phase 2 (top-20 advancement) scores the full 50,000-sequence
library with ``seqme`` for **diversity, novelty against known-AMP databases, and
physicochemical property distributions**. Phase 3 then draws 25 of the top 50 at random,
so Phase 2 is the gate: miss it and the category scores never get measured. This script
reports exactly those axes so the advancement risk is quantified, not assumed.

WHY THIS IS NOT IN THE MAIN uv ENV
----------------------------------
``seqme`` pulls heavy, submission-irrelevant deps (torch, transformers, umap, modlamp).
The submission's ``uv run generate`` never imports it, and adding it would bloat the
lockfile and put the byte-reproducible entry point at risk for zero benefit. So this is a
MEASUREMENT tool run from a throwaway isolated env, never a submission dependency:

    uv venv /tmp/seqme-env --python 3.11
    VIRTUAL_ENV=/tmp/seqme-env uv pip install --python /tmp/seqme-env/bin/python \
        "seqme[aa_descriptors,esm2]"
    CUDA_VISIBLE_DEVICES=1 HF_HOME=$HF_HOME /tmp/seqme-env/bin/python \
        experiments/measure_library.py submission/generate/library.fasta

seqme is BSD-3-Clause (szczurek-lab); using it as a measurement tool imposes no licence
obligation on the MIT submission (no seqme code is vendored or shipped).

RESULT (feat-021 submission library, 2026-09-28, ESM2-650M):
    Uniqueness 1.000 | Diversity 0.839 | Novelty 1.000 | 3-gram-Jaccard 0.0019
    physchem: charge 4.7, pI 11.6, hydrophobic-moment 0.40 (AMP-like, cationic, amphipathic)
    FBD 1.94  vs  real-AMP floor 0.074 / random ceiling 5.42  ->  35% toward random:
        firmly AMP-like (3x closer to real AMPs than to random) while staying Novelty=1.0.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path: str) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", help="path to library.fasta (the 50k design library)")
    ap.add_argument("--reference", default="data/antibacterial.fasta",
                    help="known-AMP reference set for novelty (default: the training reference)")
    ap.add_argument("--n", type=int, default=4000, help="sample size for O(n^2)/embedding metrics")
    ap.add_argument("--fbd", action="store_true",
                    help="also compute FBD in ESM2-650M space with random/real controls (needs a GPU)")
    ap.add_argument("--label", default="library")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    lib = read_fasta(args.library)
    ref = read_fasta(args.reference)
    print(f"[{args.label}] library={len(lib)}  reference={len(ref)}")

    import seqme as sm
    from seqme.metrics import Diversity, NGramJaccardSimilarity, Novelty, Uniqueness

    t = time.time()
    metrics = [
        Uniqueness(),
        Diversity(k=min(args.n, 2000), seed=0),
        Novelty(reference=ref),
        NGramJaccardSimilarity(reference=ref, n=3, objective="minimize"),
    ]
    df = sm.evaluate({args.label: lib}, metrics, verbose=False)
    print(f"\n=== diversity / novelty ({time.time()-t:.0f}s) ===")
    print(df.to_string())

    # physchem distribution suite (needs seqme[aa_descriptors])
    import seqme.models as MD
    sub = list(rng.choice(lib, min(len(lib), 20000), replace=False))
    print("\n=== physchem distributions (mean +/- std) ===")
    for name, ctor in [
        ("Charge", MD.Charge), ("Gravy", MD.Gravy), ("HydrophobicMoment", MD.HydrophobicMoment),
        ("IsoelectricPoint", MD.IsoelectricPoint), ("Aromaticity", MD.Aromaticity),
        ("BomanIndex", MD.BomanIndex), ("InstabilityIndex", MD.InstabilityIndex),
        ("AliphaticIndex", MD.AliphaticIndex),
    ]:
        try:
            v = np.asarray(ctor()(sub), dtype=float).ravel()
            print(f"  {name:20s} {np.nanmean(v):8.3f} +/- {np.nanstd(v):7.3f}"
                  f"   [p10 {np.nanpercentile(v,10):7.2f}  p90 {np.nanpercentile(v,90):7.2f}]")
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  {name:20s} ERR {type(e).__name__}: {str(e)[:70]}")

    if args.fbd:
        from seqme.metrics import FBD
        from seqme.models import ESM2, ESM2Checkpoint

        refv = [s for s in ref if 8 <= len(s) <= 50 and set(s) <= set(AA)]
        rng.shuffle(refv)
        ref_a, ref_b = refv[: len(refv) // 2], refv[len(refv) // 2: len(refv) // 2 + args.n]
        lib_s = list(rng.choice(lib, min(args.n, len(lib)), replace=False))
        lens = rng.choice([len(s) for s in lib], size=args.n)
        aa = np.frombuffer(AA.encode(), dtype="S1")
        rand = ["".join(aa[rng.integers(0, 20, size=int(L))].astype(str)) for L in lens]

        t = time.time()
        emb = ESM2(ESM2Checkpoint.t33_650M, device="cuda", batch_size=256,
                   cache_dir=os.environ.get("HF_HOME"), verbose=False)
        df2 = sm.evaluate(
            {"ref-heldout (POS)": ref_b, args.label: lib_s, "random (NEG)": rand},
            [FBD(reference=ref_a, embedder=emb)], verbose=False,
        )
        print(f"\n=== FBD vs real-AMP anchor, ESM2-650M ({time.time()-t:.0f}s) ===")
        print(df2.to_string())
        print("lower = more distributionally AMP-like; we should sit near POS, far below NEG.")


if __name__ == "__main__":
    main()
