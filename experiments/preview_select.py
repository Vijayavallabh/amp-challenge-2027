"""Preview the shipped top-100 selection on a GPU-scored pool.

Reuses the EXACT shipped selection (generate.select_top: novelty <=0.80 vs the reference set,
within-list diversity cap), but with precomputed APEX scores, so we can evaluate a candidate
generator's real top-100/top-50 profile without the slow CPU-APEX shipped run. Compares
scoring objectives (shipped broad_potency - lambda*hemolysis vs the ReST reward).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import reward as R  # noqa: E402
from amp_challenge_2027 import generate as G  # noqa: E402
from amp_challenge_2027.fasta import read_sequences  # noqa: E402
from amp_challenge_2027.oracle import broad_potency_score  # noqa: E402
from amp_challenge_2027.paths import resolve_repo_path  # noqa: E402


class Precomputed:
    def __init__(self, seqs, scores):
        self.d = dict(zip(seqs, scores))

    def score(self, seqs):
        return [float(self.d[s]) for s in seqs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="npz with sequences + mic")
    ap.add_argument("--objective", choices=("broad", "reward"), default="broad")
    ap.add_argument("--hemo-lambda", type=float, default=2.0)
    ap.add_argument("--gp-w", type=float, default=0.5)
    ap.add_argument("--mdr-w", type=float, default=0.5)
    ap.add_argument("--diversity", type=float, default=0.6)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    seqs = list(d["sequences"])
    mic = d["mic"].astype(np.float64)
    from amp_challenge_2027.hemolysis import HemolysisScorer
    phemo = np.asarray(HemolysisScorer("checkpoint/hemolysis.pt").predict_proba(seqs))

    if args.objective == "broad":
        scores = broad_potency_score(mic) - args.hemo_lambda * phemo
    else:
        scores = R.reward(mic, phemo, gp_w=args.gp_w, mdr_w=args.mdr_w, hemo_lambda=args.hemo_lambda)

    from amp_challenge_2027 import constraints as C

    # Library-diversity proxy (Phase-2 screens diversity/novelty): fraction of a random pool
    # subsample that is mutually diverse at the 0.6 identity cap. A collapsed generator scores low.
    rng = np.random.default_rng(0)
    sub = [seqs[i] for i in rng.choice(len(seqs), size=min(500, len(seqs)), replace=False)]
    dsel: list[str] = []
    for s in sub:
        if not dsel or C.max_identity(s, dsel, cutoff=0.6) < 0.6:
            dsel.append(s)
    print(f"pool diversity: {len(dsel)}/{len(sub)} of a random subsample survive the 0.6 "
          f"identity cap (library-diversity proxy; higher=more diverse)")

    reference = read_sequences(resolve_repo_path("data/antibacterial.fasta"))
    ranker = Precomputed(seqs, scores)
    top = G.select_top(seqs, ranker, args.top_k, reference, diversity_max_identity=args.diversity)

    idx = {s: i for i, s in enumerate(seqs)}
    top_i = np.array([idx[s] for s in top])
    top50_i = top_i[:50]
    print(f"pool={len(seqs)} objective={args.objective} lambda={args.hemo_lambda} "
          f"gp_w={args.gp_w} div={args.diversity}")
    print("  top100 " + R.fmt(R.summarize(mic, phemo, top_i), ""))
    print("  top50  " + R.fmt(R.summarize(mic, phemo, top50_i), ""))
    # novelty vs reference (max identity), on the selected top100
    maxid = [C.max_identity(s, reference, cutoff=1.0) for s in top[:50]]
    print(f"  top50 novelty: median max-id to known={np.median(maxid):.3f} "
          f"(rule <=0.80), max={np.max(maxid):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
