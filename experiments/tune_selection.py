"""Tune the shipped selection (category_success_score with gp/mdr up-weight, minus lambda*hemo)
on a fixed scored pool -- a free knob, no retraining. Grids over gp_weight and lambda, applies
the real 0.6 diversity cap, and prints the diversity-capped top-50 profile so the breadth /
Gram-balance / selectivity knee is visible. Predicted quantities only.
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
from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import category_success_score  # noqa: E402


def greedy_diverse(seqs, scores, k, cap=0.6, scan=80000):
    order = np.argsort(-scores)
    sel, idx = [], []
    for i in order[:scan]:
        s = seqs[i]
        if not sel or C.max_identity(s, sel, cutoff=cap) < cap:
            sel.append(s)
            idx.append(int(i))
        if len(sel) >= k:
            break
    return np.array(idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--gp-weights", type=str, default="0.0,0.5,1.0")
    ap.add_argument("--lambdas", type=str, default="0.3,0.6,1.0")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    seqs = list(d["sequences"])
    mic = d["mic"].astype(np.float64)
    from amp_challenge_2027.hemolysis import HemolysisScorer
    phemo = np.asarray(HemolysisScorer("checkpoint/hemolysis.pt").predict_proba(seqs))

    print(f"pool={len(seqs)}  (diversity-capped top-50; predicted, no wet-lab claim)")
    print(f"  {'gp_w':>4} {'lam':>4} | {'broad':>5} {'GN':>5} {'GP':>5} {'MDR':>5} "
          f"{'brd':>5} {'phemo':>6}  {'sum4':>5}")
    for gpw in [float(x) for x in args.gp_weights.split(",")]:
        for lam in [float(x) for x in args.lambdas.split(",")]:
            sc = category_success_score(mic, gp_weight=gpw, mdr_weight=gpw) - lam * phemo
            idx = greedy_diverse(seqs, sc, 50)
            s = R.summarize(mic, phemo, idx)
            sum4 = s["SR_broad"] + s["SR_gn"] + s["SR_gp"] + s["SR_mdr"]
            print(f"  {gpw:>4.1f} {lam:>4.1f} | {s['SR_broad']:>5.1f} {s['SR_gn']:>5.1f} "
                  f"{s['SR_gp']:>5.1f} {s['SR_mdr']:>5.1f} {s['mean_breadth']:>5.2f} "
                  f"{s['mean_phemo']:>6.3f}  {sum4:>5.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
