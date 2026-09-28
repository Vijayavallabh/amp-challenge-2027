"""Fast archetype comparison from a cached scored pool (no full novelty screen).

Applies the 0.6 within-list diversity cap (the real shaping constraint on the top-100) via a
greedy pass checking only against the growing selected list -- fast even on a collapsed pool.
Reports the diversity-capped top-100/top-50 profile, a library-diversity proxy, and a small
reference-novelty sample. Used to pick the ReST archetype; the winner then gets the full
shipped selection + official validator.
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
from amp_challenge_2027.oracle import broad_potency_score  # noqa: E402


def greedy_diverse(seqs, scores, k, cap=0.6, scan_limit=60000):
    order = np.argsort(-scores)
    sel, sel_idx = [], []
    for i in order[:scan_limit]:
        s = seqs[i]
        if not sel or C.max_identity(s, sel, cutoff=cap) < cap:
            sel.append(s)
            sel_idx.append(int(i))
        if len(sel) >= k:
            break
    return sel, np.array(sel_idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    seqs = list(d["sequences"])
    mic = d["mic"].astype(np.float64)
    from amp_challenge_2027.hemolysis import HemolysisScorer
    phemo = np.asarray(HemolysisScorer("checkpoint/hemolysis.pt").predict_proba(seqs))

    print(f"===== {args.tag} (pool={len(seqs)}) =====")
    # library-diversity proxy
    rng = np.random.default_rng(0)
    sub = [seqs[i] for i in rng.choice(len(seqs), size=min(400, len(seqs)), replace=False)]
    dsel = []
    for s in sub:
        if not dsel or C.max_identity(s, dsel, cutoff=0.6) < 0.6:
            dsel.append(s)
    print(f"  library diversity: {len(dsel)}/{len(sub)} of a random subsample mutually diverse @0.6")

    objectives = {
        "broad-2hemo": broad_potency_score(mic) - 2.0 * phemo,
        "reward(gp1,l1)": R.reward(mic, phemo, gp_w=1.0, mdr_w=1.0, hemo_lambda=1.0),
    }
    for name, sc in objectives.items():
        _, idx100 = greedy_diverse(seqs, sc, 100)
        if len(idx100) < 100:
            print(f"  [{name}] only filled {len(idx100)}/100 diverse @0.6 -- COLLAPSED")
        print(f"  [{name}] div-top100 " + R.fmt(R.summarize(mic, phemo, idx100), ""))
        print(f"  [{name}] div-top50  " + R.fmt(R.summarize(mic, phemo, idx100[:50]), ""))

    # reference novelty on a small random sample -- TRUE max identity (rapidfuzz, no early-exit;
    # constraints.max_identity is a >cutoff boolean helper and returns 0 below cutoff).
    from rapidfuzz import distance, process

    from amp_challenge_2027.fasta import read_sequences
    from amp_challenge_2027.paths import resolve_repo_path
    ref = read_sequences(resolve_repo_path("data/antibacterial.fasta"))
    samp = [seqs[i] for i in rng.choice(len(seqs), size=min(200, len(seqs)), replace=False)]
    maxid = np.array([process.extractOne(s, ref, scorer=distance.Levenshtein.normalized_similarity)[1]
                      for s in samp])
    print(f"  reference novelty (200-sample): median max-id={np.median(maxid):.3f} "
          f"p90={np.percentile(maxid, 90):.3f} frac>0.8={float((maxid > 0.8).mean()):.3f} "
          f"(rule: top-100 must be <=0.80)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
