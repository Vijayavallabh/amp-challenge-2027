"""Which selection objective best balances the 5 scored categories on the shipped pool?

Uses the cached 150k pool scores (experiments/cache/pool_scores.npz from tune_category_weights.py).
The ceiling analysis showed the pool DOES contain GP/MDR-active *and* non-hemolytic peptides, but the
shipped category_success_score (dominated by the broad soft-mean) under-surfaces them. Here we try
several deterministic objectives, run each through the REAL select_top screens (novelty 0.80 +
diversity 0.60), and report the screened top-50's per-category mean Success Rate (as the competition
scores it) PLUS diversity/novelty health -- a Goodhart guard (a GP push must not collapse to a
near-duplicate cluster or memorised AMPs). Offline; picks an objective to wire into ApexRanker.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from amp_challenge_2027.oracle import GRAM_NEG, GRAM_POS, MDR, _soft_success  # noqa: E402


def load():
    d = np.load(REPO / "experiments" / "cache" / "pool_scores.npz", allow_pickle=True)
    return list(d["seqs"]), d["mic"], d["phemo"]


def soft(mic):
    return _soft_success(mic)  # (n,11) in [0,1]


# --- objectives: (name, fn(mic, phemo) -> per-peptide score) ---
def obj_baseline(mic, ph):
    s = soft(mic)
    act = s.mean(1) + 0.5 * s[:, list(GRAM_POS)].mean(1) + 0.5 * s[:, list(MDR)].mean(1)
    return act - 0.5 * ph

def obj_softbucket(mic, ph):
    # equal-weight the four soft bucket-SRs, then subtract hemolysis
    s = soft(mic)
    b = 0.25 * (s.mean(1) + s[:, list(GRAM_NEG)].mean(1) + s[:, list(GRAM_POS)].mean(1) + s[:, list(MDR)].mean(1))
    return b - 0.5 * ph

def obj_gpmdr_soft(mic, ph):
    # broad soft-mean + strong GP/MDR soft emphasis - hemolysis
    s = soft(mic)
    return s.mean(1) + 1.5 * s[:, list(GRAM_POS)].mean(1) + 1.5 * s[:, list(MDR)].mean(1) - 0.5 * ph

def obj_min_soft(mic, ph):
    # maximise the weakest soft bucket (broad/GN/GP/MDR) - hemolysis  (lifts the laggard category)
    s = soft(mic)
    stacked = np.stack([s.mean(1), s[:, list(GRAM_NEG)].mean(1),
                        s[:, list(GRAM_POS)].mean(1), s[:, list(MDR)].mean(1)], 1)
    return stacked.min(1) - 0.5 * ph

def obj_min_plus_mean(mic, ph):
    # weakest bucket + 0.5*mean bucket - hemolysis  (balance, but don't ignore breadth)
    s = soft(mic)
    stacked = np.stack([s.mean(1), s[:, list(GRAM_NEG)].mean(1),
                        s[:, list(GRAM_POS)].mean(1), s[:, list(MDR)].mean(1)], 1)
    return stacked.min(1) + 0.5 * stacked.mean(1) - 0.5 * ph

OBJS = [("baseline(shipped)", obj_baseline), ("soft-bucket-equal", obj_softbucket),
        ("gpmdr-soft-1.5", obj_gpmdr_soft), ("min-bucket", obj_min_soft),
        ("min+0.5mean", obj_min_plus_mean)]


def main():
    from amp_challenge_2027.generate import select_top

    seqs, mic, ph = load()
    mic_l = {s: mic[i] for i, s in enumerate(seqs)}
    ph_l = {s: float(ph[i]) for i, s in enumerate(seqs)}
    reference = [l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
                 if l.strip() and not l.startswith(">")]
    hit = (mic <= 16.0)
    gp_all = hit[:, list(GRAM_POS)].mean(1)
    mdr_all = hit[:, list(MDR)].mean(1)
    gn_all = hit[:, list(GRAM_NEG)].mean(1)
    broad_all = hit.mean(1)
    idx_of = {s: i for i, s in enumerate(seqs)}

    class R:
        name = "obj"
        def __init__(self, fn): self.v = {s: float(x) for s, x in zip(seqs, fn(mic, ph))}
        def score(self, ss): return [self.v[s] for s in ss]

    def selfid(top):  # within-top-50 novelty (median identity to nearest OTHER member): diversity health
        ids = []
        for i, s in enumerate(top):
            others = top[:i] + top[i + 1:]
            ids.append(process.extractOne(s, others, scorer=fuzz.ratio)[1] / 100.0)
        return float(np.median(ids))

    def knownid(top):  # median identity to nearest known AMP (memorisation/Goodhart guard)
        return float(np.median([process.extractOne(s, reference, scorer=fuzz.ratio)[1] / 100.0 for s in top]))

    print(f"{'objective':20s} | {'broad':>5s} {'GN':>5s} {'GP':>5s} {'MDR':>5s} {'medPh':>5s} | "
          f"{'mean5':>5s} {'min4':>5s} | {'selfID':>6s} {'knownID':>7s}")
    for name, fn in OBJS:
        top100 = select_top(seqs, R(fn), 100, reference, diversity_max_identity=0.6)
        top = top100[:50]
        ii = [idx_of[s] for s in top]
        b, gn, gp, md = broad_all[ii].mean(), gn_all[ii].mean(), gp_all[ii].mean(), mdr_all[ii].mean()
        mph = float(np.median([ph_l[s] for s in top]))
        mean5 = np.mean([b, gn, gp, md, 1 - mph]); min4 = min(b, gn, gp, md)
        print(f"{name:20s} | {b:5.2f} {gn:5.2f} {gp:5.2f} {md:5.2f} {mph:5.3f} | "
              f"{mean5:5.3f} {min4:5.3f} | {selfid(top):6.3f} {knownid(top):7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
