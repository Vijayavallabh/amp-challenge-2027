"""Is feat-021's balanced greedy top-50 already near-optimal for the *weakest category floor*?

The competition ranks five categories separately, so the weakest category is the highest-leverage
number. feat-021 selects by a weighted balanced score (a SUM). This probe instead runs an explicit
greedy MAXIMIN over the SAME feat-020 400k pool (same shipped APEX MIC + ESMC phemo, 0% hemolytic,
same 0.6 diversity + <80% novelty screens): repeatedly add the non-hemolytic, novel, diverse peptide
that most raises the current minimum of the four activity-category means. If maximin's floor beats
feat-021's Gram- 0.583 cleanly, it is a zero-risk, byte-deterministic win; if not, feat-021 is
confirmed near-optimal and stays.

CPU only; reuses experiments/cache/bigpool_400000.npz.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import GRAM_POS, GRAM_NEG, MDR, balanced_success_score  # noqa: E402

GN, GP, MD = list(GRAM_NEG), list(GRAM_POS), list(MDR)


def cats(mic_row):  # per-peptide (broad, GN, GP, MDR) success rates
    hit = (mic_row <= 16.0)
    return np.array([hit.mean(), hit[GN].mean(), hit[GP].mean(), hit[MD].mean()])


def main():
    ref = [l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
           if l.strip() and not l.startswith(">")]
    d = np.load(REPO / "experiments/cache/bigpool_400000.npz", allow_pickle=True)
    seqs, mic, ph = d["seqs"].tolist(), d["mic"].astype(float), d["phemo"].astype(float)

    # candidate space: non-hemolytic, then the top few thousand by balanced score (feat-021's space),
    # then novelty-screened (expensive, so only on that shortlist).
    bal = balanced_success_score(mic, gn_weight=0.75)
    cand = np.where(ph < 0.5)[0]
    cand = cand[np.argsort(-bal[cand])[:4000]]
    print(f"non-hemolytic {int((ph<0.5).sum())}; balanced-top-4000 shortlist -> novelty screen...", flush=True)
    novel = [int(i) for i in cand if C.is_novel_enough(seqs[i], ref)]
    print(f"novel shortlist: {len(novel)}", flush=True)
    C_cats = {i: cats(mic[i]) for i in novel}

    # --- feat-021 reference (balanced greedy, same as shipped) ---
    sel_b, rej = [], 0
    for i in sorted(novel, key=lambda i: -bal[i]):
        if len(sel_b) == 50:
            break
        if sel_b and C.max_identity(seqs[i], [seqs[j] for j in sel_b], cutoff=0.6) >= 0.6:
            continue
        sel_b.append(i)
    Bb = np.array([C_cats[i] for i in sel_b]).mean(0)
    print(f"\nbalanced greedy (=feat-021): broad {Bb[0]:.3f} GN {Bb[1]:.3f} GP {Bb[2]:.3f} MDR {Bb[3]:.3f} "
          f"| floor {Bb[1:].min():.3f} mean4 {Bb.mean():.3f}")

    # --- greedy MAXIMIN: add the peptide that most raises the running minimum category ---
    def maximin(weight_broad=True):
        sel = []
        cur_sum = np.zeros(4)
        pool = list(novel)
        while len(sel) < 50 and pool:
            best_i, best_key = None, None
            for i in pool:
                if sel and C.max_identity(seqs[i], [seqs[j] for j in sel], cutoff=0.6) >= 0.6:
                    continue
                m = (cur_sum + C_cats[i]) / (len(sel) + 1)
                # rank by (worst activity category, then total) -- lexicographic maximin
                key = (m[1:].min(), m.sum()) if weight_broad else (m[1:].min(), m[1:].sum())
                if best_key is None or key > best_key:
                    best_key, best_i = key, i
            if best_i is None:
                break
            sel.append(best_i)
            cur_sum = cur_sum + C_cats[best_i]
            pool.remove(best_i)
        return sel

    sel_m = maximin()
    Bm = np.array([C_cats[i] for i in sel_m]).mean(0)
    print(f"greedy maximin           : broad {Bm[0]:.3f} GN {Bm[1]:.3f} GP {Bm[2]:.3f} MDR {Bm[3]:.3f} "
          f"| floor {Bm[1:].min():.3f} mean4 {Bm.mean():.3f}  (n={len(sel_m)})")
    print(f"\nfloor delta (maximin - balanced): {Bm[1:].min() - Bb[1:].min():+.3f}")
    print("verdict:", "MAXIMIN WINS the floor -> worth shipping" if Bm[1:].min() - Bb[1:].min() > 0.02
          else "no meaningful floor gain -> feat-021 stays")


if __name__ == "__main__":
    raise SystemExit(main())
