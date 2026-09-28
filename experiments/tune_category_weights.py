"""Diagnostic: can re-weighting gp/mdr in the ranking lift the top-50's Gram+/MDR Success Rate,
or is the generator's pool distribution the ceiling (=> we'd need a GP/MDR-targeted ReST round)?

Reproduces the exact shipped 150k pool (deterministic), caches the APEX MIC matrix and ESMC
P(hemolytic) once (npz), then sweeps (gp_weight, mdr_weight) through the *real* select_top screens
(novelty 0.80 + diversity 0.60) and reports each resulting top-50's per-category profile as the
competition scores it (mean per-peptide Success Rate at MIC<=16 uM, per strain-bucket) plus the
selectivity proxy. Offline analysis only; does not touch the shipped files.

    CUDA_VISIBLE_DEVICES=1 HF_HOME=/path uv run python experiments/tune_category_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CACHE = REPO / "experiments" / "cache"
POOL_NPZ = CACHE / "pool_scores.npz"

from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import GRAM_NEG, GRAM_POS, MDR, category_success_score  # noqa: E402


def build_pool_and_scores():
    """Reproduce the shipped pool and its APEX MIC + ESMC P(hemo); cache to npz."""
    if POOL_NPZ.exists():
        d = np.load(POOL_NPZ, allow_pickle=True)
        return list(d["seqs"]), d["mic"], d["phemo"]

    import argparse

    from amp_challenge_2027.generate import build_library, build_model, set_determinism
    from amp_challenge_2027.paths import resolve_repo_path
    from amp_challenge_2027.selectivity_esm import EsmcSelectivityScorer
    from amp_challenge_2027.oracle import ApexScorer

    set_determinism(42)
    args = argparse.Namespace(checkpoint="checkpoint/generator.pt", temperature=1.6, top_p=1.0,
                              length=None, min_length=C.MIN_LENGTH, max_length=C.MAX_LENGTH,
                              rank="apex", device=None, baseline=False)
    ref = set(l.strip() for l in resolve_repo_path("data/antibacterial.fasta").read_text().splitlines()
              if l.strip() and not l.startswith(">"))
    model = build_model(args)
    rng = np.random.default_rng(42)
    pool = build_library(model, max(50000, round(3.0 * 50000)), rng, ref)
    print(f"pool: {len(pool)} candidates")

    mic = ApexScorer("oracle/apex", device="cpu").predict_mic(pool)  # (n,11)
    print(f"APEX MIC matrix: {mic.shape}")
    # ESMC P(hemo) for the whole pool so any weighting can be swept exactly.
    sc = EsmcSelectivityScorer()
    phemo = sc.predict_proba(pool)
    print(f"ESMC P(hemo): {phemo.shape} median={np.median(phemo):.3f}")

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(POOL_NPZ, seqs=np.array(pool, dtype=object), mic=mic, phemo=phemo)
    return pool, mic, phemo


class PrecomputedRanker:
    """A ranker with cached MIC + P(hemo); score = category_success(gp_w,mdr_w) - lam*P(hemo)."""
    name = "precomputed"

    def __init__(self, seqs, mic, phemo, gp_w, mdr_w, lam=0.5, refine_k=4000):
        self._mic = {s: mic[i] for i, s in enumerate(seqs)}
        self._ph = {s: float(phemo[i]) for i, s in enumerate(seqs)}
        self.gp_w, self.mdr_w, self.lam, self.refine_k = gp_w, mdr_w, lam, refine_k

    def score(self, sequences):
        mic = np.array([self._mic[s] for s in sequences])
        act = category_success_score(mic, self.gp_w, self.mdr_w)
        ph = np.ones(len(sequences))
        k = min(len(sequences), self.refine_k)
        order = np.argsort(-act, kind="stable")[:k]
        for i in order:
            ph[i] = self._ph[sequences[int(i)]]
        return (act - self.lam * ph).tolist()


def profile(top, mic_lookup, ph_lookup):
    """Per-category mean Success Rate (MIC<=16) + selectivity proxy for a list of peptides."""
    mic = np.array([mic_lookup[s] for s in top])
    ph = np.array([ph_lookup[s] for s in top])
    hit = (mic <= 16.0)
    return {
        "broad": hit.mean(),
        "GN": hit[:, list(GRAM_NEG)].mean(),
        "GP": hit[:, list(GRAM_POS)].mean(),
        "MDR": hit[:, list(MDR)].mean(),
        "phemo_med": float(np.median(ph)),
        "minMIC_med": float(np.median(mic.min(1))),
    }


def main():
    from amp_challenge_2027.generate import select_top

    seqs, mic, phemo = build_pool_and_scores()
    mic_lookup = {s: mic[i] for i, s in enumerate(seqs)}
    ph_lookup = {s: float(phemo[i]) for i, s in enumerate(seqs)}
    reference = [l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
                 if l.strip() and not l.startswith(">")]

    grid = [(0.5, 0.5), (1.0, 0.5), (0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (1.0, 1.5),
            (2.0, 1.0), (1.0, 2.0), (2.0, 2.0), (0.25, 0.25)]
    print(f"\n{'gp_w':>4s} {'mdr_w':>5s} | {'broad':>5s} {'GN':>5s} {'GP':>5s} {'MDR':>5s} "
          f"{'phemo':>5s} {'minMIC':>6s} | {'mean5*':>6s}")
    best = None
    for gp_w, mdr_w in grid:
        r = PrecomputedRanker(seqs, mic, phemo, gp_w, mdr_w)
        top100 = select_top(seqs, r, 100, reference, diversity_max_identity=0.6)
        top50 = top100[:50]
        p = profile(top50, mic_lookup, ph_lookup)
        # balanced objective: mean of the 4 activity SRs + selectivity (1 - phemo_med)
        mean5 = np.mean([p["broad"], p["GN"], p["GP"], p["MDR"], 1.0 - p["phemo_med"]])
        star = " *" if (best is None or mean5 > best[0]) else ""
        if best is None or mean5 > best[0]:
            best = (mean5, gp_w, mdr_w, p)
        print(f"{gp_w:4.2f} {mdr_w:5.2f} | {p['broad']:5.2f} {p['GN']:5.2f} {p['GP']:5.2f} "
              f"{p['MDR']:5.2f} {p['phemo_med']:5.3f} {p['minMIC_med']:6.2f} | {mean5:6.3f}{star}")
    print(f"\nbest balanced: gp_w={best[1]} mdr_w={best[2]} mean5={best[0]:.3f} -> {best[3]}")
    print("(shipped default is gp_w=0.5 mdr_w=0.5; mean5* = mean of broad/GN/GP/MDR SR + (1-median P_hemo))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
