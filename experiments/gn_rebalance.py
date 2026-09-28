"""Can the SELECTION objective recover Gram-negative (the one soft category, top-50 GN 0.54) without
losing the Gram+/MDR gains, now that the feat-020 generator is Gram+/MDR-rich?

feat-020's GN dip is a *selection* effect: `balanced_success_score` ranks by hard Gram+/MDR SR, so the
top-50 fills with Gram+/MDR specialists that happen to be slightly less Gram-negative. The generator's
Gram- output actually rose. So the fix (if any) is in the ranking: add a Gram-negative hard-SR term.
This caches the feat-020 generator's 400k pool once (APEX MIC + ESMC P(hemo)) and sweeps a Gram-neg
weight, reporting the screened top-50 five-category profile so we can pick a config that lifts GN at
acceptable Gram+/MDR cost -- or confirm feat-020 is already the right balance.

    CUDA_VISIBLE_DEVICES=0 HF_HOME=/path uv run python experiments/gn_rebalance.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CACHE = REPO / "experiments" / "cache"
NPZ = CACHE / "gnrebalance_feat020_400k.npz"

from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import GRAM_NEG, GRAM_POS, MDR, _soft_success  # noqa: E402


def build_and_score():
    if NPZ.exists():
        d = np.load(NPZ, allow_pickle=True)
        return list(d["seqs"]), d["mic"], d["phemo"]
    import argparse
    from amp_challenge_2027.generate import build_library, build_model, set_determinism
    from amp_challenge_2027.paths import resolve_repo_path
    from amp_challenge_2027.oracle import ApexScorer
    from amp_challenge_2027.selectivity_esm import EsmcSelectivityScorer
    set_determinism(42)
    args = argparse.Namespace(checkpoint="checkpoint/generator.pt", temperature=1.6, top_p=1.0,
                              min_length=C.MIN_LENGTH, max_length=C.MAX_LENGTH, baseline=False, device=None)
    ref = set(l.strip() for l in resolve_repo_path("data/antibacterial.fasta").read_text().splitlines()
              if l.strip() and not l.startswith(">"))
    model = build_model(args)
    pool = build_library(model, 400000, np.random.default_rng(42), ref)
    print(f"pool {len(pool)}", flush=True)
    mic = ApexScorer("oracle/apex", device="cpu").predict_mic(pool)
    print(f"APEX {mic.shape}", flush=True)
    # two-stage ESMC (like the shipped path): score selectivity only on the top-K by activity that
    # could enter any top-50 under any gn_w; the rest are assumed hemolytic (phemo=1.0). ~10x faster
    # than scoring all 400k.
    s0 = _soft_success(mic)
    act = (s0[:, list(GRAM_POS)].mean(1) + s0[:, list(MDR)].mean(1)
           + s0[:, list(GRAM_NEG)].mean(1) + s0.mean(1))
    topk = np.argsort(-act, kind="stable")[:30000]
    phemo = np.ones(len(pool))
    pv = EsmcSelectivityScorer().predict_proba([pool[int(i)] for i in topk])
    for j, i in enumerate(topk):
        phemo[int(i)] = float(pv[j])
    print(f"ESMC top-{len(topk)} scored", flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, seqs=np.array(pool, dtype=object), mic=mic, phemo=phemo)
    return pool, mic, phemo


def main():
    from amp_challenge_2027.generate import select_top
    from rapidfuzz import process, fuzz
    seqs, mic, phemo = build_and_score()
    hit = (mic <= 16.0)
    gp, md, gn, broad = (hit[:, list(GRAM_POS)].mean(1), hit[:, list(MDR)].mean(1),
                         hit[:, list(GRAM_NEG)].mean(1), hit.mean(1))
    s = _soft_success(mic)
    idx = {q: i for i, q in enumerate(seqs)}
    ref = [l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
           if l.strip() and not l.startswith(">")]

    def obj(gn_w):  # balanced + gn_w*SR_hard(GN) - 1.5*phemo
        return (hit[:, list(GRAM_POS)].mean(1) + hit[:, list(MDR)].mean(1)
                + gn_w * hit[:, list(GRAM_NEG)].mean(1) + 0.5 * s.mean(1) - 1.5 * phemo)

    class R:
        name = "r"
        def __init__(self, gn_w): self.v = {q: float(x) for q, x in zip(seqs, obj(gn_w))}
        def score(self, ss): return [self.v[q] for q in ss]

    print(f"\n{'gn_w':>4s} | {'broad':>5s} {'GN':>5s} {'GP':>5s} {'MDR':>5s} {'medPh':>5s} | {'mean5':>5s} {'min4':>5s} {'novID':>5s}")
    for gn_w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        top = select_top(seqs, R(gn_w), 100, ref, diversity_max_identity=0.6)[:50]
        ii = [idx[q] for q in top]
        b, g_n, g_p, m = broad[ii].mean(), gn[ii].mean(), gp[ii].mean(), md[ii].mean()
        mph = float(np.median(phemo[ii]))
        nov = float(np.median([process.extractOne(q, ref, scorer=fuzz.ratio)[1] / 100.0 for q in top]))
        mean5 = np.mean([b, g_n, g_p, m, 1 - mph]); min4 = min(b, g_n, g_p, m)
        print(f"{gn_w:4.2f} | {b:5.2f} {g_n:5.2f} {g_p:5.2f} {m:5.2f} {mph:5.3f} | {mean5:5.3f} {min4:5.3f} {nov:5.2f}")
    print("(gn_w=0 is the shipped feat-020 objective; looking for a gn_w that lifts GN with acceptable GP/MDR cost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
