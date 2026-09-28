"""Does a LARGER oversample pool + a GP/MDR-emphasising objective lift the top-50's Gram+/MDR
Success Rate while preserving selectivity and diversity? (The cheap, deterministic alternative to a
GP/MDR-targeted ReST round.)

Diagnostics showed: the 150k pool has only ~19 diverse non-hemolytic GP/MDR-active peptides (the
0.60 diversity screen is the binding constraint), and this diverse-champion count grows ~linearly
with pool size. So sample a bigger pool, score it (APEX on CPU = the shipped deterministic path;
ESMC P(hemo) on the top-K by activity), and measure the achievable top-50 per-category profile under
the REAL screens for a few objectives. Reports counts + Goodhart guards (diversity, novelty).

    CUDA_VISIBLE_DEVICES=2 HF_HOME=/path uv run python experiments/bigpool_gpmdr.py [POOL_SIZE]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CACHE = REPO / "experiments" / "cache"

from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import GRAM_NEG, GRAM_POS, MDR, _soft_success  # noqa: E402

POOL_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
REFINE_K = 20000  # ESMC scores this many top-activity candidates (>= any top-50 selection)


def collect_pool(target: int, ref: set) -> list[str]:
    """Incrementally sample unique valid peptides up to `target` (robust: uses whatever it gets)."""
    from amp_challenge_2027.model import TrainedGenerator
    from amp_challenge_2027.paths import resolve_repo_path
    g = TrainedGenerator(resolve_repo_path("checkpoint/generator.pt"), min_length=C.MIN_LENGTH,
                         max_length=C.MAX_LENGTH, temperature=1.6, top_p=1.0)
    rng = np.random.default_rng(42)
    seen: dict[str, None] = {}
    t0 = time.time()
    for _ in range(60):
        if len(seen) >= target:
            break
        for s in g.sample(min(300000, int((target - len(seen)) * 1.3) + 1000), rng):
            if s not in seen and s not in ref and C.is_valid_sequence(s):
                seen[s] = None
        print(f"  sampled unique={len(seen)} ({time.time()-t0:.0f}s)", flush=True)
    return list(seen)[:target]


def main():
    from amp_challenge_2027.generate import select_top, set_determinism
    from amp_challenge_2027.oracle import ApexScorer
    from amp_challenge_2027.selectivity_esm import EsmcSelectivityScorer

    set_determinism(42)
    npz = CACHE / f"bigpool_{POOL_SIZE}.npz"
    ref = set(l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
              if l.strip() and not l.startswith(">"))
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        pool, mic, phemo = list(d["seqs"]), d["mic"], d["phemo"]
        print(f"loaded cache: {len(pool)} peptides")
    else:
        print(f"target pool: {POOL_SIZE}")
        pool = collect_pool(POOL_SIZE, ref)
        print(f"pool: {len(pool)} unique valid")
        t0 = time.time()
        mic = ApexScorer("oracle/apex", device="cpu").predict_mic(pool)
        print(f"APEX MIC {mic.shape} in {time.time()-t0:.0f}s")
        # two-stage: ESMC only on the top REFINE_K by a broad+gp/mdr activity (superset of any top-50)
        s = _soft_success(mic)
        act = s.mean(1) + s[:, list(GRAM_POS)].mean(1) + s[:, list(MDR)].mean(1)
        top = np.argsort(-act, kind="stable")[:REFINE_K]
        phemo = np.ones(len(pool))
        sc = EsmcSelectivityScorer()
        pv = sc.predict_proba([pool[int(i)] for i in top])
        for j, i in enumerate(top):
            phemo[int(i)] = pv[j]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, seqs=np.array(pool, dtype=object), mic=mic, phemo=phemo)
        print(f"cached -> {npz}")

    hit = (mic <= 16.0)
    gp_all, mdr_all = hit[:, list(GRAM_POS)].mean(1), hit[:, list(MDR)].mean(1)
    gn_all, broad_all = hit[:, list(GRAM_NEG)].mean(1), hit.mean(1)
    ph_l = {s: float(phemo[i]) for i, s in enumerate(pool)}
    idx_of = {s: i for i, s in enumerate(pool)}
    reference = list(ref)

    # diverse non-hemolytic GP/MDR champions available in THIS pool
    from rapidfuzz import fuzz
    tgt = (phemo < 0.5) & ((gp_all >= 0.5) | (mdr_all >= 0.5))
    order = [i for i in np.argsort(-(gp_all + mdr_all), kind="stable") if tgt[i]]
    kept = []
    for i in order:
        if all(fuzz.ratio(pool[i], k) / 100.0 <= 0.6 for k in kept):
            kept.append(pool[i])
        if len(kept) >= 80:
            break
    print(f"\npool has {int(tgt.sum())} non-hemolytic GP/MDR-active; diverse@0.6 = {len(kept)}")

    def soft(mic_):
        return _soft_success(mic_)

    def o_baseline(m, p):
        s = soft(m); return s.mean(1) + 0.5*s[:, list(GRAM_POS)].mean(1) + 0.5*s[:, list(MDR)].mean(1) - 0.5*p
    def o_gpmdr(m, p, w=1.5):
        s = soft(m); return s.mean(1) + w*s[:, list(GRAM_POS)].mean(1) + w*s[:, list(MDR)].mean(1) - 0.5*p
    def o_gpmdr_hard(m, p):
        s = soft(m); h = (m <= 16.0)
        return (h[:, list(GRAM_POS)].mean(1) + h[:, list(MDR)].mean(1) + 0.5*s.mean(1)) - 0.5*p

    objs = [("baseline", o_baseline), ("gpmdr-soft1.5", o_gpmdr),
            ("gpmdr-soft3", lambda m, p: o_gpmdr(m, p, 3.0)), ("gpmdr-hard", o_gpmdr_hard)]

    class R:
        name = "obj"
        def __init__(self, fn): self.v = {s: float(x) for s, x in zip(pool, fn(mic, phemo))}
        def score(self, ss): return [self.v[s] for s in ss]

    print(f"\n{'objective':16s} | {'broad':>5s} {'GN':>5s} {'GP':>5s} {'MDR':>5s} {'medPh':>5s} | "
          f"{'mean5':>5s} {'min4':>5s} | {'knownID':>7s}")
    for name, fn in objs:
        top100 = select_top(pool, R(fn), 100, reference, diversity_max_identity=0.6)
        top = top100[:50]
        ii = [idx_of[s] for s in top]
        b, gn, gp, md = broad_all[ii].mean(), gn_all[ii].mean(), gp_all[ii].mean(), mdr_all[ii].mean()
        mph = float(np.median([ph_l[s] for s in top]))
        mean5 = np.mean([b, gn, gp, md, 1 - mph]); min4 = min(b, gn, gp, md)
        from rapidfuzz import process as _proc
        kid = float(np.median([_proc.extractOne(s, reference, scorer=fuzz.ratio)[1] / 100.0 for s in top]))
        print(f"{name:16s} | {b:5.2f} {gn:5.2f} {gp:5.2f} {md:5.2f} {mph:5.3f} | "
              f"{mean5:5.3f} {min4:5.3f} | {kid:7.3f}")
    print("\n(shipped baseline@150k top-50: broad .50 GN .57 GP .37 MDR .42 medPh .007 mean5 .572)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
