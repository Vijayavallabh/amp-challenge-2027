"""Ensemble of specialist generators: can a COMBINED pool + category-balanced selection beat any
single generator across all five scored categories?

feat-020 is a Gram+/MDR specialist (top-50 GP 0.75 / MDR 0.67 but GN 0.54); feat-019 was Gram-strong
(GN 0.63, GP 0.50). Neither is best at everything. Idea: sample from BOTH, merge into one pool, and
let the selection pick Gram+/MDR specialists (from feat-020) AND Gram-strong peptides (from feat-019),
optionally adding a Gram-negative weight (gn_w) so the top-50 covers every category. Uses GPU-APEX
across the free GPUs (non-contending with any CPU-APEX job).

    HF_HOME=/path <agpu-python-not-needed> uv run python experiments/ensemble_pool.py \
        --gpu 2 --apex-gpus 2,3,4,5,6,7 --gpu-python /path/to/agpu/python
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
CACHE = REPO / "experiments" / "cache"
NPZ = CACHE / "ensemble_pool.npz"

from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import GRAM_NEG, GRAM_POS, MDR, _soft_success  # noqa: E402


def sample_from(ckpt, n, seed, device):
    import torch
    from amp_challenge_2027.model import TrainedGenerator
    from amp_challenge_2027.paths import resolve_repo_path
    g = TrainedGenerator(resolve_repo_path(ckpt), min_length=C.MIN_LENGTH, max_length=C.MAX_LENGTH,
                         temperature=1.6, top_p=1.0, device=device)
    rng = np.random.default_rng(seed)
    seen = {}
    for _ in range(40):
        if len(seen) >= n:
            break
        for s in g.sample(min(200000, int((n - len(seen)) * 1.3) + 1000), rng):
            if s not in seen and C.is_valid_sequence(s):
                seen[s] = None
    del g
    torch.cuda.empty_cache()
    return list(seen)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-each", type=int, default=200000)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--apex-gpus", type=str, default="2,3,4,5,6,7")
    ap.add_argument("--gpu-python", type=str, required=True)
    args = ap.parse_args()

    from amp_challenge_2027.generate import select_top, set_determinism
    from amp_challenge_2027.selectivity_esm import EsmcSelectivityScorer
    import bulk_apex
    from rapidfuzz import process, fuzz

    set_determinism(42)
    dev = f"cuda:{args.gpu}"
    ref = set(l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
              if l.strip() and not l.startswith(">"))

    if NPZ.exists():
        d = np.load(NPZ, allow_pickle=True)
        pool, mic, phemo, origin = list(d["seqs"]), d["mic"], d["phemo"], d["origin"]
    else:
        print("sampling feat-020 (Gram+/MDR specialist)...", flush=True)
        p20 = sample_from("checkpoint/generator.pt", args.n_each, 42, dev)
        print(f"  feat-020: {len(p20)}", flush=True)
        print("sampling feat-019 (Gram-strong)...", flush=True)
        p19 = sample_from("experiments/cache/generator_feat019.pt", args.n_each, 43, dev)
        print(f"  feat-019: {len(p19)}", flush=True)
        # merge, dedup (keep first-seen origin), drop reference
        seen, origin_l = {}, {}
        for s in p20:
            if s not in seen and s not in ref:
                seen[s] = None; origin_l[s] = 0
        for s in p19:
            if s not in seen and s not in ref:
                seen[s] = None; origin_l[s] = 1
        pool = list(seen)
        origin = np.array([origin_l[s] for s in pool], dtype=np.int8)
        print(f"combined unique pool: {len(pool)} ({(origin==0).sum()} feat-020, {(origin==1).sum()} feat-019)", flush=True)
        gpus = [g for g in args.apex_gpus.split(",") if g]
        _, mic = bulk_apex.score(pool, device="cuda", workers=8, gpu_python=args.gpu_python, gpus=gpus)
        mic = mic.astype(np.float64)
        print(f"APEX {mic.shape}", flush=True)
        s0 = _soft_success(mic)
        act = (s0[:, list(GRAM_POS)].mean(1) + s0[:, list(MDR)].mean(1) + s0[:, list(GRAM_NEG)].mean(1) + s0.mean(1))
        topk = np.argsort(-act, kind="stable")[:40000]
        phemo = np.ones(len(pool))
        pv = EsmcSelectivityScorer(device=dev).predict_proba([pool[int(i)] for i in topk])
        for j, i in enumerate(topk):
            phemo[int(i)] = float(pv[j])
        print(f"ESMC top-{len(topk)}", flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(NPZ, seqs=np.array(pool, dtype=object), mic=mic, phemo=phemo, origin=origin)

    hit = (mic <= 16.0)
    gp, md, gn, broad = (hit[:, list(GRAM_POS)].mean(1), hit[:, list(MDR)].mean(1),
                         hit[:, list(GRAM_NEG)].mean(1), hit.mean(1))
    s = _soft_success(mic)
    idx = {q: i for i, q in enumerate(pool)}
    ref_l = list(ref)

    class R:
        name = "r"
        def __init__(self, gn_w):
            v = (hit[:, list(GRAM_POS)].mean(1) + hit[:, list(MDR)].mean(1)
                 + gn_w * hit[:, list(GRAM_NEG)].mean(1) + 0.5 * s.mean(1) - 1.5 * phemo)
            self.v = {q: float(x) for q, x in zip(pool, v)}
        def score(self, ss): return [self.v[q] for q in ss]

    print(f"\n{'gn_w':>4s} | {'broad':>5s} {'GN':>5s} {'GP':>5s} {'MDR':>5s} {'medPh':>5s} | {'mean5':>5s} {'min4':>5s} {'novID':>5s} | from19")
    for gn_w in [0.0, 0.5, 1.0, 1.5, 2.0]:
        top = select_top(pool, R(gn_w), 100, ref_l, diversity_max_identity=0.6)[:50]
        ii = [idx[q] for q in top]
        b, g_n, g_p, m = broad[ii].mean(), gn[ii].mean(), gp[ii].mean(), md[ii].mean()
        mph = float(np.median(phemo[ii]))
        nov = float(np.median([process.extractOne(q, ref_l, scorer=fuzz.ratio)[1] / 100.0 for q in top]))
        frm19 = int(sum(origin[i] == 1 for i in ii))
        mean5 = np.mean([b, g_n, g_p, m, 1 - mph]); min4 = min(b, g_n, g_p, m)
        print(f"{gn_w:4.2f} | {b:5.2f} {g_n:5.2f} {g_p:5.2f} {m:5.2f} {mph:5.3f} | {mean5:5.3f} {min4:5.3f} {nov:5.2f} | {frm19}/50")
    print("(feat-020 alone top-50: broad .62 GN .54 GP .75 MDR .67; from19 = how many top-50 came from the feat-019 generator)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
