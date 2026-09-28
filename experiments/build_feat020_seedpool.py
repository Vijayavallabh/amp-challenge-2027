"""Build a directed-evolution seed pool from the ACTUAL shipped (feat-020) generator.

The first feat-022 run was seeded from bigpool_400000.npz, which turned out to be an OLDER generator's
pool (0/50 overlap with feat-021's top-50). This regenerates seeds from checkpoint/generator.pt (the
feat-020 generator that produces feat-021), scores them with APEX (ensemble mean) + ESMC selectivity,
and saves {seqs, mic, phemo} so directed_evolution.py --seed-pool can start from feat-021-grade,
BALANCED-strong peptides (GP ~0.75) instead of Gram-specialists.

    HF_HOME=/... .venv/bin/python experiments/build_feat020_seedpool.py \
        --gpu 1 --apex-gpus 1,2,3,4,5,6,7 --gpu-python /.../agpu/bin/python --n 150000 \
        --out experiments/cache/feat020_seedpool.npz
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import balanced_success_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--apex-gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--gpu-python", required=True)
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--esmc-k", type=int, default=15000, help="ESMC-score this many top-by-balanced")
    ap.add_argument("--out", default="experiments/cache/feat020_seedpool.npz")
    args = ap.parse_args()

    import torch
    from amp_challenge_2027.model import TrainedGenerator
    from amp_challenge_2027.paths import resolve_repo_path
    import bulk_apex
    from amp_challenge_2027.selectivity_esm import EsmcSelectivityScorer

    dev = f"cuda:{args.gpu}"
    gen = TrainedGenerator(resolve_repo_path("checkpoint/generator.pt"),
                           min_length=C.MIN_LENGTH, max_length=C.MAX_LENGTH,
                           temperature=1.6, top_p=1.0, device=dev)
    rng = np.random.default_rng(42)
    ref = set(l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
              if l.strip() and not l.startswith(">"))
    seen: dict[str, None] = {}
    for _ in range(60):
        if len(seen) >= args.n:
            break
        for s in gen.sample(min(200000, int((args.n - len(seen)) * 1.3) + 1000), rng):
            if s not in seen and s not in ref and C.is_valid_sequence(s):
                seen[s] = None
    pool = list(seen)[: args.n]
    del gen; torch.cuda.empty_cache()
    print(f"sampled {len(pool)} unique from feat-020 generator", flush=True)

    gpus = [g for g in args.apex_gpus.split(",") if g]
    uniq, mic = bulk_apex.score(pool, device="cuda", workers=8, gpu_python=args.gpu_python, gpus=gpus)
    mic = mic.astype(np.float64)
    idx = {s: i for i, s in enumerate(uniq)}
    mic = np.array([mic[idx[s]] for s in pool])
    print(f"APEX scored {mic.shape}", flush=True)

    bal = balanced_success_score(mic, gn_weight=0.75)
    top = np.argsort(-bal)[: args.esmc_k]
    phemo = np.ones(len(pool))
    pv = EsmcSelectivityScorer(device=dev).predict_proba([pool[int(i)] for i in top])
    for j, i in enumerate(top):
        phemo[int(i)] = float(pv[j])
    nonhemo = int((phemo < 0.5).sum())
    print(f"ESMC top-{len(top)}: {nonhemo} non-hemolytic; balanced max {bal.max():.3f}", flush=True)

    np.savez_compressed(REPO / args.out, seqs=np.array(pool, dtype=object),
                        mic=mic.astype(np.float32), phemo=phemo.astype(np.float32))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
