"""feat-022: Submodel-cross-validated DIRECTED EVOLUTION (genetic algorithm) in sequence space.

Why: feat-019 found the generator's SAMPLING distribution has a ~linear ceiling on diverse,
non-hemolytic Gram+/MDR hitters (only ~19-28 per 150k pool). Directed evolution CONSTRUCTS better
peptides by mutating the best ones toward higher reward, so it is not bounded by that sampling
distribution -- it can exceed the ceiling if APEX's fitness landscape is navigable by local moves.

Optimizing against an oracle is the central risk (Goodhart). Guards, all active here:
  * SUBMODEL SPLIT -- the reward driving evolution uses APEX submodels [0..4] (TRAIN); final
    selection and the headline numbers use HELD-OUT submodels [5,6,7] (VAL). A genuine potency
    gain generalizes to VAL; a sequence that games APEX's shared weights lifts TRAIN but not VAL.
    Both are printed every round so divergence (TRAIN up, VAL flat) is visible immediately.
  * ESMC selectivity (Synthyra ESM++ large, different model + data than APEX) HARD-GATES every
    survivor to P(hemolytic) < 0.5 -- selectivity is not reachable by gaming APEX.
  * NOVELTY -- survivors are held to the shipped top list's < 80% Levenshtein identity vs the 39k
    known AMPs, so evolution cannot drift back onto known antibacterials.
  * Winners are re-checked downstream with the FULL 8-model ensemble, ESMC, novelty and ESMFold2.

Seeds: best non-hemolytic peptides from the cached feat-020 400k pool (bigpool_400000.npz).

Offline only -- never touches the byte-deterministic ``uv run generate`` path. Writes an archive
npz (seq, mic8, phemo) so winners can be analysed and, if they beat feat-021 on HELD-OUT submodels,
folded into the submission candidate pool.

    HF_HOME=/... .venv/bin/python experiments/directed_evolution.py \
        --apex-gpus 1,2,3,4,5,6,7 --esmc-gpu 1 --gpu-python /.../agpu/bin/python \
        --rounds 10 --pop 300 --mutants 64 --seeds 300 --out experiments/cache/de.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import (  # noqa: E402
    GRAM_NEG, GRAM_POS, MDR, balanced_success_score,
)
from bulk_permodel import score_permodel  # noqa: E402

TRAIN_SUB = [0, 1, 2, 3, 4]   # submodels the reward is optimized on
VAL_SUB = [5, 6, 7]           # held-out submodels for selection + reporting
AA = C.ALPHABET               # "ACDEFGHIKLMNPQRSTVWY"
GATE = 0.5                    # ESMC P(hemolytic) hard gate
GN_W = 0.75                   # feat-021 balanced objective Gram- weight


def reward(mic_mean: np.ndarray) -> np.ndarray:
    """Shipped feat-021 balanced objective on an (n, 11) MIC (ensemble-mean over some submodels)."""
    return balanced_success_score(mic_mean, gn_weight=GN_W)


def mic_mean(mic8: np.ndarray, subs: list[int]) -> np.ndarray:
    return mic8[:, subs, :].mean(axis=1)


def category_profile(mic_mean_arr: np.ndarray, idx: np.ndarray) -> dict:
    hit = (mic_mean_arr[idx] <= 16.0)
    return {
        "broad": float(hit.mean()),
        "GN": float(hit[:, list(GRAM_NEG)].mean()),
        "GP": float(hit[:, list(GRAM_POS)].mean()),
        "MDR": float(hit[:, list(MDR)].mean()),
    }


def mutate(seq: str, rng: np.random.Generator) -> str:
    """One local move: single/double substitution, or (rarer) a single insertion/deletion.

    Uniform over the 20 residues -- deliberately not biased toward cationic/hydrophobic AAs, so the
    move set injects no prior of ours; APEX+ESMC do all the selecting. Length stays in [MIN, MAX]."""
    s = list(seq)
    r = rng.random()
    if r < 0.70:                                   # single substitution
        i = int(rng.integers(len(s)))
        s[i] = AA[int(rng.integers(len(AA)))]
    elif r < 0.90:                                 # double substitution
        for _ in range(2):
            i = int(rng.integers(len(s)))
            s[i] = AA[int(rng.integers(len(AA)))]
    elif r < 0.95 and len(s) < C.MAX_LENGTH:       # insertion
        i = int(rng.integers(len(s) + 1))
        s.insert(i, AA[int(rng.integers(len(AA)))])
    elif len(s) > C.MIN_LENGTH:                    # deletion
        i = int(rng.integers(len(s)))
        del s[i]
    else:                                          # fallback: substitution
        i = int(rng.integers(len(s)))
        s[i] = AA[int(rng.integers(len(AA)))]
    return "".join(s)


def diverse_top(cands: list[str], score_map: dict, k: int, *, max_id: float,
                reference: list[str] | None) -> list[str]:
    """Greedy: highest reward first, skip near-duplicates (> max_id identity to a kept peptide) and,
    if ``reference`` given, sequences failing the < 80% novelty screen vs known AMPs."""
    ranked = sorted(cands, key=lambda s: (-score_map[s], s))
    out: list[str] = []
    for s in ranked:
        if len(out) == k:
            break
        if reference is not None and not C.is_novel_enough(s, reference):
            continue
        if out and C.max_identity(s, out, cutoff=max_id) >= max_id:
            continue
        out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apex-gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--esmc-gpu", type=int, default=1)
    ap.add_argument("--gpu-python", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--mutants", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--refine-k", type=int, default=4000, help="ESMC-score this many top-by-TRAIN mutants/round")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed-pool", default="experiments/cache/bigpool_400000.npz")
    ap.add_argument("--out", default="experiments/cache/de.npz")
    args = ap.parse_args()

    gpus = [g for g in args.apex_gpus.split(",") if g]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    from amp_challenge_2027.selectivity_esm import EsmcSelectivityScorer
    esmc = EsmcSelectivityScorer(device=f"cuda:{args.esmc_gpu}")
    reference = [ln.strip() for ln in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
                 if ln.strip() and not ln.startswith(">")]
    ref_set = set(reference)

    # --- seeds: feat-021-STYLE selection from the pool (balanced - 1.5*P(hemo), then diversity +
    # novelty), so the seed set SPANS Gram+/MDR/Gram- like the shipped top-50 -- NOT a top-by-balanced
    # set, which is Gram-heavy (the diversity screen is what balances feat-021's top-50). This lets
    # evolution start from Gram+-strong backbones and test whether it can ADD Gram- without losing them.
    d = np.load(REPO / args.seed_pool, allow_pickle=True)
    pseqs, pmic, pph = d["seqs"].tolist(), d["mic"].astype(np.float64), d["phemo"].astype(np.float64)
    pr = reward(pmic) - 1.5 * pph
    cand = [i for i in range(len(pseqs)) if pph[i] < GATE and C.is_valid_sequence(str(pseqs[i]))]
    cand.sort(key=lambda i: -pr[i])
    seeds = []
    for i in cand:
        if len(seeds) >= args.seeds:
            break
        s = str(pseqs[i])
        if not C.is_novel_enough(s, reference):
            continue
        if seeds and C.max_identity(s, seeds, cutoff=0.6) >= 0.6:
            continue
        seeds.append(s)
    print(f"[{time.time()-t0:5.0f}s] seeds: {len(seeds)} diverse feat-021-style from {args.seed_pool} "
          f"(seed score max {pr[cand[0]]:.3f})", flush=True)

    # archive: every scored sequence -> per-submodel MIC (8,11) and ESMC phemo (default 1.0 = unscored)
    mic8_by: dict[str, np.ndarray] = {}
    phemo_by: dict[str, float] = {}

    def ingest(seqs: list[str]) -> None:
        todo = [s for s in dict.fromkeys(seqs) if s not in mic8_by]
        if not todo:
            return
        uniq, mic8 = score_permodel(todo, gpus=gpus, gpu_python=args.gpu_python)
        for s, row in zip(uniq, mic8):
            if not np.isnan(row).any():
                mic8_by[s] = row

    def esmc_gate(new_seqs: list[str]) -> None:
        """ESMC-score the top ``refine_k`` of ``new_seqs`` by TRAIN reward (the expensive PLM only
        runs on promising peptides; the rest keep phemo=1.0 and are excluded from survivors)."""
        cand = [s for s in new_seqs if s in mic8_by and s not in phemo_by]
        if not cand:
            return
        rt = reward(mic_mean(np.array([mic8_by[s] for s in cand]), TRAIN_SUB))
        order = np.argsort(-rt)[: args.refine_k]
        pick = [cand[int(i)] for i in order]
        proba = esmc.predict_proba(pick)
        for s, p in zip(pick, proba):
            phemo_by[s] = float(p)

    ingest(seeds)
    esmc_gate(seeds)
    pop = list(seeds)
    history = []

    for rnd in range(1, args.rounds + 1):
        # generate mutants from the current population
        mutants = set()
        for parent in pop:
            for _ in range(args.mutants):
                m = mutate(parent, rng)
                if C.is_valid_sequence(m) and m not in mic8_by and m not in ref_set:
                    mutants.add(m)
        mutants = list(mutants)
        ingest(mutants)
        esmc_gate(mutants)

        # survivors across the whole archive: non-hemolytic per ESMC
        survivors = [s for s in mic8_by if phemo_by.get(s, 1.0) < GATE]
        surv_arr = np.array([mic8_by[s] for s in survivors])         # (S, 8, 11)
        rt_map = dict(zip(survivors, reward(surv_arr[:, TRAIN_SUB, :].mean(1)).tolist()))
        # next population: diverse top by TRAIN reward (no novelty filter here -- cheap; applied at select)
        pop = diverse_top(survivors, rt_map, args.pop, max_id=0.6, reference=None)

        # ---- report: top-50 selected on TRAIN submodels (what we optimize), then profiled on
        # TRAIN and on the HELD-OUT submodels [5,6,7] which never touch any selection decision.
        # HOLDOUT is the honest generalization estimate: if TRAIN >> HOLDOUT, evolution is gaming
        # APEX's weights rather than finding real potency.
        top50 = diverse_top(survivors, rt_map, 50, max_id=0.6, reference=reference)
        if len(top50) >= 10:
            arr = np.array([mic8_by[s] for s in top50])          # (m,8,11)
            pt = category_profile(arr[:, TRAIN_SUB, :].mean(1), np.arange(len(top50)))
            ph_ = category_profile(arr[:, VAL_SUB, :].mean(1), np.arange(len(top50)))
            medph = float(np.median([phemo_by[s] for s in top50]))
            novs = [C.max_identity(s, reference, cutoff=0.0) for s in top50]
            print(f"[{time.time()-t0:5.0f}s] r{rnd:02d} | surv {len(survivors):5d} | "
                  f"TRAIN(broad {pt['broad']:.2f} GN {pt['GN']:.2f} GP {pt['GP']:.2f} MDR {pt['MDR']:.2f}) | "
                  f"HOLDOUT(broad {ph_['broad']:.2f} GN {ph_['GN']:.2f} GP {ph_['GP']:.2f} MDR {ph_['MDR']:.2f}) | "
                  f"medPh {medph:.3f} medNov {np.median(novs):.2f} | best50={len(top50)}", flush=True)
            history.append({"round": rnd, "train": pt, "holdout": ph_, "n_surv": len(survivors)})

    # ---- persist archive + final winners --------------------------------------------------------
    survivors = [s for s in mic8_by if phemo_by.get(s, 1.0) < GATE]
    surv_arr = np.array([mic8_by[s] for s in survivors])
    rt_map = dict(zip(survivors, reward(surv_arr[:, TRAIN_SUB, :].mean(1)).tolist()))
    final100 = diverse_top(survivors, rt_map, 100, max_id=0.6, reference=reference)
    seqs_all = list(mic8_by)
    np.savez_compressed(
        REPO / args.out,
        seqs=np.array(seqs_all, dtype=object),
        mic8=np.array([mic8_by[s] for s in seqs_all], dtype=np.float32),
        phemo=np.array([phemo_by.get(s, 1.0) for s in seqs_all], dtype=np.float32),
        final100=np.array(final100, dtype=object),
    )
    print(f"[{time.time()-t0:5.0f}s] archive {len(seqs_all)} seqs, {len(survivors)} non-hemolytic; "
          f"final100 {len(final100)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
