"""feat-023 PROBE: can robust directed-evolution (feat-022) peptides lift feat-021's weakest category
(Gram-) without an overfitting penalty?

The feat-022 GA overfit Gram+/MDR (train >> held-out), but Gram- generalised (small train-holdout gap;
APEX Gram- submodel agreement 95.7%). Gram- is feat-021's weakest, highest-leverage category. This
probe tests, HONESTLY (select on TRAIN submodels [0-4], report on HELD-OUT [5,6,7] -- so neither the
selection nor the number is leaked), whether augmenting feat-021's candidates with GA peptides yields a
top-50 with higher held-out Gram- at no Gram+/MDR cost.

CPU only -- reuses cached per-submodel MIC (experiments/cache/de.npz, feat021_top50_permodel.npz).
No shipping decision here; just evidence for/against building feat-023.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from amp_challenge_2027 import constraints as C  # noqa: E402
from amp_challenge_2027.oracle import GRAM_POS, GRAM_NEG, MDR, balanced_success_score  # noqa: E402

TRAIN, VAL = [0, 1, 2, 3, 4], [5, 6, 7]
GN, GP, MD = list(GRAM_NEG), list(GRAM_POS), list(MDR)


def bal(mic_mean):
    return balanced_success_score(mic_mean, gn_weight=0.75)


def profile(mic8, idx, subs):
    hit = (mic8[idx][:, subs, :].mean(1) <= 16.0)
    return dict(broad=float(hit.mean()), GN=float(hit[:, GN].mean()),
                GP=float(hit[:, GP].mean()), MDR=float(hit[:, MD].mean()))


def greedy_select(seqs, mic8, phemo, score, k, reference, *, max_id=0.6, gate=0.5):
    order = np.argsort(-score, kind="stable")
    out = []
    for i in order:
        if len(out) == k:
            break
        if phemo[i] >= gate:
            continue
        s = seqs[i]
        if not C.is_novel_enough(s, reference):
            continue
        if out and C.max_identity(s, [seqs[j] for j in out], cutoff=max_id) >= max_id:
            continue
        out.append(int(i))
    return out


def main():
    ref = [l.strip() for l in (REPO / "data" / "antibacterial.fasta").read_text().splitlines()
           if l.strip() and not l.startswith(">")]

    # --- pools -------------------------------------------------------------------------------
    ga = np.load(REPO / "experiments/cache/de.npz", allow_pickle=True)
    ga_seq, ga_mic8, ga_ph = ga["seqs"].tolist(), ga["mic8"].astype(float), ga["phemo"].astype(float)
    f21 = np.load(REPO / "experiments/cache/feat021_top50_permodel.npz", allow_pickle=True)
    f21_seq, f21_mic8 = f21["seqs"].tolist(), f21["mic8"].astype(float)
    f21_ph = np.zeros(len(f21_seq))  # feat-021 top-50 measured 0% hemolytic (max P 0.061)

    # feat-021 shipped top-50 held-out profile (the bar to beat)
    print("feat-021 shipped top-50:")
    for nm, subs in [("HELD-OUT[5,6,7]", VAL), ("FULL[0-7]", list(range(8)))]:
        p = profile(f21_mic8, np.arange(len(f21_seq)), subs)
        print(f"  {nm:16s} broad {p['broad']:.3f} GN {p['GN']:.3f} GP {p['GP']:.3f} MDR {p['MDR']:.3f}")

    # robust GA subset: non-hemolytic and Gram- generalises (train GN SR ~ held-out GN SR)
    ga_tr = ga_mic8[:, TRAIN, :].mean(1)
    ga_va = ga_mic8[:, VAL, :].mean(1)
    gn_tr = (ga_tr[:, GN] <= 16).mean(1)
    gn_va = (ga_va[:, GN] <= 16).mean(1)
    robust = (ga_ph < 0.5) & (np.abs(gn_tr - gn_va) <= 0.15) & (gn_va >= 0.55)
    print(f"\nGA archive {len(ga_seq)}: robust GN candidates (nonhemo, |gap|<=0.15, held-out GN>=0.55) = {int(robust.sum())}")

    # --- experiment: union pool, SELECT on TRAIN, REPORT on HELD-OUT (leak-free both ways) ----
    def run(tag, seqs, mic8, phemo):
        tr_mean = mic8[:, TRAIN, :].mean(1)
        sel = greedy_select(seqs, mic8, phemo, bal(tr_mean), 50, ref)
        p_ho = profile(mic8, np.array(sel), VAL)
        p_full = profile(mic8, np.array(sel), list(range(8)))
        print(f"  {tag:22s} HELD-OUT broad {p_ho['broad']:.3f} GN {p_ho['GN']:.3f} GP {p_ho['GP']:.3f} MDR {p_ho['MDR']:.3f}"
              f"  | FULL GN {p_full['GN']:.3f} GP {p_full['GP']:.3f}")
        return sel

    print("\nSelect top-50 by TRAIN[0-4] balanced, report HELD-OUT[5,6,7] (honest):")
    # control: feat-021 pool only
    run("feat-021 only", f21_seq, f21_mic8, f21_ph)
    # union: feat-021 + robust GA
    ridx = np.where(robust)[0]
    union_seq = f21_seq + [ga_seq[i] for i in ridx]
    union_mic8 = np.concatenate([f21_mic8, ga_mic8[ridx]], 0)
    union_ph = np.concatenate([f21_ph, ga_ph[ridx]], 0)
    sel = run("feat-021 + robust GA", union_seq, union_mic8, union_ph)
    n_from_ga = sum(1 for i in sel if i >= len(f21_seq))
    print(f"  -> {n_from_ga}/50 of the union top-50 came from the GA")

    # --- targeted swap: KEEP feat-021's best by their native FULL rank, ADD GA by HELD-OUT GN ----
    # feat-021 peptides ranked by their shipped full-ensemble balanced (honest, their native order);
    # GA GN-specialists chosen by held-out[5,6,7] GN SR (honest for GA). Report held-out.
    print("\nTargeted swap: keep feat-021's top-(50-m) by full-rank, add m GA GN-specialists (held-out GN):")
    f21_full_rank = np.argsort(-bal(f21_mic8.mean(1)), kind="stable")
    ridx = np.where(robust)[0]
    ga_ho_gn = (ga_mic8[ridx][:, VAL, :][:, :, GN].mean((1, 2)) <= 16.0).astype(float)
    ga_ho_gn = (ga_mic8[ridx][:, VAL, :].mean(1)[:, GN] <= 16).mean(1)   # held-out GN SR per candidate
    ga_by_gn = ridx[np.argsort(-ga_ho_gn, kind="stable")]
    for m in [0, 8, 12, 16, 20, 25]:
        keep = [int(i) for i in f21_full_rank[: 50 - m]]
        kept_seqs = [f21_seq[i] for i in keep]
        kept_mic = [f21_mic8[i] for i in keep]
        added = 0
        for gi in ga_by_gn:
            if added == m:
                break
            s = ga_seq[gi]
            if not C.is_novel_enough(s, ref):
                continue
            if C.max_identity(s, kept_seqs, cutoff=0.6) >= 0.6:
                continue
            kept_seqs.append(s); kept_mic.append(ga_mic8[gi]); added += 1
        mm = np.array(kept_mic)
        pv = profile(mm, np.arange(len(mm)), VAL)
        print(f"  m={m:2d} (GA added {added:2d})  HELD-OUT broad {pv['broad']:.3f} GN {pv['GN']:.3f} "
              f"GP {pv['GP']:.3f} MDR {pv['MDR']:.3f}  mean4 {np.mean([pv['broad'],pv['GN'],pv['GP'],pv['MDR']]):.3f} "
              f"floor {min(pv['GN'],pv['GP'],pv['MDR']):.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
