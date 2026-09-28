"""Objective bake-off on a cached APEX MIC matrix.

Compares the shipped margin objective (broad_potency) against a category-aligned success-rate
objective, reporting the predicted per-category profile of the resulting top-50 (what the
competition actually scores: mean per-peptide Success Rate within Broad / Gram+/ Gram- / MDR,
plus a hemolysis/selectivity view). No wet-lab claims -- APEX predictions only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# APEX pathogen buckets (indices into the 11-column MIC matrix).
GN = [0, 1, 2, 3, 4, 5, 6]        # A. baumannii, E. coli x3, K. pneumoniae, P. aeruginosa x2
GP = [7, 8, 9, 10]                # S. aureus x2, VRE faecalis, VRE faecium
MDR = [8, 9, 10]                  # MRSA, VRE faecalis, VRE faecium
ALL = list(range(11))
THRESH = 16.0
TAU = 0.5  # log10 units for the smooth success indicator


def soft_success(mic: np.ndarray) -> np.ndarray:
    """Smooth P(MIC<=16): sigmoid of the log-margin below threshold. (n,11)->(n,11)."""
    margin = (np.log10(THRESH) - np.log10(np.clip(mic, 1e-6, None))) / TAU
    return 1.0 / (1.0 + np.exp(-margin))


def hard_sr(mic: np.ndarray, idx: list[int]) -> np.ndarray:
    """Per-peptide Success Rate over a strain bucket (fraction with MIC<=16)."""
    return (mic[:, idx] <= THRESH).mean(axis=1)


def o_broad(mic: np.ndarray) -> np.ndarray:
    per = np.log10(THRESH / np.clip(mic, 1e-6, None))
    return np.clip(per, 0.0, None).sum(axis=1)


def o_cat(mic: np.ndarray, w=(1.0, 1.0, 1.0, 1.0)) -> np.ndarray:
    """Equal-weight mean of soft Success Rate across Broad / GN / GP / MDR."""
    s = soft_success(mic)
    sr_all = s[:, ALL].mean(axis=1)
    sr_gn = s[:, GN].mean(axis=1)
    sr_gp = s[:, GP].mean(axis=1)
    sr_mdr = s[:, MDR].mean(axis=1)
    return w[0] * sr_all + w[1] * sr_gn + w[2] * sr_gp + w[3] * sr_mdr


def profile(mic: np.ndarray, sel: np.ndarray, phemo: np.ndarray | None, label: str) -> dict:
    m = mic[sel]
    minmic = m.min(axis=1)
    active = minmic <= THRESH
    prof = dict(
        label=label, n=len(sel),
        pct_active=100 * active.mean(),
        med_minmic=float(np.median(minmic)),
        SR_broad=100 * hard_sr(m, ALL).mean(),
        SR_gn=100 * hard_sr(m, GN).mean(),
        SR_gp=100 * hard_sr(m, GP).mean(),
        SR_mdr=100 * hard_sr(m, MDR).mean(),
        mean_breadth=float((m <= THRESH).sum(axis=1).mean()),
    )
    if phemo is not None:
        prof["mean_phemo"] = float(phemo[sel].mean())
    return prof


def print_row(p: dict) -> None:
    extra = f" phemo={p['mean_phemo']:.3f}" if "mean_phemo" in p else ""
    print(f"  {p['label']:<28} act={p['pct_active']:5.1f}% minMIC~{p['med_minmic']:5.1f} "
          f"| SR broad={p['SR_broad']:4.1f} GN={p['SR_gn']:4.1f} GP={p['SR_gp']:4.1f} "
          f"MDR={p['SR_mdr']:4.1f} | breadth={p['mean_breadth']:.2f}/11{extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--hemolysis", action="store_true", help="also compute P(hemolytic)")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    seqs = list(d["sequences"])
    mic = d["mic"].astype(np.float64)
    print(f"pool: {len(seqs)} peptides, MIC matrix {mic.shape}")

    phemo = None
    if args.hemolysis:
        from amp_challenge_2027.hemolysis import HemolysisScorer
        hs = HemolysisScorer("checkpoint/hemolysis.pt")
        phemo = np.asarray(hs.predict_proba(seqs))

    k = args.top_k
    sel_broad = np.argsort(-o_broad(mic))[:k]
    sel_cat = np.argsort(-o_cat(mic))[:k]

    # Pool-level baseline for reference.
    print("\nTop-%d selection under each objective (predicted; no wet-lab claim):" % k)
    print_row(profile(mic, sel_broad, phemo, "O_broad (shipped)"))
    print_row(profile(mic, sel_cat, phemo, "O_cat (category-aligned)"))

    # Overlap between the two selections.
    ov = len(set(sel_broad.tolist()) & set(sel_cat.tolist()))
    print(f"\noverlap of the two top-{k} sets: {ov}/{k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
