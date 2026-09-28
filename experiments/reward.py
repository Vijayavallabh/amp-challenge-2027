"""Shared reward / metric definitions for offline optimization (ReST, analysis).

Predicted quantities only -- APEX MIC + a hemolysis model. No wet-lab claims. The reward
mirrors what the competition scores (per-peptide Success Rate = fraction of strains with
MIC<=16 uM), aggregated across Broad / Gram+/ Gram- / MDR buckets, minus a hemolysis penalty
for the Optimal Selectivity category. Success is measured with a *saturating* soft indicator
so the optimizer clears the 16 uM bar broadly instead of chasing APEX's (poorly calibrated)
sub-uM tail -- an explicit anti-Goodhart choice.
"""

from __future__ import annotations

import numpy as np

# APEX pathogen buckets (indices into the 11-column MIC matrix); see oracle.PATHOGENS.
GN = [0, 1, 2, 3, 4, 5, 6]        # A. baumannii, E. coli x3, K. pneumoniae, P. aeruginosa x2
GP = [7, 8, 9, 10]                # S. aureus x2, VRE faecalis, VRE faecium
MDR = [8, 9, 10]                  # MRSA, VRE faecalis, VRE faecium (hard Gram+ MDR)
ALL = list(range(11))
THRESH = 16.0
TAU = 0.5  # log10 units


def soft_success(mic: np.ndarray) -> np.ndarray:
    """Smooth, saturating P(MIC<=16 uM). (n,11) -> (n,11)."""
    margin = (np.log10(THRESH) - np.log10(np.clip(mic, 1e-6, None))) / TAU
    return 1.0 / (1.0 + np.exp(-margin))


def hard_sr(mic: np.ndarray, idx=ALL) -> np.ndarray:
    return (mic[:, idx] <= THRESH).mean(axis=1)


def activity_reward(mic: np.ndarray, gp_w: float = 0.5, mdr_w: float = 0.5) -> np.ndarray:
    """Broad soft-Success-Rate with the hard Gram+/MDR buckets up-weighted.

    r_act = SR_all + gp_w*SR_gp + mdr_w*SR_mdr   (all soft, in [0, 1+gp_w+mdr_w]).
    Up-weighting the currently-weak buckets pulls coverage toward the hard strains without
    abandoning the strong Gram-negative side.
    """
    s = soft_success(mic)
    return s[:, ALL].mean(1) + gp_w * s[:, GP].mean(1) + mdr_w * s[:, MDR].mean(1)


def reward(mic: np.ndarray, phemo: np.ndarray | None = None, *,
           gp_w: float = 0.5, mdr_w: float = 0.5, hemo_lambda: float = 1.0) -> np.ndarray:
    r = activity_reward(mic, gp_w=gp_w, mdr_w=mdr_w)
    if phemo is not None and hemo_lambda > 0:
        r = r - hemo_lambda * np.asarray(phemo)
    return r


def summarize(mic: np.ndarray, phemo: np.ndarray | None = None, idx: np.ndarray | None = None) -> dict:
    """Predicted profile of a peptide set (idx into mic), for logging."""
    m = mic if idx is None else mic[idx]
    minmic = m.min(axis=1)
    out = dict(
        n=len(m),
        pct_active=float(100 * (minmic <= THRESH).mean()),
        med_minmic=float(np.median(minmic)),
        SR_broad=float(100 * hard_sr(m, ALL).mean()),
        SR_gn=float(100 * hard_sr(m, GN).mean()),
        SR_gp=float(100 * hard_sr(m, GP).mean()),
        SR_mdr=float(100 * hard_sr(m, MDR).mean()),
        mean_breadth=float((m <= THRESH).sum(1).mean()),
    )
    if phemo is not None:
        ph = phemo if idx is None else np.asarray(phemo)[idx]
        out["mean_phemo"] = float(np.mean(ph))
    return out


def fmt(s: dict, label: str = "") -> str:
    extra = f" phemo={s['mean_phemo']:.3f}" if "mean_phemo" in s else ""
    return (f"{label:<20} n={s['n']:>6} act={s['pct_active']:5.1f}% minMIC~{s['med_minmic']:5.1f} "
            f"| SR broad={s['SR_broad']:4.1f} GN={s['SR_gn']:4.1f} GP={s['SR_gp']:4.1f} "
            f"MDR={s['SR_mdr']:4.1f} | breadth={s['mean_breadth']:.2f}/11{extra}")
