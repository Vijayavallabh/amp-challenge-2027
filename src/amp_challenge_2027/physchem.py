"""Deterministic physicochemical descriptors for peptide sequences.

Pure NumPy, no external data or heavy deps, so it runs inside ``generate`` and stays
byte-reproducible. Used for two things: the feature vector behind the hemolysis/selectivity
model (``model.HemolysisScorer``), and characterising the library's property distributions
(the competition's phase-1 screen looks at physicochemical distributions).

Scales are standard published values (Kyte-Doolittle and Eisenberg hydrophobicity). The
hydrophobic moment is Eisenberg's, at the 100 deg/residue periodicity of an alpha-helix -- the
classic amphipathicity measure, and (with hydrophobicity) the main driver of hemolysis, which
is why these features separate hemolytic from non-hemolytic peptides so well.
"""

from __future__ import annotations

import math

import numpy as np

# Kyte-Doolittle hydropathy (higher = more hydrophobic).
_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Eisenberg consensus hydrophobicity (used for the hydrophobic moment).
_EISENBERG = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74,
    "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19,
    "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
_CATIONIC = set("KR")
_ANIONIC = set("DE")
_AROMATIC = set("FWY")
_HYDROPHOBIC = set("AILMFWVC")

#: Names of the feature vector components, in order (see :func:`feature_vector`).
FEATURE_NAMES: tuple[str, ...] = (
    "length", "net_charge", "charge_density", "mean_kd_hydrophobicity",
    "hydrophobic_moment", "frac_hydrophobic", "frac_cationic", "frac_anionic",
    "frac_aromatic", "frac_glycine", "frac_proline",
)


def net_charge(seq: str, ph: float = 7.4) -> float:
    """Approximate net charge at ``ph`` (Lys/Arg +1, Asp/Glu -1, His partial, plus termini)."""
    # Henderson-Hasselbalch on the ionisable groups; pKa values are standard side-chain values.
    def pos(pk: float) -> float:
        return 1.0 / (1.0 + 10 ** (ph - pk))

    def neg(pk: float) -> float:
        return 1.0 / (1.0 + 10 ** (pk - ph))

    charge = pos(9.0)  # N-terminus
    charge += neg(2.0)  # C-terminus (contributes negative)
    for aa in seq:
        if aa == "K":
            charge += pos(10.5)
        elif aa == "R":
            charge += pos(12.5)
        elif aa == "H":
            charge += pos(6.0)
        elif aa == "D":
            charge -= neg(3.9)
        elif aa == "E":
            charge -= neg(4.1)
        elif aa == "C":
            charge -= neg(8.3)
        elif aa == "Y":
            charge -= neg(10.1)
    return charge


def hydrophobic_moment(seq: str, angle_deg: float = 100.0) -> float:
    """Eisenberg hydrophobic moment at the given periodicity (100 deg = alpha-helix).

    The magnitude of the vector sum of per-residue hydrophobicities placed around a helical
    wheel, normalised by length -- high for amphipathic (one hydrophobic face) peptides.
    """
    if not seq:
        return 0.0
    angle = math.radians(angle_deg)
    sx = sum(_EISENBERG.get(aa, 0.0) * math.cos(i * angle) for i, aa in enumerate(seq))
    sy = sum(_EISENBERG.get(aa, 0.0) * math.sin(i * angle) for i, aa in enumerate(seq))
    return math.hypot(sx, sy) / len(seq)


def _frac(seq: str, members: set[str]) -> float:
    return sum(1 for aa in seq if aa in members) / len(seq) if seq else 0.0


def descriptors(seq: str) -> dict[str, float]:
    """Return the named physicochemical descriptors for one sequence."""
    n = len(seq)
    charge = net_charge(seq)
    return {
        "length": float(n),
        "net_charge": charge,
        "charge_density": charge / n if n else 0.0,
        "mean_kd_hydrophobicity": (sum(_KD.get(aa, 0.0) for aa in seq) / n) if n else 0.0,
        "hydrophobic_moment": hydrophobic_moment(seq),
        "frac_hydrophobic": _frac(seq, _HYDROPHOBIC),
        "frac_cationic": _frac(seq, _CATIONIC),
        "frac_anionic": _frac(seq, _ANIONIC),
        "frac_aromatic": _frac(seq, _AROMATIC),
        "frac_glycine": _frac(seq, {"G"}),
        "frac_proline": _frac(seq, {"P"}),
    }


def feature_vector(seq: str) -> np.ndarray:
    """The descriptors as an array in :data:`FEATURE_NAMES` order."""
    d = descriptors(seq)
    return np.array([d[name] for name in FEATURE_NAMES], dtype=float)


def feature_matrix(sequences: list[str]) -> np.ndarray:
    """Stack :func:`feature_vector` over many sequences into an ``(n, len(FEATURE_NAMES))`` array."""
    if not sequences:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return np.vstack([feature_vector(s) for s in sequences])
