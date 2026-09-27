"""Tests for the physicochemical descriptors (deterministic, no external deps)."""

from __future__ import annotations

import numpy as np

from amp_challenge_2027 import physchem as P


def test_feature_vector_matches_names_length():
    v = P.feature_vector("KWKLFKKIGAVLKVL")
    assert v.shape == (len(P.FEATURE_NAMES),)
    assert np.all(np.isfinite(v))


def test_net_charge_sign():
    # poly-Lys strongly cationic; poly-Glu strongly anionic; poly-Ala ~ neutral (termini only)
    assert P.net_charge("K" * 12) > 8
    assert P.net_charge("E" * 12) < -8
    assert abs(P.net_charge("A" * 12)) < 1.5


def test_hydrophobic_moment_higher_for_amphipathic():
    # an idealised amphipathic helix vs a uniform stretch
    amphipathic = P.hydrophobic_moment("LKLLKKLLKLLKKL")
    uniform = P.hydrophobic_moment("LLLLLLLLLLLLLL")
    assert amphipathic > uniform


def test_fractions_in_unit_range_and_deterministic():
    d1 = P.descriptors("FWYKRDEAILG")
    d2 = P.descriptors("FWYKRDEAILG")
    assert d1 == d2
    for key in ("frac_hydrophobic", "frac_cationic", "frac_anionic", "frac_aromatic"):
        assert 0.0 <= d1[key] <= 1.0
    assert d1["frac_aromatic"] == 3 / 11  # F, W, Y


def test_feature_matrix_stacks():
    m = P.feature_matrix(["KWKLFKKIGAVLKVL", "GGGGGGGGGG"])
    assert m.shape == (2, len(P.FEATURE_NAMES))
