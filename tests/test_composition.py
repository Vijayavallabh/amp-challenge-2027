"""Tests for the SHIPPED-DEFAULT composition ranking (feat-033): the composition primitives, the
``ApexRanker`` composition-rank path (the APEX-active-band gate, the aromatic-penalty sign, the
graceful overflow, determinism, and the weight guards), and the ``build_ranker`` wiring (the
no-selectivity-model fallback and the superseded-legacy-flag disclosure).

Before feat-033 this default-on path had no coverage; these lock its behaviour so a regression in
the penalty sign, the gate, the stable-sort determinism, or the CLI plumbing cannot ship silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from amp_challenge_2027 import oracle as O
from amp_challenge_2027.generate import build_ranker, parse_args
from amp_challenge_2027.model import ApexRanker


def _bare_ranker(*, compw=1.0, aromw=0.5, refine_k=100, hemo=None, lam=0.0):
    """An ApexRanker with only the composition attributes set (no APEX oracle loaded), for
    unit-testing the pure ranking logic (_composition_rank / _active_band / _band_phemo)."""
    r = object.__new__(ApexRanker)
    r._compw, r._aromw, r._refine_k, r._hemo, r._lam = compw, aromw, refine_k, hemo, lam
    return r


class _StubScorer:
    """Stand-in for oracle.ApexScorer so ApexRanker constructs without a real oracle project."""

    def __init__(self, apex_dir, device="cpu"):
        self.apex_dir = apex_dir


class _StubHemo:
    def __init__(self, *a, **k):
        pass

    def predict_proba(self, sequences):
        return [0.0 for _ in sequences]


def _raise(*a, **k):
    raise RuntimeError("model unavailable")


class TestCompositionPrimitives:
    def test_lys_fraction(self):
        np.testing.assert_allclose(O.lys_fraction(["KKKAA", "AAAAA", "K"]), [0.6, 0.0, 1.0])

    def test_lys_fraction_empty_is_zero_not_error(self):
        assert O.lys_fraction([""])[0] == 0.0

    def test_aromatic_fraction_counts_FWY_only(self):
        np.testing.assert_allclose(O.aromatic_fraction(["FWYAA", "KKKKK"]), [0.6, 0.0])
        assert O.aromatic_fraction(["HHHH"])[0] == 0.0  # His is NOT aromatic here (matches physchem)

    def test_aromatic_fraction_empty_is_zero(self):
        assert O.aromatic_fraction([""])[0] == 0.0

    def test_composition_score_is_lys_minus_weighted_aromatic(self):
        # KKKFF: lys 3/5=0.6, aromatic 2/5=0.4 -> 0.6 - 0.5*0.4 = 0.4
        assert O.composition_score(["KKKFF"])[0] == pytest.approx(0.4)
        assert O.composition_score(["KKKFF"], aromatic_weight=1.0)[0] == pytest.approx(0.2)
        assert O.composition_score(["KKKFF"], aromatic_weight=0.0)[0] == pytest.approx(0.6)

    def test_aromatic_set_is_shared_with_physchem(self):
        # single source of truth (review #9): the two definitions cannot silently diverge
        from amp_challenge_2027 import physchem
        assert set(O._AROMATIC) == set(physchem._AROMATIC) == set("FWY")

    def test_composition_deterministic(self):
        seqs = ["KWKLFKKIGAVLKVL", "RRRWWRR", "KKKKKAAAAA"]
        np.testing.assert_array_equal(O.composition_score(seqs), O.composition_score(seqs))


class TestCompositionRank:
    def test_lys_rich_outranks_aromatic_rich_in_band(self):
        seqs = ["KKKKKKKKKK", "FFFFFWWWWW", "KKKKKFFFFF"]
        activity = np.array([1.0, 1.0, 1.0])  # all in band
        scores = _bare_ranker(refine_k=3)._composition_rank(seqs, activity)
        assert scores[0] > scores[2] > scores[1]  # pure-Lys > mixed > pure-aromatic

    def test_gate_excludes_apex_inactive_and_overflow_is_activity_ordered(self):
        seqs = ["KKKKK", "KKKKK", "KKKKK", "KKKKK"]  # identical composition
        activity = np.array([0.9, 0.8, 0.1, 0.05])
        scores = np.array(_bare_ranker(refine_k=2)._composition_rank(seqs, activity))
        assert scores[0] == scores[1]                       # both in band, identical comp
        assert scores[2] < scores[0] and scores[3] < scores[0]   # out-of-band strictly below the band
        assert scores[2] > scores[3]                        # overflow ordered by APEX activity (#1 fix)

    def test_hemolysis_penalty_pushes_down_hemolytic_peptide(self):
        # KKKKKKKKKK (lys 1.0) is hemolytic (P=0.9); KKKKKKKKKA (lys 0.9) is clean -> clean one wins
        phemo = {"KKKKKKKKKK": 0.9, "KKKKKKKKKA": 0.0}

        class _H:
            def predict_proba(self, seqs):
                return [phemo[s] for s in seqs]

        seqs = ["KKKKKKKKKK", "KKKKKKKKKA"]
        activity = np.array([1.0, 1.0])
        no_pen = _bare_ranker(refine_k=2)._composition_rank(seqs, activity)
        assert no_pen[0] > no_pen[1]                         # without penalty, higher-Lys wins
        pen = _bare_ranker(refine_k=2, hemo=_H(), lam=1.5)._composition_rank(seqs, activity)
        assert pen[1] > pen[0]                               # with penalty, the clean peptide wins

    def test_deterministic(self):
        seqs = ["KKKKKFFFFF", "KKKKKKKKKK", "FFFFFFFFFF", "KKKAAKKKAA"]
        activity = np.array([1.0, 0.9, 0.8, 0.7])
        r = _bare_ranker(refine_k=3)
        assert r._composition_rank(seqs, activity) == r._composition_rank(seqs, activity)


class TestRankerWeightGuards:
    def test_aromatic_weight_guard_clamps_bad_values(self, monkeypatch):
        monkeypatch.setattr("amp_challenge_2027.oracle.ApexScorer", _StubScorer)
        for bad in (float("inf"), float("-inf"), float("nan"), -1.0):
            assert ApexRanker("x", composition_weight=1.0, aromatic_weight=bad)._aromw == 0.0
        assert ApexRanker("x", composition_weight=1.0, aromatic_weight=0.5)._aromw == 0.5

    def test_composition_weight_guard_clamps_bad_values(self, monkeypatch):
        monkeypatch.setattr("amp_challenge_2027.oracle.ApexScorer", _StubScorer)
        for bad in (float("inf"), float("nan"), -1.0):
            assert ApexRanker("x", composition_weight=bad)._compw == 0.0


class TestBuildRankerCompositionWiring:
    def test_falls_back_to_apex_ranking_when_no_selectivity_model(self, monkeypatch, capsys):
        # A pure-composition top-50 (no selectivity model) is measured 22% ESMC-hemolytic; build_ranker
        # must fall back to APEX activity ranking (comp_w -> 0), never ship that (review #2).
        monkeypatch.setattr("amp_challenge_2027.oracle.ApexScorer", _StubScorer)
        monkeypatch.setattr("amp_challenge_2027.selectivity_esm.EsmcSelectivityScorer", _raise)
        monkeypatch.setattr("amp_challenge_2027.hemolysis.HemolysisScorer", _raise)
        args = parse_args(["--rank", "apex", "--composition-weight", "1.0", "--hemolysis-penalty", "1.5"])
        ranker = build_ranker(object(), args)
        assert isinstance(ranker, ApexRanker)
        assert ranker._compw == 0.0
        assert "falling back to APEX activity ranking" in capsys.readouterr().err

    def test_composition_mode_discloses_superseded_legacy_flags(self, monkeypatch, capsys):
        # #8: composition mode ignores the enabled-by-default amphipathicity-bonus / lys-hedge; say so.
        monkeypatch.setattr("amp_challenge_2027.oracle.ApexScorer", _StubScorer)
        monkeypatch.setattr("amp_challenge_2027.selectivity_esm.EsmcSelectivityScorer", _StubHemo)
        args = parse_args(["--rank", "apex", "--composition-weight", "1.0"])  # amphi 0.2, lys-hedge 0.4
        ranker = build_ranker(object(), args)
        assert ranker._compw == 1.0
        err = capsys.readouterr().err
        assert "supersedes and ignores" in err and "amphipathicity" in err and "lys-hedge" in err
