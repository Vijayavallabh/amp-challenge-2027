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


def _bare_ranker(*, compw=1.0, aromw=0.5, refine_k=100, hemo=None, lam=0.0, maxcat=None):
    """An ApexRanker with only the composition attributes set (no APEX oracle loaded), for
    unit-testing the pure ranking logic (_composition_rank / _active_band / _band_phemo)."""
    r = object.__new__(ApexRanker)
    r._compw, r._aromw, r._refine_k, r._hemo, r._lam = compw, aromw, refine_k, hemo, lam
    r._maxcat = maxcat
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

    def test_fraction_formula_matches_physchem_frac(self):
        # #5: pin the fraction FORMULA (not just the aromatic SET) to physchem._frac, the shared
        # count/len definition, so lys_fraction/aromatic_fraction cannot silently diverge from it --
        # including the empty-string -> 0.0 convention. Divergence here would shift the shipped signal.
        from amp_challenge_2027 import physchem
        for s in ["KWKLFKKIGAVLKVL", "RRRWWRR", "KKKKKAAAAA", "FWYFWY", "", "K", "AAAA"]:
            assert O.lys_fraction([s])[0] == physchem._frac(s, {"K"})
            assert O.aromatic_fraction([s])[0] == physchem._frac(s, physchem._AROMATIC)


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
        err = capsys.readouterr().err
        assert "falling back to APEX activity ranking" in err
        assert "it is unavailable" in err  # a genuine load failure IS reported as unavailable

    def test_penalty_zero_reports_disabled_not_unavailable(self, monkeypatch, capsys):
        # #2: with --hemolysis-penalty 0 the selectivity model is intentionally NOT loaded, so hemo is
        # None for a different reason than a load failure. The fallback message must say the penalty is
        # disabled (and how to re-enable it), NOT misreport the model as "unavailable".
        monkeypatch.setattr("amp_challenge_2027.oracle.ApexScorer", _StubScorer)
        args = parse_args(["--rank", "apex", "--composition-weight", "1.0", "--hemolysis-penalty", "0"])
        ranker = build_ranker(object(), args)
        assert ranker._compw == 0.0
        err = capsys.readouterr().err
        assert "--hemolysis-penalty is 0" in err
        assert "unavailable" not in err

    def test_composition_mode_discloses_superseded_legacy_flags(self, monkeypatch, capsys):
        # #8: composition mode ignores the enabled-by-default amphipathicity-bonus / lys-hedge; say so.
        monkeypatch.setattr("amp_challenge_2027.oracle.ApexScorer", _StubScorer)
        monkeypatch.setattr("amp_challenge_2027.selectivity_esm.EsmcSelectivityScorer", _StubHemo)
        args = parse_args(["--rank", "apex", "--composition-weight", "1.0"])  # amphi 0.2, lys-hedge 0.4
        ranker = build_ranker(object(), args)
        assert ranker._compw == 1.0
        err = capsys.readouterr().err
        assert "supersedes and ignores" in err and "amphipathicity" in err and "lys-hedge" in err


class TestCompositionEnvelopeCap:
    """The --max-cationic-fraction envelope cap (feat-037): over-cationic band members are demoted
    below every in-envelope one, so the top list stays inside the wet-lab-validated cationic envelope
    (the max (K+R)/len among the 46 actives is 4/9). None or >= 1.0 disables it, recovering feat-033."""

    def test_cap_demotes_over_cationic_below_every_in_envelope(self):
        # seqA is pure-Lys (top composition) but over the ceiling; seqB/seqC are in-envelope.
        seqs = ["KKKKKKKKKK", "KKKLLLLLLL", "KKLLLLLLLL"]  # fcat 1.0 (over), 0.3 (in), 0.2 (in)
        activity = np.array([1.0, 0.9, 0.8])
        uncapped = _bare_ranker(refine_k=10)._composition_rank(seqs, activity)
        assert uncapped[0] > uncapped[1] > uncapped[2]     # pure-Lys wins uncapped
        capped = _bare_ranker(refine_k=10, maxcat=0.444)._composition_rank(seqs, activity)
        assert capped[0] < min(capped[1], capped[2])       # over-cationic below EVERY in-envelope one

    def test_cap_boundary_keeps_peptide_at_the_validated_max(self):
        # A peptide at exactly 4/9 (the max (K+R)/len among the 46 wet-lab actives) must be KEPT by the
        # shipped 0.4445 cap and only DEMOTED by a naive 0.444 truncation (code-review #1).
        seqs = ["KKKKLLLLL", "KKKLLLLLL"]     # fcat 4/9=0.4444 (the ceiling) vs 3/9=0.333 (in)
        activity = np.array([1.0, 0.9])
        kept = _bare_ranker(refine_k=10, maxcat=0.4445)._composition_rank(seqs, activity)
        assert kept[0] > kept[1]              # at the ceiling -> kept; higher composition still wins
        truncated = _bare_ranker(refine_k=10, maxcat=0.444)._composition_rank(seqs, activity)
        assert truncated[1] > truncated[0]    # 0.444 wrongly demotes the ceiling peptide

    def test_cap_offset_holds_under_large_composition_weight(self):
        # The demotion is derived from the in-envelope score minimum, not a fixed constant, so a large
        # --composition-weight cannot lift an over-envelope peptide back in (code-review #2: a fixed
        # -10 offset broke here).
        seqs = ["KKKKKKKKKK", "KKKLLLLLLL"]   # fcat 1.0 (over) vs 0.3 (in)
        activity = np.array([1.0, 0.9])
        capped = _bare_ranker(refine_k=10, compw=20.0, maxcat=0.444)._composition_rank(seqs, activity)
        assert capped[1] > capped[0]          # in-envelope still outranks the extrapolation at compw=20

    def test_cap_backfill_order_is_least_cationic_first(self):
        # Among demoted over-envelope members the LEAST cationic (closest to the envelope) ranks highest,
        # so overflow backfills with the closest-to-envelope peptide (code-review #5).
        seqs = ["KRKRKRKRKR", "KKKKKKKKKA", "KKKLLLLLLL"]  # fcat 1.0, 0.9 (both over), 0.3 (in)
        activity = np.array([1.0, 0.9, 0.8])
        capped = _bare_ranker(refine_k=10, maxcat=0.444)._composition_rank(seqs, activity)
        assert capped[1] > capped[0]                       # less-cationic (0.9) outranks more (1.0)
        assert min(capped[0], capped[1]) < capped[2]       # both demoted below the in-envelope member

    def test_cap_leaves_in_envelope_scores_unchanged(self):
        seqs = ["KKKLLLLLLL", "KLLLLLLLLL"]   # fcat 0.3, 0.1 -- both in-envelope, untouched by the cap
        activity = np.array([1.0, 0.9])
        assert (_bare_ranker(refine_k=10, maxcat=0.4445)._composition_rank(seqs, activity)
                == _bare_ranker(refine_k=10)._composition_rank(seqs, activity))

    def test_cap_none_recovers_feat033_order(self):
        seqs = ["KKKKKKKKKK", "KKKLLLLLLL"]
        activity = np.array([1.0, 0.9])
        assert (_bare_ranker(refine_k=10, maxcat=None)._composition_rank(seqs, activity)
                == _bare_ranker(refine_k=10)._composition_rank(seqs, activity))

    def test_max_cationic_fraction_ge_one_disables_and_names_off(self, monkeypatch):
        # >= 1.0 disables the cap (code-review #3): _maxcat maps to None and the name must not claim a cap.
        monkeypatch.setattr("amp_challenge_2027.oracle.ApexScorer", _StubScorer)
        assert ApexRanker("x", composition_weight=1.0, max_cationic_fraction=1.0)._maxcat is None
        assert "capped" not in ApexRanker("x", composition_weight=1.0, max_cationic_fraction=1.0).name
        assert ApexRanker("x", composition_weight=1.0, max_cationic_fraction=0.4445)._maxcat == 0.4445
