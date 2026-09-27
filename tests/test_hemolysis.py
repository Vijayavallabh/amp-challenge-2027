"""Tests for the hemolysis/selectivity model and its wiring into the APEX ranker.

These are CPU-cheap: the shipped hemolysis checkpoint is a tiny MLP, and constructing the
ranker does not invoke the APEX subprocess (that only happens on ``.score``). So they run in
CI. The APEX+hemolysis scoring path end-to-end is covered by ``./init.sh`` and the validator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027.generate import build_ranker, parse_args
from amp_challenge_2027.hemolysis import HemolysisScorer
from amp_challenge_2027.model import ApexRanker

REPO_ROOT = Path(__file__).resolve().parents[1]
HEMO_CKPT = REPO_ROOT / "checkpoint" / "hemolysis.pt"
APEX_DIR = REPO_ROOT / "oracle" / "apex"
needs_hemo = pytest.mark.skipif(not HEMO_CKPT.is_file(), reason="no hemolysis checkpoint committed")
needs_apex = pytest.mark.skipif(
    not (APEX_DIR / "APEX_predict.py").is_file(), reason="no APEX oracle committed"
)

MELITTIN = "GIGAVLKVLTTGLPALISWIKRKRQQ"
POLY_E = "EEEEEEEEEEEE"


@needs_hemo
class TestHemolysisScorer:
    def test_probabilities_in_unit_interval(self):
        p = HemolysisScorer().predict_proba([MELITTIN, POLY_E, "KWKLFKKIGAVLKVL"])
        assert p.shape == (3,)
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_deterministic(self):
        h = HemolysisScorer()
        assert np.array_equal(h.predict_proba([MELITTIN, POLY_E]), h.predict_proba([MELITTIN, POLY_E]))

    def test_ranks_a_membrane_active_peptide_above_polyanion(self):
        # poly-Glu is not membrane-active; melittin is. The model should order them so.
        p = HemolysisScorer().predict_proba([MELITTIN, POLY_E])
        assert p[0] > p[1]

    def test_reports_its_held_out_auroc(self):
        assert 0.5 < HemolysisScorer().val_auroc <= 1.0


@needs_apex
@needs_hemo
class TestRankerWiring:
    # Constructing the ranker is cheap (no APEX subprocess until .score is called).
    def test_penalty_engages_hemolysis(self):
        ranker = build_ranker(None, parse_args(["--rank", "apex", "--hemolysis-penalty", "2"]))
        assert isinstance(ranker, ApexRanker)
        assert "hemolysis" in ranker.name

    def test_zero_penalty_is_activity_only(self):
        ranker = build_ranker(None, parse_args(["--rank", "apex", "--hemolysis-penalty", "0"]))
        assert isinstance(ranker, ApexRanker)
        assert "hemolysis" not in ranker.name
