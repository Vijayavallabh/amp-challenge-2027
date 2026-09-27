"""Tests for the APEX activity oracle and the ranking seam.

The MIC-matrix aggregations and the ``build_ranker`` fallback are pure/cheap and always run.
The live APEX subprocess is heavy (it syncs an isolated torch env on first call), so that test
is gated behind ``AMP_RUN_APEX_TESTS=1`` -- it runs locally, and is skipped in CPU CI where the
env is not provisioned. The full APEX path is also exercised end-to-end by ``./init.sh`` and the
official validator.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from amp_challenge_2027 import oracle as O
from amp_challenge_2027.generate import build_ranker, parse_args
from amp_challenge_2027.model import RandomBaseline

# a 3-peptide x 11-pathogen MIC matrix: potent / mixed / inactive
MIC = np.array([
    [2.0] * 11,                                   # broadly potent
    [4.0, 4.0, 4.0] + [500.0] * 8,                # narrow-spectrum (3 pathogens)
    [400.0] * 11,                                 # inactive
])


class TestAggregations:
    def test_min_and_mean(self):
        assert list(O.min_mic(MIC)) == [2.0, 4.0, 400.0]
        assert O.mean_mic(MIC)[0] == pytest.approx(2.0)
        assert O.mean_mic(MIC)[2] == pytest.approx(400.0)

    def test_breadth_counts_pathogens_under_threshold(self):
        assert list(O.breadth(MIC, threshold=32.0)) == [11, 3, 0]

    def test_potency_score_higher_is_more_potent(self):
        s = O.potency_score(MIC)
        assert s[0] > s[1] > s[2]                 # potent > narrow > inactive
        assert s[0] == pytest.approx(-np.log10(2.0))

    def test_potency_uses_best_pathogen_not_mean(self):
        # the narrow-spectrum peptide is ranked by its best MIC (4), not its mean (~365)
        assert O.potency_score(MIC)[1] == pytest.approx(-np.log10(4.0))


class TestBuildRanker:
    def test_likelihood_returns_the_model_itself(self):
        model = RandomBaseline()
        assert build_ranker(model, parse_args(["--rank", "likelihood"])) is model

    def test_apex_falls_back_to_likelihood_when_oracle_missing(self, tmp_path):
        model = RandomBaseline()
        # a directory with no APEX_predict.py -> ApexRanker construction raises -> fall back
        ranker = build_ranker(model, parse_args(["--rank", "apex", "--apex-dir", str(tmp_path)]))
        assert ranker is model

    def test_apex_ranker_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(O.ApexUnavailable):
            O.ApexScorer(tmp_path)


@pytest.mark.skipif(
    os.environ.get("AMP_RUN_APEX_TESTS") != "1",
    reason="live APEX subprocess is heavy; set AMP_RUN_APEX_TESTS=1 to run",
)
class TestApexLive:
    def test_ranks_active_above_inactive_and_is_deterministic(self):
        sc = O.ApexScorer(device="cpu")
        seqs = [
            "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # LL-37, active
            "GGGGGGGGGGGGGGGG",                       # inactive control
        ]
        mic = sc.predict_mic(seqs)
        assert mic.shape == (2, 11)
        assert O.min_mic(mic)[0] < O.min_mic(mic)[1]        # LL-37 more potent
        assert np.array_equal(mic, sc.predict_mic(seqs))    # deterministic

    def test_maps_results_back_to_input_order_with_duplicates(self):
        sc = O.ApexScorer(device="cpu")
        seqs = ["KWKLFKKIGAVLKVL", "GGGGGGGGGGGGGGGG", "KWKLFKKIGAVLKVL"]
        mic = sc.predict_mic(seqs)
        assert np.array_equal(mic[0], mic[2])               # duplicate rows identical
