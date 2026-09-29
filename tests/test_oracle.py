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
from amp_challenge_2027.generate import build_ranker, parse_args, select_maximin, select_top
from amp_challenge_2027.model import RandomBaseline


class _FixedRanker:
    """Ranks by a fixed score map (higher = better); for testing selection logic."""

    name = "fixed"

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, sequences: list[str]) -> list[float]:
        return [self._scores[s] for s in sequences]

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

    def test_broad_potency_rewards_breadth_and_zeroes_inactive(self):
        s = O.broad_potency_score(MIC)                    # threshold 16 uM
        assert s[0] > s[1] > s[2]                         # broad > narrow > inactive
        assert s[2] == pytest.approx(0.0)                 # nothing below 16 -> zero
        # broad peptide: 11 strains x log10(16/2); narrow: 3 strains x log10(16/4)
        assert s[0] == pytest.approx(11 * np.log10(8.0))
        assert s[1] == pytest.approx(3 * np.log10(4.0))

    def test_broad_potency_ignores_strains_above_threshold(self):
        # a strain at exactly the threshold contributes zero; only sub-threshold margin counts
        one = np.array([[16.0] + [1000.0] * 10])
        assert O.broad_potency_score(one)[0] == pytest.approx(0.0)

    def test_balanced_success_rewards_hard_gp_mdr_hits(self):
        # 'balanced' = gp*SR_hard(Gram+) + mdr*SR_hard(MDR) + gn*SR_hard(Gram-) + broad*mean_soft
        # (defaults gp=mdr=1, gn=0.75, broad=0.5)
        s = O.balanced_success_score(MIC)
        assert s[0] > s[1] > s[2]  # broad (clears all buckets) > narrow (a few Gram-) > inactive
        soft = O._soft_success(MIC)
        # broadly-potent peptide clears all 4 Gram+, all 3 MDR, all 7 Gram- -> full hard SR on each
        assert s[0] == pytest.approx(1.0 + 1.0 + 0.75 * 1.0 + 0.5 * soft[0].mean())
        # narrow peptide clears 3 of 7 Gram- strains (indices 0,1,2), zero Gram+/MDR
        assert s[1] == pytest.approx(0.75 * (3.0 / 7.0) + 0.5 * soft[1].mean())

    def test_balanced_uses_hard_threshold_not_soft_margin(self):
        # Two peptides that tie on hard Gram+ SR (both clear all 4) but differ in depth: 'balanced'
        # rewards the hard hit fully and does not chase sub-uM depth -- the deep one wins only via
        # the small broad soft tie-break, not by a large margin (unlike an unbounded potency score).
        deep = np.array([[0.1] * 11])       # far below threshold on every strain
        just_in = np.array([[15.0] * 11])   # just below the 16 uM threshold on every strain
        gap = O.balanced_success_score(deep)[0] - O.balanced_success_score(just_in)[0]
        assert 0.0 < gap < 0.5              # both get full hard SR; only the soft tie-break differs


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


class TestDiversityScreen:
    # highest-scored two are near-duplicates (0.9 identity); third is distinct.
    LIB = ["AAAAAAAAAA", "AAAAAAAAAC", "KLWKKLLKKL"]
    SCORES = {"AAAAAAAAAA": 3.0, "AAAAAAAAAC": 2.0, "KLWKKLLKKL": 1.0}

    def test_without_cap_takes_the_near_duplicate(self):
        top = select_top(self.LIB, _FixedRanker(self.SCORES), 2, [])
        assert top == ["AAAAAAAAAA", "AAAAAAAAAC"]        # by score, duplicates kept

    def test_cap_skips_near_duplicate_keeps_more_active(self):
        top = select_top(self.LIB, _FixedRanker(self.SCORES), 2, [],
                         diversity_max_identity=0.8)
        # the 0.9-identity runner-up is skipped; the distinct third fills the slot
        assert top == ["AAAAAAAAAA", "KLWKKLLKKL"]

    def test_cap_is_deterministic(self):
        a = select_top(self.LIB, _FixedRanker(self.SCORES), 2, [], diversity_max_identity=0.8)
        b = select_top(self.LIB, _FixedRanker(self.SCORES), 2, [], diversity_max_identity=0.8)
        assert a == b


class _MaximinRanker:
    """A ranker exposing both seams: ``maximin_data`` (per-category rates + phemo) for the maximin
    selector and ``score`` (a balanced weighted sum) for the older single-score path."""

    name = "maximin-mock"

    def __init__(self, rates: dict[str, list[float]], phemo: dict[str, float] | None = None) -> None:
        self._r = rates
        self._p = phemo or {}

    def maximin_data(self, sequences: list[str]):
        return (np.array([self._r[s] for s in sequences], dtype=float),
                np.array([self._p.get(s, 0.0) for s in sequences], dtype=float))

    def score(self, sequences: list[str]) -> list[float]:
        # balanced-style weighted sum: Gram+ + MDR + 0.75*Gram- + 0.5*Broad
        return [c[2] + c[3] + 0.75 * c[1] + 0.5 * c[0] for c in (self._r[s] for s in sequences)]


class TestMaximinSelection:
    def test_category_rates_are_per_bucket_success_fractions(self):
        mic = np.array([[2.0] * 11, [400.0] * 11, [4.0, 4.0, 4.0] + [500.0] * 8])
        r = O.category_rates(mic)                       # [Broad, Gram-, Gram+, MDR]
        assert r.shape == (3, 4)
        assert list(r[0]) == [1.0, 1.0, 1.0, 1.0]       # clears everything
        assert list(r[1]) == [0.0, 0.0, 0.0, 0.0]       # clears nothing
        # third peptide clears strains 0,1,2 (all Gram-negative): Broad 3/11, Gram- 3/7, Gram+/MDR 0
        assert r[2][1] == pytest.approx(3 / 7)
        assert r[2][2] == 0.0 and r[2][3] == 0.0

    # Two Gram+/MDR specialists and one Gram--specialist. A weighted-sum ranker takes the two
    # highest-scoring (both Gram+/MDR) and leaves Gram- at the floor; maximin must instead cover the
    # weak Gram- category by including the Gram--specialist in the assayed set.
    LIB = ["KKKKKKKKKK", "RRRRRRRRRR", "DDDDDDDDDD", "EEEEEEEEEE"]
    RATES = {
        "KKKKKKKKKK": [0.5, 0.2, 1.0, 1.0],   # Gram+/MDR strong
        "RRRRRRRRRR": [0.5, 0.2, 1.0, 1.0],   # Gram+/MDR strong
        "DDDDDDDDDD": [0.5, 1.0, 0.2, 0.2],   # Gram- strong
        "EEEEEEEEEE": [0.3, 0.3, 0.3, 0.3],   # mediocre
    }

    def test_maximin_covers_the_weak_category(self):
        r = _MaximinRanker(self.RATES)
        top = select_maximin(self.LIB, r, 2, [], assayed_k=2, diversity_max_identity=None)
        assert "DDDDDDDDDD" in top                       # the Gram--specialist is selected
        # the older single-score path grabs the two Gram+/MDR specialists and neglects Gram-
        assert "DDDDDDDDDD" not in select_top(self.LIB, r, 2, [])

    def test_maximin_gates_hemolytic_and_is_deterministic(self):
        # make the Gram--specialist predicted-hemolytic: it must be gated out despite covering Gram-
        r = _MaximinRanker(self.RATES, phemo={"DDDDDDDDDD": 0.9})
        top = select_maximin(self.LIB, r, 2, [], assayed_k=2, diversity_max_identity=None, gate=0.5)
        assert "DDDDDDDDDD" not in top
        assert top == select_maximin(self.LIB, r, 2, [], assayed_k=2,
                                     diversity_max_identity=None, gate=0.5)  # deterministic


class TestAmphipathicity:
    def test_hydrophobic_moment_matches_membrane_lytic_controls(self):
        # Eisenberg muH: melittin ~0.35, magainin-2 ~0.45 (canonical amphipathic AMPs); polyG ~0.
        mu = O.hydrophobic_moment([
            "GIGAVLKVLTTGLPALISWIKRKRQQ",  # melittin
            "GIGKFLHSAKKFGKAFVGEIMNS",      # magainin-2
            "GGGGGGGGGG",                   # non-amphipathic control
        ])
        assert 0.30 <= mu[0] <= 0.40
        assert 0.40 <= mu[1] <= 0.50
        assert mu[2] < 0.10  # uniform-hydrophobicity peptide -> nearly non-amphipathic, well below band

    def test_amphipathicity_bonus_is_smooth_monotonic_and_saturates(self):
        # smooth [0,1] reward rising with muH: polyG (muH ~0.04) << magainin (muH ~0.45); a strongly
        # amphipathic peptide (muH > the 0.5 saturation) gets the full 1.0 -- NOT excluded as a band
        # would have (regression guard for the feat-025 review: aurein/LL-37 must still be rewarded).
        b = O.amphipathicity_bonus(["GGGGGGGGGG", "GIGKFLHSAKKFGKAFVGEIMNS", "GLFDIVKKVVGALGSL"])
        assert 0.0 <= b[0] < b[1] <= 1.0
        assert all(0.0 <= x <= 1.0 for x in b)
        mu = O.hydrophobic_moment(["GIGKFLHSAKKFGKAFVGEIMNS"])[0]
        assert O.amphipathicity_bonus(["GIGKFLHSAKKFGKAFVGEIMNS"])[0] == pytest.approx(
            float(np.clip((mu - 0.25) / 0.25, 0.0, 1.0)))

    def test_arg_excess_penalises_arg_over_lys_above_floor(self):
        # feat-031 wet-lab Gram- hedge: max(0, R/(R+K) - 0.4). All-Lys and no-cationic score 0 (nothing
        # to correct); all-Arg scores 1.0-0.4=0.6; a balanced RRKK (R/(R+K)=0.5) scores 0.1. Monotonic
        # in Arg-fraction. This is the deterministic composition term subtracted from the APEX activity.
        ae = O.arg_excess(["KKKKK", "GGGG", "RRKK", "RRRRR", "RRRK"])
        assert ae[0] == 0.0 and ae[1] == 0.0
        assert ae[2] == pytest.approx(0.1)
        assert ae[3] == pytest.approx(0.6)
        assert ae[4] == pytest.approx(0.35)
        # floor is configurable and defaults to ARG_EXCESS_FLOOR
        assert O.arg_excess(["RRRRR"], floor=0.0)[0] == pytest.approx(1.0)
        assert O.ARG_EXCESS_FLOOR == 0.4
        # deterministic and never negative
        seqs = ["KWKLFKKIGAVLKVL", "RRRWWRR", "GIGKFLHSAKKFGKAFVGEIMNS"]
        assert np.array_equal(O.arg_excess(seqs), O.arg_excess(seqs))
        assert all(x >= 0.0 for x in O.arg_excess(seqs))

    def test_batched_moment_matches_physchem_scalar(self):
        # single source of truth: oracle's vectorised muH must equal physchem's scalar muH exactly,
        # so the shared Eisenberg scale / formula cannot silently diverge (feat-025 review #7).
        from amp_challenge_2027 import physchem
        seqs = ["KWKLFKKIGAVLKVL", "GIGKFLHSAKKFGKAFVGEIMNS", "RRRRKKKKDDEE", "ILPWKWPWWPWRR"]
        np.testing.assert_allclose(
            O.hydrophobic_moment(seqs),
            np.array([physchem.hydrophobic_moment(s) for s in seqs]),
            rtol=1e-9, atol=1e-12,
        )

    def test_bonus_is_deterministic(self):
        seqs = ["KWKLFKKIGAVLKVL", "GIGKFLHSAKKFGKAFVGEIMNS", "RRRRRRRRRR"]
        assert np.array_equal(O.hydrophobic_moment(seqs), O.hydrophobic_moment(seqs))

    def test_single_string_is_rejected_not_iterated_per_char(self):
        # a bare string would silently iterate character-by-character and return a bogus per-residue
        # array; the guard must turn that footgun into a loud error (feat-025 review #14).
        with pytest.raises(TypeError):
            O.hydrophobic_moment("GIGKFLHSAKKFGKAFVGEIMNS")


class TestRewardRankerParity:
    """The offline ReST reward (``experiments/reward.balanced_reward``) must MIRROR the shipped
    selection ranker (``oracle.balanced_success_score``) at default weights -- otherwise the generator
    is fine-tuned toward a different objective than selection scores by, which is exactly what happened
    for three feature cycles (the reward regained the Gram- term only in feat-024). This pins the
    invariant so the two cannot silently diverge again on the next objective tweak.
    """

    def _reward_module(self):
        import importlib
        import os
        import sys
        exp = os.path.join(os.path.dirname(__file__), "..", "experiments")
        if exp not in sys.path:
            sys.path.insert(0, exp)
        return importlib.import_module("reward")

    def test_balanced_reward_matches_ranker_at_default_weights(self):
        R = self._reward_module()
        mic = np.array([
            [2.0] * 11,
            [4.0, 4.0, 4.0] + [500.0] * 8,
            [400.0] * 11,
            [1.0, 20.0, 3.0, 50.0, 8.0, 100.0, 2.0, 4.0, 300.0, 6.0, 9.0],
        ])
        # phemo=None -> pure activity term; must equal the ranker's balanced_success_score exactly.
        np.testing.assert_allclose(
            R.balanced_reward(mic, None), O.balanced_success_score(mic), rtol=1e-9, atol=1e-12
        )

    def test_gn_w_zero_recovers_feat020_pure_gp_mdr_target(self):
        R = self._reward_module()
        mic = np.array([[2.0] * 11, [4.0] * 7 + [500.0] * 4])
        np.testing.assert_allclose(
            R.balanced_reward(mic, None, gn_w=0.0),
            O.balanced_success_score(mic, gn_weight=0.0),
            rtol=1e-9, atol=1e-12,
        )


class TestApexThreadPinning:
    def test_both_apex_paths_pin_threads_for_byte_reproducibility(self):
        # Regression guard (adversarial audit, 2026-09-29): the sequential path _run_apex once OMITTED
        # the OMP/MKL/OPENBLAS/NUMEXPR=1 pinning that the parallel path _run_pool sets, so on any grader
        # that took the sequential path (small input, device='cuda', or a low-mem/low-core box where
        # _auto_workers -> 1) APEX ran multi-threaded and the top-100's active-band gate was NOT
        # bit-reproducible -- risking the validator's two-run byte comparison (check #8 = disqualifying).
        # Both paths must build the subprocess env through the single _apex_env() helper. No live oracle
        # needed: _apex_env is static and the wiring is verified by source.
        import inspect
        env = O.ApexScorer._apex_env()
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert env[var] == "1", f"{var} not pinned to 1"
        assert "VIRTUAL_ENV" not in env, "VIRTUAL_ENV must be dropped so uv resolves the APEX project env"
        assert "_apex_env()" in inspect.getsource(O.ApexScorer._run_apex), "_run_apex must use _apex_env()"
        assert "_apex_env()" in inspect.getsource(O.ApexScorer._run_pool), "_run_pool must use _apex_env()"


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

    def test_parallel_cpu_scoring_is_byte_reproducible(self):
        # feat-016: CPU scoring shards over single-threaded workers. The submission's two-run
        # byte comparison depends on this being deterministic AND independent of worker count.
        import random
        rng = random.Random(1)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        seqs = sorted({
            "".join(rng.choice(aa) for _ in range(rng.randint(8, 30))) for _ in range(4000)
        })[:3500]                                            # n >= 3000 triggers the parallel path
        par = O.ApexScorer(device="cpu", workers=6)
        m1 = par.predict_mic(seqs)
        assert m1.shape == (len(seqs), 11)
        assert np.array_equal(m1, par.predict_mic(seqs))     # deterministic across runs
        # a different worker count must give byte-identical results (single-threaded shards)
        assert np.array_equal(m1, O.ApexScorer(device="cpu", workers=12).predict_mic(seqs))
