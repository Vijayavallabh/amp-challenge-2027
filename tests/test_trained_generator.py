"""Tests for the trained generator and build_model selection.

The generator *machinery* (sampling, ranking, determinism, the np-rng -> torch-seed bridge)
is exercised through TrainedGenerator loaded from a **tiny** checkpoint, so the tests are
CPU-fast and run in CI without a GPU. The real 41MB checkpoint is only loaded (cheap) and
its scoring path checked; its full sampling is covered by the end-to-end `./init.sh` run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from amp_challenge_2027 import constraints as C
from amp_challenge_2027.generate import build_model, parse_args
from amp_challenge_2027.model import RandomBaseline, TrainedGenerator
from amp_challenge_2027.nn import LMConfig, PeptideLM

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "checkpoint" / "generator.pt"
needs_ckpt = pytest.mark.skipif(not CHECKPOINT.is_file(), reason="no trained checkpoint committed")


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory) -> Path:
    """A tiny, randomly-initialised generator saved in the shipped checkpoint format."""
    cfg = LMConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64)
    model = PeptideLM(cfg)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict(),
                "seed": 0, "val_loss": 0.0, "epoch": 0}, path)
    return path


@pytest.fixture(scope="module")
def gen(tiny_ckpt) -> TrainedGenerator:
    return TrainedGenerator(tiny_ckpt, device="cpu")


class TestSampling:
    def test_samples_are_valid_peptides(self, gen):
        seqs = gen.sample(16, np.random.default_rng(0))
        assert len(seqs) == 16
        assert all(C.is_valid_sequence(s) for s in seqs)

    def test_sampling_is_deterministic_for_a_fixed_rng_seed(self, gen):
        assert gen.sample(16, np.random.default_rng(123)) == gen.sample(16, np.random.default_rng(123))

    def test_successive_draws_from_the_same_rng_differ(self, gen):
        rng = np.random.default_rng(0)
        assert gen.sample(16, rng) != gen.sample(16, rng)

    def test_respects_the_length_window(self, tiny_ckpt):
        g = TrainedGenerator(tiny_ckpt, device="cpu", min_length=10, max_length=20)
        assert all(10 <= len(s) <= 20 for s in g.sample(16, np.random.default_rng(1)))


class TestScoring:
    def test_one_finite_score_per_sequence(self, gen):
        seqs = gen.sample(8, np.random.default_rng(2))
        scores = gen.score(seqs)
        assert len(scores) == len(seqs)
        assert all(np.isfinite(s) for s in scores)

    def test_scoring_is_deterministic(self, gen):
        seqs = ["KWKLFKKIGAVLKVL", "GLFDIVKKVVGALGSL", "AAAAAAAAAA"]
        assert gen.score(seqs) == gen.score(seqs)

    def test_score_is_per_sequence_not_position_dependent(self, gen):
        seqs = ["KWKLFKKIGAVLKVL", "GLFDIVKKVVGALGSL", "ILPWKWPWWPWRR"]
        forward = dict(zip(seqs, gen.score(seqs)))
        rev = dict(zip(seqs[::-1], gen.score(seqs[::-1])))
        for s in seqs:
            assert forward[s] == pytest.approx(rev[s], abs=1e-4)


def _args(**over):
    argv: list[str] = []
    for key, value in over.items():
        flag = "--" + key.replace("_", "-")
        argv += [flag] if value is True else [flag, str(value)]
    return parse_args(argv)


class TestBuildModelSelection:
    @needs_ckpt
    def test_uses_trained_generator_when_checkpoint_present(self):
        model = build_model(_args())
        assert isinstance(model, TrainedGenerator)
        assert model.name == "trained-ar-transformer"

    def test_falls_back_to_baseline_when_forced(self):
        assert isinstance(build_model(_args(baseline=True)), RandomBaseline)

    def test_falls_back_when_checkpoint_missing(self, tmp_path):
        model = build_model(_args(checkpoint=str(tmp_path / "nope.pt")))
        assert isinstance(model, RandomBaseline)


@needs_ckpt
def test_real_checkpoint_loads_and_scores():
    # Loading + a single forward-pass score is cheap even for the full model; sampling is not,
    # so it is left to the end-to-end run.
    gen = TrainedGenerator(CHECKPOINT, device="cpu")
    scores = gen.score(["KWKLFKKIGAVLKVL", "GLFDIVKKVVGALGSL"])
    assert len(scores) == 2 and all(np.isfinite(s) for s in scores)
