"""The seam where a real generative model plugs in.

Two implementations of :class:`PeptideGenerator` live here:

* :class:`TrainedGenerator` -- the real model: the trained AR-Transformer (``nn.py``),
  sampling deterministically on GPU-if-available (else CPU) and ranking by per-residue
  likelihood. This is what ``build_model`` returns when a checkpoint is present.
* :class:`RandomBaseline` -- a uniform-random placeholder kept as a graceful fallback so
  ``uv run generate`` still produces a contract-valid submission if the checkpoint or
  ``torch`` is missing. Its peptides have no expected antimicrobial activity.

Both must be deterministic so the organizers' twice-run byte comparison passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .constraints import ALPHABET, MAX_LENGTH, MIN_LENGTH

DEFAULT_CHECKPOINT = "checkpoint/generator.pt"


@runtime_checkable
class PeptideGenerator(Protocol):
    """What ``generate.py`` needs from a model.

    Both methods must be deterministic given the same ``rng`` / inputs -- the organizers
    run the entry point twice and compare the output byte for byte.
    """

    name: str

    def sample(self, n: int, rng: np.random.Generator) -> list[str]:
        """Return ``n`` candidate sequences. Duplicates and invalid sequences are fine;
        the caller filters them. Returning more than ``n`` is fine too."""
        ...

    def score(self, sequences: list[str]) -> list[float]:
        """Return one score per sequence, higher meaning a better candidate.

        This is what the ranked top-100 list is built from, so it is the part of a
        submission that the "Optimal Selectivity" and potency categories actually test.
        """
        ...


class RandomBaseline:
    """Uniform random peptides over the 20 standard residues. A placeholder.

    Lengths are drawn uniformly from ``[MIN_LENGTH, MAX_LENGTH]``. Scores are a fixed
    deterministic ordering and carry no biological meaning whatsoever.
    """

    name = "random-baseline"

    def __init__(self, min_length: int = MIN_LENGTH, max_length: int = MAX_LENGTH) -> None:
        if not MIN_LENGTH <= min_length <= max_length <= MAX_LENGTH:
            raise ValueError(
                f"lengths must satisfy {MIN_LENGTH} <= min <= max <= {MAX_LENGTH}, "
                f"got min={min_length}, max={max_length}"
            )
        self.min_length = min_length
        self.max_length = max_length

    def sample(self, n: int, rng: np.random.Generator) -> list[str]:
        alphabet = np.frombuffer(ALPHABET.encode("ascii"), dtype="S1")
        lengths = rng.integers(self.min_length, self.max_length + 1, size=n)
        # One flat draw then slice, so the byte stream consumed from `rng` depends only
        # on `n` and the length draw -- keeps output stable as the sampler is edited.
        flat = rng.integers(0, len(ALPHABET), size=int(lengths.sum()))
        residues = alphabet[flat]

        sequences: list[str] = []
        offset = 0
        for length in lengths.tolist():
            sequences.append(residues[offset : offset + length].tobytes().decode("ascii"))
            offset += length
        return sequences

    def score(self, sequences: list[str]) -> list[float]:
        # Arbitrary, deterministic, and explicitly not a predictor of activity.
        # Replace this with your model's predicted potency / selectivity.
        return [float(len(sequences) - i) for i in range(len(sequences))]


class TrainedGenerator:
    """The trained AR-Transformer generator (``nn.PeptideLM``).

    Samples deterministically on GPU when one is available (the challenge validator has a
    GPU -- the official AMP-Diffusion baseline needs it), falling back to CPU. Each
    ``sample`` call reseeds a torch generator from the caller's numpy ``rng`` so repeated
    calls advance yet stay reproducible for a fixed run seed.

    ``score`` ranks by negative mean per-residue NLL under the model (higher = more
    AMP-like). This is an interim ranking signal, replaced by the APEX activity oracle in
    feat-013; it is honest about being a proxy, not a potency measurement.
    """

    name = "trained-ar-transformer"

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        min_length: int = MIN_LENGTH,
        max_length: int = MAX_LENGTH,
        temperature: float = 1.0,
        top_p: float = 1.0,
        device: str | None = None,
    ) -> None:
        import torch  # lazy: keeps the module importable without torch

        from . import nn as _nn
        from .paths import resolve_repo_path

        self._torch = torch
        self._nn = _nn
        self.min_length = min_length
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = _nn.load_generator(resolve_repo_path(checkpoint), self.device)

    def sample(self, n: int, rng: np.random.Generator) -> list[str]:
        # Derive a fresh torch seed from the caller's rng so successive top-up rounds draw
        # new sequences while the whole run stays reproducible for a fixed --seed.
        seed = int(rng.integers(0, 2**63 - 1))
        g = self._torch.Generator(device=self.device).manual_seed(seed)
        return self._nn.sample(
            self.model, n, device=self.device, temperature=self.temperature,
            top_p=self.top_p, min_residues=self.min_length, max_residues=self.max_length,
            generator=g, batch_size=4096,
        )

    def score(self, sequences: list[str]) -> list[float]:
        nll = self._nn.sequence_nll(self.model, sequences, device=self.device)
        return [-x for x in nll]  # higher is better


class ApexRanker:
    """Ranks candidates by APEX-predicted antimicrobial potency (higher = more potent).

    A *ranker*, not a generator: it only implements ``score`` (the seam ``select_top`` uses),
    delegating to :class:`~amp_challenge_2027.oracle.ApexScorer`. The score is
    :func:`~amp_challenge_2027.oracle.broad_potency_score` -- a smooth breadth-of-coverage
    signal aligned with the competition's Success Rate metric (fraction of strains inhibited at
    <= 16 uM), which the validation in ``docs/RESEARCH.md`` found ranks a broader, still-potent
    top-100 than best-strain MIC alone. Scoring is deterministic (APEX on CPU, ``.eval()``), so
    the two-run byte comparison still holds. Used for the top-100 ranking, which is what the wet
    lab tests; the library itself is unaffected.
    """

    name = "apex-mic"

    def __init__(self, apex_dir: str | Path = "oracle/apex", *, device: str = "cpu") -> None:
        from .oracle import ApexScorer  # lazy: keeps model.py importable without the oracle

        self._oracle = ApexScorer(apex_dir, device=device)

    def score(self, sequences: list[str]) -> list[float]:
        from .oracle import broad_potency_score

        mic = self._oracle.predict_mic(sequences)
        return broad_potency_score(mic).tolist()
