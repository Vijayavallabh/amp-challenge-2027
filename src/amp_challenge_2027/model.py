"""The seam where a real generative model plugs in.

``RandomBaseline`` exists so that ``uv run generate`` produces a *contract-valid*
submission from the first commit: 50,000 unique sequences, a ranked top 100, byte-identical
across runs. It is a placeholder, not a method. Uniform random peptides have no expected
antimicrobial activity -- they will score near zero on every competition category.

To submit real work, implement :class:`PeptideGenerator` and return it from
``build_model`` in ``generate.py``. Nothing else in the package needs to change: the
compliance filtering, novelty screening, ranking and file writing all stay the same.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .constraints import ALPHABET, MAX_LENGTH, MIN_LENGTH


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
