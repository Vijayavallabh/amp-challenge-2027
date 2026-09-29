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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .constraints import ALPHABET, MAX_LENGTH, MIN_LENGTH

if TYPE_CHECKING:
    from .hemolysis import HemolysisScorer

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

    def __init__(
        self,
        apex_dir: str | Path = "oracle/apex",
        *,
        device: str = "cpu",
        objective: str = "balanced",
        gp_weight: float = 1.0,
        mdr_weight: float = 1.0,
        broad_weight: float = 0.5,
        gn_weight: float = 0.75,
        hemolysis_scorer: "HemolysisScorer | None" = None,
        hemolysis_penalty: float = 0.0,
        refine_k: int = 20000,
        amphipathicity_bonus: float = 0.0,
        lys_hedge: float = 0.0,
    ) -> None:
        from .oracle import ApexScorer  # lazy: keeps model.py importable without the oracle

        self._oracle = ApexScorer(apex_dir, device=device)
        # The selectivity model (esp. the ESMC PLM) is expensive, so it is applied only to the
        # ``refine_k`` most-active candidates; the rest are assumed hemolytic so they rank below.
        # The shipped top-100 is drawn from the refined set, so this is exact for selection.
        self._refine_k = int(refine_k)
        # Ranking objective: "category" = success-rate aligned with the hard Gram+/MDR buckets
        # up-weighted (oracle.category_success_score), which balances the five scored categories;
        # "broad" = the older unbounded broad_potency margin. "category" is on a [0, ~2] scale, so
        # the hemolysis lambda is O(1), not O(2) as it was for the broad margin.
        self._objective = objective
        self._gpw = float(gp_weight)
        self._mdrw = float(mdr_weight)
        self._broadw = float(broad_weight)
        self._gnw = float(gn_weight)
        # Optional selectivity penalty: subtract lambda * P(hemolytic) from the activity score,
        # so that among comparably active peptides the less hemolytic ones rank higher. This
        # serves the Optimal Selectivity category and removes likely-toxic peptides, at a small
        # activity cost. The hemolysis signal is moderate (see docs/RESEARCH.md), so it is a
        # nudge, not an authority -- lambda is kept modest and the peptides stay APEX-active.
        self._hemo = hemolysis_scorer
        self._lam = float(hemolysis_penalty)
        # Optional closed-form amphipathic-moment bonus (feat-025): add this to the activity for
        # peptides whose Eisenberg hydrophobic moment is higher (a smooth 0..1 reward), preferring
        # mechanistically-amphipathic designs among the APEX-active ones. Deterministic (no fold), so
        # byte-reproducibility holds. This class defaults to 0.0 = off (a neutral library primitive);
        # the submission entry point generate.py sets the shipped default to 0.2 (feat-025), which lifts
        # the top-50 muH median 0.31->0.40 at <=0.014 category cost. 0.0 recovers the feat-021 selection.
        # A non-finite or negative value is treated as off rather than poisoning the scores with NaN
        # (``inf * 0.0``).
        amp = float(amphipathicity_bonus)
        self._amphi = amp if (amp == amp and amp not in (float("inf"), float("-inf")) and amp >= 0.0) else 0.0
        # Optional wet-lab Gram-negative hedge (feat-031): subtract lys_hedge * arg_excess (how
        # Arg-over-Lys-biased a peptide is above R/(R+K)=0.4) from the activity, de-biasing the top-50
        # away from the Arg-dominance that APEX over-rates on Gram-negative activity (validated against
        # the 46 wet-lab MICs; the bias is shared across all 8 submodels, so the held-out guard is blind
        # to it). Near-free on APEX (Gram- SR is flat across R/(R+K) 0.2-0.8). Deterministic. 0.0 = off.
        lh = float(lys_hedge)
        self._lyshedge = lh if (lh == lh and lh not in (float("inf"), float("-inf")) and lh >= 0.0) else 0.0
        base = {"category": "apex-success", "balanced": "apex-balanced-success"}.get(
            objective, "apex-broad-potency"
        )
        self.name = (
            f"{base} - {self._lam:g}*hemolysis"
            if self._hemo is not None and self._lam > 0 else base
        )
        if self._amphi > 0:  # provenance: an enabled bonus must be visible in the printed ranker name
            self.name += f" + {self._amphi:g}*amphipathicity"
        if self._lyshedge > 0:  # provenance: an enabled hedge must be visible in the printed ranker name
            self.name += f" - {self._lyshedge:g}*arg_excess"

    def score(self, sequences: list[str]) -> list[float]:
        import numpy as np

        from .oracle import (
            amphipathicity_bonus, arg_excess, balanced_success_score, broad_potency_score,
            category_success_score,
        )

        mic = self._oracle.predict_mic(sequences)
        if self._objective == "broad":
            activity = broad_potency_score(mic)
        elif self._objective == "category":
            activity = category_success_score(mic, self._gpw, self._mdrw)
        else:  # "balanced" -- hard Gram+/MDR/Gram- Success Rate + broad soft tie-break (shipped default)
            activity = balanced_success_score(mic, self._gpw, self._mdrw, self._broadw, self._gnw)
        activity = np.asarray(activity, dtype=float)
        if self._amphi > 0:  # closed-form amphipathic-moment nudge (deterministic; off by default)
            activity = activity + self._amphi * amphipathicity_bonus(sequences)
        if self._lyshedge > 0:  # wet-lab Gram- de-bias: penalise Arg-over-Lys excess (deterministic)
            activity = activity - self._lyshedge * arg_excess(sequences)
        if self._hemo is None or self._lam <= 0:
            return activity.tolist()
        # Two-stage: score selectivity only on the ``refine_k`` most-active candidates (the PLM
        # model is expensive), leaving the rest assumed hemolytic (penalty 1.0) so they rank below.
        # A stable argsort of the fixed activity array keeps this byte-deterministic.
        n = len(sequences)
        k = min(n, self._refine_k) if self._refine_k > 0 else n
        order = np.argsort(-activity, kind="stable")[:k]
        phemo = np.ones(n, dtype=float)
        phemo[order] = np.asarray(
            self._hemo.predict_proba([sequences[int(i)] for i in order]), dtype=float
        )
        return (activity - self._lam * phemo).tolist()  # activity is already a float ndarray

    def maximin_data(self, sequences: list[str]):
        """Return ``(category_rates (n, 4), phemo (n,))`` for the maximin top-list selector.

        ``category_rates`` columns are ``[Broad, Gram-, Gram+, MDR]`` hard Success Rates. ``phemo`` is
        the ESMC P(hemolytic) computed exactly as in :meth:`score` -- on the ``refine_k`` most-active
        candidates (ranked by the balanced activity), the rest left at 1.0 (assumed hemolytic, so the
        selector gates them out). With no selectivity model, ``phemo`` is 0 (no gate).
        """
        import numpy as np

        from .oracle import balanced_success_score, category_rates

        mic = self._oracle.predict_mic(sequences)
        rates = category_rates(mic)
        n = len(sequences)
        if self._hemo is None:
            return rates, np.zeros(n, dtype=float)
        activity = balanced_success_score(mic, self._gpw, self._mdrw, self._broadw, self._gnw)
        k = min(n, self._refine_k) if self._refine_k > 0 else n
        order = np.argsort(-activity, kind="stable")[:k]
        phemo = np.ones(n, dtype=float)
        phemo[order] = np.asarray(
            self._hemo.predict_proba([sequences[int(i)] for i in order]), dtype=float
        )
        return rates, phemo
