"""Activity oracle: predicted MIC across the 11-pathogen panel via APEX-pathogen.

The ranked top-100 is what the competition wet-lab tests, so ranking candidates by a real
activity predictor -- not by generator likelihood -- is the highest-leverage lever we have.
This module wraps **APEX-pathogen** (Wan / de la Fuente lab, *Nat. Microbiol.* 2025; the exact
release used to score the AMP-Diffusion library in Torres et al., *Cell Biomaterials* 2025),
vendored under ``oracle/apex`` as an **isolated uv project** so its pinned, conflicting deps
(``torch==2.5.1``, ``numpy<2``, Python 3.10) never co-resolve with this package's environment.
It is invoked as a subprocess, exactly as the organizers' own baseline does it.

Why a subprocess and not an in-process import: APEX ships pickled *whole-module* checkpoints
that only load under ``torch<2.6`` (``weights_only`` default), and its ``numpy<2`` pin is
incompatible with ours. Isolation is the only clean option.

Determinism: APEX runs in ``.eval()`` mode (dropout off) on **CPU**, so its output is
byte-stable across runs -- which the submission's two-run reproducibility check requires.

Validation of the oracle itself (against the 46 wet-lab-measured peptides in
``data/experimental/mic.csv``) lives in ``docs/RESEARCH.md``: APEX is a *moderate*,
wet-lab-aligned signal (AUROC ~0.76 known-AMP vs random; ~0.62 per-(peptide,strain)
inhibition on novel peptides), so it is used to reach the active band and rank into it, not
treated as ground truth -- selection layers novelty, diversity and selectivity on top.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .paths import resolve_repo_path

#: The 11 clinical pathogens APEX-pathogen scores, in the column order it emits.
PATHOGENS: tuple[str, ...] = (
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
)

DEFAULT_APEX_DIR = "oracle/apex"


class ApexUnavailable(RuntimeError):
    """Raised when the APEX subprocess cannot be run (missing env, sync failure, ...).

    Callers catch this to fall back to a lighter ranking so a valid submission is always
    produced, mirroring the ``TrainedGenerator`` -> ``RandomBaseline`` fallback.
    """


class ApexScorer:
    """Score peptides by APEX-predicted MIC (uM) against the 11-pathogen panel.

    One instance manages the isolated APEX environment and batches all sequences through a
    single subprocess call. Sequences are de-duplicated before scoring and results are mapped
    back to the caller's order, so scoring a list with repeats is safe.
    """

    def __init__(
        self,
        apex_dir: str | Path = DEFAULT_APEX_DIR,
        *,
        device: str = "cpu",
        batch_size: int = 20000,
    ) -> None:
        # Absolute, so it is unambiguous both as ``--project`` and as the subprocess ``cwd``.
        self.apex_dir = resolve_repo_path(apex_dir).resolve()
        script = self.apex_dir / "APEX_predict.py"
        if not script.is_file():
            raise ApexUnavailable(f"APEX inference script not found at {script}")
        # device: "cpu" is deterministic (required for the shipped ranking); "cuda" is only
        # for fast *offline* bulk scoring, where byte-reproducibility is not required.
        self._gpu_flag = "1" if device == "cuda" else "0"
        self.batch_size = batch_size

    def predict_mic(self, sequences: list[str]) -> np.ndarray:
        """Return an ``(n, 11)`` array of predicted MIC (uM); lower is more potent.

        Rows align with ``sequences`` (duplicates included). Raises :class:`ApexUnavailable`
        if the subprocess fails for any reason.
        """
        if not sequences:
            return np.empty((0, len(PATHOGENS)), dtype=float)

        # Score each distinct sequence once, then broadcast back to the input order.
        uniq = list(dict.fromkeys(sequences))
        mic_by_seq: dict[str, np.ndarray] = {}
        for start in range(0, len(uniq), self.batch_size):
            chunk = uniq[start : start + self.batch_size]
            mic_by_seq.update(self._run_apex(chunk))

        missing = [s for s in uniq if s not in mic_by_seq]
        if missing:
            raise ApexUnavailable(
                f"APEX returned no prediction for {len(missing)} sequence(s), "
                f"e.g. {missing[0]!r}"
            )
        return np.array([mic_by_seq[s] for s in sequences], dtype=float)

    def _run_apex(self, sequences: list[str]) -> dict[str, np.ndarray]:
        with tempfile.TemporaryDirectory() as tmp:
            in_fa = Path(tmp) / "in.fasta"
            out_csv = Path(tmp) / "out.csv"
            with in_fa.open("w") as fh:
                for i, seq in enumerate(sequences):
                    fh.write(f">s{i}\n{seq}\n")

            cmd = [
                "uv", "run", "--project", str(self.apex_dir),
                "python", "APEX_predict.py",
                "-i", str(in_fa), "-o", str(out_csv), "-g", self._gpu_flag,
            ]
            # Run from the APEX dir so its ``from APEX_models import ...`` resolves and the
            # weights load relative to the script. Keep VIRTUAL_ENV out of uv's way.
            env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
            try:
                subprocess.run(
                    cmd, cwd=self.apex_dir, env=env, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                raise ApexUnavailable(f"APEX subprocess failed: {detail[-500:]}") from exc

            return self._read_predictions(out_csv)

    @staticmethod
    def _read_predictions(out_csv: Path) -> dict[str, np.ndarray]:
        rows = list(csv.reader(out_csv.open()))
        if not rows:
            raise ApexUnavailable("APEX produced an empty predictions file")
        out: dict[str, np.ndarray] = {}
        for row in rows[1:]:  # row[0] is the sequence, then 11 MIC columns
            out[row[0]] = np.array([float(x) for x in row[1:]], dtype=float)
        return out


# --- aggregations from the per-pathogen MIC matrix to a single ranking signal -------------
# Evidence (docs/RESEARCH.md): the best-pathogen (min) MIC discriminates active from inactive
# far better than the mean, which over-penalises real narrow-spectrum actives. We expose both
# plus a breadth count so selection (feat-014) can trade potency against spectrum.

def min_mic(mic: np.ndarray) -> np.ndarray:
    """Best-pathogen MIC per peptide (the most sensitive potency signal)."""
    return mic.min(axis=1)


def mean_mic(mic: np.ndarray) -> np.ndarray:
    """Mean MIC across the panel (a broad-spectrum, conservative signal)."""
    return mic.mean(axis=1)


def breadth(mic: np.ndarray, threshold: float = 32.0) -> np.ndarray:
    """Number of pathogens each peptide is predicted to inhibit at or below ``threshold``."""
    return (mic <= threshold).sum(axis=1)


def potency_score(mic: np.ndarray) -> np.ndarray:
    """A higher-is-better activity score for ranking: ``-log10(min_MIC)``.

    Log scale because MIC spans orders of magnitude; using the best-pathogen MIC follows the
    validation finding that it is the most discriminating aggregate.
    """
    return -np.log10(np.clip(min_mic(mic), 1e-6, None))
