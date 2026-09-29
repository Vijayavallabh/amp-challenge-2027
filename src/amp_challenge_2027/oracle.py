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
        workers: int | None = None,
    ) -> None:
        # Absolute, so it is unambiguous both as ``--project`` and as the subprocess ``cwd``.
        self.apex_dir = resolve_repo_path(apex_dir).resolve()
        script = self.apex_dir / "APEX_predict.py"
        if not script.is_file():
            raise ApexUnavailable(f"APEX inference script not found at {script}")
        # device: "cpu" is deterministic (required for the shipped ranking); "cuda" is only
        # for fast *offline* bulk scoring, where byte-reproducibility is not required.
        self._gpu_flag = "1" if device == "cuda" else "0"
        self._device = device
        self.batch_size = batch_size
        # CPU scoring is embarrassingly parallel across sequences (each peptide's MIC is
        # independent of its batch-mates -- no cross-sequence ops), so we shard it over
        # several subprocesses. Determinism is preserved: results are keyed by sequence and
        # reassembled in input order, so output is identical regardless of worker count.
        # ``workers=None`` auto-picks from CPU count and *available memory* (each worker holds
        # the full 8-model ensemble, ~8 GB), so it stays safe on a small validator machine.
        self.workers = workers

    def predict_mic(self, sequences: list[str]) -> np.ndarray:
        """Return an ``(n, 11)`` array of predicted MIC (uM); lower is more potent.

        Rows align with ``sequences`` (duplicates included). Raises :class:`ApexUnavailable`
        if the subprocess fails for any reason.
        """
        if not sequences:
            return np.empty((0, len(PATHOGENS)), dtype=float)

        # Score each distinct sequence once, then broadcast back to the input order.
        uniq = list(dict.fromkeys(sequences))
        n_workers = self._resolve_workers(len(uniq))
        if n_workers <= 1:
            mic_by_seq: dict[str, np.ndarray] = {}
            for start in range(0, len(uniq), self.batch_size):
                mic_by_seq.update(self._run_apex(uniq[start : start + self.batch_size]))
        else:
            import math

            shard_size = max(2000, min(self.batch_size, math.ceil(len(uniq) / (n_workers * 3))))
            shards = [uniq[i : i + shard_size] for i in range(0, len(uniq), shard_size)]
            try:
                mic_by_seq = self._run_pool(shards, n_workers)
            except ApexUnavailable:
                # Parallel scoring failed mid-way -> fall back to the sequential path so a
                # valid ranking is still produced (mirrors the wider graceful-degradation policy).
                mic_by_seq = {}
                for start in range(0, len(uniq), self.batch_size):
                    mic_by_seq.update(self._run_apex(uniq[start : start + self.batch_size]))

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

    def _resolve_workers(self, n_uniq: int) -> int:
        """Pick a safe number of parallel CPU workers (1 = sequential path)."""
        if self._device == "cuda" or n_uniq < 3000:
            return 1  # GPU is already fast; tiny inputs are not worth the fan-out
        if self.workers is not None:
            return max(1, int(self.workers))
        return self._auto_workers()

    @staticmethod
    def _auto_workers() -> int:
        env = os.environ.get("APEX_WORKERS")
        if env and env.isdigit():
            return max(1, int(env))
        cpu = os.cpu_count() or 1
        # each worker's resident set is ~8 GB (torch + the 8 APEX models); never exceed memory.
        by_mem = max(1, int(ApexScorer._available_gb() // 9))
        return max(1, min(cpu // 2, by_mem, 24))

    @staticmethod
    def _available_gb() -> float:
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024 * 1024)
        except OSError:
            pass
        return 8.0

    def _run_pool(self, shards: list[list[str]], n_workers: int) -> dict[str, np.ndarray]:
        """Run APEX on many shards with bounded concurrency; merge results by sequence.

        Determinism holds because each sequence's MIC is independent of the others (no
        cross-sequence ops in APEX), so the merged mapping is identical no matter how the
        shards are split or how the subprocesses interleave.
        """
        import time

        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        # Pin each worker to ONE thread. Multi-threaded oneDNN/MKL GRU reductions are not
        # bit-reproducible (thread-timing-dependent accumulation order), which would break the
        # validator's two-run byte comparison. Single-threaded scoring is fully deterministic
        # AND independent of how many workers/shards we use, so the merged result is identical
        # on any machine; parallelism comes from running many single-threaded workers at once.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = "1"

        results: dict[str, np.ndarray] = {}
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            queue: list[tuple[Path, Path]] = []
            for i, shard in enumerate(shards):
                in_fa = tmp / f"in_{i}.fasta"
                with in_fa.open("w") as fh:
                    for j, seq in enumerate(shard):
                        fh.write(f">s{j}\n{seq}\n")
                queue.append((in_fa, tmp / f"out_{i}.csv"))
            queue.reverse()  # pop() then yields shards in original order (cosmetic)

            running: dict[subprocess.Popen, Path] = {}

            def launch(job: tuple[Path, Path]) -> None:
                in_fa, out_csv = job
                cmd = [
                    "uv", "run", "--project", str(self.apex_dir),
                    "python", "APEX_predict.py",
                    "-i", str(in_fa), "-o", str(out_csv), "-g", self._gpu_flag,
                ]
                proc = subprocess.Popen(
                    cmd, cwd=self.apex_dir, env=env,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                )
                running[proc] = out_csv

            while queue and len(running) < n_workers:
                launch(queue.pop())
            while running:
                for proc in list(running):
                    if proc.poll() is None:
                        continue
                    out_csv = running.pop(proc)
                    if proc.returncode != 0:
                        detail = (proc.stderr.read() if proc.stderr else "")[-500:]
                        for p in running:
                            p.kill()
                        raise ApexUnavailable(f"APEX subprocess failed: {detail}")
                    results.update(self._read_predictions(out_csv))
                    if queue:
                        launch(queue.pop())
                if running:
                    time.sleep(0.05)
        return results

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
    validation finding that it is the most discriminating aggregate. Best for "active on at
    least one strain" (the Selectivity category's eligibility gate); for broad-spectrum
    ranking prefer :func:`broad_potency_score`.
    """
    return -np.log10(np.clip(min_mic(mic), 1e-6, None))


#: The competition's Potency Threshold: a peptide "counts as active" on a strain at MIC <= 16 uM,
#: and every activity category is scored by *Success Rate* = the fraction of strains it clears.
POTENCY_THRESHOLD_UM = 16.0


def broad_potency_score(mic: np.ndarray, threshold: float = POTENCY_THRESHOLD_UM) -> np.ndarray:
    """Higher-is-better broad-spectrum score aligned with the competition's Success Rate.

    Each strain contributes ``max(0, log10(threshold / MIC))`` -- positive by how many log-units
    the predicted MIC sits below the 16 uM potency threshold, zero above it. Summing over the
    panel rewards **breadth** (inhibiting many strains) and margin, which is exactly what the
    Broad-Spectrum / Gram / MDR categories measure (mean per-peptide Success Rate).

    A smooth margin is used rather than a hard count of strains below 16 uM because APEX's
    absolute MIC scale is compressed and only moderately calibrated (see ``docs/RESEARCH.md``);
    a hard threshold would be brittle to that miscalibration, while the margin degrades
    gracefully. On a 20k pool this ranks a top-100 averaging ~5.3 of 11 strains covered, versus
    ~4.1 for best-strain (min-MIC) ranking.
    """
    per_strain = np.log10(threshold / np.clip(mic, 1e-6, None))
    return np.clip(per_strain, 0.0, None).sum(axis=1)


# APEX pathogen buckets, mapping the 11-column panel onto the competition's category structure
# (docs/COMPETITION.md): 7 Gram-negative, 4 Gram-positive, of which 3 are the hard MDR isolates.
GRAM_NEG = (0, 1, 2, 3, 4, 5, 6)   # A. baumannii, E. coli x3, K. pneumoniae, P. aeruginosa x2
GRAM_POS = (7, 8, 9, 10)           # S. aureus x2, VRE faecalis, VRE faecium
MDR = (8, 9, 10)                   # MRSA, VRE faecalis, VRE faecium


def _soft_success(mic: np.ndarray, threshold: float = POTENCY_THRESHOLD_UM, tau: float = 0.5) -> np.ndarray:
    """Smooth, *saturating* P(MIC <= threshold): sigmoid of the sub-threshold log-margin.

    Unlike :func:`broad_potency_score`'s unbounded margin, this saturates once a strain is
    cleared, so ranking rewards *breadth* of coverage (the Success-Rate metric) rather than
    ever-deeper potency on a few easy strains -- and it does not chase APEX's poorly-calibrated
    sub-uM tail. See ``docs/RESEARCH.md``.
    """
    margin = (np.log10(threshold) - np.log10(np.clip(mic, 1e-6, None))) / tau
    return 1.0 / (1.0 + np.exp(-margin))


def category_success_score(
    mic: np.ndarray, gp_weight: float = 0.5, mdr_weight: float = 0.5,
    threshold: float = POTENCY_THRESHOLD_UM,
) -> np.ndarray:
    """Success-Rate-aligned ranking score, with the hard Gram+/MDR buckets up-weighted.

    ``mean soft-success over all strains + gp_weight * SR(Gram+) + mdr_weight * SR(MDR)``.
    The competition ranks five *separate* categories (Broad, Gram+, Gram-, MDR, Selectivity),
    and 15 of 20 real strains are Gram-negative -- so a bare mean over-serves the (easy, for
    cationic peptides) Gram-negative side. Up-weighting the currently-weak Gram+/MDR buckets
    pulls the ranked top-100 toward peptides that cover *all* categories, which raised measured
    Gram+ coverage of the top-50 with little Gram-negative cost (see ``docs/RESEARCH.md``).
    """
    s = _soft_success(mic, threshold)
    return (
        s.mean(axis=1)
        + gp_weight * s[:, list(GRAM_POS)].mean(axis=1)
        + mdr_weight * s[:, list(MDR)].mean(axis=1)
    )


def balanced_success_score(
    mic: np.ndarray, gp_weight: float = 1.0, mdr_weight: float = 1.0,
    broad_weight: float = 0.5, gn_weight: float = 0.75,
    threshold: float = POTENCY_THRESHOLD_UM,
) -> np.ndarray:
    """Balanced 5-category ranking score: **hard** Gram+/MDR/Gram- Success Rate + a broad soft tie-break.

    ``gp_weight * SR_hard(Gram+) + mdr_weight * SR_hard(MDR) + gn_weight * SR_hard(Gram-)
    + broad_weight * mean soft-success``, where ``SR_hard(bucket)`` is the fraction of that bucket's
    strains cleared at <= ``threshold`` (the exact metric the competition scores).
    :func:`category_success_score` averages *soft* success, which is dominated by the many (easy, for
    cationic peptides) Gram-negative strains and *dilutes* the rare peptides that truly clear the hard
    Gram+/MDR strains -- so it plateaus at ~0.37 Gram+ Success Rate no matter how the soft weights are
    set. Ranking by the hard Gram+/MDR rate instead surfaces those hard-hitters directly. The
    ``gn_weight`` term then adds back an explicit hard Gram-negative Success Rate: without it the
    Gram+/MDR-specialist selection leaves Gram- slightly low (0.54); a modest ``gn_weight`` (0.75)
    lifts the top-50 Gram- 0.54 -> 0.58 and Broad 0.62 -> 0.64 at *no* Gram+/MDR cost (the pool already
    holds those Gram-strong peptides; it just needed surfacing). The soft-mean term is a smooth
    tie-break (hard Success Rate is coarse). See ``docs/RESEARCH.md``.
    """
    s = _soft_success(mic, threshold)
    hit = (mic <= threshold).astype(float)
    return (
        gp_weight * hit[:, list(GRAM_POS)].mean(axis=1)
        + mdr_weight * hit[:, list(MDR)].mean(axis=1)
        + gn_weight * hit[:, list(GRAM_NEG)].mean(axis=1)
        + broad_weight * s.mean(axis=1)
    )


def category_rates(mic: np.ndarray, threshold: float = POTENCY_THRESHOLD_UM) -> np.ndarray:
    """Per-peptide hard Success Rate in each scored *activity* category, as an ``(n, 4)`` array with
    columns ``[Broad-spectrum, Gram-negative, Gram-positive, MDR]`` -- each the fraction of that
    bucket's strains cleared at <= ``threshold`` (Broad = all 11 strains).

    This is what the maximin top-list selector (``generate.select_maximin``) needs: the competition
    ranks the five categories *separately*, so the standing turns on the **weakest** one. A fixed
    weighted sum (:func:`balanced_success_score`) cannot maximise a minimum; greedy maximin over these
    per-category rates raises the floor directly. Selectivity (the 5th category) is handled separately
    as the P(hemolytic) gate, not an activity rate.
    """
    hit = (mic <= threshold).astype(float)
    return np.stack([
        hit.mean(axis=1),
        hit[:, list(GRAM_NEG)].mean(axis=1),
        hit[:, list(GRAM_POS)].mean(axis=1),
        hit[:, list(MDR)].mean(axis=1),
    ], axis=1)


# The amphipathic-moment ranking bonus reuses the single Eisenberg scale from ``physchem`` (one source
# of truth; a parity test pins ``hydrophobic_moment`` here to ``physchem.hydrophobic_moment``).
from .physchem import _EISENBERG  # noqa: E402

#: The amphipathicity bonus ramps linearly from ``AMPHI_FLOOR`` (below which a peptide is too weakly
#: amphipathic to reward -- this deprioritises the confidently-folded-but-non-amphipathic designs the
#: structure check flagged, e.g. muH ~0.10) up to ``AMPHI_SATURATION`` (at/above which it is full). It
#: rises MONOTONICALLY and rewards even the strongest amphipaths (melittin 0.35, magainin 0.45, aurein
#: 0.67, LL-37 0.88 all saturate to 1.0) -- a fixed [0.30, 0.65] *band* would have excluded aurein/LL-37,
#: the very class it means to promote. The hemolytic-amphipathic extreme is removed by the ESMC
#: hemolysis penalty applied later in the same score, not by a muH cap.
AMPHI_FLOOR, AMPHI_SATURATION = 0.25, 0.50


def hydrophobic_moment(sequences: list[str], angle_deg: float = 100.0) -> np.ndarray:
    """Vectorised Eisenberg mean hydrophobic moment (muH) per sequence -- a closed-form, DETERMINISTIC
    measure of alpha-helical amphipathicity, the membrane-disruption determinant of AMP activity.
    Needs no structure prediction, so it can enter the byte-reproducible ranking (unlike the ESMFold2
    offline check, whose muH it matches: melittin ~0.35, magainin ~0.45). Numerically identical to the
    scalar :func:`physchem.hydrophobic_moment` (pinned by a parity test); this batched form exists only
    because the bonus runs over the whole ~150k pool. Whole-sequence normalisation by length is the
    repo-wide convention (physchem + structure_validate share it), not a windowed maximum."""
    if isinstance(sequences, str):  # a bare string would silently iterate character-by-character
        raise TypeError("hydrophobic_moment expects a list of sequences, not a single string")
    d = np.deg2rad(angle_deg)
    lengths = np.fromiter((len(s) for s in sequences), dtype=int, count=len(sequences))
    lmax = int(lengths.max()) if len(sequences) else 0
    basis_c = np.cos(np.arange(lmax) * d)  # length-only helical-wheel basis, computed once (not per-seq)
    basis_s = np.sin(np.arange(lmax) * d)
    out = np.empty(len(sequences), dtype=float)
    for k, s in enumerate(sequences):
        n = len(s)
        if n == 0:
            out[k] = 0.0
            continue
        h = np.fromiter((_EISENBERG.get(a, 0.0) for a in s), dtype=float, count=n)
        out[k] = np.hypot((h * basis_c[:n]).sum(), (h * basis_s[:n]).sum()) / n
    return out


def amphipathicity_bonus(
    sequences: list[str], floor: float = AMPHI_FLOOR, saturation: float = AMPHI_SATURATION
) -> np.ndarray:
    """Smooth ``[0, 1]`` amphipathicity reward: ``clip((muH - floor) / (saturation - floor), 0, 1)`` -- a
    MONOTONIC gradient toward stronger amphipathic-helix character, so among comparably APEX-active
    peptides the ranking prefers the more membrane-active ones (an orthogonal, structure-free hedge
    against APEX's sequence-only transfer error, which the ESMFold2 check found feat-021's top-50 at the
    weak edge of). Unlike a fixed band it gives a true gradient (a muH 0.45 peptide is preferred over a
    0.31 one), zeroes only the genuinely non-amphipathic tail (below ``floor``), and rewards the
    strongest amphipaths (saturating, not excluded); the hemolytic-amphipathic extreme is handled by the
    ESMC penalty, not here. Deterministic, so byte-reproducibility holds."""
    mu = hydrophobic_moment(sequences)
    return np.clip((mu - floor) / max(saturation - floor, 1e-9), 0.0, 1.0)


#: Arg-excess floor for the wet-lab Gram-negative hedge (feat-031). Below this Arg-fraction the hedge
#: is inert; above it, ranking is penalised in proportion to the Arg-over-Lys excess. 0.4 keeps the
#: hedged top-50 in the R/(R+K) ~0.4-0.5 band, which the wet-lab data and the ESMC selectivity model
#: both prefer while APEX's Gram-negative Success Rate is flat (so the hedge costs ~nothing on APEX).
ARG_EXCESS_FLOOR = 0.4


def arg_excess(sequences: list[str], floor: float = ARG_EXCESS_FLOOR) -> np.ndarray:
    """``max(0, R/(R+K) - floor)`` per sequence -- how Arg-over-Lys-biased a peptide is above ``floor``.

    APEX systematically OVER-rates Arg-rich peptides on Gram-negative activity (validated against the 46
    wet-lab MICs: residual APEX_GN - REAL_GN vs R/(R+K) Spearman +0.41, p=0.003), while the real Gram-
    Success Rate FAVOURS Lys-richness (rho -0.40, p=0.004) -- and 75% of the competition panel is
    Gram-negative. The anti-Goodhart held-out-submodel guard is blind to this because the bias is shared
    across all 8 APEX submodels (same training data). Subtracting ``lys_hedge * arg_excess`` from the
    ranking score de-biases the top-50 toward the wet-lab-favourable composition. It is near-free on
    APEX: Gram-negative Success Rate is flat across R/(R+K) 0.2-0.8, and Lys-rich strong actives are
    abundant, so the hedge shifts composition at ~zero APEX activity cost (see docs/RESEARCH.md feat-031).
    Peptides with no Arg and no Lys score 0 (no cationic imbalance to correct). Deterministic --
    composition is a pure function of sequence, so byte-reproducibility holds."""
    out = np.zeros(len(sequences), dtype=float)
    for i, s in enumerate(sequences):
        r = s.count("R")
        rk = r + s.count("K")
        if rk > 0:
            out[i] = max(0.0, r / rk - floor)
    return out
