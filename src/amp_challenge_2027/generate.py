"""Entry point for the AMP Challenge 2027 submission: ``uv run generate``.

Writes two files that the organizers' validator reads:

    generate/library.fasta   50,000 unique candidate peptides
    generate/top.fasta       the ranked top 100, a strict subset of the library

Every argument has a default, so a bare ``uv run generate`` is a complete run, and the
default seed is fixed so two runs produce byte-identical files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import constraints as C
from .fasta import read_sequences, write_fasta
from .model import (
    DEFAULT_CHECKPOINT,
    ApexRanker,
    PeptideGenerator,
    RandomBaseline,
    TrainedGenerator,
)
from .paths import resolve_repo_path

DEFAULT_SEED = 42
DEFAULT_OUT_DIR = "generate"
DEFAULT_REFERENCE = "data/antibacterial.fasta"

# How many candidates to draw per round while topping up the library, and how many
# rounds to allow before giving up. Generous for any sane generator.
_BATCH_OVERDRAW = 1.2
_MAX_ROUNDS = 100


def build_model(args: argparse.Namespace) -> PeptideGenerator:
    """Return the generator used for a submission.

    Prefers the trained AR-Transformer when its checkpoint is present and ``torch`` imports;
    otherwise falls back to :class:`RandomBaseline` so a submission is always produced. The
    choice is printed so a run's provenance is never ambiguous.
    """
    try:
        checkpoint = resolve_repo_path(args.checkpoint)
    except FileNotFoundError:
        checkpoint = None

    if checkpoint is not None and not args.baseline:
        try:
            model = TrainedGenerator(
                checkpoint, min_length=args.min_length, max_length=args.max_length,
                temperature=args.temperature, top_p=args.top_p,
            )
            print(f"Model: {model.name} from {checkpoint} on {model.device}")
            return model
        except Exception as exc:  # torch missing, bad checkpoint, no compute -> fall back
            print(f"WARNING: could not load trained generator ({exc}); using baseline",
                  file=sys.stderr)

    model = RandomBaseline(min_length=args.min_length, max_length=args.max_length)
    print(f"Model: {model.name} (fallback placeholder -- no trained checkpoint in use)")
    return model


def build_ranker(model: PeptideGenerator, args: argparse.Namespace):
    """Return the object whose ``.score`` ranks the top-100 (the seam ``select_top`` uses).

    ``--rank apex`` ranks by APEX-predicted potency -- the wet-lab-aligned signal, and a large
    improvement over model likelihood (see ``docs/RESEARCH.md``). If the APEX oracle cannot be
    started (its isolated env fails to sync, no compute, ...), we fall back to the model's own
    likelihood ranking so a valid submission is still produced. ``--rank likelihood`` uses the
    generator's likelihood directly.
    """
    if args.rank == "apex":
        try:
            hemo = None
            if args.hemolysis_penalty > 0:
                # Prefer the ESMC-600M selectivity model (latest PLM, held-out AUROC ~0.89); fall
                # back to the lightweight physicochemical model, then to activity-only ranking, so
                # a valid submission is always produced.
                try:
                    from .selectivity_esm import EsmcSelectivityScorer
                    hemo = EsmcSelectivityScorer(args.selectivity_checkpoint)
                    print("Selectivity: ESMC-600M (ESM++) embeddings")
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: ESMC selectivity unavailable ({exc}); trying physchem model",
                          file=sys.stderr)
                    try:
                        from .hemolysis import HemolysisScorer
                        hemo = HemolysisScorer(args.hemolysis_checkpoint)
                        print("Selectivity: physicochemical hemolysis model (fallback)")
                    except Exception as exc2:  # noqa: BLE001 -- degrade to activity-only ranking
                        print(f"WARNING: hemolysis model unavailable ({exc2}); ranking on activity "
                              f"only", file=sys.stderr)
            ranker = ApexRanker(
                args.apex_dir,
                objective=args.rank_objective,
                gp_weight=args.gp_weight,
                mdr_weight=args.mdr_weight,
                broad_weight=args.broad_weight,
                gn_weight=args.gn_weight,
                hemolysis_scorer=hemo,
                hemolysis_penalty=args.hemolysis_penalty if hemo is not None else 0.0,
                refine_k=args.refine_k,
                amphipathicity_bonus=args.amphipathicity_bonus,
                lys_hedge=args.lys_hedge,
                composition_weight=args.composition_weight,
                aromatic_weight=args.aromatic_weight,
            )
            print(f"Ranking: {ranker.name}")
            return ranker
        except Exception as exc:  # noqa: BLE001 -- any failure must degrade, not crash
            print(f"WARNING: APEX ranker unavailable ({exc}); ranking by likelihood",
                  file=sys.stderr)
    print(f"Ranking: {model.name} likelihood")
    return model


def set_determinism(seed: int) -> None:
    """Make sampling reproducible across two runs on the same machine (validator check).

    Seeds torch (CPU and CUDA) and requests deterministic kernels. ``warn_only`` avoids a
    hard error on any op lacking a deterministic implementation while still pinning the ones
    that matter; the sampling RNG itself is a seeded generator, so output is stable.
    """
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # torch not installed -> baseline path, nothing to seed


def build_library(
    model: PeptideGenerator,
    n_sequences: int,
    rng: np.random.Generator,
    excluded: set[str],
) -> list[str]:
    """Draw unique, constraint-valid sequences until ``n_sequences`` are collected.

    Order is insertion order, never set order, so the result is reproducible. Sequences
    in ``excluded`` (the known antibacterial reference) are dropped: the library is
    required to contain none of them.
    """
    collected: dict[str, None] = {}
    batch_size = max(1, int(n_sequences * _BATCH_OVERDRAW))

    for round_index in range(_MAX_ROUNDS):
        remaining = n_sequences - len(collected)
        if remaining <= 0:
            break
        draw = batch_size if round_index == 0 else max(1, int(remaining * _BATCH_OVERDRAW))
        for seq in model.sample(draw, rng):
            if len(collected) >= n_sequences:
                break
            if seq in collected or seq in excluded:
                continue
            if not C.is_valid_sequence(seq):
                continue
            collected[seq] = None

    if len(collected) < n_sequences:
        raise RuntimeError(
            f"only produced {len(collected)} of {n_sequences} unique valid sequences after "
            f"{_MAX_ROUNDS} rounds -- the generator's output is not diverse enough"
        )

    return list(collected)


def select_top(
    library: list[str],
    ranker,
    top_k: int,
    reference: list[str],
    *,
    diversity_max_identity: float | None = None,
) -> list[str]:
    """Rank the library and return the best ``top_k`` that clear the novelty screen.

    ``ranker`` is anything with a ``.score(sequences) -> list[float]`` (higher = better):
    the generator itself (likelihood) or :class:`~amp_challenge_2027.model.ApexRanker`.

    The top list is held to a stricter standard than the library: no sequence may exceed
    80% Levenshtein identity with any known antibacterial peptide. Candidates that fail
    are skipped and the next-best takes the slot, which is what the rules prescribe.

    ``diversity_max_identity`` (when set) additionally enforces *within-list* diversity: a
    candidate is skipped if its Levenshtein identity to an already-selected peptide exceeds
    the threshold. This matters because a strong activity ranker (APEX) concentrates the top
    of the list into one sequence motif, and only 25 of the top 50 are assayed at random --
    a redundant list wastes draws on near-duplicates. Selection stays greedy and
    deterministic: the more-active peptide of any near-duplicate pair is kept.
    """
    scores = ranker.score(library)
    if len(scores) != len(library):
        raise ValueError(
            f"ranker.score returned {len(scores)} scores for {len(library)} sequences"
        )

    # Sort by descending score, then by sequence to break ties deterministically.
    ranked = sorted(zip(scores, library), key=lambda pair: (-pair[0], pair[1]))

    selected: list[str] = []
    rejected = 0
    rejected_div = 0
    for _, seq in ranked:
        if len(selected) == top_k:
            break
        if not C.is_novel_enough(seq, reference):
            rejected += 1
            continue
        if diversity_max_identity is not None and selected and (
            C.max_identity(seq, selected, cutoff=diversity_max_identity)
            >= diversity_max_identity
        ):
            rejected_div += 1
            continue
        selected.append(seq)

    if len(selected) < top_k:
        extra = (
            f" and {rejected_div} for exceeding the {diversity_max_identity:.0%} diversity cap"
            if diversity_max_identity is not None else ""
        )
        raise RuntimeError(
            f"only {len(selected)} of {top_k} ranked candidates cleared the "
            f"{C.MAX_TOP_IDENTITY:.0%} novelty screen ({rejected} rejected{extra}) -- the "
            f"library is too close to known antibacterial peptides"
        )

    if rejected:
        print(f"  novelty screen rejected {rejected} higher-ranked candidate(s)")
    if rejected_div:
        print(f"  diversity screen rejected {rejected_div} near-duplicate candidate(s) "
              f"(> {diversity_max_identity:.0%} identity to a kept peptide)")
    return selected


def select_maximin(
    library: list[str],
    ranker,
    top_k: int,
    reference: list[str],
    *,
    assayed_k: int = 50,
    diversity_max_identity: float | None = None,
    gate: float = 0.5,
    shortlist: int = 8000,
) -> list[str]:
    """Select the top list to maximise the WEAKEST scored category (maximin), not a weighted sum.

    The competition ranks Broad / Gram- / Gram+ / MDR / Selectivity *separately*, so a team's standing
    turns on its weakest category. :func:`~amp_challenge_2027.oracle.balanced_success_score` is a fixed
    weighted sum and cannot maximise a minimum. This selector greedily fills the *assayed* set (the top
    ``assayed_k`` = 50, of which 25 are assayed at random) with the candidate that most raises the
    running **minimum** per-category Success Rate, then pads to ``top_k`` with the least-hemolytic
    strong actives (slots 51-100 are not assayed; they exist only so ``top`` is a subset of the
    library). Candidates are gated to ESMC P(hemolytic) < ``gate`` (the Selectivity category) and held
    to the same < 80% novelty and within-list diversity screens as :func:`select_top`. Deterministic:
    stable ordering with a sequence tie-break, so the organizers' twice-run byte comparison holds.

    Needs a ranker exposing ``maximin_data`` (:class:`~amp_challenge_2027.model.ApexRanker`); callers
    fall back to :func:`select_top` when it is absent (e.g. the likelihood/random fallbacks).
    """
    rates, phemo = ranker.maximin_data(library)
    weights = np.array([0.5, 0.75, 1.0, 1.0])  # Broad, Gram-, Gram+, MDR -- shortlist ordering only

    # Shortlist: non-hemolytic, most-active first (so the expensive novelty screen vs 39k refs runs on
    # a few thousand), ranked by summed category rates so generalists AND single-category specialists
    # both enter -- maximin needs Gram-specialists to fill Gram- and Gram+ specialists to fill Gram+.
    activity = rates @ weights
    gated = [i for i in range(len(library)) if phemo[i] < gate]
    gated.sort(key=lambda i: (-activity[i], library[i]))
    short = gated[:shortlist]
    novel = [i for i in short if C.is_novel_enough(library[i], reference)]

    selected: list[int] = []
    selected_seqs: set[str] = set()
    running = np.zeros(4)

    def diverse_ok(i: int) -> bool:
        if diversity_max_identity is None or not selected:
            return True
        return C.max_identity(library[i], [library[j] for j in selected],
                              cutoff=diversity_max_identity) < diversity_max_identity

    # Phase 1 -- greedily fill the currently WEAKEST category. Each step, take the category with the
    # least coverage so far (``argmin`` of the running totals; Broad at the start) and add the
    # candidate strongest in it, breaking ties by the resulting floor, then total, then sequence. This
    # is less myopic than "maximise the running minimum", which can grab a balanced-mediocre peptide
    # first and lock in a worse set; filling the weak category directly raises the eventual floor.
    remaining = list(novel)
    while len(selected) < min(assayed_k, top_k) and remaining:
        weak = int(np.argmin(running))
        cand = np.asarray(remaining)
        cr = rates[cand]
        newmean = (running[None, :] + cr) / (len(selected) + 1)  # (m, 4)
        key_weak = cr[:, weak]
        key_min = newmean.min(axis=1)
        key_sum = newmean.sum(axis=1)
        order = sorted(range(len(remaining)),
                       key=lambda j: (-key_weak[j], -key_min[j], -key_sum[j], library[remaining[j]]))
        picked = None
        for j in order:
            if diverse_ok(remaining[j]):
                picked = remaining[j]
                break
        if picked is None:
            break
        selected.append(picked)
        selected_seqs.add(library[picked])
        running = running + rates[picked]
        remaining.remove(picked)

    # Phase 2 -- pad to top_k with the least-hemolytic strong actives (not assayed).
    for i in sorted(short, key=lambda i: (phemo[i], -activity[i], library[i])):
        if len(selected) >= top_k:
            break
        if library[i] in selected_seqs:
            continue
        if C.is_novel_enough(library[i], reference) and diverse_ok(i):
            selected.append(i)
            selected_seqs.add(library[i])

    if len(selected) < top_k:
        raise RuntimeError(
            f"maximin selected only {len(selected)} of {top_k}: the novelty/diversity/selectivity "
            f"screens are too strict for this pool -- raise --oversample or relax the screens"
        )
    floor = (running / min(assayed_k, top_k)).min()
    print(f"  maximin top-{min(assayed_k, top_k)} category floor (min of Broad/Gram-/Gram+/MDR SR): {floor:.3f}")
    return [library[i] for i in selected]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate",
        description="Generate the AMP Challenge 2027 submission library and ranked top list.",
    )
    parser.add_argument("--n-sequences", type=int, default=C.LIBRARY_SIZE,
                        help="sequences in the library (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=C.TOP_SIZE,
                        help="sequences in the ranked top list (default: %(default)s)")
    parser.add_argument("--oversample", type=float, default=8.0,
                        help="generate this multiple of --n-sequences candidates and pick the "
                             "top list from the whole pool; the library is then the top list plus "
                             "a diverse fill to --n-sequences. Broad-spectrum actives are rare, so "
                             "a larger pool yields a stronger top list (measured +17%% breadth at "
                             "3x). 1.0 disables (default: %(default)s). Raises generation time.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed; fixed so runs are reproducible (default: %(default)s)")
    parser.add_argument("--min-length", type=int, default=C.MIN_LENGTH,
                        help="shortest peptide to generate (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=C.MAX_LENGTH,
                        help="longest peptide to generate (default: %(default)s)")
    parser.add_argument("--length", type=int, default=None,
                        help="generate every peptide at exactly this length, overriding "
                             "--min-length/--max-length (default: %(default)s, meaning a range). "
                             "Accepted for parity with the official template's interface.")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR),
                        help="output directory (default: %(default)s)")
    parser.add_argument("--reference", type=str, default=DEFAULT_REFERENCE,
                        help="FASTA of known antibacterial peptides (default: %(default)s)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="write the files without running the local compliance check")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="trained generator checkpoint (default: %(default)s)")
    parser.add_argument("--temperature", type=float, default=1.6,
                        help="sampling temperature for the 50k LIBRARY body. Hot sampling (1.6) keeps "
                             "the library diverse and novel -- the Phase-2 advancement axes -- since the "
                             "activity-tuned generator otherwise concentrates on a few modes "
                             "(default: %(default)s)")
    parser.add_argument("--top-temperature", type=float, default=0.8,
                        help="sampling temperature for the TOP-100 CANDIDATE pool (feat-028/029 mixed-temperature "
                             "sampling). Lower than --temperature: the ranked pool is drawn cooler so its best "
                             "candidates sit on the generator's high-activity modes. feat-029 swept 0.6-1.0 and "
                             "found 0.8 a clean interior optimum (top-50 Gram- 0.57->0.66, Broad 0.63->0.70 vs a "
                             "single hot pool; +0.02 Gram-/+0.02 Broad over top-temperature 1.0, cross-validated "
                             "on held-out APEX submodels at seeds 42/43/44; more amphipathic, novelty/selectivity "
                             "and GP/MDR held), while the library body stays at --temperature so Phase-2 diversity "
                             "is unchanged (0.84). Set equal to --temperature to disable and use one pool "
                             "(default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="nucleus sampling cutoff for the trained generator (default: %(default)s)")
    parser.add_argument("--baseline", action="store_true",
                        help="force the random-baseline generator even if a checkpoint exists")
    parser.add_argument("--rank", choices=("likelihood", "apex"), default="apex",
                        help="how to rank the top list: 'apex' = APEX-predicted MIC "
                             "(wet-lab-aligned; falls back to likelihood if the oracle can't "
                             "start), 'likelihood' = generator likelihood (default: %(default)s)")
    parser.add_argument("--apex-dir", type=str, default="oracle/apex",
                        help="APEX oracle project directory, used when --rank apex "
                             "(default: %(default)s)")
    parser.add_argument("--rank-objective", choices=("balanced", "category", "broad"),
                        default="balanced",
                        help="APEX activity score for ranking: 'balanced' = HARD Gram+/MDR Success "
                             "Rate + a broad soft-potency tie-break (directly optimises the scored "
                             "metric and surfaces the rare true Gram+/MDR hard-hitters that the soft "
                             "'category' score dilutes -- the shipped default); 'category' = mean "
                             "soft-success with Gram+/MDR up-weighted; 'broad' = the older "
                             "broad_potency margin (default: %(default)s). Only used with --rank apex.")
    parser.add_argument("--gp-weight", type=float, default=1.0,
                        help="weight for the Gram-positive Success Rate in the ranking objective "
                             "(default: %(default)s)")
    parser.add_argument("--mdr-weight", type=float, default=1.0,
                        help="weight for the MDR Success Rate in the ranking objective "
                             "(default: %(default)s)")
    parser.add_argument("--broad-weight", type=float, default=0.5,
                        help="weight for the broad soft-potency tie-break in the 'balanced' "
                             "objective (default: %(default)s)")
    parser.add_argument("--gn-weight", type=float, default=0.75,
                        help="weight for the hard Gram-negative Success Rate in the 'balanced' "
                             "objective; surfaces Gram-strong peptides so the top-50 covers Gram- "
                             "too, at no Gram+/MDR cost (default: %(default)s)")
    parser.add_argument("--hemolysis-penalty", type=float, default=1.5,
                        help="selectivity weight lambda: rank by the APEX activity score minus "
                             "lambda * P(hemolytic), so less-hemolytic actives rank higher "
                             "(serves the Optimal Selectivity category). Calibrated for the "
                             "'balanced' score's larger [0,~2.5] scale: 1.5 holds the top-50 at ~0% "
                             "predicted-hemolytic while keeping the Gram+/MDR gains (a smaller lambda "
                             "lets hard-to-avoid hemolytic Gram+ hitters slip in). 0 disables "
                             "(default: %(default)s). Only used with --rank apex.")
    parser.add_argument("--hemolysis-checkpoint", type=str, default="checkpoint/hemolysis.pt",
                        help="physicochemical hemolysis model weights, used as a fallback if the "
                             "ESMC selectivity model is unavailable (default: %(default)s)")
    parser.add_argument("--selectivity-checkpoint", type=str,
                        default="checkpoint/selectivity_esmc.pt",
                        help="ESMC-600M selectivity head (latest PLM); preferred over the physchem "
                             "model for the hemolysis penalty (default: %(default)s)")
    parser.add_argument("--refine-k", type=int, default=20000,
                        help="apply the (expensive) selectivity model to only the top-K "
                             "most-active candidates; the top-100 is drawn from these "
                             "(default: %(default)s)")
    parser.add_argument("--diversity-max-identity", type=float, default=0.6,
                        help="cap within-top-list Levenshtein identity: skip a candidate too "
                             "similar to an already-selected one, keeping the more-active of a "
                             "near-duplicate pair. Set to a value >= 1 to disable "
                             "(default: %(default)s)")
    parser.add_argument("--select", choices=("score", "maximin"), default="score",
                        help="top-list selection. 'score' (default, shipped) ranks by the balanced "
                             "hard Success-Rate score minus the hemolysis penalty -- it keeps the "
                             "Gram+/MDR and Selectivity STANDOUTS that win those separately-ranked "
                             "categories. 'maximin' instead fills the assayed top-50 to maximise the "
                             "WEAKEST category's Success Rate (a robust, no-weak-category profile); it "
                             "raises the Gram- floor but trades away the Gram+/MDR/Selectivity "
                             "standouts, so it is NOT the default -- see docs/RESEARCH.md (feat-023). "
                             "maximin needs the APEX ranker; it falls back to 'score' otherwise "
                             "(default: %(default)s)")
    parser.add_argument("--amphipathicity-bonus", type=float, default=0.2,
                        help="scale of a smooth [0,1] amphipathicity reward added to the APEX activity "
                             "score, rising with the (closed-form, deterministic) Eisenberg hydrophobic "
                             "moment (floor-ramp from muH 0.25, saturating at 0.50) -- an orthogonal, "
                             "structure-free hedge that prefers mechanistically-amphipathic designs among "
                             "the APEX-active ones (the ESMFold2 check found the bonus-0 top-50 at the weak "
                             "edge of amphipathicity, 26% non-amphipathic). The shipped default 0.2 lifts "
                             "top-50 muH median 0.31->0.40 (non-amphipathic 26%->12%) at <=0.014 cost to "
                             "every scored category (within APEX's 0.62-AUROC noise) and 0% predicted-"
                             "hemolytic held; 0.0 recovers the pre-feat-025 (feat-021) selection. Applies "
                             "to --select score. See docs/RESEARCH.md (feat-025) (default: %(default)s)")
    parser.add_argument("--lys-hedge", type=float, default=0.4,
                        help="scale of a wet-lab-grounded Gram-negative de-bias subtracted from the APEX "
                             "activity score: lys_hedge * max(0, R/(R+K) - 0.4), penalising Arg-over-Lys "
                             "excess. APEX over-rates Arg-rich peptides on Gram-negative activity "
                             "(validated on the 46 wet-lab MICs: residual vs R/(R+K) Spearman +0.41, "
                             "p=0.003) while the REAL Gram- Success Rate favours Lys-richness (-0.40, "
                             "p=0.004); the bias is shared across all 8 APEX submodels so the held-out "
                             "guard is blind to it, and 75% of the competition panel is Gram-negative. "
                             "Near-free on APEX (Gram- SR is flat across R/(R+K) 0.2-0.8; Lys-rich strong "
                             "actives are abundant), and the ESMC safety-window is best in the resulting "
                             "0.4-0.5 band. Applies to --select score. See docs/RESEARCH.md (feat-031); "
                             "0.0 recovers the feat-029 selection (default: %(default)s)")
    parser.add_argument("--composition-weight", type=float, default=1.0,
                        help="feat-033 wet-lab COMPOSITION ranking (SHIPPED DEFAULT). When > 0, the top-100 is ranked NOT by "
                             "the APEX activity score (which anti-ranks real activity within the band it "
                             "selects -- validated on the 46 wet-lab MICs, docs/RESEARCH.md feat-033) but by "
                             "composition_weight * (lys_fraction - --aromatic-weight * aromatic_fraction) minus "
                             "the ESMC hemolysis penalty, restricted to the APEX-active band (top --refine-k by "
                             "APEX; APEX kept only as the coarse active-band GATE it is good at). Lys-richness / "
                             "low aromatic content is the strongest real-activity predictor on the 46 (top-15 "
                             "real broad SR 0.46 vs 0.31 baseline, Gram- 0.54 vs 0.34) and lands in the Lys/Arg "
                             "blend zone the literature endorses. Supersedes --amphipathicity-bonus and "
                             "--lys-hedge (no-ops when > 0). Applies to --select score. 0.0 = off, the "
                             "APEX-ranked feat-031 path (default: %(default)s)")
    parser.add_argument("--aromatic-weight", type=float, default=0.5,
                        help="weight on the aromatic-fraction penalty inside --composition-weight ranking "
                             "(default: %(default)s). Only used when --composition-weight > 0.")

    args = parser.parse_args(argv)
    if args.length is not None:
        args.min_length = args.max_length = args.length
    if not (0.0 <= args.amphipathicity_bonus < float("inf")):
        parser.error("--amphipathicity-bonus must be a finite value >= 0")
    if not (0.0 <= args.lys_hedge < float("inf")):
        parser.error("--lys-hedge must be a finite value >= 0")
    if not (0.0 <= args.composition_weight < float("inf")):
        parser.error("--composition-weight must be a finite value >= 0")
    if not (0.0 <= args.aromatic_weight < float("inf")):
        parser.error("--aromatic-weight must be a finite value >= 0")
    if not (0.0 < args.top_temperature < float("inf")):
        parser.error("--top-temperature must be a finite value > 0")
    if not (0.0 < args.temperature < float("inf")):
        parser.error("--temperature must be a finite value > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_determinism(args.seed)

    reference_path = resolve_repo_path(args.reference)
    reference = read_sequences(reference_path)
    reference_set = set(reference)
    print(f"Reference: {len(reference)} known antibacterial peptides from {reference_path}")

    model = build_model(args)
    rng = np.random.default_rng(args.seed)

    # Oversample: draw a larger candidate pool, rank the whole pool, and pick the top list from
    # it. Broad-spectrum actives are rare, so a bigger pool surfaces a stronger top list. The
    # submitted library is then the top list plus a diverse fill up to --n-sequences, so the
    # top list is guaranteed to be a subset of the library. Order is insertion order throughout,
    # so the result stays byte-reproducible.
    pool_size = max(args.n_sequences, round(args.oversample * args.n_sequences))
    ranker = build_ranker(model, args)
    # feat-028 mixed-temperature sampling: draw the RANKED top-candidate pool at the cooler
    # --top-temperature (its best peptides then sit on the generator's high-activity modes -> a
    # stronger, cross-validated top-50), but build the 50k LIBRARY BODY at the hot --temperature so
    # Phase-2 diversity/novelty are unchanged. The top list is prepended to the library body, so the
    # top-100-subset rule still holds and only the ~100 selected peptides are cool. Both draws use the
    # same rng in sequence, so the run stays byte-reproducible. Disabled (single pool) when the two
    # temperatures are equal or the model has no temperature (the RandomBaseline fallback).
    mixed = (args.top_temperature != args.temperature) and hasattr(model, "temperature")
    if mixed:
        model.temperature = args.top_temperature
        rank_pool = build_library(model, pool_size, rng, reference_set)
        model.temperature = args.temperature
        library_body = build_library(model, args.n_sequences, rng, reference_set)
        print(f"Pool: {len(rank_pool)} candidates ({args.oversample:g}x) @ top-temperature "
              f"{args.top_temperature:g} for the top list; {len(library_body)} @ temperature "
              f"{args.temperature:g} for the library body (mixed-temperature)")
    else:
        rank_pool = build_library(model, pool_size, rng, reference_set)
        library_body = rank_pool
        if pool_size > args.n_sequences:
            print(f"Pool: {len(rank_pool)} candidates ({args.oversample:g}x) for top-list selection")

    if args.select == "maximin" and hasattr(ranker, "maximin_data"):
        if args.amphipathicity_bonus > 0:
            print("  NOTE: --amphipathicity-bonus applies to --select score only; the maximin "
                  "selector ranks by per-category rates and ignores it (no bonus applied)")
        if args.lys_hedge > 0:
            print("  NOTE: --lys-hedge applies to --select score only; the maximin selector ranks "
                  "by per-category rates and ignores it (no hedge applied)")
        if args.composition_weight > 0:
            print("  NOTE: --composition-weight applies to --select score only; the maximin selector "
                  "ranks by per-category rates and ignores it (no composition ranking applied)")
        top = select_maximin(rank_pool, ranker, args.top_k, reference,
                             diversity_max_identity=args.diversity_max_identity)
    else:
        if args.select == "maximin":
            print("  maximin needs the APEX ranker (needs per-category MIC); using score ranking")
        top = select_top(rank_pool, ranker, args.top_k, reference,
                         diversity_max_identity=args.diversity_max_identity)

    # Library = the top list first (guarantees the subset rule), then the library body in order,
    # de-duplicated and truncated to the required size.
    library = list(dict.fromkeys(top + library_body))[: args.n_sequences]
    library_path = args.out_dir / "library.fasta"
    write_fasta(library, library_path)
    print(f"Library: {len(library)} sequences -> {library_path}")

    top_path = args.out_dir / "top.fasta"
    write_fasta(top, top_path)
    print(f"Top list: {len(top)} sequences -> {top_path}")

    if args.skip_validation:
        print("Skipped local compliance check (--skip-validation)")
        return 0

    errors = C.check_library(
        library,
        reference=reference_set,
        expected_size=args.n_sequences,
    )
    errors += [
        f"top list: {problem}"
        for problem in C.check_top(
            top, library, reference=reference, expected_size=args.top_k
        )
    ]

    if errors:
        print("\nCompliance check FAILED:", file=sys.stderr)
        for problem in errors:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("Compliance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
