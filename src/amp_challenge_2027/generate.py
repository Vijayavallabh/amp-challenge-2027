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
            ranker = ApexRanker(args.apex_dir)
            print(f"Ranking: {ranker.name} (APEX-predicted MIC)")
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate",
        description="Generate the AMP Challenge 2027 submission library and ranked top list.",
    )
    parser.add_argument("--n-sequences", type=int, default=C.LIBRARY_SIZE,
                        help="sequences in the library (default: %(default)s)")
    parser.add_argument("--top-k", type=int, default=C.TOP_SIZE,
                        help="sequences in the ranked top list (default: %(default)s)")
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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature for the trained generator (default: %(default)s)")
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
    parser.add_argument("--diversity-max-identity", type=float, default=0.6,
                        help="cap within-top-list Levenshtein identity: skip a candidate too "
                             "similar to an already-selected one, keeping the more-active of a "
                             "near-duplicate pair. Set to a value >= 1 to disable "
                             "(default: %(default)s)")

    args = parser.parse_args(argv)
    if args.length is not None:
        args.min_length = args.max_length = args.length
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

    library = build_library(model, args.n_sequences, rng, reference_set)
    library_path = args.out_dir / "library.fasta"
    write_fasta(library, library_path)
    print(f"Library: {len(library)} sequences -> {library_path}")

    ranker = build_ranker(model, args)
    top = select_top(library, ranker, args.top_k, reference,
                     diversity_max_identity=args.diversity_max_identity)
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
