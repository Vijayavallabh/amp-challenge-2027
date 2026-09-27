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
from .model import PeptideGenerator, RandomBaseline

DEFAULT_SEED = 42
DEFAULT_OUT_DIR = "generate"
DEFAULT_REFERENCE = "data/antibacterial.fasta"

# How many candidates to draw per round while topping up the library, and how many
# rounds to allow before giving up. Generous for any sane generator.
_BATCH_OVERDRAW = 1.2
_MAX_ROUNDS = 100


def build_model(args: argparse.Namespace) -> PeptideGenerator:
    """Return the generator used for a submission.

    This is the one function to change when swapping in a real model. Load weights from
    ``checkpoint/`` here, using :func:`resolve_repo_path` so the path works regardless of
    the working directory the organizers run from.
    """
    return RandomBaseline(min_length=args.min_length, max_length=args.max_length)


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve ``path`` against the CWD, falling back to the repository root.

    The validator runs the entry point with the repo root as CWD, but developers run it
    from anywhere. Relative data and checkpoint paths must work in both cases.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    fallback = repo_root / path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"{path!r} not found -- looked in {Path.cwd()} and {repo_root}"
    )


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
    model: PeptideGenerator,
    top_k: int,
    reference: list[str],
) -> list[str]:
    """Rank the library and return the best ``top_k`` that clear the novelty screen.

    The top list is held to a stricter standard than the library: no sequence may exceed
    80% Levenshtein identity with any known antibacterial peptide. Candidates that fail
    are skipped and the next-best takes the slot, which is what the rules prescribe.
    """
    scores = model.score(library)
    if len(scores) != len(library):
        raise ValueError(
            f"model.score returned {len(scores)} scores for {len(library)} sequences"
        )

    # Sort by descending score, then by sequence to break ties deterministically.
    ranked = sorted(zip(scores, library), key=lambda pair: (-pair[0], pair[1]))

    selected: list[str] = []
    rejected = 0
    for _, seq in ranked:
        if len(selected) == top_k:
            break
        if C.is_novel_enough(seq, reference):
            selected.append(seq)
        else:
            rejected += 1

    if len(selected) < top_k:
        raise RuntimeError(
            f"only {len(selected)} of {top_k} ranked candidates cleared the "
            f"{C.MAX_TOP_IDENTITY:.0%} novelty screen ({rejected} rejected) -- the library "
            f"is too close to known antibacterial peptides"
        )

    if rejected:
        print(f"  novelty screen rejected {rejected} higher-ranked candidate(s)")
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

    args = parser.parse_args(argv)
    if args.length is not None:
        args.min_length = args.max_length = args.length
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    reference_path = resolve_repo_path(args.reference)
    reference = read_sequences(reference_path)
    reference_set = set(reference)
    print(f"Reference: {len(reference)} known antibacterial peptides from {reference_path}")

    model = build_model(args)
    rng = np.random.default_rng(args.seed)
    print(f"Model: {model.name} (seed {args.seed})")

    library = build_library(model, args.n_sequences, rng, reference_set)
    library_path = args.out_dir / "library.fasta"
    write_fasta(library, library_path)
    print(f"Library: {len(library)} sequences -> {library_path}")

    top = select_top(library, model, args.top_k, reference)
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
