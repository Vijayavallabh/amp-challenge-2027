"""The competition's hard constraints, as code.

Every rule here is taken from the official rules and from
``scripts/verify_submission.py`` in szczurek-lab/amp-challenge-2027. Keep this module
in agreement with that script -- it is the thing that decides whether a submission is
accepted at all. If the upstream validator changes, change this module first, then the
tests, then the generator.

Rules enforced
--------------
Library (``generate/library.fasta``)
  * exactly 50,000 records, each with a non-empty header
  * alphabet restricted to the 20 standard proteinogenic amino acids
  * length between 8 and 50 residues inclusive
  * no duplicate sequences
  * no sequence identical to any known antibacterial peptide in the reference set

Top list (``generate/top.fasta``)
  * exactly 100 records
  * every sequence also present in the library
  * no duplicates
  * no sequence with Levenshtein ratio > 0.80 against any reference sequence

Constraints that live outside this file because they are chemical, not computational:
linear peptides only, free termini (no amidation or other terminal modification), and
no non-canonical residues, staples, lipidation, glycosylation, PEGylation or
dendrimeric constructs. Restricting the alphabet and emitting plain linear strings is
what satisfies them -- there is nothing to check at runtime.
"""

from __future__ import annotations

import Levenshtein

STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
"""The 20 standard proteinogenic amino acids -- the only permitted characters."""

# Deterministic ordering for sampling. frozenset iteration order is not guaranteed
# across processes, so never sample from STANDARD_AMINO_ACIDS directly.
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

MIN_LENGTH = 8
MAX_LENGTH = 50
LIBRARY_SIZE = 50_000
TOP_SIZE = 100
MAX_TOP_IDENTITY = 0.80
"""Top-100 sequences must not exceed this Levenshtein ratio against any reference."""


def is_valid_sequence(seq: str) -> bool:
    """True if ``seq`` satisfies the per-sequence alphabet and length rules."""
    return (
        MIN_LENGTH <= len(seq) <= MAX_LENGTH
        and not (set(seq) - STANDARD_AMINO_ACIDS)
    )


def max_identity(seq: str, reference: list[str], cutoff: float = MAX_TOP_IDENTITY) -> float:
    """Highest Levenshtein ratio between ``seq`` and any sequence in ``reference``.

    Returns as soon as the cutoff is exceeded, so the result is only exact when it is
    ``<= cutoff``. That is all the ``> cutoff`` rule needs, and it keeps the scan cheap.
    ``score_cutoff`` makes rapidfuzz abandon hopeless pairs early; it returns 0.0 below
    the cutoff, which leaves the ``> cutoff`` comparison unchanged.
    """
    best = 0.0
    for ref in reference:
        ratio = Levenshtein.ratio(seq, ref, score_cutoff=cutoff)
        if ratio > best:
            best = ratio
            if best > cutoff:
                return best
    return best


def is_novel_enough(seq: str, reference: list[str], cutoff: float = MAX_TOP_IDENTITY) -> bool:
    """True if ``seq`` stays at or below ``cutoff`` identity against every reference."""
    return max_identity(seq, reference, cutoff) <= cutoff


def check_library(
    sequences: list[str],
    *,
    headers: list[str] | None = None,
    reference: set[str] | None = None,
    expected_size: int = LIBRARY_SIZE,
) -> list[str]:
    """Return a list of human-readable problems with the library. Empty means valid."""
    errors: list[str] = []

    if not sequences:
        errors.append("library is empty -- no sequences found")
    elif len(sequences) != expected_size:
        errors.append(f"expected {expected_size} sequences, got {len(sequences)}")

    seen: set[str] = set()
    duplicates = 0
    bad_alphabet = 0
    too_short = 0
    too_long = 0
    empty_headers = 0

    for i, seq in enumerate(sequences, start=1):
        if headers is not None and i <= len(headers) and not headers[i - 1].strip():
            empty_headers += 1
        if not seq:
            errors.append(f"record {i}: empty sequence")
            continue
        if set(seq) - STANDARD_AMINO_ACIDS:
            bad_alphabet += 1
            if bad_alphabet == 1:
                invalid = sorted(set(seq) - STANDARD_AMINO_ACIDS)
                errors.append(f"record {i}: non-standard residues {invalid} (first of several)")
        if len(seq) < MIN_LENGTH:
            too_short += 1
        if len(seq) > MAX_LENGTH:
            too_long += 1
        if seq in seen:
            duplicates += 1
        seen.add(seq)

    if empty_headers:
        errors.append(f"{empty_headers} record(s) have an empty header")
    if bad_alphabet:
        errors.append(f"{bad_alphabet} record(s) contain non-standard residues")
    if too_short:
        errors.append(f"{too_short} record(s) shorter than {MIN_LENGTH}")
    if too_long:
        errors.append(f"{too_long} record(s) longer than {MAX_LENGTH}")
    if duplicates:
        errors.append(f"{duplicates} duplicate sequence(s)")

    if reference:
        overlap = seen & reference
        if overlap:
            errors.append(
                f"{len(overlap)} sequence(s) are identical to a known antibacterial peptide"
            )

    return errors


def check_top(
    top: list[str],
    library: list[str] | set[str],
    *,
    reference: list[str] | None = None,
    expected_size: int = TOP_SIZE,
    cutoff: float = MAX_TOP_IDENTITY,
) -> list[str]:
    """Return a list of human-readable problems with the top list. Empty means valid."""
    errors: list[str] = []
    library_set = library if isinstance(library, set) else set(library)

    if len(top) != expected_size:
        errors.append(f"expected {expected_size} sequences, got {len(top)}")

    seen: set[str] = set()
    missing = 0
    duplicates = 0
    for seq in top:
        if seq not in library_set:
            missing += 1
        if seq in seen:
            duplicates += 1
        seen.add(seq)

    if missing:
        errors.append(f"{missing} sequence(s) in the top list are not in the library")
    if duplicates:
        errors.append(f"{duplicates} duplicate sequence(s) in the top list")

    if reference:
        too_similar = [seq for seq in top if not is_novel_enough(seq, reference, cutoff)]
        if too_similar:
            errors.append(
                f"{len(too_similar)} sequence(s) exceed {cutoff:.0%} identity with a known "
                f"antibacterial peptide (e.g. {too_similar[0]})"
            )

    return errors
