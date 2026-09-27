"""Training corpus: loading, metadata, and disclosure.

The competition provides no dataset (its only Kaggle data file is a note saying so), so
the training corpus is sourced externally and must be disclosed in full. This module is
that corpus made programmatic: it loads the peptides, parses the per-record metadata, and
exposes a clean, deterministic sequence list for a generative model to train on.

Source
------
`data/antibacterial.fasta`, vendored from the official challenge template
(szczurek-lab/amp-challenge-2027, BSD-3-Clause). It is a curated aggregation of 39,448
antimicrobial peptides drawn from public AMP databases -- dbAMP, DRAMP, DBAASP, CAMP,
SATPdb, APD and others (see ``SOURCE_DATABASES``). The organizers pre-filtered it to the
competition's own constraints: every record is 8-50 residues over the 20 standard amino
acids, and all are unique. See ``docs/DATA.md`` for the full data card and licence terms.

The dual role of this file, and why it is deliberate
----------------------------------------------------
The same file is *also* the novelty screen reference in ``generate.py``: the submitted
library may contain no sequence identical to it, and the top 100 must stay at or below
80% Levenshtein identity to every sequence in it. So a model trained here cannot win by
memorizing -- reproducing a training peptide puts a forbidden sequence in the library.
The screen forces genuine generalization, which is the point of the challenge. This is a
disclosed, intentional choice, not an oversight; a real model (feat-006) must generate
novel variants, and the already-built screen enforces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .constraints import is_valid_sequence
from .fasta import read_fasta
from .paths import resolve_repo_path

DEFAULT_CORPUS = "data/antibacterial.fasta"

#: Public databases the corpus aggregates, with the count of corpus peptides recorded in
#: each (from the vendored file's ``dbs=`` header field). Every one offers open access to
#: peptide data for research; see docs/DATA.md for per-source terms.
SOURCE_DATABASES: tuple[str, ...] = (
    "dbAMP", "DRAMP", "DBAASP", "CAMP", "SATPdb", "APD",
    "AMPDB", "DADP", "InverPep", "CancerPPD", "BaAMPs", "CyBase", "ParaPep",
)

_HEADER_ID = re.compile(r"^(\S+)")
_FIELD = re.compile(r"(\w+)=(\S+)")


@dataclass(frozen=True)
class Peptide:
    """One corpus peptide and the metadata parsed from its FASTA header.

    Fields beyond ``sequence`` come straight from the vendored header; ``charge`` and
    ``disulfide`` are ``None`` when absent. ``source_dbs`` and ``activities`` are the
    pipe-separated ``dbs=`` and ``activity=`` fields, split into tuples.
    """

    sequence: str
    identifier: str = ""
    charge: float | None = None
    disulfide: int | None = None
    source_dbs: tuple[str, ...] = field(default_factory=tuple)
    activities: tuple[str, ...] = field(default_factory=tuple)

    @property
    def length(self) -> int:
        return len(self.sequence)


def _parse_header(header: str) -> dict:
    """Pull ``id``, ``charge``, ``disulfide``, ``dbs`` and ``activity`` out of a header."""
    out: dict = {}
    id_match = _HEADER_ID.match(header)
    if id_match:
        out["identifier"] = id_match.group(1)

    for key, value in _FIELD.findall(header):
        if key == "charge":
            try:
                out["charge"] = float(value)
            except ValueError:
                pass
        elif key == "disulfide":
            try:
                out["disulfide"] = int(value)
            except ValueError:
                pass
        elif key == "dbs":
            out["source_dbs"] = tuple(value.split("|"))
        elif key == "activity":
            out["activities"] = tuple(value.split("|"))
    return out


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[Peptide]:
    """Load every peptide in the corpus with its parsed metadata, in file order.

    The path resolves against the CWD then the repo root, so this works both when the
    validator runs from the repo root and when a developer runs from elsewhere.
    """
    resolved = resolve_repo_path(path)
    headers, sequences = read_fasta(resolved)
    return [
        Peptide(sequence=seq, **_parse_header(header))
        for header, seq in zip(headers, sequences)
    ]


def training_sequences(
    path: str | Path = DEFAULT_CORPUS,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
) -> list[str]:
    """Return the training sequences: competition-valid, deduplicated, deterministically ordered.

    Filtering to :func:`~amp_challenge_2027.constraints.is_valid_sequence` is a safety net
    -- the vendored corpus is already 100% valid -- so that a model is never trained on a
    sequence it could not legally emit. De-duplication preserves first-seen order via a
    dict, never set iteration, so the result is stable across processes. Optional length
    bounds narrow the corpus further (e.g. to train a fixed-length model).
    """
    seen: dict[str, None] = {}
    for peptide in load_corpus(path):
        seq = peptide.sequence
        if not is_valid_sequence(seq):
            continue
        if min_length is not None and len(seq) < min_length:
            continue
        if max_length is not None and len(seq) > max_length:
            continue
        seen.setdefault(seq, None)
    return list(seen)


@dataclass(frozen=True)
class CorpusStats:
    """Summary statistics for the data card and for sanity checks in tests."""

    count: int
    unique: int
    valid: int
    min_length: int
    max_length: int
    mean_length: float
    net_positive: int
    source_databases: dict[str, int]
    activities: dict[str, int]


def corpus_stats(path: str | Path = DEFAULT_CORPUS) -> CorpusStats:
    """Compute summary statistics over the corpus (used by docs/DATA.md and tests)."""
    peptides = load_corpus(path)
    lengths = [p.length for p in peptides]
    sequences = [p.sequence for p in peptides]

    db_counts: dict[str, int] = {}
    act_counts: dict[str, int] = {}
    net_positive = 0
    for p in peptides:
        for db in p.source_dbs:
            db_counts[db] = db_counts.get(db, 0) + 1
        for act in p.activities:
            act_counts[act] = act_counts.get(act, 0) + 1
        if p.charge is not None and p.charge > 0:
            net_positive += 1

    return CorpusStats(
        count=len(peptides),
        unique=len(set(sequences)),
        valid=sum(1 for s in sequences if is_valid_sequence(s)),
        min_length=min(lengths) if lengths else 0,
        max_length=max(lengths) if lengths else 0,
        mean_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        net_positive=net_positive,
        source_databases=dict(sorted(db_counts.items(), key=lambda kv: -kv[1])),
        activities=dict(sorted(act_counts.items(), key=lambda kv: -kv[1])),
    )
