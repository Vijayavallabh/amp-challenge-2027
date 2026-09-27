"""FASTA reading and writing.

The reader deliberately mirrors ``scripts/verify_submission.py::_read_fasta`` so that
what we validate locally is exactly what the organizers' validator will parse.
"""

from __future__ import annotations

from pathlib import Path


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    """Return ``(headers, sequences)``. Multi-line records are joined, sequences upper-cased."""
    headers: list[str] = []
    sequences: list[str] = []
    header: str | None = None
    seq_parts: list[str] = []

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                headers.append(header)
                sequences.append("".join(seq_parts))
            header, seq_parts = line[1:], []
        else:
            seq_parts.append(line.upper())

    if header is not None:
        headers.append(header)
        sequences.append("".join(seq_parts))

    return headers, sequences


def read_sequences(path: str | Path) -> list[str]:
    """Return just the sequences, preserving file order."""
    return read_fasta(path)[1]


def write_fasta(sequences: list[str], path: str | Path, *, prefix: str = "seq") -> None:
    """Write ``sequences`` as ``>{prefix}{i}`` records, one sequence per line.

    Order is preserved exactly as given -- never pass an unordered ``set`` here, or
    output will differ between runs and fail the reproducibility check.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for i, seq in enumerate(sequences, start=1):
            handle.write(f">{prefix}{i}\n{seq}\n")
