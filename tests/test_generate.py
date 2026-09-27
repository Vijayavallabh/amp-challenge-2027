"""End-to-end tests of the generate entry point (fast baseline path, no APEX/GPU).

Uses --baseline --rank likelihood so the run is CPU-cheap and CI-safe; the APEX + hemolysis
path is covered by ./init.sh and the official validator.
"""

from __future__ import annotations

from amp_challenge_2027.fasta import read_sequences
from amp_challenge_2027.generate import main


def _run(tmp_path, **flags) -> tuple[list[str], list[str]]:
    argv = ["--baseline", "--rank", "likelihood", "--out-dir", str(tmp_path)]
    for k, v in flags.items():
        argv += ["--" + k.replace("_", "-"), str(v)]
    assert main(argv) == 0
    return read_sequences(tmp_path / "library.fasta"), read_sequences(tmp_path / "top.fasta")


def test_oversample_keeps_library_size_and_top_subset(tmp_path):
    lib, top = _run(tmp_path, n_sequences=300, top_k=10, oversample=4)
    assert len(lib) == 300 and len(set(lib)) == 300      # library is exactly n, unique
    assert len(top) == 10
    assert set(top) <= set(lib)                          # top-list subset rule holds
    assert lib[:10] == top                               # top placed first, guaranteeing the subset


def test_oversample_is_byte_reproducible(tmp_path):
    a_lib, a_top = _run(tmp_path / "a", n_sequences=300, top_k=10, oversample=4)
    b_lib, b_top = _run(tmp_path / "b", n_sequences=300, top_k=10, oversample=4)
    assert a_lib == b_lib and a_top == b_top


def test_oversample_one_is_plain_generation(tmp_path):
    lib, top = _run(tmp_path, n_sequences=200, top_k=10, oversample=1)
    assert len(lib) == 200 and len(top) == 10 and set(top) <= set(lib)
