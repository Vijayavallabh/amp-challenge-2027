"""Tests for the competition's repository-level rules.

`test_constraints.py` covers the rules about sequences. This file covers the rules about
the *submission* -- licence, entry point, pinned environment, defaulted arguments -- so
that a change which would disqualify the entry fails CI instead of failing at review.

Every assertion here traces to a published requirement; see docs/COMPLIANCE.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from amp_challenge_2027 import constraints as C
from amp_challenge_2027.generate import DEFAULT_SEED, parse_args

REPO_ROOT = Path(__file__).resolve().parents[1]

# The full requirements name these three explicitly.
PERMISSIVE_LICENCES = {"MIT", "BSD-3-Clause", "Apache-2.0"}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


class TestEntryPoint:
    """'Entry point runnable via `uv run generate`.'"""

    def test_generate_script_is_declared(self, pyproject):
        scripts = pyproject["project"]["scripts"]
        assert "generate" in scripts, "the entry point must be named 'generate'"

    def test_generate_points_at_our_main(self, pyproject):
        assert pyproject["project"]["scripts"]["generate"] == (
            "amp_challenge_2027.generate:main"
        )


class TestLicence:
    """'A permissive OSI-approved license (e.g. MIT, BSD-3-Clause, or Apache 2.0).'"""

    def test_declared_licence_is_permissive(self, pyproject):
        assert pyproject["project"]["license"] in PERMISSIVE_LICENCES

    def test_licence_file_exists_and_matches(self, pyproject):
        text = (REPO_ROOT / "LICENSE").read_text()
        assert "MIT License" in text
        assert pyproject["project"]["license"] == "MIT"


class TestPinnedEnvironment:
    """'Uses uv for dependency management (include uv.lock and a defined Python version).'"""

    def test_lock_file_is_committed(self):
        assert (REPO_ROOT / "uv.lock").is_file(), "uv.lock must be committed"

    def test_python_version_is_pinned(self):
        pinned = (REPO_ROOT / ".python-version").read_text().strip()
        assert pinned, ".python-version must not be empty"
        assert pinned[0].isdigit(), f"expected a version number, got {pinned!r}"

    def test_requires_python_agrees_with_the_pin(self, pyproject):
        pinned = (REPO_ROOT / ".python-version").read_text().strip()
        major_minor = ".".join(pinned.split(".")[:2])
        assert major_minor in pyproject["project"]["requires-python"]

    def test_uses_the_uv_build_backend(self, pyproject):
        assert pyproject["build-system"]["build-backend"] == "uv_build"


class TestArgumentsAllHaveDefaults:
    """'Any additional arguments must have defaults.'"""

    def test_bare_invocation_parses(self):
        # The validator runs `uv run generate` with no arguments at all.
        assert parse_args([]) is not None

    def test_defaults_request_the_required_sizes(self):
        args = parse_args([])
        assert args.n_sequences == C.LIBRARY_SIZE == 50_000
        assert args.top_k == C.TOP_SIZE == 100

    def test_default_output_paths_are_where_the_validator_looks(self):
        args = parse_args([])
        assert args.out_dir.name == "generate"
        assert args.reference == "data/antibacterial.fasta"


class TestFixedSeed:
    """'A fixed default random seed such that running the generation script twice
    produces identical output.'"""

    def test_seed_default_is_a_fixed_integer(self):
        assert isinstance(DEFAULT_SEED, int)
        assert parse_args([]).seed == DEFAULT_SEED

    def test_seed_is_stable_across_calls(self):
        assert parse_args([]).seed == parse_args([]).seed


class TestTemplateInterfaceParity:
    """The official template documents --n-sequences, --top-k, --seed and --length."""

    @pytest.mark.parametrize("flag", ["--n-sequences", "--top-k", "--seed", "--length"])
    def test_documented_flag_is_accepted(self, flag):
        assert parse_args([flag, "10"]) is not None

    def test_length_pins_both_ends_of_the_range(self):
        args = parse_args(["--length", "24"])
        assert args.min_length == args.max_length == 24

    def test_length_defaults_to_a_range(self):
        args = parse_args([])
        assert args.length is None
        assert args.min_length == C.MIN_LENGTH
        assert args.max_length == C.MAX_LENGTH


class TestConstantsMatchTheOfficialValidator:
    """constraints.py must not drift from scripts/verify_submission.py."""

    def test_constants_agree(self):
        source = (REPO_ROOT / "scripts" / "verify_submission.py").read_text()
        assert 'STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")' in source
        assert f"MIN_LENGTH = {C.MIN_LENGTH}" in source
        assert f"MAX_LENGTH = {C.MAX_LENGTH}" in source
        assert f"TOP_SIZE = {C.TOP_SIZE}" in source
        assert "LIBRARY_SIZE = 50_000" in source
        assert C.LIBRARY_SIZE == 50_000
        assert 'ENTRY_POINT = "generate"' in source

    def test_similarity_threshold_agrees(self):
        source = (REPO_ROOT / "scripts" / "verify_submission.py").read_text()
        # Upstream passes the cutoff as a default argument: threshold: float = 0.8
        assert "threshold: float = 0.8" in source
        assert C.MAX_TOP_IDENTITY == 0.80

    def test_alphabet_has_exactly_twenty_residues(self):
        assert len(C.STANDARD_AMINO_ACIDS) == 20
        assert set(C.ALPHABET) == set(C.STANDARD_AMINO_ACIDS)
        assert len(C.ALPHABET) == 20


class TestReferenceSetIsPresent:
    """The library is screened against the known-antibacterial reference set."""

    def test_reference_fasta_is_committed(self):
        assert (REPO_ROOT / "data" / "antibacterial.fasta").is_file()

    def test_reference_fasta_is_non_trivial(self):
        from amp_challenge_2027.fasta import read_sequences

        sequences = read_sequences(REPO_ROOT / "data" / "antibacterial.fasta")
        assert len(sequences) > 10_000
