"""Tests for the training corpus module.

These lock down the two properties feat-006 will rely on: that the corpus loads with its
metadata intact, and that ``training_sequences`` returns a clean, deterministic,
competition-valid list -- so a model is never trained on a sequence it could not emit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from amp_challenge_2027 import constraints as C
from amp_challenge_2027.data import (
    DEFAULT_CORPUS,
    Peptide,
    SOURCE_DATABASES,
    _parse_header,
    corpus_stats,
    load_corpus,
    training_sequences,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# The vendored corpus is fixed; these are its known statistics as of the BSD-3 template
# snapshot. If the file is re-vendored and these change, update them deliberately.
EXPECTED_COUNT = 39_448


@pytest.fixture(scope="module")
def corpus() -> list[Peptide]:
    return load_corpus()


class TestHeaderParsing:
    def test_full_header_is_parsed(self):
        header = (
            "MLAMP0003076 len=15 charge=4.73 disulfide=0 "
            "dbs=APD|DBAASP|dbAMP activity=antibacterial|antimicrobial"
        )
        parsed = _parse_header(header)
        assert parsed["identifier"] == "MLAMP0003076"
        assert parsed["charge"] == pytest.approx(4.73)
        assert parsed["disulfide"] == 0
        assert parsed["source_dbs"] == ("APD", "DBAASP", "dbAMP")
        assert parsed["activities"] == ("antibacterial", "antimicrobial")

    def test_negative_charge_is_parsed(self):
        assert _parse_header("X charge=-3.5")["charge"] == pytest.approx(-3.5)

    def test_missing_fields_are_absent(self):
        parsed = _parse_header("JustAnID")
        assert parsed["identifier"] == "JustAnID"
        assert "charge" not in parsed
        assert "source_dbs" not in parsed

    def test_malformed_numbers_are_dropped_not_raised(self):
        parsed = _parse_header("X charge=abc disulfide=nan")
        assert "charge" not in parsed
        assert "disulfide" not in parsed


class TestPeptide:
    def test_length_matches_the_sequence(self):
        assert Peptide(sequence="KWKLFKKI").length == 8

    def test_defaults_are_empty_not_none_collections(self):
        p = Peptide(sequence="KWKLFKKI")
        assert p.source_dbs == ()
        assert p.activities == ()
        assert p.charge is None

    def test_is_hashable_and_frozen(self):
        p = Peptide(sequence="KWKLFKKI", identifier="a")
        assert p in {p}  # frozen dataclass -> hashable
        with pytest.raises(Exception):
            p.sequence = "X"  # type: ignore[misc]


class TestLoadCorpus:
    def test_loads_the_expected_count(self, corpus):
        assert len(corpus) == EXPECTED_COUNT

    def test_every_record_has_a_sequence_and_id(self, corpus):
        assert all(p.sequence for p in corpus)
        assert all(p.identifier for p in corpus)

    def test_metadata_is_populated(self, corpus):
        # The vendored corpus annotates every record with source databases and activities.
        assert all(p.source_dbs for p in corpus)
        assert all("antimicrobial" in p.activities for p in corpus)

    def test_default_corpus_path_points_at_the_vendored_file(self):
        assert DEFAULT_CORPUS == "data/antibacterial.fasta"
        assert (REPO_ROOT / DEFAULT_CORPUS).is_file()


class TestTrainingSequences:
    def test_all_sequences_are_competition_valid(self):
        assert all(C.is_valid_sequence(s) for s in training_sequences())

    def test_no_duplicates(self):
        seqs = training_sequences()
        assert len(seqs) == len(set(seqs))

    def test_deterministic_across_calls(self):
        assert training_sequences() == training_sequences()

    def test_length_bounds_are_applied(self):
        bounded = training_sequences(min_length=10, max_length=20)
        assert all(10 <= len(s) <= 20 for s in bounded)
        assert len(bounded) < EXPECTED_COUNT

    def test_the_whole_vendored_corpus_is_usable(self):
        # The organizers pre-filtered to the competition constraints, so nothing is dropped.
        assert len(training_sequences()) == EXPECTED_COUNT

    def test_returns_a_list_not_a_view(self):
        seqs = training_sequences()
        assert isinstance(seqs, list)


class TestCorpusStats:
    def test_stats_are_internally_consistent(self, corpus):
        stats = corpus_stats()
        assert stats.count == len(corpus) == EXPECTED_COUNT
        assert stats.unique == stats.count           # all unique
        assert stats.valid == stats.count             # all competition-valid
        assert 8 <= stats.min_length <= stats.max_length <= 50

    def test_corpus_is_predominantly_cationic(self):
        # AMPs are typically net-positive; a sanity check that charge parsing works.
        stats = corpus_stats()
        assert stats.net_positive / stats.count > 0.5

    def test_every_named_source_database_is_present(self):
        stats = corpus_stats()
        for db in SOURCE_DATABASES:
            assert db in stats.source_databases, f"{db} missing from corpus"

    def test_source_databases_are_sorted_by_frequency(self):
        counts = list(corpus_stats().source_databases.values())
        assert counts == sorted(counts, reverse=True)
