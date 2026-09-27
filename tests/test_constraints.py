"""Tests for the compliance layer.

These are the checks that decide whether a submission is accepted, so they are worth
testing directly rather than only through a full generation run.
"""

from __future__ import annotations

import numpy as np
import pytest

from amp_challenge_2027 import constraints as C
from amp_challenge_2027.fasta import read_fasta, write_fasta
from amp_challenge_2027.generate import build_library, select_top
from amp_challenge_2027.model import RandomBaseline

VALID = "KWKLFKKIGAVLKVL"  # 15 residues, standard alphabet


class TestIsValidSequence:
    def test_accepts_a_well_formed_peptide(self):
        assert C.is_valid_sequence(VALID)

    def test_accepts_the_length_boundaries(self):
        assert C.is_valid_sequence("A" * C.MIN_LENGTH)
        assert C.is_valid_sequence("A" * C.MAX_LENGTH)

    @pytest.mark.parametrize("length", [0, 1, C.MIN_LENGTH - 1, C.MAX_LENGTH + 1])
    def test_rejects_lengths_outside_the_window(self, length):
        assert not C.is_valid_sequence("A" * length)

    @pytest.mark.parametrize("residue", ["B", "J", "O", "U", "X", "Z", "*", "-", "k"])
    def test_rejects_non_standard_residues(self, residue):
        assert not C.is_valid_sequence(VALID[:-1] + residue)


class TestCheckLibrary:
    def _library(self, n):
        # Distinct 10-mers built from a counter, so no accidental duplicates.
        return [f"KWKLFKKI{C.ALPHABET[i // 20]}{C.ALPHABET[i % 20]}" for i in range(n)]

    def test_valid_library_has_no_errors(self):
        library = self._library(50)
        assert C.check_library(library, expected_size=50) == []

    def test_wrong_size_is_reported(self):
        errors = C.check_library(self._library(49), expected_size=50)
        assert any("expected 50" in e for e in errors)

    def test_empty_library_is_reported(self):
        assert any("empty" in e for e in C.check_library([], expected_size=50))

    def test_duplicates_are_reported(self):
        library = self._library(49) + [self._library(1)[0]]
        errors = C.check_library(library, expected_size=50)
        assert any("duplicate" in e for e in errors)

    def test_non_standard_residues_are_reported(self):
        library = self._library(49) + ["KWKLFKKIBB"]
        errors = C.check_library(library, expected_size=50)
        assert any("non-standard" in e for e in errors)

    def test_out_of_range_lengths_are_reported(self):
        library = self._library(48) + ["SHORT", "A" * 60]
        errors = C.check_library(library, expected_size=50)
        assert any("shorter than" in e for e in errors)
        assert any("longer than" in e for e in errors)

    def test_empty_header_is_reported(self):
        library = self._library(2)
        errors = C.check_library(library, headers=["ok", "   "], expected_size=2)
        assert any("empty header" in e for e in errors)

    def test_overlap_with_known_antibacterials_is_reported(self):
        library = self._library(50)
        errors = C.check_library(library, reference={library[7]}, expected_size=50)
        assert any("identical to a known antibacterial" in e for e in errors)


class TestCheckTop:
    def test_valid_top_list_has_no_errors(self):
        library = [f"KWKLFKKI{a}{b}" for a in C.ALPHABET for b in C.ALPHABET]
        assert C.check_top(library[:10], library, expected_size=10) == []

    def test_wrong_size_is_reported(self):
        library = [f"KWKLFKKI{a}{b}" for a in C.ALPHABET for b in C.ALPHABET]
        errors = C.check_top(library[:9], library, expected_size=10)
        assert any("expected 10" in e for e in errors)

    def test_sequence_outside_the_library_is_reported(self):
        library = [f"KWKLFKKI{a}{b}" for a in C.ALPHABET for b in C.ALPHABET]
        top = library[:9] + ["WWWWWWWWWW"]
        errors = C.check_top(top, library, expected_size=10)
        assert any("not in the library" in e for e in errors)

    def test_duplicate_in_top_list_is_reported(self):
        library = [f"KWKLFKKI{a}{b}" for a in C.ALPHABET for b in C.ALPHABET]
        top = library[:9] + [library[0]]
        errors = C.check_top(top, library, expected_size=10)
        assert any("duplicate" in e for e in errors)

    def test_sequence_too_similar_to_a_reference_is_reported(self):
        errors = C.check_top([VALID], [VALID], reference=[VALID], expected_size=1)
        assert any("identity" in e for e in errors)


class TestIdentity:
    def test_identical_sequence_scores_one(self):
        assert C.max_identity(VALID, [VALID]) == pytest.approx(1.0)

    def test_identical_sequence_is_not_novel_enough(self):
        assert not C.is_novel_enough(VALID, [VALID])

    def test_unrelated_sequence_is_novel_enough(self):
        assert C.is_novel_enough("AAAAAAAAAA", ["WWWWWWWWWWWW"])

    def test_empty_reference_leaves_everything_novel(self):
        assert C.is_novel_enough(VALID, [])

    def test_single_substitution_stays_above_the_cutoff(self):
        # 14/15 identity is ~0.93, comfortably over the 0.80 bar.
        assert not C.is_novel_enough("KWKLFKKIGAVLKVA", [VALID])


class TestFasta:
    def test_round_trip_preserves_order(self, tmp_path):
        sequences = ["KWKLFKKIGA", "VLKVLTTGLP", "ALISWIKRKR"]
        path = tmp_path / "out.fasta"
        write_fasta(sequences, path)
        headers, parsed = read_fasta(path)
        assert parsed == sequences
        assert headers == ["seq1", "seq2", "seq3"]

    def test_multi_line_records_are_joined_and_upper_cased(self, tmp_path):
        path = tmp_path / "wrapped.fasta"
        path.write_text(">a\nKWKL\nfkki\n\n>b\nGAVL\n")
        _, parsed = read_fasta(path)
        assert parsed == ["KWKLFKKI", "GAVL"]


class TestGenerationIsReproducible:
    def test_same_seed_gives_the_same_library(self):
        model = RandomBaseline()
        first = build_library(model, 200, np.random.default_rng(42), set())
        second = build_library(model, 200, np.random.default_rng(42), set())
        assert first == second

    def test_different_seed_gives_a_different_library(self):
        model = RandomBaseline()
        first = build_library(model, 200, np.random.default_rng(42), set())
        second = build_library(model, 200, np.random.default_rng(43), set())
        assert first != second

    def test_library_is_valid_and_excludes_the_reference(self):
        model = RandomBaseline()
        library = build_library(model, 200, np.random.default_rng(42), set())
        excluded = {library[0], library[1]}
        filtered = build_library(model, 200, np.random.default_rng(42), excluded)
        assert not excluded & set(filtered)
        assert C.check_library(filtered, expected_size=200) == []

    def test_top_selection_is_a_deterministic_subset(self):
        model = RandomBaseline()
        library = build_library(model, 200, np.random.default_rng(42), set())
        first = select_top(library, model, 10, [])
        second = select_top(library, model, 10, [])
        assert first == second
        assert set(first) <= set(library)

    def test_top_selection_skips_candidates_that_fail_the_novelty_screen(self):
        model = RandomBaseline()
        library = build_library(model, 200, np.random.default_rng(42), set())
        blocked = select_top(library, model, 1, [])[0]
        replacement = select_top(library, model, 1, [blocked])[0]
        assert replacement != blocked

    def test_impossible_request_raises_rather_than_writing_a_short_list(self):
        model = RandomBaseline()
        library = build_library(model, 20, np.random.default_rng(42), set())
        with pytest.raises(RuntimeError, match="novelty screen"):
            select_top(library, model, 10, library)


class TestModel:
    def test_rejects_an_inverted_length_window(self):
        with pytest.raises(ValueError, match="lengths must satisfy"):
            RandomBaseline(min_length=30, max_length=20)

    def test_rejects_a_length_outside_the_competition_window(self):
        with pytest.raises(ValueError, match="lengths must satisfy"):
            RandomBaseline(min_length=2, max_length=60)

    def test_samples_respect_the_length_window(self):
        model = RandomBaseline(min_length=10, max_length=12)
        for seq in model.sample(200, np.random.default_rng(0)):
            assert 10 <= len(seq) <= 12
            assert C.is_valid_sequence(seq)

    def test_score_returns_one_value_per_sequence(self):
        model = RandomBaseline()
        sequences = model.sample(30, np.random.default_rng(0))
        assert len(model.score(sequences)) == len(sequences)
