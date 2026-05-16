"""Tests for src/cdr_gen/hard_filters.py."""

from cdr_gen.hard_filters import (
    ACTIVE_LIABILITY_MOTIFS,
    all_hard_filters,
    check_alphabet,
    check_length,
    check_motifs,
    has_n_glyc_sequon,
)
from cdr_gen.schema import CDRSpan, GenerationTask


def _task(cdr_len=13, forbidden=()):
    return GenerationTask(
        parent_id="ab", parent_vh="A" * 120, parent_vl="D" * 110,
        cdr_span=CDRSpan(chain="H", start=96, end=96 + cdr_len),
        forbidden_motifs=forbidden,
    )


class TestAlphabet:
    def test_standard_aas_pass(self):
        assert check_alphabet("ACDEFGHIKL").passes

    def test_non_standard_rejected(self):
        r = check_alphabet("ABXY")
        assert not r.passes
        assert "B" in r.reason or "X" in r.reason


class TestLength:
    def test_within_bounds(self):
        assert check_length("ACDEFGHIJKLMN", _task(cdr_len=13)).passes

    def test_too_short(self):
        assert not check_length("AB", _task(cdr_len=13)).passes

    def test_too_long(self):
        assert not check_length("A" * 20, _task(cdr_len=13)).passes


class TestMotifs:
    def test_clean_cdr_passes(self):
        # All lysines: no NG, DG, etc.
        assert check_motifs("KKKKKKKK").passes

    def test_ng_rejected(self):
        r = check_motifs("AANGHL")
        assert not r.passes
        assert "NG" in r.reason

    def test_cys_rejected(self):
        r = check_motifs("ACKQPLR")
        assert not r.passes
        assert "C" in r.reason

    def test_n_glyc_sequon_rejected(self):
        r = check_motifs("QQNASV")  # N-A-S
        assert not r.passes

    def test_np_not_sequon(self):
        # N-P-S is NOT a valid sequon (Pro breaks it).
        r = check_motifs("QQNPSV")
        assert r.passes  # not a motif we have; NP is fine

    def test_methionine_tolerated_by_default(self):
        # M is in DEFAULT but removed from ACTIVE to be tolerant.
        assert "M" not in ACTIVE_LIABILITY_MOTIFS
        assert check_motifs("AMAMAV").passes


class TestNGlycSequon:
    def test_detects_classic_sequon(self):
        assert has_n_glyc_sequon("QQNASV")
        assert has_n_glyc_sequon("NGT")  # NG-T

    def test_misses_proline_breaks(self):
        # N-P-S/T is NOT a sequon (Pro breaks).
        assert not has_n_glyc_sequon("QQNPS")
        assert not has_n_glyc_sequon("NPT")

    def test_no_false_positives(self):
        assert not has_n_glyc_sequon("KKKKKK")


class TestAllHardFilters:
    def test_clean_passes(self):
        assert all_hard_filters("KKQLHAPKQLLRH", _task(cdr_len=13)).passes

    def test_returns_first_failure(self):
        # Has both wrong alphabet AND NG.
        r = all_hard_filters("BBXNGSS", _task(cdr_len=7))
        assert not r.passes
        # Alphabet fails first.
        assert "non-standard" in r.reason

    def test_user_motifs_checked(self):
        r = all_hard_filters("AAAAAAAAAWWWA", _task(cdr_len=13, forbidden=("WW",)))
        assert not r.passes
        assert "WW" in r.reason
