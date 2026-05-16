"""Tests for src/cdr_gen/surrogate.py — the physicochemical guidance."""

import pytest

from cdr_gen.surrogate import (
    mean_hydrophobicity,
    net_charge,
    score_full,
    score_residue_in_context,
)


class TestBuildingBlocks:
    def test_mean_hydrophobicity_poly_ile(self):
        assert mean_hydrophobicity("IIIIII") == pytest.approx(4.5)

    def test_mean_hydrophobicity_poly_arg(self):
        assert mean_hydrophobicity("RRRRRR") == pytest.approx(-4.5)

    def test_net_charge_lys_positive(self):
        assert net_charge("KKKKK") > 3

    def test_net_charge_glu_negative(self):
        assert net_charge("EEEEE") < -3


class TestScoreFull:
    def test_balanced_sequence_near_zero_penalty(self):
        # A sequence with modest hydrophobicity and balanced charge
        # should score near zero.
        s = score_full("QVQLQESGGG")
        assert s >= -1.0

    def test_hydrophobic_patch_penalized(self):
        s_hydrophobic = score_full("VVVVVVVVVV")
        s_mixed = score_full("VQVQVQVQVQ")
        assert s_hydrophobic < s_mixed

    def test_highly_charged_penalized(self):
        s_pos = score_full("KKKKKKKKKK")
        s_neutral = score_full("QQQQQQQQQQ")
        assert s_pos < s_neutral

    def test_empty_returns_zero(self):
        assert score_full("") == 0.0


class TestScoreResidueInContext:
    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of CDR range"):
            score_residue_in_context("SRWGGDGFYAMDY", 999, "A")

    def test_non_standard_is_poison(self):
        v = score_residue_in_context("SRWGGDGFYAMDY", 0, "X")
        assert v < -1e8

    def test_mutation_changes_score(self):
        parent = "SRWGGDGFYAMDY"
        s_orig = score_full(parent)
        s_mut = score_residue_in_context(parent, 0, "K")
        assert s_orig != s_mut  # the substitution changed the score
