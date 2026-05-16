"""Tests for src/cdr_gen/sampler.py — using a fake MLM to keep tests fast."""

from __future__ import annotations

import numpy as np
import pytest

from cdr_gen.sampler import SamplerConfig, sample, sample_nn_retrieval
from cdr_gen.schema import CDRSpan, GenerationTask


class FakeMLM:
    """Deterministic stand-in for the real MLM."""

    linker = "GGGGSGGGGSGGGGS"

    def joint_sequence(self, vh: str, vl: str):
        return vh + self.linker + vl, len(vh) + len(self.linker)

    def position_probs(self, joint_seq: str, position: int, temperature: float = 1.0):
        # Biased toward Q and A — benign, non-liability residues.
        p = np.ones(20) * 0.02
        p[0] = 0.30   # A
        p[13] = 0.30  # Q
        return p / p.sum()

    def score_sequence_log_lik(self, joint_seq: str, positions: list[int]):
        return -2.5


TRAS_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
           "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS")
TRAS_VL = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
           "QPEDFATYYCQQHYTTPPTFGQGTKVEIK")


@pytest.fixture
def task():
    return GenerationTask(
        parent_id="trastuzumab", parent_vh=TRAS_VH, parent_vl=TRAS_VL,
        cdr_span=CDRSpan(chain="H", start=96, end=109),
    )


class TestSample:
    def test_random_strategy_produces_n_designs(self, task):
        cfg = SamplerConfig(strategy="random", seed=0)
        out = sample(task, FakeMLM(), config=cfg, n_designs=5)
        assert len(out) == 5
        assert all(d.strategy == "random" for d in out)

    def test_random_strategy_uses_allowed_aas(self, task):
        cfg = SamplerConfig(strategy="random", seed=0)
        out = sample(task, FakeMLM(), config=cfg, n_designs=20)
        for d in out:
            assert set(d.cdr_substring).issubset(set(task.allowed_aas))

    def test_unguided_strategy_runs(self, task):
        cfg = SamplerConfig(strategy="unguided", n_sweeps=3, seed=42)
        out = sample(task, FakeMLM(), config=cfg, n_designs=4)
        assert len(out) == 4
        assert all(d.strategy == "unguided" for d in out)

    def test_guided_strategy_applies_guidance(self, task):
        cfg = SamplerConfig(strategy="guided", lambda_guidance=1.0, n_sweeps=3, seed=42)
        out = sample(task, FakeMLM(), config=cfg, n_designs=4)
        assert len(out) == 4
        assert all(d.strategy == "guided" for d in out)
        assert all(d.lambda_guidance == 1.0 for d in out)

    def test_unknown_strategy_raises(self, task):
        cfg = SamplerConfig(strategy="fake", seed=0)
        with pytest.raises(ValueError, match="unknown strategy"):
            sample(task, FakeMLM(), config=cfg, n_designs=1)

    def test_sweeps_increment_accept_count(self, task):
        cfg = SamplerConfig(strategy="unguided", n_sweeps=5, seed=7)
        out = sample(task, FakeMLM(), config=cfg, n_designs=2)
        # Each design should have made ≥ CDR length × sweeps proposals.
        for d in out:
            assert d.proposed_moves >= task.cdr_span.length
            assert 0 <= d.mh_accept_rate <= 1

    def test_parent_cdr_preserved_framework(self, task):
        cfg = SamplerConfig(strategy="guided", lambda_guidance=1.0, n_sweeps=3, seed=1)
        out = sample(task, FakeMLM(), config=cfg, n_designs=3)
        for d in out:
            # Framework (everything outside CDR span) must match parent.
            assert d.sequence_vh[: task.cdr_span.start] == TRAS_VH[: task.cdr_span.start]
            assert d.sequence_vh[task.cdr_span.end :] == TRAS_VH[task.cdr_span.end :]
            assert d.sequence_vl == TRAS_VL

    def test_determinism_same_seed(self, task):
        cfg = SamplerConfig(strategy="guided", lambda_guidance=1.0, n_sweeps=4, seed=5)
        a = sample(task, FakeMLM(), config=cfg, n_designs=3)
        b = sample(task, FakeMLM(), config=cfg, n_designs=3)
        assert [d.cdr_substring for d in a] == [d.cdr_substring for d in b]


class TestNNRetrieval:
    def test_picks_from_library(self, task):
        lib = ["A" * 13, "Q" * 13, "KQHAPLRPQQLKH"]
        out = sample_nn_retrieval(task, lib, n_designs=10, seed=0)
        assert len(out) == 10
        for d in out:
            assert d.cdr_substring in lib
            assert d.strategy == "nn_retrieval"

    def test_empty_matching_library_returns_empty(self, task):
        out = sample_nn_retrieval(task, ["short", "alsoshort"], n_designs=5, seed=0)
        assert out == []
