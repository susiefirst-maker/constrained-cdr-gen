"""Tests for src/cdr_gen/schema.py."""

import pytest

from cdr_gen.schema import (
    CDRSpan,
    GenerationTask,
    GeneratedSequence,
    OracleResult,
)


class TestCDRSpan:
    def test_valid(self):
        s = CDRSpan(chain="H", start=96, end=109)
        assert s.length == 13

    def test_rejects_bad_chain(self):
        with pytest.raises(ValueError, match="chain"):
            CDRSpan(chain="X", start=0, end=10)

    def test_rejects_empty_span(self):
        with pytest.raises(ValueError, match="invalid span"):
            CDRSpan(chain="H", start=5, end=5)


class TestGenerationTask:
    def _task(self, **k):
        defaults = dict(
            parent_id="ab",
            parent_vh="A" * 120,
            parent_vl="D" * 110,
            cdr_span=CDRSpan(chain="H", start=96, end=109),
        )
        defaults.update(k)
        return GenerationTask(**defaults)

    def test_defaults_infer_length_bounds(self):
        t = self._task()
        assert t.length_min == 13
        assert t.length_max == 13

    def test_parent_cdr(self):
        t = self._task()
        assert t.parent_cdr == "A" * 13

    def test_rejects_span_out_of_range(self):
        with pytest.raises(ValueError, match="cdr_span end"):
            GenerationTask(
                parent_id="ab", parent_vh="A" * 50, parent_vl="D" * 50,
                cdr_span=CDRSpan(chain="H", start=96, end=109),
            )

    def test_rejects_bad_allowed_aas(self):
        with pytest.raises(ValueError, match="allowed_aas"):
            GenerationTask(
                parent_id="ab", parent_vh="A" * 120, parent_vl="D" * 110,
                cdr_span=CDRSpan(chain="H", start=0, end=10),
                allowed_aas="XYZ",
            )


class TestGeneratedSequence:
    def test_mh_accept_rate_with_zero_proposed(self):
        s = GeneratedSequence(
            sequence_vh="A", sequence_vl="D", cdr_substring="",
            strategy="random", lambda_guidance=0.0, sweep_index=0,
            esm_log_likelihood=0.0,
        )
        assert s.mh_accept_rate == 1.0

    def test_mh_accept_rate(self):
        s = GeneratedSequence(
            sequence_vh="A", sequence_vl="D", cdr_substring="",
            strategy="guided", lambda_guidance=1.0, sweep_index=0,
            esm_log_likelihood=0.0,
            accepted_moves=80, proposed_moves=100,
        )
        assert s.mh_accept_rate == 0.8


class TestOracleResult:
    def test_pass_with_reasons(self):
        o = OracleResult(
            passes=True,
            tap_risk_flag_count=1, di_seq_proxy=0.5,
            camsol_intrinsic_mean=-0.1, humanness_identity=0.6,
            pass_reasons=["all checks"],
        )
        assert o.passes
        assert o.pass_reasons == ["all checks"]
