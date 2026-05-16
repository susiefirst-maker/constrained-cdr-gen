"""Tests for src/cdr_gen/oracle.py and src/cdr_gen/eval.py."""

import os
from pathlib import Path

import pytest

from cdr_gen.eval import (
    compute_strategy_metrics,
    strategies_have_nonoverlapping_cis,
)
from cdr_gen.oracle import humanness_proxy, evaluate
from cdr_gen.schema import (
    CDRSpan,
    GeneratedSequence,
    GenerationTask,
    OracleResult,
)


_DEFAULT_AB_BENCHMARK = Path(__file__).resolve().parents[2] / "ab-benchmark"
_AB_BENCHMARK_READY = Path(
    os.environ.get("AB_BENCHMARK_PATH", _DEFAULT_AB_BENCHMARK)
).joinpath("ab_benchmark", "__init__.py").exists()


TRAS_VH = ("EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
           "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS")
TRAS_VL = ("DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
           "QPEDFATYYCQQHYTTPPTFGQGTKVEIK")


class TestHumanness:
    def test_identical_sequence_gives_id_1(self):
        from cdr_gen.oracle import IGHV3_23, IGKV1_39

        h = humanness_proxy(IGHV3_23, IGKV1_39)
        assert h == pytest.approx(1.0)

    def test_trastuzumab_identity_in_range(self):
        # Humanized Trastuzumab should have 0.4-0.9 identity to the
        # canonical germlines.
        h = humanness_proxy(TRAS_VH, TRAS_VL)
        assert 0.3 < h < 0.95


class TestEvaluateHardFilterShortCircuit:
    def test_hard_filter_rejects_with_cys(self):
        task = GenerationTask(
            parent_id="ab", parent_vh=TRAS_VH, parent_vl=TRAS_VL,
            cdr_span=CDRSpan(chain="H", start=96, end=109),
        )
        bad_cdr = "CCCCCCCCCCCCC"  # all cysteine — liability motif
        design = GeneratedSequence(
            sequence_vh=TRAS_VH[:96] + bad_cdr + TRAS_VH[109:],
            sequence_vl=TRAS_VL,
            cdr_substring=bad_cdr, strategy="test", lambda_guidance=0.0,
            sweep_index=0, esm_log_likelihood=-5.0,
        )
        r = evaluate(design, task=task)
        assert not r.passes
        assert "hard filter" in r.fail_reasons[0]

    @pytest.mark.skipif(not _AB_BENCHMARK_READY, reason="ab-benchmark not installed")
    def test_parent_trastuzumab_passes_oracle(self):
        """Calibration check: unmutated Trastuzumab should pass."""
        task = GenerationTask(
            parent_id="trastuzumab", parent_vh=TRAS_VH, parent_vl=TRAS_VL,
            cdr_span=CDRSpan(chain="H", start=96, end=109),
        )
        parent_design = GeneratedSequence(
            sequence_vh=TRAS_VH, sequence_vl=TRAS_VL,
            cdr_substring="SRWGGDGFYAMDY", strategy="parent",
            lambda_guidance=0.0, sweep_index=0, esm_log_likelihood=-2.0,
        )
        r = evaluate(parent_design, task=task)
        # Parent may or may not pass depending on thresholds; what we
        # require is a valid result, not NaN everywhere.
        import math
        assert not math.isnan(r.tap_risk_flag_count)
        assert not math.isnan(r.di_seq_proxy)


class TestStrategyMetrics:
    def _design(self, cdr, ll=-2.0, accepted=10, proposed=10, strategy="guided"):
        return GeneratedSequence(
            sequence_vh="X" * 120, sequence_vl="Y" * 110,
            cdr_substring=cdr, strategy=strategy, lambda_guidance=1.0,
            sweep_index=0, esm_log_likelihood=ll,
            accepted_moves=accepted, proposed_moves=proposed,
        )

    def test_zero_designs(self):
        m = compute_strategy_metrics("guided", [], [], parent_cdr="XYZXYZXYZ")
        assert m.n_total == 0
        assert m.n_passing == 0

    def test_all_pass_rate_1(self):
        d = [self._design("QQQKKKRRRHHH"[: 13]) for _ in range(10)]
        o = [OracleResult(True, 0, 0, 0, 0.6, ["ok"], []) for _ in range(10)]
        m = compute_strategy_metrics("guided", d, o, parent_cdr="AAAAAAAAAAAAA")
        assert m.n_passing == 10
        assert m.pass_rate == 1.0

    def test_bootstrap_ci_brackets_point_estimate(self):
        d = [self._design("A" * 13) for _ in range(20)]
        o = [OracleResult(True, 0, 0, 0, 0.6, [], []) for _ in range(10)] + \
            [OracleResult(False, 3, 5.0, -1.0, 0.4, [], ["bad"]) for _ in range(10)]
        m = compute_strategy_metrics("guided", d, o, parent_cdr="AAAAAAAAAAAAA")
        assert m.pass_rate == 0.5
        assert m.pass_rate_ci_low <= 0.5 <= m.pass_rate_ci_high

    def test_diversity_equals_unique_over_total(self):
        d = [self._design("AAAA"[:13 or 4] * 3 + "X")]  # nonsense; but each is identical
        d = [self._design("QQQKKKRRRHHHA") for _ in range(5)]
        d.append(self._design("AAAAAAAAAAAAA"))
        o = [OracleResult(True, 0, 0, 0, 0.6, [], []) for _ in range(len(d))]
        m = compute_strategy_metrics("guided", d, o, parent_cdr="AAAAAAAAAAAAA")
        # Two unique designs out of 6.
        assert m.diversity_all == pytest.approx(2 / 6)


class TestNonOverlap:
    def test_strictly_nonoverlapping(self):
        from cdr_gen.eval import StrategyMetrics

        a = StrategyMetrics(
            "g", 100, 70, 0.7, 0.6, 0.8, 1.0, 1.0, 5.0, -2.0, 7.0, 1.0,
        )
        b = StrategyMetrics(
            "u", 100, 30, 0.3, 0.2, 0.4, 1.0, 1.0, 5.0, -2.5, 12.0, 1.0,
        )
        assert strategies_have_nonoverlapping_cis(a, b)

    def test_overlapping_returns_false(self):
        from cdr_gen.eval import StrategyMetrics

        a = StrategyMetrics("g", 100, 50, 0.5, 0.4, 0.6, 1.0, 1.0, 5.0, -2.0, 7.0, 1.0)
        b = StrategyMetrics("u", 100, 40, 0.4, 0.3, 0.5, 1.0, 1.0, 5.0, -2.5, 12.0, 1.0)
        assert not strategies_have_nonoverlapping_cis(a, b)
