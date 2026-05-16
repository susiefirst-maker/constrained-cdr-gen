"""Independent developability oracle.

Called only after generation. Uses the sibling ab-benchmark package
for TAP / DI / CamSol; adds a simple humanness proxy.

Thresholds are calibrated so that parent Trastuzumab passes. If a new
parent doesn't pass, the thresholds can be loosened per-task.
"""

from __future__ import annotations

import sys
from pathlib import Path
import os

from cdr_gen.hard_filters import all_hard_filters
from cdr_gen.schema import (
    GeneratedSequence,
    GenerationTask,
    OracleResult,
)


_PROJECTS_ROOT = Path(__file__).resolve().parents[3]
_AB_BENCHMARK = Path(
    os.environ.get("AB_BENCHMARK_PATH", str(_PROJECTS_ROOT / "ab-benchmark"))
)


def _ensure_ab_benchmark_on_path() -> None:
    if _AB_BENCHMARK.is_dir() and str(_AB_BENCHMARK) not in sys.path:
        sys.path.append(str(_AB_BENCHMARK))


# Canonical germlines for humanness proxy (matches struct-devpred Phase 4).
IGHV3_23 = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTIS"
            "RDNSKNTLYLQMNSLRAEDTAVYYCAK")
IGKV1_39 = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQ"
            "SYSTP")


def humanness_proxy(vh: str, vl: str) -> float:
    """Average Hamming identity of VH vs IGHV3-23 and VL vs IGKV1-39."""
    def _id(a: str, b: str) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        return sum(1 for i in range(n) if a[i] == b[i]) / n
    return 0.5 * (_id(vh, IGHV3_23) + _id(vl, IGKV1_39))


# ---------------------------------------------------------------------------


# Calibrated so Trastuzumab parent passes.
DEFAULT_THRESHOLDS = {
    "tap_risk_flag_count_max": 2,          # passes if count <= 2
    "di_seq_proxy_max": 3.0,                # passes if <= 3.0
    "camsol_intrinsic_mean_min": -0.3,      # passes if >= -0.3
    "humanness_min": 0.45,                  # passes if >= 0.45
}


def evaluate(
    design: GeneratedSequence,
    task: GenerationTask | None = None,
    thresholds: dict | None = None,
) -> OracleResult:
    """Score a single generated sequence against the oracle.

    If `task` is provided, hard filters are applied first (matching
    the sampler's own filters, as a sanity check). Failed hard filters
    result in passes=False with a note.
    """
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    if task is not None:
        hf = all_hard_filters(design.cdr_substring, task)
        if not hf.passes:
            return OracleResult(
                passes=False,
                tap_risk_flag_count=float("nan"),
                di_seq_proxy=float("nan"),
                camsol_intrinsic_mean=float("nan"),
                humanness_identity=float("nan"),
                fail_reasons=[f"hard filter: {hf.reason}"],
                notes="rejected before oracle",
            )

    _ensure_ab_benchmark_on_path()
    try:
        from ab_benchmark.baselines.camsol import compute_camsol_intrinsic
        from ab_benchmark.baselines.developability_index import compute_developability_index
        from ab_benchmark.baselines.tap import compute_tap
        from ab_benchmark.schema import AntibodyRecord, SourceDataset
    except ImportError as e:
        return OracleResult(
            passes=False,
            tap_risk_flag_count=float("nan"),
            di_seq_proxy=float("nan"),
            camsol_intrinsic_mean=float("nan"),
            humanness_identity=float("nan"),
            fail_reasons=["oracle unavailable"],
            notes=f"ab_benchmark import failed: {e}",
        )

    record = AntibodyRecord(
        ab_id=f"{task.parent_id if task else 'design'}_gen",
        source=SourceDataset.JAIN_2017,
        vh=design.sequence_vh, vl=design.sequence_vl,
    )
    tap_res = compute_tap(record)
    di_res = compute_developability_index(record)
    cam_res = compute_camsol_intrinsic(record)
    hum = humanness_proxy(design.sequence_vh, design.sequence_vl)

    tap_flags = (
        float(tap_res.metrics.get("tap_risk_flag_count", 99)) if tap_res.available else float("nan")
    )
    di_sp = float(di_res.metrics.get("di_seq_proxy", 99)) if di_res.available else float("nan")
    cam_mean = float(cam_res.metrics.get("camsol_intrinsic_mean", -99)) if cam_res.available else float("nan")

    pass_reasons: list[str] = []
    fail_reasons: list[str] = []

    import math
    tap_ok = not math.isnan(tap_flags) and tap_flags <= thr["tap_risk_flag_count_max"]
    di_ok = not math.isnan(di_sp) and di_sp <= thr["di_seq_proxy_max"]
    cam_ok = not math.isnan(cam_mean) and cam_mean >= thr["camsol_intrinsic_mean_min"]
    hum_ok = hum >= thr["humanness_min"]

    (pass_reasons if tap_ok else fail_reasons).append(
        f"TAP {'≤' if tap_ok else '>'} {thr['tap_risk_flag_count_max']} (got {tap_flags:.0f})"
    )
    (pass_reasons if di_ok else fail_reasons).append(
        f"DI {'≤' if di_ok else '>'} {thr['di_seq_proxy_max']:.1f} (got {di_sp:+.2f})"
    )
    (pass_reasons if cam_ok else fail_reasons).append(
        f"CamSol {'≥' if cam_ok else '<'} {thr['camsol_intrinsic_mean_min']:+.2f} (got {cam_mean:+.3f})"
    )
    (pass_reasons if hum_ok else fail_reasons).append(
        f"humanness {'≥' if hum_ok else '<'} {thr['humanness_min']:.2f} (got {hum:.2f})"
    )

    passes_all = tap_ok and di_ok and cam_ok and hum_ok
    return OracleResult(
        passes=passes_all,
        tap_risk_flag_count=tap_flags,
        di_seq_proxy=di_sp,
        camsol_intrinsic_mean=cam_mean,
        humanness_identity=hum,
        pass_reasons=pass_reasons,
        fail_reasons=fail_reasons,
    )


def evaluate_batch(
    designs: list[GeneratedSequence],
    task: GenerationTask | None = None,
    thresholds: dict | None = None,
) -> list[OracleResult]:
    return [evaluate(d, task=task, thresholds=thresholds) for d in designs]
