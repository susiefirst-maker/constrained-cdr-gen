"""Core schema: GenerationTask, GeneratedSequence, OracleResult.

A `GenerationTask` defines WHERE to generate (which CDR on which parent)
and WHAT is valid. `GeneratedSequence` is one sampler output; the
evaluator adds oracle pass/fail, property scores, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class CDRSpan:
    """Half-open [start, end) positions within a chain sequence."""

    chain: str          # "H" or "L"
    start: int          # 0-indexed
    end: int            # exclusive

    def __post_init__(self) -> None:
        if self.chain not in {"H", "L"}:
            raise ValueError(f"chain must be 'H' or 'L', got {self.chain!r}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class GenerationTask:
    """Specifies a generation job.

    `parent_vh` + `parent_vl` are the framework; positions within
    `cdr_span` are redesigned. All positions outside the span are held
    fixed (no framework mutations in Phase 5).
    """

    parent_id: str
    parent_vh: str
    parent_vl: str
    cdr_span: CDRSpan

    # Length bounds for the generated CDR (for future insertions/deletions).
    # Phase 5 keeps length fixed; these default to the parent CDR length.
    length_min: int | None = None
    length_max: int | None = None

    # Motif-level hard filters; case-sensitive substring matches are banned.
    forbidden_motifs: tuple[str, ...] = ()

    # Which AAs are allowed at each position (defaults to all 20).
    allowed_aas: str = STANDARD_AA

    def __post_init__(self) -> None:
        if self.cdr_span.chain == "H":
            chain = self.parent_vh
        else:
            chain = self.parent_vl
        if self.cdr_span.end > len(chain):
            raise ValueError(
                f"cdr_span end {self.cdr_span.end} > chain length {len(chain)}"
            )
        if self.length_min is None:
            self.length_min = self.cdr_span.length
        if self.length_max is None:
            self.length_max = self.cdr_span.length
        bad = set(self.allowed_aas) - set(STANDARD_AA)
        if bad:
            raise ValueError(f"allowed_aas contains non-standard: {bad}")

    @property
    def parent_cdr(self) -> str:
        chain = self.parent_vh if self.cdr_span.chain == "H" else self.parent_vl
        return chain[self.cdr_span.start : self.cdr_span.end]


@dataclass
class GeneratedSequence:
    """One generation result (pre-oracle)."""

    sequence_vh: str
    sequence_vl: str
    cdr_substring: str           # the redesigned CDR residues
    strategy: str                # "random" | "unguided" | "guided" | "nn_retrieval"
    lambda_guidance: float       # guidance strength (0.0 for non-guided)
    sweep_index: int             # which Gibbs sweep produced this
    esm_log_likelihood: float    # mean log-prob of CDR positions under ESM-2
    accepted_moves: int = 0      # MH accept count during sampling
    proposed_moves: int = 0      # MH propose count during sampling
    seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def mh_accept_rate(self) -> float:
        if self.proposed_moves == 0:
            return 1.0
        return self.accepted_moves / self.proposed_moves


@dataclass
class OracleResult:
    """Independent oracle's verdict on a generated sequence."""

    passes: bool
    tap_risk_flag_count: float
    di_seq_proxy: float
    camsol_intrinsic_mean: float
    humanness_identity: float
    pass_reasons: list[str] = field(default_factory=list)  # which checks passed
    fail_reasons: list[str] = field(default_factory=list)  # which failed
    notes: str = ""
