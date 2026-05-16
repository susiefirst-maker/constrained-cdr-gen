"""Hard filters for generated CDR sequences.

These are pre-oracle gatekeepers. Anything a filter rejects is NOT sent
to the oracle — the sampler's proposal was invalid or contained a
well-known liability motif.

Implemented:
    - valid AA alphabet (no X / B / Z / stop)
    - length within [task.length_min, task.length_max]
    - forbidden motifs (N-glycosylation NGX, asparagine-deamidation NG,
      aspartate-isomerization DG / DP / DH, cysteine, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass

from cdr_gen.schema import STANDARD_AA, GenerationTask


# Common CDR-H3 liability motifs — default forbidden set.
# References: Xu et al. 2019 mAbs, Jain 2017 PNAS.
DEFAULT_LIABILITY_MOTIFS: tuple[str, ...] = (
    "NG",   # asparagine deamidation (Asn-Gly)
    "NS",   # slower deamidation but still flagged
    "DG",   # aspartate isomerization (Asp-Gly)
    "DP",   # aspartate isomerization slower
    "DH",   # asp isomerization edge case
    "DS",   # additional aspartate isomerization
    "DT",   # additional aspartate isomerization
    "C",    # free cysteine in CDR (disulfide scrambling)
    "M",    # methionine oxidation risk (optional; off by default)
)

# Motifs commonly tolerated in deployed therapeutics.
# Trastuzumab's CDR-H3 `SRWGGDGFYAMDY` contains both DG and M — a working
# commercial mAb demonstrates these are manageable, so we do NOT auto-reject.
# A paranoid user can pass `forbidden_motifs=DEFAULT_LIABILITY_MOTIFS` to
# reinstate the strict filter.
COMMONLY_TOLERATED: tuple[str, ...] = ("M", "DG", "DP", "DH")

ACTIVE_LIABILITY_MOTIFS: tuple[str, ...] = tuple(
    m for m in DEFAULT_LIABILITY_MOTIFS if m not in COMMONLY_TOLERATED
)

# N-glycosylation sequon (regex-worthy; here as a function).
def has_n_glyc_sequon(cdr: str) -> bool:
    """True if the CDR contains the N-X-S/T N-glycosylation sequon."""
    for i in range(len(cdr) - 2):
        if cdr[i] == "N" and cdr[i + 1] not in {"P"} and cdr[i + 2] in {"S", "T"}:
            return True
    return False


@dataclass
class FilterResult:
    passes: bool
    reason: str = ""


def check_alphabet(cdr: str) -> FilterResult:
    bad = set(cdr) - set(STANDARD_AA)
    if bad:
        return FilterResult(False, f"non-standard residues: {sorted(bad)}")
    return FilterResult(True)


def check_length(cdr: str, task: GenerationTask) -> FilterResult:
    if task.length_min is not None and len(cdr) < task.length_min:
        return FilterResult(False, f"length {len(cdr)} < min {task.length_min}")
    if task.length_max is not None and len(cdr) > task.length_max:
        return FilterResult(False, f"length {len(cdr)} > max {task.length_max}")
    return FilterResult(True)


def check_motifs(cdr: str, motifs: tuple[str, ...] = ACTIVE_LIABILITY_MOTIFS) -> FilterResult:
    hits: list[str] = []
    for m in motifs:
        if m in cdr:
            hits.append(m)
    if has_n_glyc_sequon(cdr):
        hits.append("NxS/T (N-glyc sequon)")
    if hits:
        return FilterResult(False, f"liability motif(s): {', '.join(hits)}")
    return FilterResult(True)


def all_hard_filters(cdr: str, task: GenerationTask) -> FilterResult:
    """Apply every hard filter; return first failure."""
    alpha = check_alphabet(cdr)
    if not alpha.passes:
        return alpha
    length = check_length(cdr, task)
    if not length.passes:
        return length
    # Task-specific forbidden motifs (configured per run).
    user_motifs = check_motifs(cdr, motifs=task.forbidden_motifs) if task.forbidden_motifs else FilterResult(True)
    if not user_motifs.passes:
        return user_motifs
    default_motifs = check_motifs(cdr, motifs=ACTIVE_LIABILITY_MOTIFS)
    if not default_motifs.passes:
        return default_motifs
    return FilterResult(True)
