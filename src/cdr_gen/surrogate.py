"""Lightweight physicochemical surrogate (guidance-only; not circular).

Scores a CDR residue (or full CDR) using first-principles biochemistry:
  - Kyte-Doolittle hydrophobicity (Kyte & Doolittle 1982)
  - Henderson-Hasselbalch net charge at pH 7

No ML training. No fitting to any developability dataset. This keeps the
guidance signal independent from the evaluation oracle (which is
ab-benchmark's TAP/DI/CamSol).

Semantics: `score_full` returns a scalar where HIGHER = more
developable. The sampler multiplies exp(λ · score) onto the ESM-2
proposal distribution; with λ = 0 the sampler is pure ESM-2.
"""

from __future__ import annotations

# Kyte-Doolittle 1982 (J. Mol. Biol. 157:105).
KYTE_DOOLITTLE = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}

# Side-chain pKa values.
PKA = {"D": 3.65, "E": 4.25, "H": 6.0, "K": 10.53, "R": 12.48, "C": 8.33, "Y": 10.07}


def _charge_at(aa: str, ph: float) -> float:
    """Signed partial charge contribution at pH (Henderson-Hasselbalch)."""
    pk = PKA.get(aa)
    if pk is None:
        return 0.0
    if aa in {"K", "R", "H"}:
        return 1.0 / (1.0 + 10 ** (ph - pk))
    return -1.0 / (1.0 + 10 ** (pk - ph))


def net_charge(seq: str, ph: float = 7.0) -> float:
    q = 1.0 / (1.0 + 10 ** (ph - 8.0))  # N-terminal amine
    q -= 1.0 / (1.0 + 10 ** (3.1 - ph))  # C-terminal carboxyl
    for aa in seq:
        q += _charge_at(aa, ph)
    return q


def mean_hydrophobicity(seq: str) -> float:
    if not seq:
        return 0.0
    vals = [KYTE_DOOLITTLE.get(a, 0.0) for a in seq]
    return sum(vals) / len(vals)


def score_full(cdr: str) -> float:
    """Global CDR developability score. Higher = better.

    Continuous-penalty formulation so every substitution perturbs the
    score (stepped `max(0, …)` penalties produced zero gradient for
    gentle substitutions and made the sampler unable to choose between
    candidates). Coefficients by inspection; NOT fit to any biologics
    dataset.
    """
    if not cdr:
        return 0.0
    mh = mean_hydrophobicity(cdr)
    q = net_charge(cdr, ph=7.0)

    # longest consecutive run above hydrophobicity threshold (patchiness)
    run = best = 0
    for a in cdr:
        if KYTE_DOOLITTLE.get(a, 0) > 1.0:
            run += 1
            best = max(best, run)
        else:
            run = 0

    # Continuous soft penalties.
    hydro_cost = 0.10 * mh * mh + 0.30 * max(0.0, mh)
    charge_cost = 0.05 * q * q
    patch_cost = 0.40 * max(0, best - 3) + 0.02 * best

    return -(hydro_cost + charge_cost + patch_cost)


def score_residue_in_context(cdr: str, position: int, candidate: str) -> float:
    """Score what happens if we place `candidate` at `position` in the CDR.

    This is the per-residue surrogate used by the Gibbs sampler: it
    computes the full-CDR score with the substitution applied and
    returns it. Fast (O(len(cdr)) per call).
    """
    if position < 0 or position >= len(cdr):
        raise ValueError(f"position {position} out of CDR range [0, {len(cdr)})")
    if candidate not in KYTE_DOOLITTLE:
        return -1e9  # poison non-standard proposals
    mutated = cdr[:position] + candidate + cdr[position + 1 :]
    return score_full(mutated)
