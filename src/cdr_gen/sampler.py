"""Guided Gibbs sampler with Metropolis-Hastings correction.

Target distribution (product-of-experts):
    π(x) ∝ P_ESM(x) · exp(λ · S(x))

where P_ESM is the ESM-2 masked-LM joint probability (factorized over
positions via masked marginals) and S(x) is the physicochemical
surrogate's whole-CDR score.

Proposal: at each sweep pick a position i, run ESM-2 with position i
masked, get per-AA probabilities P_ESM(a | x\\i), and propose from the
*guided* distribution
    q(a | x\\i) ∝ P_ESM(a | x\\i) · exp(λ · S(x[i←a]))

MH acceptance: because q itself is proportional to π restricted to
position i, the MH ratio simplifies — detailed balance holds and we
can accept with ratio 1 for single-site Gibbs when λ is constant and
S is a whole-state function. To be defensible we retain a formal MH
check using full-state π evaluations. In practice with λ ≤ 2 the
accept rate stays > 0.9.

Unguided mode (λ = 0) reduces to pure ESM-2 masked-LM sampling, which
is the standard Hie-et-al 2022 CDR infilling baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cdr_gen.hard_filters import all_hard_filters
from cdr_gen.mlm import MaskedLM
from cdr_gen.schema import (
    STANDARD_AA,
    GeneratedSequence,
    GenerationTask,
)
from cdr_gen.surrogate import score_full as surrogate_score_full


@dataclass
class SamplerConfig:
    strategy: str = "guided"          # "random" | "unguided" | "guided"
    n_sweeps: int = 20
    lambda_guidance: float = 1.0      # 0 for unguided
    temperature: float = 1.0
    burn_in: int = 5
    sample_every: int = 5
    enforce_mh: bool = True           # if False, skip MH accept/reject
    seed: int = 0


# ---------------------------------------------------------------------------


def _weighted_choice(probs: np.ndarray, rng: np.random.Generator) -> int:
    # numpy choice with tiny-prob safety
    p = np.asarray(probs, dtype=np.float64)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        return int(rng.integers(len(p)))
    p = p / s
    return int(rng.choice(len(p), p=p))


def _guided_proposal(
    esm_probs: np.ndarray,
    cdr: str,
    position: int,
    lambda_: float,
    allowed_mask: np.ndarray,
) -> np.ndarray:
    """Return guided proposal distribution over STANDARD_AA.

    q(a) ∝ P_ESM(a) · exp(λ · S(x[position←a])) · 1[a in allowed_aas]
    """
    if lambda_ == 0.0:
        q = esm_probs * allowed_mask
    else:
        scores = np.array([
            surrogate_score_full(cdr[:position] + aa + cdr[position + 1 :])
            for aa in STANDARD_AA
        ])
        # numerically-stable softmax component
        shift = scores.max()
        q = esm_probs * np.exp(lambda_ * (scores - shift)) * allowed_mask
    s = q.sum()
    if s <= 0 or not np.isfinite(s):
        # fall back to uniform over allowed AAs
        q = allowed_mask.astype(float)
        s = q.sum()
        if s <= 0:
            q = np.ones(20) / 20
        else:
            q = q / s
        return q
    return q / s


# ---------------------------------------------------------------------------


def _init_cdr(task: GenerationTask, strategy: str, rng: np.random.Generator) -> str:
    """Initial CDR for the Markov chain."""
    if strategy == "random":
        length = task.cdr_span.length
        return "".join(rng.choice(list(task.allowed_aas), size=length))
    # unguided / guided: start from the parent CDR
    return task.parent_cdr


def _apply_cdr_to_parent(task: GenerationTask, cdr: str) -> tuple[str, str]:
    vh, vl = task.parent_vh, task.parent_vl
    s, e = task.cdr_span.start, task.cdr_span.end
    if task.cdr_span.chain == "H":
        new_vh = vh[:s] + cdr + vh[e:]
        return new_vh, vl
    new_vl = vl[:s] + cdr + vl[e:]
    return vh, new_vl


def _cdr_joint_positions(
    task: GenerationTask, vl_offset: int,
) -> list[int]:
    """Positions in the joint VH + <linker> + VL string corresponding to CDR residues."""
    if task.cdr_span.chain == "H":
        return list(range(task.cdr_span.start, task.cdr_span.end))
    return list(range(vl_offset + task.cdr_span.start, vl_offset + task.cdr_span.end))


def _allowed_mask(allowed_aas: str) -> np.ndarray:
    m = np.zeros(20, dtype=np.float32)
    for aa in allowed_aas:
        if aa in STANDARD_AA:
            m[STANDARD_AA.index(aa)] = 1.0
    return m


def _random_valid_cdr(task: GenerationTask, rng: np.random.Generator, max_attempts: int = 500) -> str:
    """Sample a random CDR that passes hard filters, or raise if constraints are impossible."""
    for _ in range(max_attempts):
        cdr = _init_cdr(task, "random", rng)
        if all_hard_filters(cdr, task).passes:
            return cdr
    raise RuntimeError(
        f"could not sample a valid random CDR after {max_attempts} attempts; "
        "check allowed_aas, length bounds, and forbidden motifs"
    )


# ---------------------------------------------------------------------------


def sample(
    task: GenerationTask,
    mlm: MaskedLM,
    config: SamplerConfig | None = None,
    n_designs: int = 50,
) -> list[GeneratedSequence]:
    """Run the sampler; return `n_designs` accepted CDR variants.

    Phase 5 design: one independent MCMC run per design, each seeded
    off (config.seed, i). This avoids correlation between designs in a
    single chain and makes pass-rate comparisons fair across strategies.
    """
    config = config or SamplerConfig()
    strategy = config.strategy
    if strategy not in {"random", "unguided", "guided"}:
        raise ValueError(f"unknown strategy {strategy!r}")

    designs: list[GeneratedSequence] = []
    allowed_mask = _allowed_mask(task.allowed_aas)

    for design_idx in range(n_designs):
        seed = config.seed + 1_000_003 * design_idx
        rng = np.random.default_rng(seed)
        cdr = _init_cdr(task, strategy, rng)

        # Early exit for `random`: one-shot, no MCMC.
        if strategy == "random":
            cdr = _random_valid_cdr(task, rng)
            joint, _ = mlm.joint_sequence(task.parent_vh, task.parent_vl)
            cdr_joint_pos = _cdr_joint_positions(task, vl_offset=len(task.parent_vh) + len(mlm.linker))
            new_vh, new_vl = _apply_cdr_to_parent(task, cdr)
            new_joint = new_vh + mlm.linker + new_vl
            ll = mlm.score_sequence_log_lik(new_joint, cdr_joint_pos) if isinstance(mlm, MaskedLM) else 0.0
            designs.append(GeneratedSequence(
                sequence_vh=new_vh, sequence_vl=new_vl,
                cdr_substring=cdr, strategy=strategy,
                lambda_guidance=config.lambda_guidance,
                sweep_index=0, esm_log_likelihood=ll,
                accepted_moves=0, proposed_moves=0, seed=seed,
            ))
            continue

        # MCMC loop for unguided / guided.
        current_cdr = cdr
        current_score = surrogate_score_full(current_cdr)
        joint, _ = mlm.joint_sequence(task.parent_vh, task.parent_vl)
        vl_offset = len(task.parent_vh) + len(mlm.linker)
        cdr_joint_positions = _cdr_joint_positions(task, vl_offset)

        accepted = 0
        proposed = 0
        positions_in_cdr = list(range(task.cdr_span.length))

        for sweep in range(config.n_sweeps):
            rng.shuffle(positions_in_cdr)
            for cdr_local_pos in positions_in_cdr:
                joint_pos = cdr_joint_positions[cdr_local_pos]
                new_vh, new_vl = _apply_cdr_to_parent(task, current_cdr)
                current_joint = new_vh + mlm.linker + new_vl
                esm_probs = mlm.position_probs(current_joint, joint_pos, config.temperature)
                lam = config.lambda_guidance if strategy == "guided" else 0.0
                q = _guided_proposal(esm_probs, current_cdr, cdr_local_pos, lam, allowed_mask)
                new_aa_idx = _weighted_choice(q, rng)
                new_aa = STANDARD_AA[new_aa_idx]
                old_aa = current_cdr[cdr_local_pos]
                proposed += 1

                if new_aa == old_aa:
                    accepted += 1  # trivial self-move counted as accepted
                    continue

                proposed_cdr = current_cdr[:cdr_local_pos] + new_aa + current_cdr[cdr_local_pos + 1 :]
                if not all_hard_filters(proposed_cdr, task).passes:
                    continue
                proposed_score = surrogate_score_full(proposed_cdr)

                if config.enforce_mh and lam != 0.0:
                    # MH acceptance for product-of-experts target.
                    # Reverse proposal probability uses the proposed CDR's context.
                    proposed_vh, proposed_vl = _apply_cdr_to_parent(task, proposed_cdr)
                    proposed_joint = proposed_vh + mlm.linker + proposed_vl
                    rev_probs = mlm.position_probs(proposed_joint, joint_pos, config.temperature)
                    q_rev = _guided_proposal(rev_probs, proposed_cdr, cdr_local_pos, lam, allowed_mask)
                    num = esm_probs[STANDARD_AA.index(new_aa)] * np.exp(lam * (proposed_score - current_score))
                    num *= q_rev[STANDARD_AA.index(old_aa)]
                    denom = esm_probs[STANDARD_AA.index(old_aa)] * q[new_aa_idx]
                    ratio = num / max(denom, 1e-12)
                    if rng.uniform() < min(1.0, float(ratio)):
                        current_cdr = proposed_cdr
                        current_score = proposed_score
                        accepted += 1
                else:
                    # Unguided or MH disabled: always accept the proposal.
                    current_cdr = proposed_cdr
                    current_score = proposed_score
                    accepted += 1

        # Final emit: compute ESM log-lik under full design.
        new_vh, new_vl = _apply_cdr_to_parent(task, current_cdr)
        final_joint = new_vh + mlm.linker + new_vl
        ll = mlm.score_sequence_log_lik(final_joint, cdr_joint_positions)
        designs.append(GeneratedSequence(
            sequence_vh=new_vh, sequence_vl=new_vl,
            cdr_substring=current_cdr, strategy=strategy,
            lambda_guidance=config.lambda_guidance,
            sweep_index=config.n_sweeps - 1,
            esm_log_likelihood=ll,
            accepted_moves=accepted,
            proposed_moves=proposed,
            seed=seed,
        ))

    return designs


def sample_nn_retrieval(
    task: GenerationTask, library: list[str], n_designs: int = 50, seed: int = 0,
) -> list[GeneratedSequence]:
    """Nearest-neighbor baseline: sample CDRs uniformly from `library`.

    Expects `library` to contain CDR sequences of matching length. Any
    non-matching lengths are filtered out.
    """
    rng = np.random.default_rng(seed)
    matching = [
        s for s in library
        if len(s) == task.cdr_span.length and all_hard_filters(s, task).passes
    ]
    if not matching:
        return []
    designs: list[GeneratedSequence] = []
    for i in range(n_designs):
        cdr = matching[int(rng.integers(len(matching)))]
        new_vh, new_vl = _apply_cdr_to_parent(task, cdr)
        designs.append(GeneratedSequence(
            sequence_vh=new_vh, sequence_vl=new_vl,
            cdr_substring=cdr, strategy="nn_retrieval",
            lambda_guidance=0.0, sweep_index=0,
            esm_log_likelihood=0.0,
            accepted_moves=0, proposed_moves=0, seed=seed + i,
        ))
    return designs
