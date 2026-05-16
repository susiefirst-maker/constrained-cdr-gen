"""Generation-metrics evaluation + strategy comparison.

Given a list of (GeneratedSequence, OracleResult) pairs per strategy,
compute:
    - pass_rate  (fraction passing the oracle) with bootstrap 95% CI
    - diversity  (unique CDRs / total) among passing designs
    - novelty    (mean Hamming distance to parent CDR)
    - mean_esm_log_likelihood   (quality proxy; related to perplexity)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cdr_gen.schema import GeneratedSequence, OracleResult


@dataclass
class StrategyMetrics:
    strategy: str
    n_total: int
    n_passing: int
    pass_rate: float
    pass_rate_ci_low: float
    pass_rate_ci_high: float
    diversity_passing: float       # unique CDRs / |passing|
    diversity_all: float           # unique CDRs / |all|
    novelty_mean_hamming: float    # to parent CDR, all designs
    mean_esm_log_lik: float
    mean_esm_perplexity: float
    mh_accept_rate: float          # aggregate across MCMC designs

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _bootstrap_rate_ci(k: int, n: int, n_boot: int = 5000, alpha: float = 0.05,
                      random_state: int = 0) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    successes = np.concatenate([np.ones(k, dtype=int), np.zeros(n - k, dtype=int)])
    rates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rates[i] = successes[idx].mean()
    return (float(np.quantile(rates, alpha / 2)),
            float(np.quantile(rates, 1 - alpha / 2)))


def _hamming(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def compute_strategy_metrics(
    strategy: str,
    designs: list[GeneratedSequence],
    oracle_results: list[OracleResult],
    parent_cdr: str,
    bootstrap_seed: int = 0,
) -> StrategyMetrics:
    n = len(designs)
    if n == 0:
        return StrategyMetrics(
            strategy=strategy, n_total=0, n_passing=0,
            pass_rate=float("nan"), pass_rate_ci_low=float("nan"), pass_rate_ci_high=float("nan"),
            diversity_passing=float("nan"), diversity_all=float("nan"),
            novelty_mean_hamming=float("nan"),
            mean_esm_log_lik=float("nan"), mean_esm_perplexity=float("nan"),
            mh_accept_rate=float("nan"),
        )

    passing_designs = [d for d, r in zip(designs, oracle_results) if r.passes]
    k = len(passing_designs)
    pass_rate = k / n
    lo, hi = _bootstrap_rate_ci(k, n, random_state=bootstrap_seed)

    unique_passing = len({d.cdr_substring for d in passing_designs})
    div_pass = unique_passing / max(k, 1) if k > 0 else 0.0

    unique_all = len({d.cdr_substring for d in designs})
    div_all = unique_all / n

    novelty = float(np.mean([_hamming(d.cdr_substring, parent_cdr) for d in designs]))

    lls = [d.esm_log_likelihood for d in designs if d.esm_log_likelihood != 0.0]
    mean_ll = float(np.mean(lls)) if lls else float("nan")
    # Perplexity = exp(-mean log lik). Log-lik is mean per residue already.
    mean_ppl = float(np.exp(-mean_ll)) if not np.isnan(mean_ll) else float("nan")

    # MCMC strategies have nonzero proposed_moves; aggregate.
    accepts = [d.accepted_moves for d in designs if d.proposed_moves > 0]
    proposes = [d.proposed_moves for d in designs if d.proposed_moves > 0]
    mh = (sum(accepts) / max(sum(proposes), 1)) if proposes else float("nan")

    return StrategyMetrics(
        strategy=strategy,
        n_total=n, n_passing=k,
        pass_rate=pass_rate, pass_rate_ci_low=lo, pass_rate_ci_high=hi,
        diversity_passing=div_pass, diversity_all=div_all,
        novelty_mean_hamming=novelty,
        mean_esm_log_lik=mean_ll, mean_esm_perplexity=mean_ppl,
        mh_accept_rate=mh,
    )


def strategies_have_nonoverlapping_cis(
    a: StrategyMetrics, b: StrategyMetrics,
) -> bool:
    """True if A's pass_rate 95% CI does not overlap B's."""
    if np.isnan(a.pass_rate_ci_low) or np.isnan(b.pass_rate_ci_low):
        return False
    return a.pass_rate_ci_high < b.pass_rate_ci_low or b.pass_rate_ci_high < a.pass_rate_ci_low


def format_strategy_comparison(metrics_list: list[StrategyMetrics]) -> str:
    lines = [
        f"{'strategy':<16s} {'n':>4s} {'pass':>5s} {'pass_rate':>10s}  {'95% CI':<22s} "
        f"{'div_all':>8s} {'novelty':>8s} {'PPL':>7s}  {'MH%':>5s}"
    ]
    for m in metrics_list:
        lines.append(
            f"{m.strategy:<16s} {m.n_total:>4d} {m.n_passing:>5d} "
            f"{m.pass_rate:>9.3f}  [{m.pass_rate_ci_low:.2f}, {m.pass_rate_ci_high:.2f}]   "
            f"{m.diversity_all:>7.3f} {m.novelty_mean_hamming:>7.2f} "
            f"{m.mean_esm_perplexity:>7.2f}  "
            f"{m.mh_accept_rate * 100:>4.1f}"
            if not np.isnan(m.mh_accept_rate) else
            f"{m.strategy:<16s} {m.n_total:>4d} {m.n_passing:>5d} "
            f"{m.pass_rate:>9.3f}  [{m.pass_rate_ci_low:.2f}, {m.pass_rate_ci_high:.2f}]   "
            f"{m.diversity_all:>7.3f} {m.novelty_mean_hamming:>7.2f} "
            f"{m.mean_esm_perplexity:>7.2f}    n/a"
        )
    return "\n".join(lines)
