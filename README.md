# constrained-cdr-gen

> Property-guided Gibbs-style sampling for antibody CDR-H3 design with independent oracle evaluation.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)

## Key results

| Strategy | Pass rate | 95% CI | ESM-2 log-likelihood |
|----------|-----------|--------|----------------------|
| Random AA substitution | 44% (35/80) | -- | -3.44 |
| Unguided ESM-2 MLM | 65% (52/80) | -- | -2.68 |
| **Guided Gibbs + MH** | **85% (68/80)** | non-overlapping | **-2.21** |

Guided sampling achieves a 31% absolute improvement over random in the reproducible Trastuzumab run, with non-overlapping 95% CIs. Evaluation uses an **independent oracle** (TAP, DI, CamSol, humanness) that the sampler never sees.

## Problem

CDR-H3 loop engineering requires balancing sequence naturalness with biophysical developability. Naive random mutagenesis produces mostly non-viable candidates. Evaluating guided designs with the same surrogate used for guidance creates circular validation that inflates apparent performance.

## Approach

Three-layer architecture with strict oracle independence. ESM-2 t12 masked LM provides sequence proposals. A physicochemical surrogate (Kyte-Doolittle hydrophobicity + charge) provides in-sampler guidance. A Metropolis-Hastings-style accept/reject check is used for guided moves. An independent oracle (TAP, DI, CamSol, humanness from ab-benchmark) evaluates candidates without ever being seen by the sampler.

Scope: the built-in CLI currently targets fixed-length Trastuzumab CDR-H3 redesign. It is a reproducible research benchmark, not a general antibody design platform.

## Experimental status

This repository reports computational validation only. The generated CDR-H3 candidates have **not** been wet-lab validated.

## Quick start

    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    # Optional UI: pip install -e ".[app]"
    python -m cdr_gen.cli --parent trastuzumab --n-designs 80 \
      --strategies random unguided guided --lambda 1.0 --n-sweeps 20 \
      --out-dir reports

## Reproduction

    python -m cdr_gen.cli --parent trastuzumab --n-designs 200 \
      --strategies random unguided guided --lambda 1.0 --n-sweeps 20 \
      --out-dir reports
    streamlit run app.py

Outputs: reports/trastuzumab_results.json (pass rates, CIs), reports/trastuzumab_candidates.csv (per-design oracle scores).

To create a candidate panel for follow-up review:

    python scripts/select_validation_panel.py \
      --candidates reports/trastuzumab_candidates.csv \
      --out reports/validation_panel.csv

If `ab-benchmark` is not installed as a package, set `AB_BENCHMARK_PATH=/path/to/ab-benchmark` before running the oracle. The default fallback assumes the sibling layout `../ab-benchmark`.

## Citation

    @software{wu2026constrainedcdrgen,
      author = {Wu, Di},
      title  = {constrained-cdr-gen: Property-Guided Gibbs Sampling for Antibody CDR Design},
      year   = {2026},
      url    = {https://github.com/diwuhub/constrained-cdr-gen}
    }

## References

- Hie et al. 2021 -- Evolutionary-scale masked language model proposals
- Metropolis et al. 1953; Hastings 1970 -- Metropolis-Hastings correction
- Raybould et al. 2019 PNAS -- TAP (Therapeutic Antibody Profiler)
- Sormanni et al. 2015 J. Mol. Biol. -- CamSol-intrinsic
- Lauer et al. 2012 J. Pharm. Sci. -- Developability Index

## License

MIT. See LICENSE.
