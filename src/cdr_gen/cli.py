"""CLI: generate + evaluate + compare strategies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cdr_gen.eval import (
    compute_strategy_metrics,
    format_strategy_comparison,
    strategies_have_nonoverlapping_cis,
)
from cdr_gen.mlm import DEFAULT_LINKER, MaskedLM
from cdr_gen.oracle import evaluate_batch
from cdr_gen.sampler import SamplerConfig, sample, sample_nn_retrieval
from cdr_gen.schema import CDRSpan, GenerationTask


# Trastuzumab fixtures (match Phases 1–4).
TRAS_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNT"
    "AYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
TRAS_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSL"
    "QPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)

# Trastuzumab CDR-H3: SRWGGDGFYAMDY, span [96, 109), 13 residues.
# (Confirmed by ab_benchmark.seqprops.extract_cdrs on the sequence above.)
TRAS_CDR_H3_START = 96
TRAS_CDR_H3_END = 109


def _get_trastuzumab_task() -> GenerationTask:
    return GenerationTask(
        parent_id="trastuzumab",
        parent_vh=TRAS_VH,
        parent_vl=TRAS_VL,
        cdr_span=CDRSpan(chain="H", start=TRAS_CDR_H3_START, end=TRAS_CDR_H3_END),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", default="trastuzumab", help="Only 'trastuzumab' is built-in.")
    ap.add_argument("--n-designs", type=int, default=200)
    ap.add_argument(
        "--strategies", nargs="+",
        default=["random", "unguided", "guided"],
        help="Subset of {random, unguided, guided, nn_retrieval}.",
    )
    ap.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    ap.add_argument("--n-sweeps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)

    if args.parent != "trastuzumab":
        print("Only 'trastuzumab' parent is built-in for Phase 5.", file=sys.stderr)
        return 1

    task = _get_trastuzumab_task()
    parent_cdr = task.parent_cdr
    print(f"Parent: {task.parent_id}")
    print(f"CDR-H3 span: [{task.cdr_span.start}, {task.cdr_span.end})  len={task.cdr_span.length}")
    print(f"Parent CDR: {parent_cdr!r}\n")

    mlm = MaskedLM(device=args.device, linker=DEFAULT_LINKER)

    metrics_list = []
    oracle_rows = []

    nn_library = [parent_cdr]  # tiny library; real runs would pull OAS CDR-H3s

    for strategy in args.strategies:
        print(f"[{strategy}] generating {args.n_designs} designs...")
        if strategy == "nn_retrieval":
            designs = sample_nn_retrieval(task, nn_library, n_designs=args.n_designs, seed=args.seed)
        else:
            cfg = SamplerConfig(
                strategy=strategy,
                n_sweeps=args.n_sweeps if strategy != "random" else 0,
                lambda_guidance=args.lambda_ if strategy == "guided" else 0.0,
                temperature=args.temperature,
                seed=args.seed,
            )
            designs = sample(task, mlm, config=cfg, n_designs=args.n_designs)

        print(f"[{strategy}] oracle evaluation...")
        results = evaluate_batch(designs, task=task)
        metrics = compute_strategy_metrics(strategy, designs, results, parent_cdr)
        metrics_list.append(metrics)

        for d, r in zip(designs, results):
            oracle_rows.append({
                "strategy": strategy,
                "cdr": d.cdr_substring,
                "passes": r.passes,
                "tap_risk": r.tap_risk_flag_count,
                "di_seq": r.di_seq_proxy,
                "camsol_mean": r.camsol_intrinsic_mean,
                "humanness": r.humanness_identity,
                "esm_log_lik": d.esm_log_likelihood,
                "mh_accept": d.mh_accept_rate,
                "fail_reasons": "; ".join(r.fail_reasons),
            })

    # Output CSV + summary JSON.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.parent}_candidates.csv"
    pd.DataFrame(oracle_rows).to_csv(csv_path, index=False)

    summary = {
        "parent_id": task.parent_id,
        "cdr_span": [task.cdr_span.start, task.cdr_span.end],
        "parent_cdr": parent_cdr,
        "n_designs_per_strategy": args.n_designs,
        "lambda_guidance": args.lambda_,
        "n_sweeps": args.n_sweeps,
        "strategies": [asdict(m) for m in metrics_list],
    }
    # Non-overlapping CI flags.
    if all(s.strategy in {m.strategy for m in metrics_list} for s in metrics_list):
        by = {m.strategy: m for m in metrics_list}
        comparisons = []
        if "guided" in by and "unguided" in by:
            g, u = by["guided"], by["unguided"]
            comparisons.append({
                "compare": "guided_vs_unguided",
                "pass_rate_guided": g.pass_rate,
                "pass_rate_unguided": u.pass_rate,
                "ratio": (g.pass_rate / u.pass_rate) if u.pass_rate > 0 else None,
                "ci_nonoverlap": strategies_have_nonoverlapping_cis(g, u),
                "meets_2x_target": (u.pass_rate > 0 and (g.pass_rate / u.pass_rate) >= 2.0
                                    and strategies_have_nonoverlapping_cis(g, u)),
            })
        if "guided" in by and "random" in by:
            g, r = by["guided"], by["random"]
            comparisons.append({
                "compare": "guided_vs_random",
                "pass_rate_guided": g.pass_rate,
                "pass_rate_random": r.pass_rate,
                "ci_nonoverlap": strategies_have_nonoverlapping_cis(g, r),
            })
        summary["comparisons"] = comparisons

    json_path = args.out_dir / f"{args.parent}_results.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72)
    print("Strategy comparison")
    print("=" * 72)
    print(format_strategy_comparison(metrics_list))

    if "comparisons" in summary:
        print("\nComparisons vs targets:")
        for c in summary["comparisons"]:
            print(f"  {c}")

    print(f"\nWritten: {csv_path}")
    print(f"         {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
