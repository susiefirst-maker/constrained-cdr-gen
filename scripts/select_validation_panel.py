#!/usr/bin/env python3
"""Create a wet-lab validation-panel CSV from generated candidate output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cdr_gen.validation_panel import PanelConfig, select_validation_panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("reports/trastuzumab_candidates.csv"),
        help="Candidate CSV emitted by `python -m cdr_gen.cli`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/validation_panel.csv"),
        help="Output validation-panel CSV.",
    )
    parser.add_argument("--guided-top-n", type=int, default=10)
    parser.add_argument("--guided-diverse-n", type=int, default=5)
    parser.add_argument("--unguided-control-n", type=int, default=4)
    parser.add_argument("--negative-control-n", type=int, default=3)
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    panel = select_validation_panel(
        candidates,
        PanelConfig(
            guided_top_n=args.guided_top_n,
            guided_diverse_n=args.guided_diverse_n,
            unguided_control_n=args.unguided_control_n,
            negative_control_n=args.negative_control_n,
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    print(f"Wrote {len(panel)} validation-panel rows to {args.out}")
    print(panel["panel_group"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
