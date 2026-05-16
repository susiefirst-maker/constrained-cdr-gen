"""Select an experiment-ready validation panel from generated candidates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


PARENT_REFERENCE_ID = "parent_trastuzumab"


@dataclass(frozen=True)
class PanelConfig:
    """Target counts for a first-pass validation panel."""

    guided_top_n: int = 10
    guided_diverse_n: int = 5
    unguided_control_n: int = 4
    negative_control_n: int = 3
    parent_cdr: str = "SRWGGDGFYAMDY"


def hamming_distance(a: str, b: str) -> int:
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def _numeric_column(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        df[name] = default
    return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(float)


def _bool_column(df: pd.DataFrame, name: str) -> pd.Series:
    if df[name].dtype == bool:
        return df[name]
    return df[name].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passing"}
    )


def _with_derived_columns(candidates: pd.DataFrame, parent_cdr: str) -> pd.DataFrame:
    df = candidates.copy()
    if "candidate_id" not in df.columns:
        df.insert(0, "candidate_id", [f"cand_{i:04d}" for i in range(len(df))])
    df["passes"] = _bool_column(df, "passes")
    df["parent_hamming"] = df["cdr"].map(lambda cdr: hamming_distance(str(cdr), parent_cdr))
    # Higher is better: language-model likelihood, solubility proxy, humanness;
    # lower is better: TAP risk count and DI proxy.
    df["panel_score"] = (
        _numeric_column(df, "esm_log_lik", 0.0)
        + 0.20 * _numeric_column(df, "camsol_mean", 0.0)
        + 0.50 * _numeric_column(df, "humanness", 0.0)
        - 0.25 * _numeric_column(df, "tap_risk", 0.0)
        - 0.10 * _numeric_column(df, "di_seq", 0.0)
    )
    return df


def _rank(df: pd.DataFrame, *, highest_score_first: bool = True) -> pd.DataFrame:
    return df.sort_values(
        by=["panel_score", "esm_log_lik", "parent_hamming", "candidate_id"],
        ascending=[not highest_score_first, False, False, True],
    )


def _add_group(rows: list[pd.Series], group: str, reason: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).copy()
    out.insert(0, "panel_group", group)
    out.insert(1, "panel_rank", range(1, len(out) + 1))
    out.insert(2, "selection_reason", reason)
    return out


def _select_diverse(pool: pd.DataFrame, already_selected_cdrs: list[str], n: int) -> list[pd.Series]:
    selected: list[pd.Series] = []
    selected_cdrs = list(already_selected_cdrs)
    ranked_pool = _rank(pool)
    for _, row in ranked_pool.iterrows():
        cdr = str(row["cdr"])
        if cdr in selected_cdrs:
            continue
        if selected_cdrs:
            min_distance = min(hamming_distance(cdr, other) for other in selected_cdrs)
        else:
            min_distance = int(row.get("parent_hamming", 0))
        # Prefer candidates that are not near-duplicates of the top tier.
        if min_distance < 3:
            continue
        selected.append(row)
        selected_cdrs.append(cdr)
        if len(selected) >= n:
            break
    if len(selected) < n:
        for _, row in ranked_pool.iterrows():
            cdr = str(row["cdr"])
            if cdr in selected_cdrs:
                continue
            selected.append(row)
            selected_cdrs.append(cdr)
            if len(selected) >= n:
                break
    return selected


def select_validation_panel(candidates: pd.DataFrame, config: PanelConfig | None = None) -> pd.DataFrame:
    """Create a validation-panel table from CLI candidate output.

    The output intentionally mixes high-scoring guided designs, diverse guided
    designs, unguided controls, negative controls, and the parent reference.
    This is an experiment-selection artifact; it is not wet-lab evidence.
    """
    config = config or PanelConfig()
    required = {"strategy", "cdr", "passes"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table is missing required columns: {sorted(missing)}")

    df = _with_derived_columns(candidates, config.parent_cdr)
    df = df.drop_duplicates(subset=["strategy", "cdr"]).reset_index(drop=True)

    guided_pass = _rank(df[(df["strategy"] == "guided") & df["passes"]])
    guided_top_rows = [row for _, row in guided_pass.head(config.guided_top_n).iterrows()]
    top_cdrs = [str(row["cdr"]) for row in guided_top_rows]

    diverse_pool = guided_pass[~guided_pass["cdr"].isin(top_cdrs)]
    guided_diverse_rows = _select_diverse(diverse_pool, top_cdrs, config.guided_diverse_n)

    unguided_pass = _rank(df[(df["strategy"] == "unguided") & df["passes"]])
    unguided_rows = [row for _, row in unguided_pass.head(config.unguided_control_n).iterrows()]

    negative_pool = _rank(df[~df["passes"]], highest_score_first=False)
    negative_rows = [row for _, row in negative_pool.head(config.negative_control_n).iterrows()]

    parent_row = pd.Series({
        "candidate_id": PARENT_REFERENCE_ID,
        "strategy": "parent_reference",
        "cdr": config.parent_cdr,
        "passes": True,
        "tap_risk": pd.NA,
        "di_seq": pd.NA,
        "camsol_mean": pd.NA,
        "humanness": pd.NA,
        "esm_log_lik": pd.NA,
        "mh_accept": pd.NA,
        "fail_reasons": "",
        "parent_hamming": 0,
        "panel_score": pd.NA,
    })

    groups = [
        _add_group(guided_top_rows, "guided_top_tier", "highest-ranked passing guided candidates"),
        _add_group(guided_diverse_rows, "guided_diversity_tier", "passing guided candidates selected for sequence diversity"),
        _add_group(unguided_rows, "unguided_controls", "passing unguided masked-LM controls"),
        _add_group(negative_rows, "negative_controls", "oracle-failing controls to test assay discrimination"),
        _add_group([parent_row], "parent_reference", "Trastuzumab parent CDR-H3 reference"),
    ]
    panel = pd.concat([g for g in groups if not g.empty], ignore_index=True)

    preferred = [
        "panel_group", "panel_rank", "selection_reason", "candidate_id", "strategy", "cdr",
        "passes", "parent_hamming", "panel_score", "esm_log_lik", "tap_risk", "di_seq",
        "camsol_mean", "humanness", "mh_accept", "fail_reasons",
    ]
    existing = [col for col in preferred if col in panel.columns]
    remainder = [col for col in panel.columns if col not in existing]
    return panel[existing + remainder]
