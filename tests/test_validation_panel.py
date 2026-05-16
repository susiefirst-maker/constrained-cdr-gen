from __future__ import annotations

import pandas as pd

from cdr_gen.validation_panel import PanelConfig, hamming_distance, select_validation_panel


def _row(strategy: str, cdr: str, passes: bool, ll: float, idx: int) -> dict:
    return {
        "candidate_id": f"input_{idx:04d}",
        "strategy": strategy,
        "cdr": cdr,
        "passes": passes,
        "tap_risk": 0 if passes else 4,
        "di_seq": 1.0 if passes else 5.0,
        "camsol_mean": 0.2 if passes else -1.0,
        "humanness": 0.7 if passes else 0.2,
        "esm_log_lik": ll,
        "mh_accept": 0.9,
        "fail_reasons": "" if passes else "oracle fail",
    }


def test_hamming_distance_handles_length_difference():
    assert hamming_distance("ABC", "AXCD") == 2


def test_select_validation_panel_builds_expected_groups():
    rows = []
    for i in range(8):
        rows.append(_row("guided", f"AAAAAAAAAAA{i % 10}A"[:13], True, -2.0 + i * 0.01, i))
    for i in range(4):
        rows.append(_row("unguided", f"QQQQQQQQQQQ{i % 10}Q"[:13], True, -2.4 + i * 0.01, 20 + i))
    for i in range(4):
        rows.append(_row("random", f"CCCCCCCCCCC{i % 10}C"[:13], False, -4.0, 40 + i))

    panel = select_validation_panel(
        pd.DataFrame(rows),
        PanelConfig(guided_top_n=3, guided_diverse_n=2, unguided_control_n=2, negative_control_n=2),
    )

    assert set(panel["panel_group"]) == {
        "guided_top_tier",
        "guided_diversity_tier",
        "unguided_controls",
        "negative_controls",
        "parent_reference",
    }
    assert len(panel[panel["panel_group"] == "guided_top_tier"]) == 3
    assert len(panel[panel["panel_group"] == "unguided_controls"]) == 2
    assert len(panel[panel["panel_group"] == "negative_controls"]) == 2
    assert panel[panel["panel_group"] == "parent_reference"]["cdr"].iloc[0] == "SRWGGDGFYAMDY"


def test_select_validation_panel_handles_minimal_columns_and_string_booleans():
    candidates = pd.DataFrame([
        {"strategy": "guided", "cdr": "AAAAAAAAAAAAA", "passes": "true"},
        {"strategy": "unguided", "cdr": "QQQQQQQQQQQQQ", "passes": "true"},
        {"strategy": "random", "cdr": "CCCCCCCCCCCCC", "passes": "false"},
    ])

    panel = select_validation_panel(
        candidates,
        PanelConfig(guided_top_n=1, guided_diverse_n=0, unguided_control_n=1, negative_control_n=1),
    )

    assert len(panel) == 4
    assert set(panel["panel_group"]) == {
        "guided_top_tier",
        "unguided_controls",
        "negative_controls",
        "parent_reference",
    }
    assert not bool(panel.loc[panel["panel_group"] == "negative_controls", "passes"].iloc[0])
