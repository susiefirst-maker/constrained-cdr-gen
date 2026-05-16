# Optional UI dependency: pip install -e ".[app]"
from pathlib import Path
import json

import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
REPORTS = BASE / "reports"
RESULTS_PATH = REPORTS / "trastuzumab_results.json"
CANDIDATES_PATH = REPORTS / "trastuzumab_candidates.csv"
FRAME1 = (
    "Does property-guided Gibbs sampling over CDR positions produce more "
    "developable candidate antibodies than unguided masked-LM sampling or random "
    "AA substitution under evaluation by an independent developability oracle "
    "that the sampler never saw?"
)
FRAME2 = (
    "Guided Gibbs CDR sampling with Metropolis-Hastings correction is evaluated "
    "by an independent TAP / DI / CamSol / humanness oracle, never by the "
    "sampler's own surrogate."
)


def highlight_rows(row):
    color = "#e8f5e9" if str(row.get("passes")).lower() == "true" else "#fdecea"
    return [f"background-color: {color}"] * len(row)


st.set_page_config(page_title="Trastuzumab CDR Browser", layout="wide")
st.title("Trastuzumab CDR Results Browser")
st.write(FRAME1)
st.write(FRAME2)
st.caption("Read-only app. Start with `streamlit run app.py` and stop it with `Ctrl-C`.")

if not RESULTS_PATH.exists() or not CANDIDATES_PATH.exists():
    st.warning("No generated report files found yet.")
    st.code(
        "python -m cdr_gen.cli --parent trastuzumab --n-designs 80 "
        "--strategies random unguided guided --lambda 1.0 --n-sweeps 20 --out-dir reports",
        language="bash",
    )
    st.stop()

RESULTS = json.loads(RESULTS_PATH.read_text())
CANDIDATES = pd.read_csv(CANDIDATES_PATH)
STRATS = pd.DataFrame(RESULTS["strategies"])

cols = st.columns(len(STRATS))
for col, row in zip(cols, STRATS.itertuples(index=False)):
    with col:
        st.metric(row.strategy.title(), f"{row.pass_rate:.1%}")
        st.caption(
            f"95% CI {row.pass_rate_ci_low:.1%} to {row.pass_rate_ci_high:.1%} | "
            f"{row.n_passing}/{row.n_total} pass"
        )

st.subheader("Pass Rate by Strategy")
st.vega_lite_chart(
    STRATS,
    {
        "layer": [
            {
                "mark": "bar",
                "encoding": {
                    "x": {"field": "strategy", "type": "nominal", "sort": ["random", "unguided", "guided"]},
                    "y": {"field": "pass_rate", "type": "quantitative", "axis": {"format": "%"}, "scale": {"domain": [0, 1]}},
                    "color": {"field": "strategy", "legend": None},
                    "tooltip": [{"field": c} for c in ["strategy", "pass_rate", "pass_rate_ci_low", "pass_rate_ci_high"]],
                },
            },
            {
                "mark": {"type": "rule", "strokeWidth": 3},
                "encoding": {
                    "x": {"field": "strategy", "type": "nominal", "sort": ["random", "unguided", "guided"]},
                    "y": {"field": "pass_rate_ci_low", "type": "quantitative"},
                    "y2": {"field": "pass_rate_ci_high"},
                },
            },
        ]
    },
    use_container_width=True,
)

st.subheader("Candidate Designs")
options = ["all"] + sorted(CANDIDATES["strategy"].dropna().unique().tolist())
choice = st.selectbox("Strategy filter", options, index=0)
view = CANDIDATES if choice == "all" else CANDIDATES[CANDIDATES["strategy"] == choice]
metric_cols = [c for c in ["tap_risk", "di_seq", "camsol_mean", "humanness"] if c in view.columns]
missing = [c for c in ["tap_risk", "di_seq", "camsol_mean", "humanness"] if c not in view.columns]
if missing:
    st.warning(f"Missing expected oracle columns: {', '.join(missing)}")
cfg = {
    "passes": st.column_config.CheckboxColumn("Oracle pass"),
    **{c: st.column_config.NumberColumn(c.replace("_", " ").title(), format="%.3f") for c in metric_cols},
}
shown = view
try:
    shown = view.style.apply(highlight_rows, axis=1) if "passes" in view.columns else view
except Exception:
    shown = view.assign(status=view["passes"].map({True: "PASS", False: "FAIL"})) if "passes" in view.columns else view
st.dataframe(shown, use_container_width=True, hide_index=True, column_config=cfg)
st.caption("Visible oracle metrics: " + (", ".join(metric_cols) if metric_cols else "none found in CSV"))
