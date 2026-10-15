"""Home page: where the pipeline stands — data, models, portfolio — and the next
step (FYP-218).

Reads only what the other pages left in session state and the active-model store:
it never fits a model or runs a backtest. The data quality report itself is on
the Data page; Home only counts what it flagged.
"""

from __future__ import annotations

import streamlit as st

import bloomberg_extraction_panel
import model_runs
import portfolio_summary
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import select_signals
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.model_metrics import MODEL_ORDER, ModelRunResult
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.store import active_model
from forecasting_engine.validation.gates import evaluate_candidate

ROLES: tuple[TargetRole, ...] = (TargetRole.EQUITY, TargetRole.BOND)
ROLE_NAMES = {TargetRole.EQUITY: "Equity", TargetRole.BOND: "Bond"}
NAIVE = MODEL_ORDER[0]

ui.inject()
st.title("Forecasting Engine")
st.caption(
    "Forecasts short-horizon returns for liquid equity and bond indices, validates them "
    "against overfitting, and turns them into an equity/bond allocation."
)

merged = st.session_state.get(bloomberg_extraction_panel.COMMITTED_KEY)
report = st.session_state.get(bloomberg_extraction_panel.COMMITTED_REPORT_KEY)
targets: dict[TargetRole, str] = st.session_state.get(
    bloomberg_extraction_panel.COMMITTED_TARGETS_KEY, {}
)
sources = st.session_state.get(bloomberg_extraction_panel.COMMITTED_SOURCES_KEY, {})
fingerprint = None if merged is None else str(dataset_fingerprint(merged))
stored: model_runs.StoredRuns | None = st.session_state.get(model_runs.RUNS_KEY)
current_runs = (
    stored
    if stored is not None
    and merged is not None
    and stored.shared_settings[0] == dataset_fingerprint(merged)
    else None
)
actives = {role: active_model.get_active_model(role) for role in ROLES}
summary: portfolio_summary.PortfolioSummary | None = st.session_state.get(
    portfolio_summary.SUMMARY_KEY
)
label = labeller([c for c in merged.columns if c != DATE_COLUMN]) if merged is not None else str


def _runs(role: TargetRole) -> dict[str, model_runs.ModelRun]:
    if current_runs is None or role not in current_runs.tabs:
        return {}
    return current_runs.tabs[role].runs


def _stale(fingerprint_then: str | None) -> bool:
    return fingerprint is not None and fingerprint_then != fingerprint


def _next_step() -> str:
    """The first thing left to do, in pipeline order."""
    if merged is None:
        return "upload Bloomberg exports on the **Data** page and click **Use Updated Data**."
    missing = [ROLE_NAMES[r].lower() for r in ROLES if r not in targets]
    if missing:
        return f"assign the {' and '.join(missing)} target on the **Data** page."
    for role in ROLES:
        active = actives[role]
        if active is None:
            if not _runs(role):
                return f"run the {ROLE_NAMES[role].lower()} models on the **Models** page."
            return f"set an active {ROLE_NAMES[role].lower()} model on the **Models** page."
        if _stale(active.dataset_fingerprint):
            return (
                f"re-run the {ROLE_NAMES[role].lower()} models on the committed data and set "
                "one active on the **Models** page."
            )
    if summary is None or _stale(summary.dataset_fingerprint):
        return "open the **Portfolio Optimizer** to allocate between the active models."
    return "nothing — every step is up to date with the committed data."


def _tick(ok: bool) -> str:
    return ui.lozenge("✓", "success") if ok else ui.lozenge("✗", "danger")


def _data_card() -> None:
    st.subheader("Data")
    if merged is None:
        not_yet = ui.status_row("Committed", ui.lozenge("Not yet", "neutral"))
        st.markdown(not_yet, unsafe_allow_html=True)
        st.caption("Nothing committed yet.")
        return
    dates = merged[DATE_COLUMN].dropna()
    numeric = [c for c in merged.columns if c != DATE_COLUMN and merged[c].dtype.kind in "fi"]
    signals = select_signals(numeric, list(targets.values()), sources)
    rows = [
        ui.status_row("Committed", ui.lozenge("Yes", "success")),
        ui.status_row("Equity target", _tick(TargetRole.EQUITY in targets)),
        ui.status_row("Bond target", _tick(TargetRole.BOND in targets)),
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)
    if not dates.empty:
        st.caption(f"{dates.min():%d %b %Y} to {dates.max():%d %b %Y}")
    st.caption(f"{len(signals)} signal{'s' if len(signals) != 1 else ''}")
    if report is not None:
        issues = bloomberg_extraction_panel.quality_issue_count(report)
        st.caption(f"{issues} quality issue{'s' if issues != 1 else ''} flagged → Data")


def _result_line(name: str, result: ModelRunResult) -> str:
    if name == NAIVE:
        return ui.status_row(name, ui.lozenge("Baseline", "neutral"))
    figures = f"Signal Rank IC {_fmt(result.gated_rank_ic, 3)} · t {_fmt(result.signal_t, 2)}"
    outcome = evaluate_candidate(result.gated_rank_ic, result.signal_t)
    badge = ui.lozenge("Passed", "success") if outcome.promoted else ui.lozenge("Failed", "danger")
    return ui.status_row(name, badge, figures)


def _fmt(value: float, decimals: int) -> str:
    return "—" if value != value else f"{value:.{decimals}f}"


def _models_card() -> None:
    st.subheader("Models")
    if merged is None:
        st.caption("Commit data first.")
        return
    for role in ROLES:
        if role not in targets:
            continue
        st.markdown(
            ui.eyebrow(f"{ROLE_NAMES[role]} · {label(targets[role])}"), unsafe_allow_html=True
        )
        runs = _runs(role)
        screening = next(
            (r.result.screening for r in runs.values() if r.result.screening is not None), None
        )
        if screening is not None:
            st.caption(
                f"{len(screening.latest_included)} of {len(screening.counts)} signals passed "
                "screening in the latest fold"
            )
        if runs:
            lines = [_result_line(n, runs[n].result) for n in MODEL_ORDER if n in runs]
            st.markdown("".join(lines), unsafe_allow_html=True)
        else:
            st.caption("No model has run on the committed data in this session.")
        active = actives[role]
        if active is None:
            st.caption("Active model: none set.")
        elif _stale(active.dataset_fingerprint):
            st.warning(
                f"Active model {active.model_name} was fitted on different data from the one "
                "committed now.",
                icon=":material/warning:",
            )
        else:
            st.caption(f"Active model: {active.model_name}, fitted on the committed data.")


def _portfolio_card() -> None:
    st.subheader("Portfolio Optimizer")
    both = all(actives[role] is not None for role in ROLES)
    st.markdown(ui.status_row("Both active models set", _tick(both)), unsafe_allow_html=True)
    if summary is None:
        st.caption("Not run yet.")
        return
    if _stale(summary.dataset_fingerprint):
        st.warning(
            "The last run used different data from the one committed now.",
            icon=":material/warning:",
        )
    st.caption(
        f"Net Sharpe {summary.net_sharpe:.2f} against {summary.baseline_net_sharpe:.2f} "
        "for 50/50."
    )
    st.caption(
        f"Latest weights ({summary.as_of:%d %b %Y}): {summary.equity_weight:.0%} equity, "
        f"{summary.bond_weight:.0%} bond."
    )


st.info(f"**Next step:** {_next_step()}", icon=":material/arrow_forward:")
data_col, models_col, portfolio_col = st.columns(3)
with data_col, st.container(border=True):
    _data_card()
with models_col, st.container(border=True):
    _models_card()
with portfolio_col, st.container(border=True):
    _portfolio_card()
