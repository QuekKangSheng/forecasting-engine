"""Home page: where the pipeline stands — data, models, portfolio — and the next
step (FYP-218).

Reads only what the other pages left in session state and the active-model store:
it never fits a model or runs a backtest. The data quality report itself is on
the Data page; Home only counts what it flagged.
"""

from __future__ import annotations

import html

import streamlit as st

import bloomberg_extraction_panel
import hero
import market_field
import model_runs
import model_settings
import portfolio_summary
import tokens
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import Transform, select_signals, transform_for
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.model_metrics import (
    MODEL_ORDER,
    ModelRunResult,
    display_model_name,
)
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.store import active_model
from forecasting_engine.validation.gates import evaluate_candidate

ROLES: tuple[TargetRole, ...] = (TargetRole.EQUITY, TargetRole.BOND)
ROLE_NAMES = {TargetRole.EQUITY: "Equity", TargetRole.BOND: "Bond"}
NAIVE = MODEL_ORDER[0]
#: What the link under the next step says, by the page it leads to.
GO_TO = {
    "data": "Go to Data",
    "models": "Go to Models",
    "portfolio": "Go to Portfolio Optimizer",
}
#: The second link in the opening: the page before the one the first leads to, or
#: the analysis that follows the last.
ALSO = {
    "data": ("models", "Open Models"),
    "models": ("data", "Open Data"),
    "portfolio": ("models", "Open Models"),
    None: ("scenarios", "Open Scenarios"),
}
TITLE = "Every forecast is tested on data it never saw."
SUMMARY = (
    "Only the ones that hold up move the portfolio. The Forecasting Engine scores forecasts "
    "of S&P 500 and US bond returns on periods they were not fitted on, then splits a "
    "portfolio between the two."
)
#: The maker's name, set large at the foot of the page.
MAKER = "Finlytics"

bloomberg_extraction_panel.restore_last_commit()
merged = st.session_state.get(bloomberg_extraction_panel.COMMITTED_KEY)
report = st.session_state.get(bloomberg_extraction_panel.COMMITTED_REPORT_KEY)
targets: dict[TargetRole, str] = st.session_state.get(
    bloomberg_extraction_panel.COMMITTED_TARGETS_KEY, {}
)
sources = st.session_state.get(bloomberg_extraction_panel.COMMITTED_SOURCES_KEY, {})
data_fingerprint = None if merged is None else dataset_fingerprint(merged)
fingerprint = None if data_fingerprint is None else str(data_fingerprint)
stored: model_runs.StoredRuns | None = st.session_state.get(model_runs.RUNS_KEY)
current_runs = (
    stored
    if stored is not None
    and merged is not None
    and stored.shared_settings[0] == data_fingerprint
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


def _active_stale(role: TargetRole) -> bool:
    """Whether ``role``'s active model was fitted on other data or another target
    column than the ones committed now."""
    active = actives[role]
    return (
        active is not None
        and fingerprint is not None
        and not active.fitted_on(fingerprint, targets.get(role))
    )


def _portfolio_current() -> bool:
    """Whether the last Portfolio run used the committed data and the models active
    now: its weights and Sharpe ratios are out of date once either is replaced, and
    once the data is cleared."""
    return (
        summary is not None
        and merged is not None
        and not _stale(summary.dataset_fingerprint)
        and summary.models_match(actives)
    )


def _next_step() -> tuple[str, str] | None:
    """The first thing left to do, in pipeline order, and the page it is done on (as
    the entry point names it), or ``None`` when every step is up to date."""
    if merged is None:
        return (
            "upload Bloomberg exports on the **Data** page and click **Use Updated Data**.",
            "data",
        )
    missing = [ROLE_NAMES[r].lower() for r in ROLES if r not in targets]
    if missing:
        return f"assign the {' and '.join(missing)} target on the **Data** page.", "data"
    for role in ROLES:
        active = actives[role]
        if active is None:
            if not _runs(role):
                return (
                    f"run the {ROLE_NAMES[role].lower()} models on the **Models** page.",
                    "models",
                )
            return (
                f"set an active {ROLE_NAMES[role].lower()} model on the **Models** page.",
                "models",
            )
        if _active_stale(role):
            return (
                f"re-run the {ROLE_NAMES[role].lower()} models on the committed data and set "
                "one active on the **Models** page.",
                "models",
            )
    if not _portfolio_current():
        return (
            "open the **Portfolio Optimizer** to allocate between the active models.",
            "portfolio",
        )
    return None


def _signals() -> list[str]:
    numeric = [c for c in merged.columns if c != DATE_COLUMN and merged[c].dtype.kind in "fi"]
    return select_signals(numeric, list(targets.values()), sources)


def _status_line() -> str:
    """What the app is working from, in one line short enough to sit on one: the
    data and how many models are set on it. The allocation is in the figures below."""
    if merged is None:
        return "waiting for data: upload exports on the Data page"
    fresh = sum(1 for role in ROLES if actives[role] and not _active_stale(role))
    return (
        f"{len(merged[DATE_COLUMN].dropna()):,} days · {len(_signals())} signals · "
        f"{fresh} of {len(ROLES)} models active"
    )


def _pill() -> str:
    if merged is None:
        return hero.pill("No data committed", "warning")
    dates = merged[DATE_COLUMN].dropna()
    if dates.empty:
        return hero.pill("Data committed, with no dated rows", "warning")
    return hero.pill(
        f"Data committed · {dates.min():%d %b %Y} to {dates.max():%d %b %Y}", "success"
    )


def _field() -> str:
    """The equity index's last months as the committed data has them, or an empty
    floor before there is an equity target to draw."""
    column = targets.get(TargetRole.EQUITY)
    if merged is None or column not in merged.columns:
        return market_field.empty_field()
    prices = merged.set_index(DATE_COLUMN)[column]
    return market_field.field(market_field.monthly_candles(prices), label(column))


def _tape() -> str:
    """Both targets, then every signal, each with its change over one horizon."""
    if merged is None:
        return ""
    columns = [targets[role] for role in ROLES if targets.get(role) in merged.columns]
    columns += [c for c in _signals() if c not in columns]
    span = model_settings.DEFAULT_HORIZON
    items = hero.ticks(
        merged.set_index(DATE_COLUMN),
        columns,
        [label(c) for c in columns],
        [transform_for(sources.get(c)) is Transform.LOG_RETURN for c in columns],
        span,
    )
    return hero.tape(items, span)


def _opening(step: tuple[str, str] | None) -> None:
    """The first screen: what the app is for and what it is working from, beside the
    equity index as the committed data has it, then every committed series."""
    where = None if step is None else step[1]
    words, picture = st.columns([5, 7], gap="medium", vertical_alignment="bottom")
    with words:
        # The stylesheet travels with the first thing seen, as on every other page.
        st.markdown(
            ui.stylesheet()
            + hero.CSS
            + hero.copy(
                _pill(),
                ui.kicker_line("00 / Status"),
                TITLE,
                SUMMARY,
                hero.typed(_status_line()),
            ),
            unsafe_allow_html=True,
        )
        # Side by side at their own widths, the second dropping under the first where
        # the column is too narrow for both.
        with st.container(key="fe-hero-go", horizontal=True):
            ui.page_link(where or "portfolio", GO_TO.get(where, "Open Portfolio Optimizer"))
            ui.page_link(*ALSO[where])
    with picture:
        st.markdown(_field(), unsafe_allow_html=True)
    strip = _tape()
    if strip:
        st.markdown(strip, unsafe_allow_html=True)
    else:
        ui.rule()


def _tick(ok: bool) -> str:
    return ui.lozenge("✓", "success") if ok else ui.lozenge("✗", "danger")


def _data_card() -> None:
    st.markdown(ui.kicker_line("01 / Ingest", small=True), unsafe_allow_html=True)
    st.subheader("Data")
    if merged is None:
        not_yet = ui.status_row("Committed", ui.lozenge("Not yet", "neutral"))
        st.markdown(not_yet, unsafe_allow_html=True)
        st.caption("Nothing committed yet.")
        return
    dates = merged[DATE_COLUMN].dropna()
    signals = _signals()
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
        # Unusual moves are shown for information and never stop a run; say so, so a
        # count of 98 doesn't read as 98 problems.
        note = "" if report.schema_errors else " (for information; none stop a run)"
        st.caption(f"{issues} quality issue{'s' if issues != 1 else ''} flagged → Data{note}")
    restored = st.session_state.get(bloomberg_extraction_panel.RESTORED_AT_KEY)
    if restored is not None:
        st.caption(f"Loaded from the commit made {restored:%d %b %Y %H:%M}.")
    ui.page_link("data", "Open Data")


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
    st.markdown(ui.kicker_line("02 / Forecast", small=True), unsafe_allow_html=True)
    st.subheader("Models")
    if merged is None:
        st.caption("Commit data first.")
        return
    for role in ROLES:
        if role not in targets:
            continue
        # Escaped: the target's name comes from a column in the uploaded file.
        st.markdown(
            ui.eyebrow(html.escape(f"{ROLE_NAMES[role]} · {label(targets[role])}")),
            unsafe_allow_html=True,
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
        elif _active_stale(role):
            st.warning(
                f"Active model {display_model_name(active.model_name)} was fitted on "
                "different data from the one committed now.",
                icon=":material/warning:",
            )
        else:
            st.caption(
                f"Active model: {display_model_name(active.model_name)}, "
                "fitted on the committed data."
            )
    ui.page_link("models", "Open Models")


def _portfolio_card() -> None:
    st.markdown(ui.kicker_line("03 / Allocate", small=True), unsafe_allow_html=True)
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
    elif not summary.models_match(actives):
        st.warning(
            "The last run used different active models from the ones set now. Open the "
            "**Portfolio Optimizer** to recompute its weights and Sharpe ratio.",
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
    ui.page_link("portfolio", "Open Portfolio Optimizer")


def _headline() -> None:
    """The latest portfolio result, first, for anyone who opens the app to see it."""
    if not _portfolio_current():
        return
    cells = [
        (
            f"Latest weights, {summary.as_of:%d %b %Y}",
            f"{summary.equity_weight:.0%} / {summary.bond_weight:.0%}",
            # The split as a bar, in the colours the two indices have on every chart.
            ui.split_bar(
                [
                    ("Equity", summary.equity_weight, tokens.SERIES_EQUITY),
                    ("Bond", summary.bond_weight, tokens.SERIES_BOND),
                ]
            ),
        ),
    ]
    if summary.risk_level is not None:
        cells.append(("Risk level", f"{summary.risk_level} of 5", ""))
    gap = summary.net_sharpe - summary.baseline_net_sharpe
    cells.append(
        (
            "Sharpe after costs",
            f"{summary.net_sharpe:.2f}",
            ui.delta(gap, f"{abs(gap):.2f}") + f" vs {summary.baseline_net_sharpe:.2f} for 50/50",
        )
    )
    if summary.forecast_value_label is not None:
        cells.append(
            (
                "Forecast value after costs",
                ui.lozenge(html.escape(summary.forecast_value_label), summary.forecast_value_tone),
                f"t = {summary.forecast_value_t}",
            )
        )
    st.markdown(ui.strip(cells), unsafe_allow_html=True)


step = _next_step()
_opening(step)
_headline()
if step is None:
    st.success(
        "**Up to date:** every step has run on the committed data.",
        icon=":material/check_circle:",
    )
else:
    # The link that does it is the first of the two in the opening, just above.
    st.info(f"**Next step:** {step[0]}", icon=":material/arrow_forward:")
# The three stages in pipeline order, numbered as the sidebar numbers them. Models
# gets the most room: it lists every model's result for both indices.
data_col, models_col, portfolio_col = st.columns([5, 7, 5])
with data_col, st.container(border=True):
    _data_card()
with models_col, st.container(border=True):
    _models_card()
with portfolio_col, st.container(border=True):
    _portfolio_card()
st.markdown(hero.wordmark(MAKER), unsafe_allow_html=True)
