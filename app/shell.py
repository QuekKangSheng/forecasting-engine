"""The frame every page sits in: the status bar across the top of the window.

It says what the app is working from, whichever page is open: the data committed,
the model active for each index, and the last allocation solved. Those are the three
things every figure in the app depends on, and before this they could only be checked
by going back to the page that owns each.

It reads what is already there (session state and the active-model log) and computes
nothing. ``status_cells`` decides what each cell says and is separate from the
drawing so it can be tested without a browser.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import bloomberg_extraction_panel
import portfolio_summary
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.reporting.model_metrics import display_model_name
from forecasting_engine.reporting.polynomial_function import dataset_fingerprint
from forecasting_engine.store import active_model

ROLE_NAMES = {TargetRole.EQUITY: "Equity", TargetRole.BOND: "Bond"}


@dataclass(frozen=True)
class StatusCell:
    """One cell of the bar: what it is about, what it says, and its state's tone
    (one of ``ui.TONES``), which colours the dot beside it. The words carry the state
    too, so the dot is never the only sign of it."""

    key: str
    value: str
    tone: str = "neutral"


def _span(dates: pd.Series) -> str:
    return f"{dates.min():%d %b %Y} → {dates.max():%d %b %Y}"


def status_cells(
    merged: pd.DataFrame | None,
    actives: Mapping[TargetRole, active_model.ActiveModelRecord | None],
    summary: portfolio_summary.PortfolioSummary | None,
) -> list[StatusCell]:
    """What the bar says, given the committed data (or ``None``), each role's active
    model (or ``None``) and the Portfolio page's last result (or ``None``)."""
    fingerprint = None if merged is None else str(dataset_fingerprint(merged))
    if merged is None:
        cells = [StatusCell("Data", "none committed", "warning")]
    else:
        dates = merged[DATE_COLUMN].dropna()
        cells = [
            StatusCell("Data", _span(dates), "success")
            if len(dates)
            else StatusCell("Data", "no dated rows", "warning")
        ]
    for role, name in ROLE_NAMES.items():
        record = actives.get(role)
        if record is None:
            cells.append(StatusCell(name, "not set"))
        elif fingerprint is not None and record.dataset_fingerprint != fingerprint:
            # Fitted on data that has since been replaced: its forecasts are not used.
            cells.append(
                StatusCell(name, f"{display_model_name(record.model_name)} · stale", "warning")
            )
        else:
            cells.append(StatusCell(name, display_model_name(record.model_name), "success"))
    if summary is None:
        cells.append(StatusCell("Allocation", "not solved yet"))
    elif fingerprint is not None and summary.dataset_fingerprint != fingerprint:
        cells.append(StatusCell("Allocation", "other data · stale", "warning"))
    else:
        level = "" if summary.risk_level is None else f"level {summary.risk_level} · "
        cells.append(
            StatusCell(
                "Allocation",
                f"{level}{summary.equity_weight:.0%} / {summary.bond_weight:.0%}"
                f" · {summary.as_of:%d %b %Y}",
                "success",
            )
        )
    return cells


def status_bar(cells: list[StatusCell]) -> str:
    """The bar, as HTML. Every value is escaped: model names come from saved runs."""
    body = "".join(
        '<div class="fe-status-cell">'
        f'<span class="fe-status-dot" style="background:var(--fe-{_tone(cell.tone)}-fg)"></span>'
        f'<span class="fe-status-key">{html.escape(cell.key)}</span>'
        f'<span class="fe-status-value" title="{html.escape(cell.value, quote=True)}">'
        f"{html.escape(cell.value)}</span></div>"
        for cell in cells
    )
    return f'<div class="fe-statusbar" role="status" aria-label="Pipeline status">{body}</div>'


def _tone(tone: str) -> str:
    return tone if tone in ui.TONES else "neutral"


def frame(run_page: Callable[[], None]) -> None:
    """Run a page under the status bar.

    The bar is drawn twice into the same place. Once before the page, so it is there
    from the first moment; and once after, because the page may have changed what the
    bar reports (the Portfolio page solving a new allocation, a model set active), and
    a bar one click behind the page under it would be worse than none. The second
    drawing happens however the page ends, ``st.stop()`` included."""
    slot = st.empty()
    render(slot)
    try:
        run_page()
    finally:
        render(slot)


def render(slot: DeltaGenerator | None = None) -> None:
    """Draw the bar for this session, into ``slot`` if given."""
    bloomberg_extraction_panel.restore_last_commit()
    merged = st.session_state.get(bloomberg_extraction_panel.COMMITTED_KEY)
    actives = {role: active_model.get_active_model(role) for role in ROLE_NAMES}
    summary = st.session_state.get(portfolio_summary.SUMMARY_KEY)
    # With the stylesheet in the same element: one of its own would be an empty row
    # above the bar, holding it off the top of the window.
    (slot or st).markdown(
        ui.stylesheet() + status_bar(status_cells(merged, actives, summary)),
        unsafe_allow_html=True,
    )
