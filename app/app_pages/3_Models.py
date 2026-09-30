"""Models page: FYP-42's Fama-French benchmark, FYP-43's polynomial forecasting
function and FYP-44's machine-learning models, run together per target through
one walk-forward harness and compared side by side.

Reads the dataset committed via "Use Updated Data" on the Data page. No maths
lives here — fitting is in forecasting_engine.models.polynomial / famafrench /
boosted; the walk-forward loop is in forecasting_engine.validation.harness;
table formatting is in forecasting_engine.reporting.model_metrics.
"""

from __future__ import annotations

import html

import streamlit as st

import bloomberg_extraction_panel
import glossary
import model_runs
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest import fama_french
from forecasting_engine.ingest.align import (
    MAX_STALENESS,
    PRODUCTION_LAG_DAYS,
    UNCLASSIFIED_TRANSFORM,
    FeaturePanel,
    align_and_lag,
    is_classified,
    select_signals,
    transform_for,
)
from forecasting_engine.ingest.fama_french import FactorFetchError
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.boosted import BoostedConfigError, run_boosted
from forecasting_engine.models.famafrench import (
    FACTOR_COLUMNS,
    FamaFrenchDataError,
    factor_coverage,
    merge_factors,
    run_famafrench,
)
from forecasting_engine.models.polynomial import (
    CANDIDATE_CONFIGS,
    MAX_DEGREE,
    DerivedPolynomial,
    PolynomialConfigError,
    run_derived_polynomial,
    run_user_polynomial,
)
from forecasting_engine.reporting.factor_labels import labeller
from forecasting_engine.reporting.model_metrics import (
    MODEL_ORDER,
    Cell,
    FoldTerms,
    ModelRunResult,
    build_metrics_rows,
)
from forecasting_engine.reporting.polynomial_function import (
    Origin,
    PolynomialFunction,
    dataset_fingerprint,
    from_description,
    term_rows,
    to_latex,
)
from forecasting_engine.validation.gates import evaluate_candidate
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward

#: Model names as the comparison table knows them (``MODEL_ORDER``).
FF5, POLYNOMIAL, ML = MODEL_ORDER

ROLE_NAMES: dict[TargetRole, str] = {TargetRole.EQUITY: "Equity", TargetRole.BOND: "Bond"}

#: The forecast horizons the validation framework asks for, reported separately
#: and never averaged.
HORIZONS: tuple[int, ...] = (1, 5)

#: One embargo shared by every horizon, equal to the longest of them, rather than
#: one per horizon. Picking h=1 used to drop the embargo to 1 as well.
EMBARGO_DAYS: int = max(HORIZONS)

TABLE_COLUMNS: tuple[str, ...] = (
    "Model",
    "IC",
    "OOS Rank IC",
    "RMSE",
    "PBO",
    "Crash Recall",
    "Crash Precision",
    "Crash F1",
    "Rows scored",
)

BADGE_LABELS = {"success": "Gate met", "danger": "Gate failed"}

#: Column heading -> the glossary term explaining it. "Model" and "Rows scored"
#: need none, and the three crash columns share one explanation.
COLUMN_TERMS = {
    "IC": "IC",
    "OOS Rank IC": "OOS Rank IC",
    "RMSE": "RMSE",
    "PBO": "PBO",
    "Crash Recall": "Crash diagnostics",
    "Crash Precision": "Crash diagnostics",
    "Crash F1": "Crash diagnostics",
}

GATE_NAMES = {"oos_rank_ic": "OOS Rank IC", "pbo": "PBO"}

st.set_page_config(page_title="Models · Forecasting Engine", page_icon=":material/functions:")
ui.inject()

st.header("Models")
st.caption(
    "Run the ticked model families together for each target through the shared "
    "walk-forward harness, then compare them side by side.",
    help=glossary.term("Promotion gate"),
)

merged = st.session_state.get(bloomberg_extraction_panel.COMMITTED_KEY)
if merged is None:
    st.info('No data committed yet — click "Use Updated Data" on the Data page first.')
    st.stop()

# Which columns are targets is a structural fact from ingestion (Task 4.1),
# not a free pick here — this is what stops a target ending up modelled as
# its own signal (e.g. SPX's own bid price screened in as a predictor for
# SPX itself, seen on the live data before this was fixed).
target_columns: dict[TargetRole, str] = st.session_state.get(
    bloomberg_extraction_panel.COMMITTED_TARGETS_KEY, {}
)
if not target_columns:
    st.info(
        "No target resolved yet — confirm at least one target index on the Data "
        'page (Target index files), then commit with "Use Updated Data".'
    )
    st.stop()

numeric_cols = [c for c in merged.columns if c != DATE_COLUMN and merged[c].dtype.kind in "fi"]
label = labeller(numeric_cols)
# Every field of every resolved target's security is excluded, not just the
# one a tab forecasts — otherwise the Equity tab would still let the Bond
# column (or SPX's own bid) ride along as a candidate signal.
sources = st.session_state.get(bloomberg_extraction_panel.COMMITTED_SOURCES_KEY, {})
signal_cols = select_signals(numeric_cols, list(target_columns.values()), sources)
transforms = {c: transform_for(sources.get(c)) for c in signal_cols}
unclassified = [c for c in signal_cols if not is_classified(sources.get(c))]
if unclassified:
    st.warning(
        f"No transform is defined for {', '.join(unclassified)}, so "
        f"{'it is' if len(unclassified) == 1 else 'they are'} treated as a "
        f"{UNCLASSIFIED_TRANSFORM}. Add the ticker to TICKER_TRANSFORMS in "
        "ingest/align.py to classify it.",
        icon=":material/warning:",
    )

with st.expander("Settings"):
    horizon = st.segmented_control(
        "Forecast horizon",
        HORIZONS,
        default=max(HORIZONS),
        required=True,
        format_func=lambda days: f"{days} day" if days == 1 else f"{days} days",
        help=glossary.term("Forecast horizon"),
    )
    cols = st.columns(2)
    train = cols[0].number_input(
        "Walk-forward train window (days)",
        min_value=10,
        value=120,
        step=10,
        help=glossary.term("Walk-forward train window (days)"),
    )
    test = cols[1].number_input(
        "Walk-forward test window (days)",
        min_value=1,
        value=20,
        step=5,
        help=glossary.term("Walk-forward test window (days)"),
    )
    st.caption(
        f"Embargo is fixed at {EMBARGO_DAYS} trading days — the longest forecast horizon — "
        "whichever horizon is selected, so training and grading never overlap.",
        help=glossary.term("Embargo"),
    )
    st.caption(
        f"Signals are lagged {PRODUCTION_LAG_DAYS} trading day, so each value is one that "
        "had already been published.",
        help=glossary.term("Signal lag"),
    )
    st.caption(
        f"The first {TUNING_ROWS} target dates are a tuning period: no model is scored "
        "on them, so machine learning's tuning never touches a reported result."
    )

check_cols = st.columns(3)
check_cols[0].checkbox("Polynomial", value=True, disabled=True)
run_ff5 = check_cols[1].checkbox(
    "Fama-French 5", value=True, help="An equity-factor benchmark, so it runs on Equity only."
)
run_ml = check_cols[2].checkbox("Machine learning", value=True)

horizon = int(horizon)
splitter = PurgedWalkForward(
    train=int(train), test=int(test), embargo=EMBARGO_DAYS, tuning_rows=TUNING_ROWS
)
stored = model_runs.stored(
    st.session_state, (dataset_fingerprint(merged), horizon, int(train), int(test))
)


def _show_alignment(panel: FeaturePanel, target_name: str) -> None:
    """Per signal: how it was made stationary, and how much of it is carried or missing."""
    with st.expander(f"Signal alignment · {target_name} · {len(panel.frame):,} target dates"):
        st.caption(
            f"Signals are read as of each date the target has a price, carried forward "
            f"for at most {MAX_STALENESS} rows. Excluded rows have no value once "
            "transformed and lagged, and are left out of fitting and scoring."
        )
        st.dataframe(
            [
                {
                    "Signal": signal,
                    "Transform": str(a.transform),
                    "Carried forward": a.carried_forward,
                    "Rows excluded": a.excluded,
                }
                for signal, a in panel.alignment.items()
            ],
            width="stretch",
            hide_index=True,
        )


def _polynomial_settings(key: str, panel: FeaturePanel) -> tuple:
    st.markdown(ui.eyebrow("Polynomial"), unsafe_allow_html=True)
    mode = st.radio(
        "Function source",
        ["Enter a function", "Derive automatically"],
        horizontal=True,
        help=glossary.term("Function source"),
        key=f"mode_{key}",
    )
    if mode == "Enter a function":
        st.caption(f"Available signal columns: {', '.join(panel.signals)}")
        formula = st.text_input(
            "Function (arithmetic on signal columns only — e.g. `2 * vix + credit_spread_hy ** 2`)",
            key=f"formula_{key}",
        )
        return (mode, formula.strip())
    max_terms = st.number_input(
        "Max terms per candidate (optional cap)", min_value=1, value=10, step=1, key=f"terms_{key}"
    )
    st.caption(
        f"Tries a small grid of degrees (1-3, of up to {MAX_DEGREE} allowed) and "
        "regularizers (Lasso, ElasticNet), compares them via PBO, and reports the one "
        "with the best out-of-sample rank IC."
    )
    return (mode, int(max_terms))


def _run_polynomial(settings: tuple, panel: FeaturePanel, price_col: str) -> model_runs.ModelRun:
    mode, value = settings
    if mode == "Enter a function":
        if not value:
            raise PolynomialConfigError("enter a function above first.")
        result, description = run_user_polynomial(value, panel, splitter)
        origin = Origin.USER_SUPPLIED
    else:
        candidates = tuple(
            DerivedPolynomial(degree=c.degree, regularizer=c.regularizer, max_terms=value)
            for c in CANDIDATE_CONFIGS
        )
        result, description = run_derived_polynomial(panel, splitter, candidates=candidates)
        origin = Origin.DERIVED
    fn = from_description(
        description, origin=origin, target=price_col, horizon=horizon, columns=panel.signals
    )
    return model_runs.ModelRun(result, description, function=fn)


def _run_famafrench(price_col: str) -> model_runs.ModelRun:
    resolved = fama_french.resolve()
    indexed = merge_factors(merged, resolved.file.frame).set_index(DATE_COLUMN)
    panel = align_and_lag(
        indexed, list(FACTOR_COLUMNS), price_col, horizon=horizon, exact=FACTOR_COLUMNS
    )
    result, description = run_famafrench(panel, splitter)
    return model_runs.ModelRun(
        result,
        description,
        coverage=factor_coverage(resolved.file.frame, panel),
        warning=resolved.warning,
    )


def _run_ml(panel: FeaturePanel) -> model_runs.ModelRun:
    result, description, tuning = run_boosted(panel, splitter)
    return model_runs.ModelRun(result, description, tuning=tuning)


def _run(models: list[str], runs: model_runs.TabRuns, *, run_one, target_name: str) -> None:
    """Fit each model in turn, each in its own status box. One failing never stops the rest."""
    for name in models:
        with st.status(f"{name} · {target_name}", expanded=False) as status:
            try:
                runs.runs[name] = run_one(name)
            except (
                PolynomialConfigError,
                FamaFrenchDataError,
                BoostedConfigError,
                FactorFetchError,
            ) as exc:
                runs.runs.pop(name, None)
                message = str(exc)
                if isinstance(exc, FactorFetchError):
                    message = (
                        "the Fama-French factors could not be downloaded and none are "
                        f"saved, so FF5 was not run: {message}"
                    )
                st.error(f"{name}: {message}", icon=":material/error:")
                status.update(label=f"{name} · {target_name} · failed", state="error")
            else:
                status.update(label=f"{name} · {target_name} · done", state="complete")


def _header_html(column: str) -> str:
    term = COLUMN_TERMS.get(column)
    if term is None:
        return f"<th>{html.escape(column)}</th>"
    hint = html.escape(glossary.term(term), quote=True)
    return (
        f'<th title="{hint}">{html.escape(column)}'
        f'<span class="fe-eyebrow-help" title="{hint}">i</span></th>'
    )


def _cell_html(cell: Cell) -> str:
    text = html.escape(cell.text)
    badge = BADGE_LABELS.get(cell.tone)
    if badge is None:
        return text
    return f"{text}&nbsp;&nbsp;{ui.lozenge(badge, cell.tone)}"


def _show_table(results: dict[str, ModelRunResult]) -> None:
    # A hand-built table has no Streamlit help=, so each heading explains
    # itself through the browser's own title tooltip.
    rows = build_metrics_rows(results)
    header = "".join(_header_html(col) for col in TABLE_COLUMNS)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell_html(row[col])}</td>" for col in TABLE_COLUMNS) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<table class="fe-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )


def _gate_line(name: str, result: ModelRunResult) -> str:
    if result.pbo is None:
        return f"**{name}**: not gated — no configuration search, so no PBO."
    outcome = evaluate_candidate(result.oos_rank_ic, result.pbo)
    if outcome.promoted:
        return f"**{name}**: gate met."
    failed = " and ".join(GATE_NAMES[g] for g in outcome.failed_gates)
    return f"**{name}**: gate failed on {failed}."


def _show_screening(runs: dict[str, model_runs.ModelRun], target_name: str) -> None:
    """Shared by the derived polynomial and ML: both screen the same panel on the
    same folds, so their screening is the same and shown once."""
    screenings = [runs[n].result.screening for n in (POLYNOMIAL, ML) if n in runs]
    screening = next((s for s in screenings if s is not None), None)
    with st.expander(f"Signal screening · {target_name}"):
        if screening is None:
            st.caption(
                "Nothing was screened: a user-supplied function and FF5 are handed their "
                "inputs. Derive the polynomial or run machine learning to screen signals."
            )
            return
        st.caption(
            f"Every one of the {screening.folds} walk-forward folds screens signals on its "
            "own training window, so a signal can be kept in some folds and dropped in others.",
            help=glossary.term("Signal inclusion across folds"),
        )
        ics = screening.latest_ics
        st.dataframe(
            [
                {
                    "Signal": signal,
                    "Transform": str(transforms[signal]) if signal in transforms else "—",
                    "Latest fold": "In" if signal in screening.latest_included else "Out",
                    "Latest-fold IC": ics.get(signal, float("nan")),
                    "Included in": f"{used} of {screening.folds} folds",
                }
                for signal, used in screening.counts
            ],
            width="stretch",
            hide_index=True,
            column_config={"Latest-fold IC": st.column_config.NumberColumn(format="%.4f")},
        )
        if screening.fell_back:
            st.caption(
                f"{screening.fell_back} of {screening.folds} folds kept no signal after "
                "screening, so they were fit on every signal instead."
            )


def _show_fold_term_count(fn: PolynomialFunction, terms: FoldTerms | None) -> None:
    """Say how typical this fold's equation is of the run.

    A derived fit is refitted per fold and regularization can zero every
    coefficient on one fold and keep several on the next. The equation above is
    the most recent fold's, so on its own it reads as the whole run's answer.
    """
    if not fn.terms:
        st.caption("No terms survived fitting — every coefficient was regularized to zero.")
    if terms is None or terms.folds <= 1:
        return
    if terms.every_fold:
        st.caption(f"Every one of the {terms.folds} walk-forward folds kept at least one term.")
    else:
        st.caption(
            f"{terms.with_terms} of {terms.folds} walk-forward folds kept any term at all. "
            "The equation above is the most recent fold's fit, not an average of them."
        )


def _show_polynomial(run: model_runs.ModelRun) -> None:
    """The fitted polynomial as a labelled equation and term table."""
    fn = run.function
    st.markdown(ui.eyebrow(fn.origin, glossary.term("Fitted terms")), unsafe_allow_html=True)
    st.caption(f"Forecasts: {label(fn.target)}, {fn.horizon}-day return")
    st.latex(to_latex(fn, label))
    if fn.formula is not None:
        st.caption(
            "This function can't be written as separate terms and exponents (it divides "
            "by a signal, or expands to more than 50 terms), so it's shown as entered."
        )
        return
    if fn.origin == Origin.DERIVED:
        _show_fold_term_count(fn, run.result.terms)
    st.dataframe(
        term_rows(fn, label),
        width="stretch",
        hide_index=True,
        column_config={
            "Factor": st.column_config.TextColumn("Factor", help=glossary.term("Factor")),
            "Exponent": st.column_config.TextColumn("Exponent", help=glossary.term("Exponent")),
            "Coefficient": st.column_config.TextColumn(
                "Coefficient", help=glossary.term("Coefficient")
            ),
        },
    )


def _show_fitted_terms(description: ModelDescription, *, is_ml: bool) -> None:
    value_col = "Mean |SHAP value|" if is_ml else "Coefficient"
    heading = "Feature attribution (SHAP)" if is_ml else "Fitted terms"
    st.markdown(ui.eyebrow(heading, glossary.term(heading)), unsafe_allow_html=True)
    rows = [
        {"Term": term, value_col: coefficient}
        for term, coefficient in zip(description.terms, description.coefficients, strict=True)
    ]
    if description.intercept is not None:
        rows.append({"Term": "(intercept)", value_col: description.intercept})
    st.dataframe(rows, width="stretch", hide_index=True)


def _show_famafrench(run: model_runs.ModelRun) -> None:
    if run.warning:
        st.warning(run.warning, icon=":material/warning:")
    coverage = run.coverage
    st.caption(
        f"Factor file covers {coverage.first:%d/%m/%Y} to {coverage.last:%d/%m/%Y}. "
        f"{coverage.rows_used:,} rows used. {coverage.missing_dates:,} target dates "
        "have no factor row — Ken French publishes one to two months late."
    )
    _show_fitted_terms(run.description, is_ml=False)


def _show_ml(run: model_runs.ModelRun) -> None:
    _show_fitted_terms(run.description, is_ml=True)
    st.caption(
        "PBO here compares only two candidates, tuned XGBoost and tuned LightGBM, so it "
        "is coarse — read it as a rough check rather than a precise probability."
    )
    tuning = run.tuning
    if tuning is None:
        return
    st.markdown(ui.eyebrow("Tuning"), unsafe_allow_html=True)
    st.caption(
        "Hyperparameters are re-tuned about once a year on the period just before the "
        "next test window, so each fold uses settings tuned only on its past."
    )
    rows = []
    for number, tune in enumerate(tuning.tunes):
        folds = [i + 1 for i, t in enumerate(tuning.fold_tunes) if t == number]
        rows.append(
            {
                "Tune": number + 1,
                "Tuned on": f"{tune.first:%d/%m/%Y} to {tune.last:%d/%m/%Y}",
                "Rows": tune.rows,
                "Trials": tune.trials,
                "Folds": f"{folds[0]}–{folds[-1]}" if len(folds) > 1 else str(folds[0]),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)
    latest = tuning.tunes[tuning.fold_tunes[-1]].params[tuning.library]
    st.caption(f"Settings from the latest tune ({tuning.library}):")
    st.dataframe(
        [{"Setting": name, "Value": f"{value:.4g}"} for name, value in latest.items()],
        width="stretch",
        hide_index=True,
    )


def _render_tab(role: TargetRole, price_col: str) -> None:
    target_name = label(price_col)
    key = role.value
    if not signal_cols:
        st.info("Need at least one other signal column, alongside the target, to model.")
        return
    indexed = merged.set_index(DATE_COLUMN)
    panel = align_and_lag(indexed, signal_cols, price_col, horizon=horizon, transforms=transforms)
    _show_alignment(panel, target_name)

    tab_runs = model_runs.tab(stored, role, _polynomial_settings(key, panel))
    models = [POLYNOMIAL]
    # FF5 is an equity-factor benchmark, not designed to predict bond returns —
    # it would technically run and produce numbers, so it never runs here.
    if run_ff5 and role == TargetRole.EQUITY:
        models.append(FF5)
    if run_ml:
        models.append(ML)

    if st.button("Run", type="primary", key=f"run_{key}"):
        runners = {
            POLYNOMIAL: lambda: _run_polynomial(tab_runs.polynomial_settings, panel, price_col),
            FF5: lambda: _run_famafrench(price_col),
            ML: lambda: _run_ml(panel),
        }
        _run(models, tab_runs, run_one=lambda name: runners[name](), target_name=target_name)

    runs = tab_runs.runs
    if not runs:
        st.caption(f"Nothing has run for {target_name} with these settings yet.")
        return

    st.subheader(f"Results · {target_name}")
    for name in MODEL_ORDER:
        if name in runs:
            st.markdown(_gate_line(name, runs[name].result))
    _show_table({name: run.result for name, run in runs.items()})

    _show_screening(runs, target_name)
    if POLYNOMIAL in runs:
        with st.expander(f"Polynomial · {target_name}"):
            _show_polynomial(runs[POLYNOMIAL])
    if FF5 in runs and role == TargetRole.EQUITY:
        with st.expander(f"Fama-French 5 · {target_name}"):
            _show_famafrench(runs[FF5])
    if ML in runs:
        with st.expander(f"Machine learning · {target_name}"):
            _show_ml(runs[ML])


roles = [role for role in TargetRole if role in target_columns]
tabs = st.tabs([f"{ROLE_NAMES[r]} · {label(target_columns[r])}" for r in roles])
for role, tab in zip(roles, tabs, strict=True):
    with tab:
        _render_tab(role, target_columns[role])
