"""Models page: FYP-42's Fama-French benchmark, FYP-43's polynomial forecasting
functions and FYP-44's machine-learning models, run together per target through
one walk-forward harness under one set of settings and compared side by side
(FYP-161), with FYP-162's check of whether a forecast's direction would have paid.

Reads the dataset committed via "Use Updated Data" on the Data page. No maths
lives here — fitting is in forecasting_engine.models.polynomial / famafrench /
boosted; the walk-forward loop is in forecasting_engine.validation.harness;
table formatting is in forecasting_engine.reporting.model_metrics.
"""

from __future__ import annotations

import html

import pandas as pd
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
from forecasting_engine.models.naive import NaiveDataError, run_naive
from forecasting_engine.models.polynomial import (
    CANDIDATE_CONFIGS,
    CLIP_SD,
    MAX_DEGREE,
    DerivedPolynomial,
    PolynomialConfigError,
    placeholders,
    run_derived_polynomial,
    run_user_polynomial,
)
from forecasting_engine.portfolio.directional import (
    DirectionalDataError,
    directional_pnl,
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
    shape_latex,
    term_rows,
    to_latex,
)
from forecasting_engine.store import active_model
from forecasting_engine.validation.gates import evaluate_candidate, is_high_risk
from forecasting_engine.validation.splitters import TUNING_ROWS, PurgedWalkForward
from model_settings import (
    DEFAULT_HORIZON,
    DEFAULT_MAX_TERMS,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    EMBARGO_DAYS,
    HORIZONS,
)

#: Model names as the comparison table knows them (``MODEL_ORDER``).
NAIVE, FF5, DERIVED, USER, ML = MODEL_ORDER

ACTIVE_MODEL_CANDIDATES = (DERIVED, USER, ML)

#: The polynomial's two sources; exactly one runs.
DERIVE_OPTION, OWN_OPTION = "Derive automatically", "Use your own function"

ROLE_NAMES: dict[TargetRole, str] = {TargetRole.EQUITY: "Equity", TargetRole.BOND: "Bond"}

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
    "Rank IC within folds",
    "Beyond 2 s.e.",
    "Constant folds",
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
    "Rank IC within folds": "Rank IC within folds",
    "Beyond 2 s.e.": "Beyond 2 s.e.",
    "Constant folds": "Constant folds",
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
        default=DEFAULT_HORIZON,
        required=True,
        format_func=lambda days: f"{days} day" if days == 1 else f"{days} days",
        help=glossary.term("Forecast horizon"),
    )
    cols = st.columns(2)
    train = cols[0].number_input(
        "Walk-forward train window (days)",
        min_value=10,
        value=DEFAULT_TRAIN_WINDOW,
        step=10,
        help=glossary.term("Walk-forward train window (days)"),
    )
    test = cols[1].number_input(
        "Walk-forward test window (days)",
        min_value=1,
        value=DEFAULT_TEST_WINDOW,
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
run_poly = check_cols[0].checkbox("Polynomial", value=True)
run_ff5 = check_cols[1].checkbox(
    "Fama-French 5", value=True, help="An equity-factor benchmark, so it runs on Equity only."
)
run_ml = check_cols[2].checkbox("Machine learning", value=True)

horizon = int(horizon)
splitter = PurgedWalkForward(
    train=int(train), test=int(test), embargo=EMBARGO_DAYS, tuning_rows=TUNING_ROWS
)
shared_settings = (dataset_fingerprint(merged), horizon, int(train), int(test))
cleared = model_runs.cleared_by(st.session_state, shared_settings)
stored = model_runs.stored(st.session_state, shared_settings)

#: What every row on the page was run under, stated beside the results (FYP-161).
SETTINGS_STAMP = (
    f"{horizon}-day horizon · walk-forward train {int(train)} / test {int(test)} days · "
    f"embargo {EMBARGO_DAYS} days · signals lagged {PRODUCTION_LAG_DAYS} day · "
    f"first {TUNING_ROWS} dates kept for tuning"
)

ACTIVE_SETTINGS: dict[str, int | str] = {
    "horizon": horizon,
    "train_window": int(train),
    "test_window": int(test),
    "dataset_fingerprint": str(dataset_fingerprint(merged)),
}


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


def _kept(widget_key: str, default: object) -> object:
    """The last value a widget held. Streamlit forgets a widget's value while it is
    hidden, so each input is kept under its own key and restored when shown again;
    hiding an input is then never a change of setting."""
    return st.session_state.get(f"{widget_key}_kept", default)


def _keep(widget_key: str, value: object) -> None:
    st.session_state[f"{widget_key}_kept"] = value


def _polynomial_settings(key: str, panel: FeaturePanel) -> tuple[str | None, dict[str, object]]:
    """Which polynomial runs, if any, and each polynomial row's own setting.

    Only the chosen source's input is shown. Each row depends only on its own
    setting, so switching source or unticking clears nothing."""
    terms_key, formula_key = f"terms_{key}", f"formula_{key}"
    settings: dict[str, object] = {
        DERIVED: int(_kept(terms_key, DEFAULT_MAX_TERMS)),
        USER: _user_function(key, str(_kept(formula_key, "")), panel),
    }
    if not run_poly:
        return None, settings
    st.markdown(
        ui.eyebrow("Polynomial", glossary.term("Function source")), unsafe_allow_html=True
    )
    source = st.radio(
        "Function source",
        (DERIVE_OPTION, OWN_OPTION),
        horizontal=True,
        label_visibility="collapsed",
        key=f"poly_source_{key}",
    )
    if source == DERIVE_OPTION:
        max_terms = st.number_input(
            "Max terms per candidate (optional cap)",
            min_value=1,
            value=settings[DERIVED],
            step=1,
            key=terms_key,
        )
        _keep(terms_key, int(max_terms))
        settings[DERIVED] = int(max_terms)
        st.caption(
            f"Tries a small grid of degrees (1-3, of up to {MAX_DEGREE} allowed) and "
            "regularizers (Lasso, ElasticNet), compares them via PBO, and reports the "
            "one with the best out-of-sample rank IC."
        )
        return DERIVED, settings
    inputs, table = st.columns([2, 3])
    with inputs:
        formula = st.text_input(
            "Function — write its shape with placeholders, then pick each one's signal",
            value=str(_kept(formula_key, "")),
            placeholder=_example(panel),
            key=formula_key,
            help=glossary.term("Function source"),
        )
        _keep(formula_key, formula)
        settings[USER] = _user_function(key, formula, panel, show=True)
    with table:
        _show_signal_table(panel)
    return USER, settings


def _example(panel: FeaturePanel) -> str:
    """An example formula, and the first two real signals its placeholders would mean."""
    first, *rest = panel.signals
    if not rest:
        return f"e.g. 2 * x, with x = {first}"
    return f"e.g. x - 0.5 * y ** 2, with x = {first} and y = {rest[0]}"


def _user_function(
    key: str, formula: str, panel: FeaturePanel, *, show: bool = False
) -> tuple[str, tuple[tuple[str, str], ...] | None]:
    """The formula and the signal each of its placeholders stands for, or no
    bindings if the formula can't be read (its error is shown under the box).

    Each placeholder gets a dropdown of the signals. A placeholder that is already
    a signal's column name defaults to it; the others default to the signals in
    order, so ``x`` and ``y`` start as the first two."""
    formula = formula.strip()
    if not formula:
        return "", ()
    try:
        names = placeholders(formula)
    except PolynomialConfigError as exc:
        if show:
            st.error(f"{USER}: {exc}", icon=":material/error:")
        return formula, None
    signals = list(panel.signals)
    bindings = []
    unnamed = iter(s for s in signals if s not in names)
    for name in names:
        widget_key = f"bind_{key}_{name}"
        default = name if name in signals else next(unnamed, signals[0])
        chosen = _kept(widget_key, default)
        if chosen not in signals:
            chosen = default
        if show:
            chosen = st.selectbox(
                f"{name} stands for",
                signals,
                index=signals.index(chosen),
                format_func=lambda s: f"{label(s)} ({s})",
                key=widget_key,
            )
            _keep(widget_key, chosen)
        bindings.append((name, chosen))
    return formula, tuple(bindings)


def _show_signal_table(panel: FeaturePanel) -> None:
    """The signals a function can use: what each is, and what the model sees."""
    st.dataframe(
        [
            {
                "Column": signal,
                "Security": sources[signal].security if signal in sources else "—",
                "Field": sources[signal].field if signal in sources else "—",
                "Transform": str(panel.alignment[signal].transform),
                "Lag": f"{PRODUCTION_LAG_DAYS} day",
                "Latest value": _latest(panel.frame[signal]),
            }
            for signal in panel.signals
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Latest value is after the transform and the lag: what the function is "
        "given, not the raw export."
    )


def _latest(series: pd.Series) -> str:
    present = series.dropna()
    return f"{present.iloc[-1]:.4g}" if len(present) else "—"


def _run_derived(max_terms: int, panel: FeaturePanel, price_col: str) -> model_runs.ModelRun:
    candidates = tuple(
        DerivedPolynomial(degree=c.degree, regularizer=c.regularizer, max_terms=max_terms)
        for c in CANDIDATE_CONFIGS
    )
    result, description = run_derived_polynomial(panel, splitter, candidates=candidates)
    fn = from_description(
        description,
        origin=Origin.DERIVED,
        target=price_col,
        horizon=horizon,
        columns=panel.signals,
    )
    return model_runs.ModelRun(result, description, function=fn)


def _run_user(
    function: tuple[str, tuple[tuple[str, str], ...]], panel: FeaturePanel, price_col: str
) -> model_runs.ModelRun:
    formula, bindings = function
    result, description = run_user_polynomial(formula, panel, splitter, dict(bindings))
    fn = from_description(
        description,
        origin=Origin.USER_SUPPLIED,
        target=price_col,
        horizon=horizon,
        columns=panel.signals,
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
                NaiveDataError,
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
        f'<div class="fe-table-wrap"><table class="fe-table"><thead><tr>{header}</tr>'
        f"</thead><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _gate_line(name: str, result: ModelRunResult) -> str:
    if name == NAIVE:
        return f"**{name}**: the baseline to beat — not gated."
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
    screenings = [runs[n].result.screening for n in (DERIVED, ML) if n in runs]
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
    if fn.origin == Origin.USER_SUPPLIED:
        st.latex(shape_latex(fn, label))
        st.caption(
            "Your function sets the shape; the scale and intercept are fitted by least "
            "squares on each fold's training window. Shown: the latest fold's fit."
        )
    else:
        st.latex(to_latex(fn, label))
    bounds = run.description.input_bounds
    if bounds:
        ranges = "; ".join(f"{label(s)} {lo:.4g} to {hi:.4g}" for s, (lo, hi) in bounds.items())
        st.caption(
            f"Each input is first clipped to its latest training window's mean ± {CLIP_SD:g} "
            f"standard deviations: {ranges}. Outside those, the equation applies at the bound."
        )
    if fn.formula is not None:
        st.caption(
            "This function can't be written as separate terms and exponents (it divides "
            "by a signal, or expands to more than 50 terms), so it has no term table."
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


def _show_active_status(role: TargetRole, target_name: str) -> None:
    current = active_model.get_active_model(role)
    if current is None:
        st.caption(
            f"No active model set yet for {target_name}.", help=glossary.term("Active model")
        )
        return
    tone = "danger" if current.high_risk else "success"
    note = " · set despite failing both gates" if current.high_risk else ""
    st.markdown(
        ui.status_row(
            f"Active model · {target_name}",
            ui.lozenge(current.model_name, tone),
            f"set {current.set_at:%d %b %Y}{note}",
        ),
        unsafe_allow_html=True,
    )


@st.dialog("Confirm high-risk model")
def _confirm_high_risk(
    role: TargetRole,
    model_name: str,
    result: ModelRunResult,
    failed: tuple[str, ...],
    target_name: str,
) -> None:
    failed_text = " and ".join(GATE_NAMES[g] for g in failed)
    st.warning(
        f"{model_name} failed both promotion gates ({failed_text}) for {target_name}. "
        "Setting it active anyway means portfolio evaluation will use a model that "
        "hasn't cleared validation.",
        icon=":material/warning:",
    )
    cols = st.columns(2)
    if cols[0].button("Set active anyway", type="primary", key=f"confirm_{role.value}"):
        active_model.set_active_model(
            role, model_name, result, high_risk=True, **ACTIVE_SETTINGS
        )
        st.rerun()
    if cols[1].button("Cancel", key=f"cancel_{role.value}"):
        st.rerun()


def _show_active_picker(
    role: TargetRole, runs: dict[str, model_runs.ModelRun], target_name: str
) -> None:
    key = role.value
    options = [n for n in ACTIVE_MODEL_CANDIDATES if n in runs]
    if not options:
        st.caption(f"No forecasting model (Polynomial or ML) has run yet for {target_name}.")
        return
    current = active_model.get_active_model(role)
    default_index = (
        options.index(current.model_name) if current and current.model_name in options else 0
    )
    st.markdown(
        ui.eyebrow("Set active model", glossary.term("Active model")), unsafe_allow_html=True
    )
    cols = st.columns([3, 1])
    selected = cols[0].selectbox(
        "Model to set active",
        options,
        index=default_index,
        key=f"active_select_{key}",
        label_visibility="collapsed",
    )
    if cols[1].button("Set as active", key=f"set_active_{key}"):
        result = runs[selected].result
        outcome = evaluate_candidate(result.oos_rank_ic, result.pbo)
        if is_high_risk(outcome):
            _confirm_high_risk(role, selected, result, outcome.failed_gates, target_name)
        else:
            active_model.set_active_model(
                role, selected, result, high_risk=False, **ACTIVE_SETTINGS
            )
            st.rerun()


def _directional_default(role: TargetRole, options: list[str], runs) -> int:
    """The active model if it ran here, else the first that met its gate, else the
    first forecasting model — the one a portfolio manager would act on."""
    current = active_model.get_active_model(role)
    if current is not None and current.model_name in options:
        return options.index(current.model_name)
    for i, name in enumerate(options):
        result = runs[name].result
        if result.pbo is not None and evaluate_candidate(result.oos_rank_ic, result.pbo).promoted:
            return i
    forecasting = [i for i, name in enumerate(options) if name in ACTIVE_MODEL_CANDIDATES]
    return forecasting[0] if forecasting else 0


def _show_directional(
    role: TargetRole, runs: dict[str, model_runs.ModelRun], target_name: str
) -> None:
    """FYP-162: holding the index only when the forecast says it will rise, against
    holding it throughout, over the run's whole out-of-sample period."""
    options = [
        n
        for n in MODEL_ORDER
        if n in runs and runs[n].result.forecast is not None and runs[n].result.realised is not None
    ]
    if not options:
        return
    key = role.value
    st.markdown(
        ui.eyebrow("Would the forecast's direction have paid?", glossary.term("Directional P&L")),
        unsafe_allow_html=True,
    )
    name = st.selectbox(
        "Forecasts from",
        options,
        index=_directional_default(role, options, runs),
        key=f"pnl_model_{key}",
    )
    result = runs[name].result
    try:
        pnl = directional_pnl(result.forecast, result.realised, horizon=horizon)
    except DirectionalDataError as exc:
        st.info(str(exc), icon=":material/info:")
        return

    steps = (
        "one call per day" if horizon == 1 else f"one call every {horizon} days, never overlapping"
    )
    st.caption(
        f"{target_name}, {pnl.start:%d/%m/%Y} to {pnl.end:%d/%m/%Y}: all {pnl.days} "
        f"out-of-sample trading days, {pnl.calls} calls ({steps}). "
        "Gross of transaction costs; cash earns nothing. On the chart, scroll to zoom, "
        "drag to pan and double-click to reset."
    )
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Long/cash strategy",
        f"{pnl.strategy_cumulative.iloc[-1]:+.2%}",
        help=glossary.term("Long/cash strategy"),
    )
    metric_cols[1].metric(
        "Buy and hold",
        f"{pnl.buy_and_hold_cumulative.iloc[-1]:+.2%}",
        help=glossary.term("Buy and hold"),
    )
    metric_cols[2].metric(
        "Hit rate",
        "—" if pnl.hit_rate != pnl.hit_rate else f"{pnl.hit_rate:.0%}",
        help=glossary.term("Hit rate"),
    )
    metric_cols[3].metric(
        "Days invested", f"{pnl.share_invested:.0%}", help=glossary.term("Days invested")
    )
    st.line_chart(
        {
            "Long/cash strategy": pnl.strategy_cumulative * 100,
            "Buy and hold": pnl.buy_and_hold_cumulative * 100,
        },
        x_label="Forecast date",
        y_label="Cumulative return (%)",
    )
    if pnl.no_forecast:
        st.caption(
            f"{pnl.no_forecast} of {pnl.calls} calls had no forecast (a signal was "
            "missing that day), so the strategy stayed in cash and the hit rate leaves them out."
        )
    other = next(h for h in HORIZONS if h != horizon) if len(HORIZONS) > 1 else None
    if other is not None:
        st.caption(
            f"This is the {horizon}-day horizon. Each horizon is reported separately: "
            f"switch to {other} {'day' if other == 1 else 'days'} in Settings and run "
            "again to see it."
        )


def _render_tab(role: TargetRole, price_col: str) -> None:
    target_name = label(price_col)
    key = role.value
    _show_active_status(role, target_name)
    if not signal_cols:
        st.info("Need at least one other signal column, alongside the target, to model.")
        return
    indexed = merged.set_index(DATE_COLUMN)
    panel = align_and_lag(indexed, signal_cols, price_col, horizon=horizon, transforms=transforms)
    _show_alignment(panel, target_name)

    polynomial, settings = _polynomial_settings(key, panel)
    tab_runs = model_runs.tab(stored, role, settings)
    models = [NAIVE]
    user_formula, user_bindings = settings[USER]
    if polynomial == DERIVED or (polynomial == USER and user_formula and user_bindings is not None):
        models.append(polynomial)
    # FF5 is an equity-factor benchmark, not designed to predict bond returns —
    # it would technically run and produce numbers, so it never runs here.
    if run_ff5 and role == TargetRole.EQUITY:
        models.append(FF5)
    if run_ml:
        models.append(ML)

    if polynomial == USER and not user_formula:
        st.caption("Enter a function for the user-supplied polynomial to run.")
    nothing_to_run = models == [NAIVE]
    if st.button("Run", type="primary", key=f"run_{key}", disabled=nothing_to_run):
        runners = {
            NAIVE: lambda: model_runs.ModelRun(*run_naive(panel, splitter)),
            DERIVED: lambda: _run_derived(settings[DERIVED], panel, price_col),
            USER: lambda: _run_user(settings[USER], panel, price_col),
            FF5: lambda: _run_famafrench(price_col),
            ML: lambda: _run_ml(panel),
        }
        _run(models, tab_runs, run_one=lambda name: runners[name](), target_name=target_name)
    if nothing_to_run:
        st.caption(
            "Tick a model besides the naive baseline to run: the baseline is only read "
            "against another model."
        )

    runs = tab_runs.runs
    if not runs:
        if cleared:
            st.caption(
                f"Earlier results for {target_name} were cleared because the data or a shared "
                "setting changed, so a table never mixes rows run under different settings."
            )
        st.caption(f"Nothing has run for {target_name} with these settings yet.")
        return

    st.subheader(f"Results · {target_name}")
    st.caption(f"Every row was run under: {SETTINGS_STAMP}.")
    for name in MODEL_ORDER:
        if name in runs:
            st.markdown(_gate_line(name, runs[name].result))
    _show_table({name: run.result for name, run in runs.items()})
    _show_active_picker(role, runs, target_name)
    _show_directional(role, runs, target_name)

    _show_screening(runs, target_name)
    for name in (DERIVED, USER):
        if name in runs:
            with st.expander(f"{name} · {target_name}"):
                _show_polynomial(runs[name])
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
