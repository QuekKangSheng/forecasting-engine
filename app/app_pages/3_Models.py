"""Models page: FYP-42's Fama-French benchmark, FYP-43's polynomial forecasting
function, and FYP-44's machine-learning models — every model family built so far,
sharing one walk-forward harness.

Reads the dataset committed via "Use Updated Data" on the Data page. No maths
lives here — fitting is in forecasting_engine.models.polynomial / famafrench /
boosted; the walk-forward loop is in forecasting_engine.validation.harness.
"""

from __future__ import annotations

import streamlit as st

import bloomberg_extraction_panel
import glossary
import ui
from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN
from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import PRODUCTION_LAG_DAYS, align_and_lag
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.models.boosted import BoostedConfigError, run_boosted
from forecasting_engine.models.famafrench import (
    FACTOR_COLUMNS,
    FamaFrenchDataError,
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
from forecasting_engine.reporting.model_metrics import FoldTerms, ModelRunResult
from forecasting_engine.reporting.polynomial_function import (
    Origin,
    PolynomialFunction,
    dataset_fingerprint,
    from_description,
    term_rows,
    to_latex,
)
from forecasting_engine.validation.splitters import PurgedWalkForward

#: Where each model family's latest result is parked for the Model Metrics page.
#: Duplicated as a matching constant in app_pages/4_Model_Metrics.py — a page
#: filename starting with a digit isn't a valid Python module name, so it can't
#: be imported from there.
POLYNOMIAL_RESULT_KEY = "polynomial_result"
POLYNOMIAL_DESCRIPTION_KEY = "polynomial_description"
FAMAFRENCH_RESULT_KEY = "famafrench_result"
FAMAFRENCH_DESCRIPTION_KEY = "famafrench_description"
ML_RESULT_KEY = "ml_result"
ML_DESCRIPTION_KEY = "ml_description"
#: The polynomial last run, as ``(PolynomialFunction, dataset_fingerprint)``.
POLYNOMIAL_FUNCTION_KEY = "polynomial_function"

#: Every result/description/function key above is namespaced by target role
#: (Task 4.2) so switching targets doesn't overwrite the other target's
#: stored result — two independent forecasting problems share this page.
TARGET_LABELS: dict[TargetRole, str] = {
    TargetRole.EQUITY: "Equity — S&P 500 (total return)",
    TargetRole.BOND: "Bond — US Aggregate (total return)",
}


def _role_key(base: str, role: TargetRole) -> str:
    return f"{base}_{role.value}"

#: The forecast horizons the validation framework asks for, reported separately
#: and never averaged.
HORIZONS: tuple[int, ...] = (1, 5)

#: One embargo shared by every horizon, equal to the longest of them, rather than
#: one per horizon. Picking h=1 used to drop the embargo to 1 as well.
EMBARGO_DAYS: int = max(HORIZONS)

st.set_page_config(page_title="Models · Forecasting Engine", page_icon=":material/functions:")
ui.inject()

st.header("Models")
st.caption(
    "Fit a model family through the shared walk-forward harness — results report to "
    "the Model Metrics page."
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

available_roles = [role for role in TargetRole if role in target_columns]
role = st.radio(
    "Target",
    available_roles,
    format_func=lambda r: TARGET_LABELS[r],
    horizontal=True,
    help=glossary.term("Target"),
)
price_col = target_columns[role]

numeric_cols = [c for c in merged.columns if c != DATE_COLUMN and merged[c].dtype.kind in "fi"]
# Every resolved target is excluded, not just the one currently selected —
# otherwise picking Equity as the target would still let the Bond column
# (or vice versa) ride along as a candidate signal.
signal_cols = [c for c in numeric_cols if c not in target_columns.values()]

family_options = ["Polynomial", "Fama-French 5-Factor", "Machine Learning"]
if role == TargetRole.BOND:
    # FF5 is an equity-factor benchmark, not designed to predict bond
    # returns — it would technically run and produce numbers, so it's kept
    # off the menu here rather than left to produce a meaningless result.
    family_options = [f for f in family_options if f != "Fama-French 5-Factor"]
family = st.radio(
    "Model family",
    family_options,
    horizontal=True,
    help=glossary.term("Model family"),
)

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

splitter = PurgedWalkForward(train=int(train), test=int(test), embargo=EMBARGO_DAYS)

result: ModelRunResult | None = None
description: ModelDescription | None = None
origin: Origin | None = None

if family == "Polynomial":
    result_key = _role_key(POLYNOMIAL_RESULT_KEY, role)
    description_key = _role_key(POLYNOMIAL_DESCRIPTION_KEY, role)

    if not signal_cols:
        st.info("Need at least one other signal column, alongside the price column, to model.")
        st.stop()
    indexed = merged.set_index(DATE_COLUMN)
    panel = align_and_lag(
        indexed, signal_cols, price_col, horizon=int(horizon)
    )

    st.subheader("Polynomial forecasting function")
    st.caption(
        "Define the function directly, or let the system derive one (degree up to "
        f"{MAX_DEGREE}, regularized)."
    )
    mode = st.radio(
        "Function source",
        ["Enter a function", "Derive automatically"],
        horizontal=True,
        help=glossary.term("Function source"),
    )

    if mode == "Enter a function":
        st.caption(f"Available signal columns: {', '.join(panel.signals)}")
        formula = st.text_input(
            "Function (arithmetic on signal columns only — e.g. `2 * vix + credit_spread_hy ** 2`)"
        )
        if st.button("Apply", type="primary") and formula:
            try:
                result, description = run_user_polynomial(formula, panel, splitter)
                origin = Origin.USER_SUPPLIED
            except PolynomialConfigError as exc:
                st.error(str(exc))
    else:
        degree_cols = st.columns(2)
        max_terms = degree_cols[0].number_input(
            "Max terms per candidate (optional cap)", min_value=1, value=10, step=1
        )
        st.caption(
            "Tries a small grid of degrees (1-3) and regularizers (Lasso, ElasticNet), "
            "compares them via PBO, and reports the one with the best out-of-sample rank IC."
        )
        if st.button("Derive", type="primary"):
            candidates = tuple(
                DerivedPolynomial(
                    degree=c.degree, regularizer=c.regularizer, max_terms=int(max_terms)
                )
                for c in CANDIDATE_CONFIGS
            )
            with st.spinner("Fitting candidate polynomials and estimating PBO…"):
                try:
                    result, description = run_derived_polynomial(
                        panel, splitter, candidates=candidates
                    )
                    origin = Origin.DERIVED
                except PolynomialConfigError as exc:
                    st.error(str(exc))

elif family == "Fama-French 5-Factor":
    result_key = _role_key(FAMAFRENCH_RESULT_KEY, role)
    description_key = _role_key(FAMAFRENCH_DESCRIPTION_KEY, role)

    st.subheader("Fama-French five-factor benchmark")
    factor_file = st.session_state.get(bloomberg_extraction_panel.FACTORS_KEY)
    if factor_file is None:
        st.info("No Fama-French factors loaded yet — download them on the Data page first.")
        st.stop()
    st.caption(
        "Regresses the target on the five Fama-French factors, lagged like any other "
        "signal, so every model's OOS Rank IC means the same thing. No configuration "
        "search happens, so no PBO is computed — same as a user-supplied polynomial."
    )
    try:
        with_factors = merge_factors(merged, factor_file.frame)
    except FamaFrenchDataError as exc:
        st.error(str(exc))
        st.stop()
    indexed = with_factors.set_index(DATE_COLUMN)
    panel = align_and_lag(
        indexed, list(FACTOR_COLUMNS), price_col, horizon=int(horizon)
    )

    if st.button("Fit", type="primary"):
        try:
            result, description = run_famafrench(panel, splitter)
        except FamaFrenchDataError as exc:
            st.error(str(exc))

else:
    result_key = _role_key(ML_RESULT_KEY, role)
    description_key = _role_key(ML_DESCRIPTION_KEY, role)

    if not signal_cols:
        st.info("Need at least one other signal column, alongside the price column, to model.")
        st.stop()
    indexed = merged.set_index(DATE_COLUMN)
    panel = align_and_lag(
        indexed, signal_cols, price_col, horizon=int(horizon)
    )

    st.subheader("Machine learning (XGBoost / LightGBM)")
    st.caption(
        "Tunes XGBoost and LightGBM once each via Optuna, refits both per walk-forward "
        "fold with those fixed hyperparameters, compares them via PBO, and reports the "
        "one with the best out-of-sample rank IC. Feature attribution below is by SHAP."
    )
    n_trials = st.number_input(
        "Optuna trials per library",
        min_value=1,
        value=10,
        step=1,
        help="Kept small by default — tuning happens once per run, not once per fold.",
    )
    if st.button("Tune & Fit", type="primary"):
        with st.spinner("Tuning XGBoost and LightGBM, then fitting per fold…"):
            try:
                result, description = run_boosted(panel, splitter, n_trials=int(n_trials))
            except BoostedConfigError as exc:
                st.error(str(exc))

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


def _show_fitted_terms(description: ModelDescription, *, is_ml: bool) -> None:
    value_col = "Mean |SHAP value|" if is_ml else "Coefficient"
    heading = "Feature attribution (SHAP)" if is_ml else "Fitted terms"
    st.markdown(ui.eyebrow(heading, glossary.term(heading)), unsafe_allow_html=True)
    if description.terms:
        rows = [
            {"Term": term, value_col: coefficient}
            for term, coefficient in zip(description.terms, description.coefficients, strict=True)
        ]
        if description.intercept is not None:
            rows.append({"Term": "(intercept)", value_col: description.intercept})
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("No terms survived fitting — every coefficient was regularized to zero.")


def _show_polynomial_function(
    fn: PolynomialFunction, fingerprint: tuple, terms: FoldTerms | None
) -> None:
    """The fitted polynomial as a labelled equation and term table."""
    label = labeller(numeric_cols)
    st.subheader(fn.origin, help=glossary.term("Fitted terms"))
    st.caption(f"Forecasts: {label(fn.target)}, {fn.horizon}-day return")
    if fingerprint != dataset_fingerprint(merged):
        st.warning("Fitted on a previous dataset. Run again to update.")
    elif fn.target != price_col or fn.horizon != int(horizon):
        st.warning(
            f"Fitted for {label(fn.target)}, {fn.horizon}-day return. The inputs above "
            "have changed, so run again to update."
        )
    st.latex(to_latex(fn, label))
    if fn.formula is not None:
        st.caption(
            "This function can't be written as separate terms and exponents (it divides "
            "by a signal, or expands to more than 50 terms), so it's shown as entered."
        )
        return
    if fn.origin == Origin.DERIVED:
        _show_fold_term_count(fn, terms)
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


if result is not None and description is not None:
    st.session_state[result_key] = result
    st.session_state[description_key] = description
    if origin is not None:
        # Saved in the same rerun as the run, so the function shown below is
        # always the one just fitted. A failed run saves nothing and leaves the
        # previous function in place.
        fn = from_description(
            description,
            origin=origin,
            target=price_col,
            horizon=int(horizon),
            columns=panel.signals,
        )
        st.session_state[_role_key(POLYNOMIAL_FUNCTION_KEY, role)] = (
            fn,
            dataset_fingerprint(merged),
        )
    st.success("Run complete — see Model Metrics for the full comparison.")

if result_key in st.session_state:
    if result is None:
        st.caption(
            "Showing the most recent run. Adjust the inputs above and run again to update it."
        )
    result = st.session_state[result_key]
    description = st.session_state[description_key]

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "IC", f"{result.ic:.4f}" if result.ic == result.ic else "—", help=glossary.term("IC")
    )
    oos_rank_ic_text = (
        f"{result.oos_rank_ic:.4f}" if result.oos_rank_ic == result.oos_rank_ic else "—"
    )
    metric_cols[1].metric("OOS Rank IC", oos_rank_ic_text, help=glossary.term("OOS Rank IC"))
    metric_cols[2].metric(
        "RMSE",
        f"{result.rmse:.4f}" if result.rmse == result.rmse else "—",
        help=glossary.term("RMSE"),
    )
    metric_cols[3].metric(
        "PBO", f"{result.pbo:.4f}" if result.pbo is not None else "N/A", help=glossary.term("PBO")
    )

    st.markdown(
        ui.eyebrow("Crash diagnostics", glossary.term("Crash diagnostics")),
        unsafe_allow_html=True,
    )
    crash = result.crash
    st.caption(
        f"Recall {crash.recall:.4f} · Precision {crash.precision:.4f} · F1 {crash.f1:.4f} "
        f"({crash.n_true_tail_days} true tail day(s) in the walk-forward test windows).",
        help=glossary.term("Crash diagnostics"),
    )

    polynomial_function_key = _role_key(POLYNOMIAL_FUNCTION_KEY, role)
    if family == "Polynomial" and polynomial_function_key in st.session_state:
        _show_polynomial_function(
            *st.session_state[polynomial_function_key], getattr(result, "terms", None)
        )
    else:
        _show_fitted_terms(description, is_ml=family == "Machine Learning")

    # Only runs that screen per fold (derived polynomial, machine learning) have
    # this. getattr, not attribute access: a result kept in session state from
    # before this field existed shouldn't take the page down.
    screening = getattr(result, "screening", None)
    if screening is not None:
        st.markdown(
            ui.eyebrow(
                "Signal inclusion across folds", glossary.term("Signal inclusion across folds")
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            f"How many of the {screening.folds} walk-forward folds were fit on each signal. "
            "Every fold screens signals on its own training window, so a signal can be "
            "kept in some folds and dropped in others.",
            help=glossary.term("Signal inclusion across folds"),
        )
        st.dataframe(
            [
                {"Signal": signal, "Folds": f"{used}/{screening.folds}"}
                for signal, used in screening.counts
            ],
            width="stretch",
            hide_index=True,
        )
        if screening.fell_back:
            st.caption(
                f"{screening.fell_back} of {screening.folds} folds kept no signal after "
                "screening, so they were fit on every signal instead."
            )
