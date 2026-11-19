"""What each model is and what it reads, in a paragraph a reader who never opens the
methodology can follow: for a run's report, under each active model.

Keyed by the names in ``reporting.model_metrics.MODEL_ORDER``. The two research models
also carry the caveat their scores are shown with, since their inputs were chosen by
looking at results over part of the period they are scored on.
"""

from __future__ import annotations

from collections.abc import Mapping

from forecasting_engine.models.hybrid import NAME as BOND_DERIVED_FIT
from forecasting_engine.reporting.model_metrics import MODEL_ORDER

NAIVE, FF5, DERIVED, RESEARCH, BOND_RESEARCH, USER, ML = MODEL_ORDER

DESCRIPTIONS: Mapping[str, str] = {
    NAIVE: (
        "The baseline every other model has to beat: it forecasts the index's average "
        "return over its training window, whatever the market is doing."
    ),
    FF5: (
        "A standard academic benchmark for equities: the index's return forecast from the "
        "latest moves of the five Fama-French factors (the market, company size, value, "
        "profitability and investment), from Ken French's published daily data, read as "
        "signals like any other."
    ),
    DERIVED: (
        "A polynomial derived from the data: in each training period it keeps the signals "
        "that have ranked future returns best, then fits a short formula in them (degree "
        "1 to 3, including terms that multiply two signals together), pushing weak terms "
        "to zero. The forecast is that formula applied to the latest values of the "
        "signals it kept."
    ),
    RESEARCH: (
        "Forecasts the equity index's return from four fixed economic inputs: the "
        "high-yield and investment-grade credit spreads, VIX and the slope between the "
        "2-year and 10-year Treasury yields, each read as a level. Each is read as part of "
        "the risk premium equity investors demand, so a higher value means a higher "
        "expected return, and the four slopes are pulled towards one shared value."
    ),
    BOND_RESEARCH: (
        "Forecasts the bond index's return from the levels of rates, spreads and "
        "volatility (the 10-year yield and breakeven inflation, the 2s10s slope, "
        "high-yield and investment-grade credit spreads, VIX and FX implied volatility), "
        "read as the yield the bond earns, its carry, rather than as a bet on how they "
        "will move."
    ),
    USER: (
        "A formula written by the user in the signals of their choice. Only its scale "
        "and intercept are fitted, on each training period, so the run tests a view the "
        "user already holds under the same rules as every other model."
    ),
    ML: (
        "Gradient-boosted decision trees: XGBoost and LightGBM, each tuned on past data "
        "only, with the one that scored better kept. Trees can pick up thresholds and "
        "interactions between signals that a polynomial can't, at the cost of a formula "
        "a reader can check; the signals driving them most are listed below (SHAP)."
    ),
}

FIT_DESCRIPTIONS: Mapping[str, str] = {
    BOND_DERIVED_FIT: (
        "The bond index's derived polynomial: the average of 14 formulas fitted to the "
        "data, none of them picked over the others. Each fits the levels of rates, credit "
        "spreads and volatility (where a bond's carry shows) by a method that corrects for "
        "how a level's own moves flatter it in a short sample, and fits the daily changes, "
        "returns and, in some, how many month-ends and similar days the forecast window "
        "holds, by one of the four fits the equity polynomial chooses between. Half use "
        "the latest 504 trading days, half up to the latest 2,520. It held up on 2008–2026 "
        "and on public data for 1992–2006, which no earlier bond model had been tested on, "
        "but it is provisional until tested on data after 12 Nov 2026, and at the 5-, "
        "10- and 20-day horizons its signal is positive but not statistically significant."
    ),
}
"""For a model whose row name has had more than one fit behind it, keyed by the stored
function's own name (``ModelDescription.name``): the bond derived polynomial is not the
grid its row name ran before 18 Nov 2026, nor equity's."""

RESEARCH_CAVEATS: Mapping[str, str] = {
    RESEARCH: (
        "Its scores are optimistic: its inputs were chosen by looking at results over "
        "Aug 2016 – Sep 2024, which overlaps the period it is scored on."
    ),
    BOND_RESEARCH: (
        "Its scores are optimistic: its inputs were chosen by looking at results on the "
        "Sep 2026 upload (Aug 2016 – Sep 2026), which overlaps the period it is scored on."
    ),
}


def describe(model: str, fit: str | None = None) -> str:
    """``model``'s description, that of its stored function ``fit`` where that has its
    own (``FIT_DESCRIPTIONS``), or a plain note that there is none for a name this module
    doesn't know (an old run's model since renamed beyond recognition)."""
    if fit in FIT_DESCRIPTIONS:
        return FIT_DESCRIPTIONS[fit]
    return DESCRIPTIONS.get(model, f"No description is kept for {model}.")
