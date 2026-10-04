"""Plain-language explanations of the jargon on the dashboard.

Every entry answers two questions in order: **what it is**, then **why it
matters** — a number a portfolio manager can't act on is a number that
shouldn't be on screen. They are rendered as Streamlit ``help=`` tooltips (the
small ⓘ beside a control, heading or metric), so they are read in passing and
must stay short.

Wording is a working default, not team-agreed. It lives here rather than
inline on the pages so a term reads the same wherever it appears, and so the
team can revise it in one place.
"""

from __future__ import annotations

from collections.abc import Mapping

#: Term -> explanation. Keys are the on-screen wording, so a reader searching
#: for what they saw finds it.
TERMS: Mapping[str, str] = {
    "Target": (
        "The price or level being forecast. Everything else in the dataset is "
        "treated as a signal that might help predict its future return. "
        "Choosing the target decides what the model is actually for: an equity "
        "forecast and a bond forecast are different models, not one model with "
        "a setting changed."
    ),
    "Function source": (
        "Choose one. **Derive automatically** searches a small grid of polynomial "
        "degrees and regularizers and reports the best one. Regularization pushes "
        "weak terms to exactly zero, so the result stays short enough to read.\n\n"
        "**Use your own function** takes the shape you write, using placeholders "
        "such as x and y that you then point at signals, and fits only a scale "
        "and an intercept to it on each training window, so the forecast is a "
        "return rather than a signal's level. A view you already hold is judged "
        "under the same settings as every other model. Each gets its own row, "
        "named for which kind it is."
    ),
    "Forecast horizon": (
        "How many trading days ahead to predict. Each horizon is run and "
        "reported separately — the two are never averaged, because a signal "
        "that works over a week often does nothing over a day."
    ),
    "Walk-forward": (
        "The model is trained on a block of history, then graded on the days "
        "immediately after it, and the whole window slides forward and repeats. "
        "Each grade therefore comes from data the model had never seen at the "
        "time — which is the only honest way to estimate how it would have "
        "done live. Grading a model on the same data it learned from always "
        "flatters it."
    ),
    "Walk-forward train window (days)": (
        "How much history the model studies before each grading period "
        "(120 days ≈ 6 months). Longer gives the fit more to learn from; "
        "shorter keeps it closer to current market conditions."
    ),
    "Walk-forward test window (days)": (
        "The days right after each training block, used only for grading. "
        "The model never sees them while fitting, so its score here is its "
        "out-of-sample score."
    ),
    "Embargo": (
        "A gap of unused days between a training block and the days it is "
        "graded on. A 5-day forecast made on the last training day is only "
        "settled 5 days later, so without the gap the model would be trained "
        "on an outcome it is about to be graded on — a leak that makes results "
        "look far better than they are. Fixed at the longest horizon."
    ),
    "Signal lag": (
        "Every signal is shifted forward a day, so a value dated today is one "
        "that had already been published today. Without it, a model can "
        "'predict' using a number nobody had yet — the most common way a "
        "backtest ends up worthless."
    ),
    "IC": (
        "Information Coefficient: the correlation between what the model "
        "predicted and what actually happened, over every grading period "
        "together. 0 means no skill. In this field even 0.02–0.05 is a real "
        "edge, so treat a large value as a reason to look for a leak rather "
        "than a cause for celebration."
    ),
    "OOS Rank IC": (
        "Out-of-sample Rank IC: the same idea as IC, but comparing the *order* "
        "of predictions with the order of outcomes, on data the model never "
        "trained on. Using ranks stops one wild day from dominating the score, "
        "which is why this is the headline number and the one the promotion "
        "gate is set on. The two s.e.s beside it are standard errors, one "
        "allowing for overlapping labels, one for errors shared within a test "
        "window: a value within about two of the larger of zero may be luck."
    ),
    "Rank IC within folds": (
        "The OOS Rank IC again, but with each fold's forecasts ranked only against "
        "each other before pooling. Pooling ranks forecasts across folds, so a model "
        "can score there just because its forecast level shifts from fold to fold — "
        "a model that uses no signal at all can pass the gate that way. Here it "
        "can't: only ranking days within a fold counts. Far below the OOS Rank IC "
        "means most of that score came from levels, not from the signals."
    ),
    "Beyond 2 s.e.": (
        "Whether the OOS Rank IC is more than two standard errors from zero, "
        "judged on the larger of the two standard errors. No means a score of "
        "this size could easily be luck, even if it meets the gate. A diagnostic "
        "only: it does not change whether the gate is met."
    ),
    "Constant folds": (
        "How many walk-forward folds forecast a single value for their whole test "
        "window. Such a fold ranks no day above another, so it adds nothing to "
        "the Rank IC within folds, but it still counts towards the pooled score "
        "through its level. The naive baseline is constant in every fold by design."
    ),
    "RMSE": (
        "Root mean squared error: the typical size of a miss, in the same "
        "units as the return being predicted. Lower is better. It says how far "
        "off the model is, whereas IC says whether it got the direction right "
        "— a model can do well on one and badly on the other."
    ),
    "PBO": (
        "Probability of Backtest Overfitting: when several configurations are "
        "tried, how often the best-looking one turns out to be below average "
        "on data held back from the search. High PBO means the winner was "
        "probably luck. Reported as N/A when only one configuration was fitted, "
        "since there was no search to overfit."
    ),
    "Crash diagnostics": (
        "How well the model's most negative predictions line up with the days "
        "that actually fell hardest. **Recall** is the share of real crash days "
        "it flagged, **precision** the share of its flags that were real, and "
        "**F1** the balance of the two. Diagnostic only — never a pass/fail "
        "bar — because crash days are rare, so these figures move a lot on very "
        "few observations."
    ),
    "Signal inclusion across folds": (
        "Each training block screens the signals on its own history and keeps "
        "the ones that look useful, so the chosen set can differ block to "
        "block. A signal kept everywhere is robust; one kept only occasionally "
        "is probably noise that happened to fit."
    ),
    "Fitted terms": (
        "The model as an equation: each term's factors, their powers, and the "
        "weight fitted to them. This is what the model would actually use if "
        "deployed today — reading it is how you sanity-check that it depends "
        "on what you expect, and in the direction you expect."
    ),
    "Factor": (
        "The signal a term is built from, by its plain-language name. Two names "
        "joined by × is an interaction: the two multiplied together. The "
        "(intercept) row is not a signal at all — it is the predicted return "
        "when every signal sits at zero, the baseline the other terms adjust."
    ),
    "Coefficient": (
        "How much the prediction moves per unit of that term. The sign is the "
        "direction of the relationship; the size depends on the units of the "
        "signal, so compare signs and relative magnitudes rather than reading "
        "one number on its own."
    ),
    "Exponent": (
        "The power a factor is raised to. 1 is a straight-line effect, 2 means "
        "the effect grows with the square, and two factors listed together is "
        "an interaction — the effect of one depends on the level of the other."
    ),
    "Feature attribution (SHAP)": (
        "A boosted model has no equation to read, so each signal is scored by "
        "how much it moved the predictions. Larger means more influential — it "
        "says nothing about direction, only weight."
    ),
    "Promotion gate": (
        "The bar a model has to clear before it is considered for use: OOS Rank "
        "IC above the threshold and PBO below it. The badge shows each "
        "separately, so a model that scores well but overfits is visibly not "
        "promotable."
    ),
    "Active model": (
        "Which model's forecast feeds portfolio evaluation for this target, "
        "set independently for equity and bond. Setting a new one replaces "
        "the prior one immediately. A model that failed both promotion gates "
        "can still be set active, but asks for confirmation first."
    ),
    "Risk aversion (λ)": (
        "How heavily risk is weighed against expected return when splitting "
        "between equity and bond. Higher pulls the allocation toward whichever "
        "is less volatile; lower chases the forecast with the better return."
    ),
    "Weight bounds": (
        "The minimum and maximum either asset may be allocated, so a small, "
        "noisy difference between the two forecasts can't swing the portfolio "
        "to one extreme."
    ),
    "Directional P&L": (
        "What you would have earned by holding this index only when the model "
        "forecast a rise, and sitting in cash otherwise, compared with simply "
        "holding it. Only the forecast's direction is used, never its size, so "
        "it is a plain check of whether the up/down call is worth acting on. "
        "Every day the model was graded on is replayed, never a day it trained "
        "on, so the result is what following it live would have looked like."
    ),
    "Long/cash strategy": (
        "Compounded return of holding the index on every call where the "
        "forecast was positive and holding cash on every other. Before "
        "transaction costs, and cash earns nothing."
    ),
    "Buy and hold": (
        "Compounded return of holding the index over the whole out-of-sample "
        "period, whatever the forecast said. The strategy has to beat this for "
        "the forecast's direction to have been worth following."
    ),
    "Hit rate": (
        "How often the forecast's direction matched what the index actually "
        "did: a forecast rise followed by a rise, or a forecast fall followed by "
        "a fall. 50% is a coin toss."
    ),
    "Days invested": (
        "The share of calls on which the strategy held the index. A low share "
        "with a good return means the model avoided the market's bad patches; a "
        "share near 100% means it behaves almost like buy and hold."
    ),
    "Annual return": (
        "The yearly growth rate that, compounded, turns the starting value into "
        "the ending value. It is the headline figure, but says nothing about the "
        "risk taken to earn it — read it with the ratios below."
    ),
    "Sharpe": (
        "Average return above the risk-free rate per unit of total volatility, "
        "annualised. Higher means more return for the ups and downs endured; it "
        "is the standard way to compare two portfolios with different risk."
    ),
    "Sortino": (
        "Like Sharpe, but only counts downside volatility, so a portfolio is not "
        "penalised for large gains. Higher is better; it matters most when "
        "returns are lopsided."
    ),
    "Calmar": (
        "Annual return divided by the size of the worst drawdown. Higher means "
        "the return was earned without deep losses along the way — the measure "
        "closest to how painful a portfolio was to hold."
    ),
    "Max drawdown": (
        "The worst fall from a previous high before recovering, as a percentage. "
        "It is the loss an investor who bought at the worst moment would have "
        "sat through, so a smaller (less negative) figure is better."
    ),
    "Equal-weight benchmark": (
        "Half in the equity index and half in the bond index, reset to 50/50 on "
        "the last trading day of each month. The optimised portfolio has to beat "
        "this simple allocation to justify the forecasting behind it."
    ),
}


def term(name: str) -> str:
    """The explanation for ``name``. Raises ``KeyError`` for an unknown term,
    so a typo on a page fails loudly instead of rendering an empty tooltip."""
    return TERMS[name]
