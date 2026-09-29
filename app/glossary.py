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
        "**Enter a function** fits the exact formula you write, and nothing "
        "else — use it to test a view you already hold.\n\n"
        "**Derive automatically** searches a small grid of polynomial degrees "
        "and regularizers and reports the best one. Regularization pushes weak "
        "terms to exactly zero, so the result stays short enough to read."
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
        "predicted and what actually happened, averaged across grading "
        "periods. 0 means no skill. In this field even 0.02–0.05 is a real "
        "edge, so treat a large value as a reason to look for a leak rather "
        "than a cause for celebration."
    ),
    "OOS Rank IC": (
        "Out-of-sample Rank IC: the same idea as IC, but comparing the *order* "
        "of predictions with the order of outcomes, on data the model never "
        "trained on. Using ranks stops one wild day from dominating the score, "
        "which is why this is the headline number and the one the promotion "
        "gate is set on."
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
}


def term(name: str) -> str:
    """The explanation for ``name``. Raises ``KeyError`` for an unknown term,
    so a typo on a page fails loudly instead of rendering an empty tooltip."""
    return TERMS[name]
