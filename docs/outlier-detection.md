# Outlier detection: method, calibration and evidence

How the engine decides a signal value is extreme, and why it is done this way.
Every number below was measured on the ten years of Bloomberg daily data
available on 2026-09-02 — `spx_close`, `agg_close`, `vix`, `credit_spread_ig`
and `breakeven_10y`, 2,610 rows — not chosen from a textbook.

## The method

For each signal, take the day-over-day change, score every change by its
distance from the median in units of the median absolute deviation, and flag
anything beyond the threshold.

```
score = 0.6745 × |change − median(change)| / median(|change − median(change)|)
```

The 0.6745 is the 75th percentile of the standard normal, which puts the median
absolute deviation on the same scale as a standard deviation, so a score reads
like a z-score.

**Threshold: 8**, as `MAD_THRESHOLD` in
`src/forecasting_engine/extraction/validation.py`, applied to every column
alike. There is no per-signal override; *Per-signal thresholds* below explains
why one hasn't been added. The threshold itself is on hold pending the sponsor
(see the 17 Sep sprint plan, Section 0).

## Why the change, not the level

A trending price series has a level distribution wide enough to swallow almost
anything. Dividing one `spx_close` by ten — the classic misplaced decimal —
scores as follows:

| Signal | Error | Score on the level | Score on the change |
|---|---|---|---|
| `spx_close` | 4219.55 → 421.96 | **2.52** | **32.67** |
| `vix` | 17.89 → 1.79 | **2.31** | **7.62** |

At any threshold high enough to be useful, level-based detection misses both.
Worse, it misses them silently: across the real data, level-based detection
flagged **zero** points in `spx_close` and `agg_close` by every method tried.

### Prices: the change in percent, not in points

A price or total-return index compounds, so its ordinary day grows with its
level. Over twenty years the S&P 500 rose about fivefold, and scored in points
its +11.6% on 13 Oct 2008 came to 5.9 and was not flagged, while an ordinary
−1.8% on 30 Oct 2026 came to 8.1 and was. On the data from 31 Aug 2006 the S&P
had no flag in 2008 and six in 2026.

So the series the models take as log returns (any total-return index field, and
the price of a ticker `ingest.align` classifies as one: SPX, DXY and the bond
indices) are scored on their log change. Rates, curve points, spreads and
volatility keep their plain change: a yield near zero would turn a move of a few
hundredths into tens of percent. A column with a value at or below zero falls
back to its plain change, so a bad value is still scored. The report still gives
each flagged move in points, as the file holds it.

On the twenty-year data, with the rebound rule of 18 Nov, this takes the flags
at threshold 8 from 280 to 245 and at 3 from 3,333 to 2,938. The S&P's flags move
to 2008 (12) and 2020 (7), and the misplaced decimal above still stands out.

## Why median absolute deviation, not standard deviation

Standard deviation is inflated by the very moves being looked for. On real
`vix` changes:

| Threshold | Flagged by classic z-score | Flagged by robust score |
|---|---|---|
| 4 | 16 | 100 |

March 2020 had dragged the standard deviation up far enough to hide everything
either side of it. The median absolute deviation does not move when a handful
of points go far away, which is the entire point of using it.

### The degenerate case

When more than half the changes are identical the median absolute deviation is
zero — a signal pegged at a constant, or a feed that has gone stale. Scaling by
zero would either divide by zero or, if naively guarded, silently declare the
one real jump unremarkable. The scale falls back to the *mean* absolute
deviation in that case, and only returns "no outliers" when the series genuinely
never moves.

## Why 8, and not the textbook 3

Financial changes are fat-tailed, so a robust score of 4 is unremarkable. Flag
counts across five signals over ten years:

| Threshold | 4 | 5 | 6 | **8** | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|
| Flags | 282 | 152 | 104 | **44** | 30 | 24 | 15 |

The portfolio manager reviews each flag by hand, so the count has to land in the
tens. 282 is not a review, it is a backlog. 15 is tight enough to start missing
things. 8 sits where the curve flattens.

For scale, real data is far fattier than a Gaussian simulation: at threshold 8,
0.357% of real cells flag against 0.015% of synthetic ones — a factor of 24. A
threshold calibrated on synthetic data would be far too tight for the real
thing.

## Spikes and their rebounds

A single bad value produces two extreme changes: the move onto it and the move
back off. Reporting both doubles the review and puts half the flags on rows
whose values are perfectly fine.

When two consecutive days are flagged in **opposite** directions, only the first
is reported — that is the row holding the odd value. Consecutive moves in the
**same** direction are both kept, because a crash is several bad days in a row
and each is a genuine observation. On the real data this collapsed 44 flags to
41: three spike-and-revert pairs, no crash days lost.

## Per-signal thresholds

The live implementation applies one threshold to every column, with no
per-signal override. (An earlier version in `quality/outliers.py` had an empty
`THRESHOLDS` override table; that module was removed on 17 Sep as a duplicate.)
On the real data, `vix` and the credit spreads flag two to three times as often
as the rest:

| Signal | Flags at threshold 8 |
|---|---|
| `vix` | 14 |
| `credit_spread_ig` | 17 |
| `spx_close` | 6 |
| `breakeven_10y` | 3 |
| `agg_close` | 1 |

That is a real difference in tail thickness, and it is tempting to raise their
thresholds. It should not be done yet: **every one of those flags was a genuine
market event**, so there is nothing to suppress. Tuning eight thresholds against
a single ten-year sample would be fitting noise. Add a per-signal override when
a signal earns one, and say in the commit which data justified it.

## What it actually flags

Run against clean institutional data, the top flags were:

| Signal | Date | Move | What it was |
|---|---|---|---|
| `vix` | 2020-03-16 | 57.83 → 82.69 | COVID crash |
| `credit_spread_ig` | 2020-03-19 | 2.85 → 3.29 | COVID credit stress |
| `vix` | 2018-02-05 | 17.31 → 37.32 | "Volmageddon" |
| `spx_close` | 2025-04-09 | 4982.77 → 5456.90 | Tariff-announcement rebound |

**Not one was a data error.** That is the expected result on good institutional
data, and it is the most important thing to understand about this check: on
clean data it finds history, not faults. Planted errors do stand out clearly —
a decimal slip scored 94× the typical move against 8–11× for genuine crash days
— but on a clean file every flag will be real.

Which is why excluding one is a decision with teeth. Those are precisely the
days a tail-risk model most needs to see, and the reason the flags are
informational, default to *included*, and require a deliberate act to drop.

## What exclusion does

Blanks that one cell. The row survives, every other signal on that date
survives, and the gap is handled by the same missing-value machinery as any
other gap. Removing the row would discard eight good observations to lose one.

The uploaded data is never modified — exclusion produces a separate frame — so
the decision is reversible and the original file remains what the sponsor
delivered.

## A known limitation

The median and deviation are computed over the whole file, so the judgement of
an early row uses information from later ones. For a data-quality gate on a
delivered file that is reasonable: nothing here feeds a forecast, and the
alternative — an expanding window — leaves the start of every sample untestable.

It is worth being explicit that this is *not* the same standard as the
look-ahead rules in the architecture design, which govern features and splits.
If outlier exclusion ever becomes automatic rather than a reviewed decision,
this needs revisiting: a rule that drops points using future information would
leak, however sensible each individual drop looked.
