# The bond derived polynomial on a hold-out nobody had scored, 18 Nov 2026

The record behind the Bond tab's derived polynomial becoming the mean of 14 fits that
fit levels apart (methodology §9.4). It adapts the research round's own write-up
(`model-research/s18-holdout-and-structure`: `RULE.md`, written before any run, with an
addendum written after the first table and before the re-test, and `FINDINGS.md`).
Runs were on the SCIS cluster: job 368268 (1,168 runs, 56 min), the re-test job
368344 (440 runs, 13 min) and diagnostics job 368347. The app's version is
`models/hybrid`; [`2026-11-18-bond-holdout.py`](2026-11-18-bond-holdout.py) runs it on
any folder of exports and builds the public hold-out.

## Why this round was different

Rounds s12–s17 scored about 2,000 bond configurations on the same test years, Oct 2008
– Nov 2026; one more regression on those years would add multiplicity and nothing else.
This round changed three things instead, all fixed in the rule before any run (no smoke
run either), and before the public data was read for anything but its dates and row
counts:

- **A. a hold-out no configuration had been scored on**: the years before 2006, rebuilt
  from public data;
- **B. averaging a grid instead of picking one winner per window** (no new fits);
- **C. a level part**: each column estimated by the method that suits its kind.

### A. The hold-out (public data, 2 Jan 1990 – 30 Aug 2006)

The Bloomberg files start on 31 Aug 2006, so nothing here overlaps a row any earlier
round trained or scored on. The files are written in the upload's format with the
upload's tickers, so the app's own pipeline reads them unchanged (`build-public`):

| Upload series | Public stand-in | Kind |
|---|---|---|
| LBUSTRUU (target) | Vanguard Total Bond Market Index fund (VBMFX): daily NAV plus its distributions | price |
| USGG10YR | FRED DGS10 | rate |
| USYC2Y10 | FRED T10Y2Y | rate |
| VIX | CBOE VIX (Yahoo ^VIX) | gauge |
| LUACOAS | Moody's Baa yield less the 10-year Treasury (FRED DBAA − DGS10) | gauge |
| DXY | ICE dollar index (Yahoo DX-Y.NYB) | price |
| LF98TRUU | Vanguard High-Yield Corporate fund (VWEHX): NAV plus distributions | price |
| SPX | S&P 500 total return (Yahoo ^SP500TR); the equity target, not a bond signal | price |
| LF98OAS, JPMVXYGL, USGGBE10 | none public for these years; left out | |

So the hold-out has six bond signals where the upload has nine, and its target is a fund
that tracks the index, not the index. It tests whether a way of deriving the polynomial
holds on other years; it cannot reproduce the upload's numbers. One check, made before
any predictive statistic, decided how the fund's income enters its total return: the NAV
does not fall on income ex-dates (income is declared daily), so each distribution is
spread over the trading days since the one before; capital gains are added on their
ex-date. Yahoo files the fund's four capital-gain payouts as dividends; they are told
apart by a rule about the payouts themselves, not about returns.

Test windows run from Jan 1992 to Aug 2006. Its two halves: **early**, windows whose
labels end before 1 Jan 1999; **late**, windows starting on or after it.

### B. Averaging instead of picking

A per-window pick takes the configuration with the best record so far, and among
150–300 configurations most of that record is luck; the forecasting literature's answer
is to combine (Bates & Granger 1969; Timmermann 2006; Aiolfi & Timmermann 2006). Three
rules were fixed: the mean, the median, and a record-weighted mean (weights from the
pooled Signal Rank IC over earlier windows only), each applied to a grid's per-window
signals (forecast less the fold's own level).

### C. A level part

Every earlier fit gave a persistent level (a volatility gauge, a spread, a yield) to the
same regression as the daily changes. For a level whose own moves drive the target, that
slope is biased in a short sample (Stambaugh 1999), with the sign of the level's same-day
correlation with the target rather than of anything it predicts. The reduced-bias fit for
that case was in the app already (`models/carry_research.py`, Amihud & Hurvich 2004), but
only as a research model on seven fixed tickers. The `hyb` fits use it by kind of series:
the fold's screened levels (gauges, by the page's transform map, and each rate's level) go
to it as it is; the fold's other screened columns (rate changes, price returns and, in
the calendar library, calendar counts) go to one of the page's four fits, unchanged; and
`forecast = level part + flow part − the flow part's mean in-sample forecast`. Nothing is
set by hand: which columns are levels follows from the kind of series, and every sign,
coefficient, penalty and AR coefficient comes from the fold's training rows.

### The candidates (11) and the bar

| | Candidate | What |
|---|---|---|
| B | `A-G0`, `M-G0`, `W-G0` | the page's four fits, combined by mean, median or record |
| B | `A-H7`, `M-H7`, `W-H7` | everything run in s16 and s17, combined the same ways |
| C | `K1` | the four `hyb` fits, page columns plus rate levels, 504 rows; picked per window |
| C | `K2` | `K1` with 2,520 rows too; picked per window |
| C | `K3` | `K2` with the calendar counts too (no degree-3 lasso there); picked per window |
| C | `A-K2`, `A-K3` | `K2` and `K3` combined by the mean |

`G0`, the page's grid with its per-window pick made from earlier windows only, is the
baseline on each data set. A candidate **qualifies** only if it passes all three:

1. on the data from 2006: at every horizon no more than 0.01 below `G0`; its mean over
   h = 1, 5, 10, 20 more than 0.02 above `G0`'s; and above zero in both decades (Oct 2008
   – Jul 2016, Aug 2016 – Sep 2026) at three or more horizons;
2. on the Sep 2026 upload alone, not below `G0` at h = 5 and 20;
3. on the hold-out, the three conditions of 1, with its early and late halves.

Passing 1 and 2 but not 3 is "qualifies on the mined years only" and is not adopted. If
several qualified the simplest would win, in the order listed. The bar was not to move
after the results were in.

### The addendum: the fund's rounding (written after the first table, before the re-test)

The old carry row scored +0.098 (t 5.9) at h = 1 on the hold-out, against +0.003 on the
index. That is a measurement artefact, not skill: the fund's NAV is to the cent, so on a
fifth of days it does not move and the day's return is the income accrual alone. Among
those tied days the return's rank is set by the level of income, which is the level of
yields, and a rank score credits any forecast that reads a level. Measured directly,
yesterday's 10-year yield has a rank correlation of +0.107 with the next day's return on
the rounded series and +0.047 once the rounding is broken at random; at 5 and 20 days
the two agree (+0.109 and +0.103, +0.215 and +0.214).

That flatters every candidate with a level part at h = 1, the one that passed included.
The first table stands as run, but a pass that leant on it was not accepted. The
stricter re-test, fixed then: five copies of the hold-out with the rounding broken at
random (each day's NAV plus an independent uniform draw within half a cent either side,
seeds 20261119 to 20261123; distributions added exactly as before; every other file
unchanged). A rounded NAV says the true one lies within half a cent of it, so each copy
is one path the unrounded NAV could have taken. `A-K3` stays qualified only if it passes
condition 3 on at least four of the five copies. The addendum also predicted the carry
row would fall to well under half its +0.098 at h = 1; it fell to about half (+0.046 to
+0.055, t 2.8 to 3.3).

## The result: one candidate qualifies, `A-K3`

`A-K3` is the mean of 14 `hyb` polynomials: the four flow fits × with and without the
calendar counts × 504 and 2,520 rows, less degree-3 lasso with the calendar counts. No
ticker is named and nothing is picked.

Signal Rank IC (t):

| | h = 1 | h = 5 | h = 10 | h = 20 |
|---|---|---|---|---|
| **Data from 2006** (tests Oct 2008 – Nov 2026) | | | | |
| `G0`, the page's grid, picked from earlier windows | +0.029 (1.8) | +0.032 (1.2) | −0.022 (−0.5) | −0.054 (−1.0) |
| `A-K3` | +0.045 (2.9) | +0.037 (1.3) | +0.019 (0.5) | +0.065 (1.1) |
| `A-K3`, 2008–16 / 2016–26 | +0.06 / +0.03 | +0.06 / +0.02 | +0.00 / +0.04 | +0.06 / +0.11 |
| **Sep 2026 upload alone** | | | | |
| `G0` | | +0.055 | | −0.022 |
| `A-K3` | | +0.084 (2.0) | | +0.112 (1.5) |
| **Hold-out, public data** (tests Jan 1992 – Aug 2006) | | | | |
| `G0` | +0.020 | +0.012 | +0.016 | −0.033 |
| `A-K3` | +0.050 | +0.058 | +0.055 | +0.048 |
| `A-K3`, 1992–98 / 1999–2006 | +0.03 / +0.07 | +0.12 / +0.01 | +0.11 / +0.01 | +0.08 / +0.03 |

The hold-out rows are means over the five re-test copies; on each copy t is 2.7–3.3 at
h = 1 and 1.7–2.3 at h = 5.

Against the bar: (1) on the data from 2006 it is above `G0` at every horizon (+0.017,
+0.005, +0.041, +0.119; mean +0.046) and above zero in both decades at all four; (2) on
the Sep 2026 upload alone it is above `G0` at h = 5 and 20; (3) on the hold-out it is
above `G0` at every horizon (mean +0.052 as first run, +0.049 on the re-test copies) and
above zero in both halves at four of four (three of four on one copy). It passed the
re-test on all five copies.

What the fits keep, by themselves and with the same sign in both samples: the 10-year
yield's level (kept in 61–75% of folds at h = 5 and 20, positive in 98–100% of those),
the month-end count (positive in 92–100%) and the last day's change in the 10-year yield
(positive in 90–100%). The jobs-Friday count changes sign between the samples.

## What did not qualify

| Candidate | From 2006 | Sep 2026 upload | Hold-out | |
|---|---|---|---|---|
| `A-G0`, `W-G0` (the page's four fits combined) | fail | fail | fail | no gain anywhere |
| `M-G0` | fail | pass | fail | |
| `K1` (four `hyb` fits, 504 rows, picked per window) | **pass** | **pass** | fail (mean gain +0.010) | mined years only |
| `K2` (+ 2,520 rows, picked) | fail (h = 1 by 0.021) | pass | fail | |
| `A-K2` (`K2` averaged; no calendar) | fail (h = 5 by 0.018) | pass | **pass**, 5 of 5 copies | near miss |
| `K3` (+ calendar, picked) | **pass** | fail (h = 5) | fail (h = 20) | |
| `A-H7` (everything in s16 + s17, mean) | **pass** | fail (h = 5) | fail (mean gain +0.006) | |
| `M-H7` (median) | **pass** | **pass** | fail (mean gain +0.006) | mined years only |
| `W-H7` (record-weighted) | **pass** | fail | fail | |

For the record, the earlier near misses on the hold-out, with their per-window picks:
H4 mean gain −0.005, H6 −0.008, H7 +0.005, each above zero in both halves at one horizon
or none.

## Reading

- **Breadth did not carry over; structure did.** Every wide grid of s16 and s17, picked
  or combined, beats the page's grid on 2008–2026 (mean gain +0.03 to +0.06) and does
  nothing on the years before 2006 (−0.01 to +0.01). That is what mining looks like.
  Every `hyb` candidate beats the page's grid on the hold-out on average (+0.010 to
  +0.052), and the two averaged ones pass it outright.
- **Averaging beats picking, but only once the fits are right.** Averaging the page's
  four fits does nothing. Averaging the `hyb` fits is what makes them hold: picked per
  window (`K1`–`K3`) they pass on the mined years and fail the hold-out.
- **Why averaging matters here** (diagnostics, nothing chosen from them): the best window
  swaps between the samples. On 2008–2026 the 504-row half carries it (+0.042, +0.072,
  +0.071, +0.098, against +0.019, +0.005, −0.033, +0.012 for 2,520 rows); on the hold-out
  the 2,520-row half does (+0.076, +0.078, +0.044, +0.027, against +0.037, +0.031,
  +0.032, +0.012). No rule that picked a window would have been right twice.
- **What the level part adds**, with the same columns, windows and averaging and only the
  way levels are fitted changed: plain fits on the page's columns and rate levels score
  +0.029, +0.022, −0.019, +0.024 from 2006 and +0.002, +0.018, −0.029, +0.017 on the
  hold-out; with the level part (`A-K2`) +0.018, +0.014, 0.000, +0.071 and +0.049,
  +0.048, +0.046, +0.049. It adds at 10 and 20 days in both samples and at every horizon
  on the hold-out; at 1 and 5 days from 2006 it costs about 0.01. The calendar counts
  supply the short horizons from 2006; the level part supplies the long ones, where the
  page's grid is negative.
- **No one configuration carries it.** Leaving out any one of the 14 moves `A-K3` by at
  most 0.013 at any horizon in any sample.

## The two bond signals on years nobody had looked at

Both were found on 2006–2026 (rounds s13 and s14). On the hold-out, with nothing fitted:

| Check | Result | |
|---|---|---|
| The fund's return on the last trading day of a month, less other days, Dec 1986 – Aug 2006 | +9.1 bp against +2.4 bp; difference +6.7 bp, t 4.2; same sign in 18 of 21 years; t 3.0 and 3.2 in the two halves | holds |
| The 10-year yield's change that day, less other days, 1987 – Aug 2006 | −1.6 bp, t −4.2 | holds |
| The same, 1962–1986 | −0.3 bp, t −0.9 | not there before the late 1980s |
| "A month-end in the next 5 days" as the forecast, h = 5 | +0.071, t 2.2 (h = 1: +0.062, t 3.6) | passes the gate |
| The 10-year yield as the forecast, h = 20 | +0.073 on 504-row folds, +0.199 (t 2.7) expanding; above zero in both halves | holds |

The month-end difference is the same size as on 2006–2026 (+6.5 bp there): the first
confirmation of either signal on data that played no part in finding it.

The two bond research models swap places, though. The earlier round (s15) chose the
calendar model over carry on 2008–2026. On the hold-out (means over the re-test copies):

| | h = 1 | h = 5 | h = 10 | h = 20 |
|---|---|---|---|---|
| Calendar model (PR #57), 2008–2026 | +0.051, t 3.3 | +0.083, t 3.2 | +0.053 | +0.043 |
| Carry model (the bond research row), 2008–2026 | +0.003 | +0.021 | +0.038 | +0.062 |
| Calendar model, hold-out | +0.032, t 1.9 | +0.005 | +0.022 | −0.016 |
| Carry model, hold-out | +0.051, t 3.1 | +0.077, t 2.2 | +0.091, t 1.8 | +0.121, t 1.7 |

By the mark fixed in the rule the calendar model does not hold on the hold-out as a
fitted model (no gate pass at h = 1 or 5), though the effect behind it does; carry passes
there at h = 1 and 5. Neither model is the better one in both samples. A model with both
parts is, which is what `A-K3` is.

## Limits, said plainly

- **The bar is not the gate.** `A-K3` clears the app's gate (Signal Rank IC above 0.02, t
  above 2) only at h = 1 on 2008–2026, and at h = 1 and 5 on the hold-out. At the page's
  default, 20 days, it is +0.065 with t 1.1: positive in both decades and on the
  hold-out, where the page's grid is negative, but not significant.
- **One of 11 passed**, and the hold-out's later half is thin at 5 days and longer
  (+0.00 to +0.03). The hold-out was scored once on the registered candidates; the
  re-test copies were added to make the test harder, not easier.
- **The hold-out is a stand-in**: a fund for the index, a Baa-less-Treasury spread for the
  index's own, six signals of nine.
- **The fund's rounding.** Above: `A-K3` had to pass on five copies with it broken, and
  did. What remains in the carry row (+0.046 to +0.055 at h = 1, t 2.8 to 3.3) is carry
  that is really there in those years.
- **Not blind on 2006–2026.** `hyb` was designed knowing that carry and the turn of the
  month held there, and that a shrinkage fit on levels plus calendar columns scored well
  in s15. The hold-out is the part that was blind.
- **Capital gains.** Two of the fund's four capital-gain payouts fall on month-ends;
  without adding them back on their day the month-end difference is +5.9 bp, t 3.4.
- **Over 2,000 bond configurations have been tried on 2008–2026.** `A-K3` is provisional
  until it is tested on data after 12 Nov 2026.
- **Not tested: whether a portfolio earns it.** In s14 forecasts this small did not pay
  after costs. Equity is untouched.

## In the app

`models/hybrid.run_bond_derived_polynomial` is `A-K3` as registered: the same columns by
kind of series, the same 14 fits, the same windows (`splitters.capped` is the research's
capped expanding window), screening per fold, and the plain mean with each fold's own
level the mean of the 14 own levels. Each window's level part is fitted once rather than
once per member, which `tests/unit/test_hybrid.py` checks gives the members fitted whole
to 1e-12. Two things differ from the research's runs, both decided before any app run:

- the calendar counts of the latest rows come from the US bond market's holiday calendar
  (`models/calendar_inputs`, as in PR #57), so the latest rows have a forecast; the
  research's had no value there, and counted the data's last day as a month-end;
- a test row on which some member lacks an input has no forecast, where the research
  averaged the members it had: only the full mean is the one polynomial the page shows.

Only the last few scored rows can differ by the first, and only a row with a missing
input by the second.

**The app reproduces the research.** `run`, on the branch that built this (cluster job
368499, 16 processes, 40 to 58 s a horizon on the data from 2006), against the
research's `A-K3`:

| Signal Rank IC (t) | h = 1 | h = 5 | h = 10 | h = 20 |
|---|---|---|---|---|
| Data from 2006, research | +0.045 (2.9) | +0.037 (1.3) | +0.019 (0.5) | +0.065 (1.1) |
| Data from 2006, app | +0.0452 (2.86) | +0.0367 (1.29) | +0.0192 (0.46) | +0.0648 (1.08) |
| Sep 2026 upload alone, research | | +0.084 (2.0) | | +0.112 (1.5) |
| Sep 2026 upload alone, app | | +0.0844 (1.97) | | +0.1120 (1.48) |
| Hold-out as first run, research | +0.066 (3.8) | +0.058 (2.1) | +0.055 (1.3) | +0.044 (0.6) |
| Hold-out as first run, app | +0.0661 (3.79) | +0.0580 (2.08) | +0.0553 (1.28) | +0.0437 (0.65) |

The two decades and the hold-out's halves agree to the same two decimals. The Models
page itself, run headless on the data from 2006 at its default 20 days, gives the same
+0.0648 (t 1.08): a 13-term equation in its latest fold, the 10y yield's level its
largest term, no fold forecasting a constant; set active, a +0.5 shock to the 10y yield
on the Scenarios page moves its change, its level and the longer window's level and
nothing else.

```
uv run --project <repo> python docs/evidence/2026-11-18-bond-holdout.py run <folder of the 2006 exports> --horizons 1,5,10,20
```

The public hold-out is rebuilt with `build-public <out folder> --dither 5` and scored with
`run <out folder>/upload_format --split 1999-01-01` (and `upload_format_dither<k>` for
each re-test copy).
