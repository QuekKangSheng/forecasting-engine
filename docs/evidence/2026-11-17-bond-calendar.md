# Bond calendar model, 17 Nov 2026

Asked for after the s13–s14 research rounds of 17 Nov: put the one bond model that held
up into the app exactly as researched, and show that the app reproduces the research's
numbers. Then, the same day, under the rule that a better research model replaces the
old one rather than adding a second row: a comparison fixed in advance (s15, part A,
below) picked it over the carry model, so it replaces carry as the Bond tab's research
row. This file records the research in brief, the replacement, the model as the app now
runs it ("Polynomial – calendar inputs (research)", methodology §9.3), and the checks.

Produced by `docs/evidence/2026-11-17-bond-calendar.py` on this branch (main at
`7a47c62` plus the model). It reads the 11 exports from 31 Aug 2006 to 11–12 Nov 2026
as the Data page merges them (their SHA-256 are in the output below) and runs the
function the Models page's runner calls, `run_calendar_research_polynomial`, at the
page's defaults: train 504 rows (rolling), test 20, embargo 20, the first 504 rows a
tuning period, labels purged; the Signal Rank IC against the model's own level, and t
on the larger of its Hansen-Hodrick and test-window standard errors. "Fresh" is the test
windows whose every label ends before 1 Aug 2016 (97, Oct 2008 – Jul 2016), "later" the
windows starting 31 Aug 2016 – 30 Sep 2026 (127), as in the research.

## The research, in brief

**s13 (the data from 2006, h = 20).** 161 runs. The one bond signal that survived on
2008–2016 was carry, the 10-year yield's level, and the yield alone, sign fixed and
nothing fitted, scores t 1.36 over Oct 2008 – Nov 2026. A 20-day label gives about 225
independent periods in 18 years, so no carry model can reach the gate's t 2 at that
horizon. A 1-day label gives about 4,500, which is why s14 looked at short horizons.

**s14 (h = 1 and 5).** The rule was written before any run (`s14-bond-short/RULE.md`):

- Ideas from the literature, chosen before looking: month-end (Treasury returns are
  higher on a month's last trading day, when index trackers extend duration; Hartley and
  Schwarz), sign positive; jobs Fridays (bonds earn more on scheduled-announcement days;
  Savor and Wilson 2013), sign positive; the bond index's own last few days; the S&P 500's
  last day and week; carry at short horizons.
- Calendar columns describe the label window, which is known in advance, so they are not
  lagged; every other new column is lagged a row like any signal.
- 98 runs: seven column sets (the page's panel, the calendar alone, the page's panel with
  the calendar, own lags or the S&P 500, all of them, levels with the calendar), three
  fits (lasso degree 1 screened and not, the page's shrinkage), two windows (504 rows,
  2,520), h = 1 and 5, and machine learning on two column sets.
- Suitable means all of: the app's gate over Oct 2008 – Nov 2026 (Signal Rank IC above
  0.02, t above 2); Signal Rank IC above zero with t above 1 in fresh and in later
  separately; the latest fold not constant and under a quarter of folds constant; the
  most often kept term one sign in at least 90% of the folds keeping it. Among several,
  the fewest moving parts wins.

**What it found.**

- The bond index's mean 1-day return is +7.4 bp on a month's last trading day against
  +0.9 bp on other days (fresh +8.5 against +1.7, later +6.2 against +0.2), and higher in
  each of the 21 calendar years 2006–2026. On those days the 10-year yield falls 1.8 bp
  (fresh) and 0.8 bp (later) more than on other days, so it is a market move, not index
  bookkeeping.
- **A-priori score, nothing fitted:** "a month-end in the next 5 days" scores +0.067,
  t 2.48 at h = 5 (fresh t 1.65, later t 2.26), the first bond signal in any round to pass
  the gate. Jobs Fridays went the wrong way (h = 5: −0.055, t −2.00).
- **Placebo:** moving the flag to other days of the month removes it. At h = 5, a flag on
  each of the last four trading days scores +0.07 to +0.10 (t 2.5–3.6); flags 5 to 15
  trading days before month-end score −0.05 to +0.03 (t −1.8 to +0.7).
- 14 of 98 runs are suitable; chance alone would give well under one. All 14 have the
  calendar columns, and no run without them is suitable. The rule's tie-break picked
  `cal / shrink / roll504`: the calendar columns only, the page's shrinkage fit screened
  per fold, the page's 504-row window, suitable at both horizons. The strongest overall
  was the levels with the calendar on 2,520 rows at h = 5 (+0.102, t 3.47).
- Offered beside the page's whole panel, the calendar columns pass at h = 1, but the
  fit's top term (the 10-year yield) takes both signs from fold to fold, and at h = 5 it
  misses the gate (t 1.89). So it works as its own small model, not as more columns in
  the derived grid.
- The research's portfolio check (`s14-bond-short/portfolio14.py`, not rerun here),
  paired with the derived equity polynomial: the optimiser's trust gate never opened, at
  h = 1 or 5, so the portfolio was the one without forecasts. Forced open as a
  diagnostic, the tilts did not pay after costs (t against no forecasts −1.11 to +0.38 at
  h = 5, −4.7 to −2.6 at h = 1).

## It replaces the carry model (s15, part A)

Asked for: bonds only; if better fixed inputs or formulas are found for the research
model, replace the old one. The rule was written before any run
(`s15-bond-grids-honest/RULE.md`, part A):

- **Candidates**, one configuration each, no selection: carry (the Bond tab's research
  row until now: the seven levels, the innovation-corrected lasso, every earlier row);
  calendar (this model: the calendar columns, shrinkage, 504 rows); carry plus calendar
  (rates and gauges as levels with the calendar columns, shrinkage) at 504 and at 2,520
  rows.
- **The replacement** is the candidate that passes the gate (Signal Rank IC above 0.02,
  t above 2) with both decades above zero (t above 1) at the most horizons. Ties: the
  higher smallest whole-period t across the four horizons, then the fewest parts.
- **It replaces carry** unless it is below carry's whole-period Signal Rank IC by more
  than 0.03 at some horizon.

Results on the data from 2006 (`part_a.txt` there; the research harness on main's code,
the model's own run of the calendar columns, so h = 5 shows the research's +0.083 where
the app gives +0.084, below). "Pass, both" = the gate with both decades above zero:

| Candidate | h | Whole period | Fresh | Later | Constant folds | |
|---|---|---|---|---|---|---|
| Carry, every earlier row | 1 | +0.003, t 0.16 | −0.006, t −0.25 | −0.002, t −0.12 | 49 | |
| | 5 | +0.021, t 0.63 | +0.016, t 0.35 | +0.026, t 0.58 | 34 | |
| | 10 | +0.038, t 0.78 | +0.030, t 0.47 | +0.046, t 0.71 | 34 | |
| | 20 | +0.062, t 0.93 | −0.026, t −0.31 | +0.117, t 1.30 | 30 | |
| Calendar, 504 rows | 1 | +0.051, t 3.25 | +0.064, t 2.52 | +0.044, t 2.11 | 0 | pass, both |
| | 5 | +0.083, t 3.24 | +0.055, t 1.41 | +0.094, t 2.72 | 0 | pass, both |
| | 10 | +0.053, t 1.73 | −0.044, t −0.97 | +0.100, t 2.52 | 5 | |
| | 20 | +0.043, t 0.92 | −0.004, t −0.06 | +0.037, t 0.63 | 38 | |
| Carry + calendar, 504 rows | 1 | +0.063, t 4.03 | +0.053, t 2.02 | +0.068, t 3.23 | 0 | pass, both |
| | 5 | +0.074, t 2.32 | +0.029, t 0.65 | +0.097, t 2.25 | 0 | pass |
| | 10 | +0.017, t 0.39 | −0.123, t −2.05 | +0.081, t 1.42 | 0 | |
| | 20 | +0.014, t 0.25 | −0.060, t −0.75 | +0.074, t 0.99 | 0 | |
| Carry + calendar, 2,520 rows | 1 | +0.038, t 2.31 | +0.048, t 1.77 | +0.020, t 0.96 | 0 | pass |
| | 5 | +0.102, t 3.47 | +0.073, t 1.69 | +0.112, t 2.99 | 0 | pass, both |
| | 10 | +0.093, t 2.34 | +0.006, t 0.09 | +0.122, t 2.48 | 0 | pass |
| | 20 | +0.087, t 1.45 | +0.053, t 0.66 | +0.114, t 1.41 | 0 | |

Of 228 folds each. **The calendar model wins**: it passes with both decades above zero
at two horizons (h = 1 and 5), each carry-plus-calendar run at one, carry at none. Its
largest shortfall against carry is 0.019, at h = 20 (+0.043 against +0.062), within the
rule's 0.03, so it replaces carry. Carry plus calendar on 2,520 rows has the highest
scores at h = 5, 10 and 20, but passes with both decades at h = 5 only.

The carry figures here differ from methodology §20's (+0.004, t 0.26 at h = 1 to
+0.063, t 0.94 at h = 20) by at most 0.002 in Signal Rank IC and 0.10 in t. The
research harness builds the same seven levels under other names and in another column
order, which the lasso path can be sensitive to; this was not checked further, since on
either figure carry passes nowhere.

What changes in the app: the Bond tab has one research row, this model. The carry row
is gone from the page; `models/carry_research.py` and its tests stay, so the evidence
that cites its figures still runs. A model still set active under the carry row's name
is never used: it is listed in `reporting.model_metrics.RETIRED_MODEL_NAMES`, so
`ActiveModelRecord.fitted_on` is false for it whatever it was fitted on, and Home, the
status bar, Portfolio and Scenarios treat it as stale and say it is no longer offered.

## The app reproduces it

| h | Period | Research (its log, main at `1bc88e0`) | The app | Constant folds (research, app) |
|---|---|---|---|---|
| 1 | whole | +0.0510, t 3.2487 | +0.0510, t 3.2487 | 0, 0 of 228 |
| 1 | fresh | +0.0637, t 2.5164 | +0.0637, t 2.5164 | |
| 1 | later | +0.0443, t 2.1094 | +0.0443, t 2.1094 | |
| 5 | whole | +0.0834, t 3.2448 | +0.0839, t 3.2669 | 0, 0 of 228 |
| 5 | fresh | +0.0550, t 1.4080 | +0.0550, t 1.4080 | |
| 5 | later | +0.0936, t 2.7164 | +0.0936, t 2.7164 | |

Exact at h = 1, and at h = 5 in both decades. The one difference is one forecast at h = 5,
4 Nov 2026's, in the last test window. The research's `calendar` used the data's dates
only, so it counted the data's last row (11 Nov 2026) as a month-end, since no row
followed it, and 4 Nov's 5-day window held it. The app takes the next trading day from
the holiday calendar (12 Nov) and counts none. Run on the research's own columns, the
same harness on this checkout gives the research's figures to four decimals (the second
table in the output), so nothing in main's code since `1bc88e0` changes this model; the
code that did change since (ingestion, the derived grid, the carry purge, the active-model
store) does not touch it. The app's columns equal the research's on every row where the
research's have a value, except that one row per horizon (the first table in the
output).

## All four horizons

| h | Signal Rank IC, t | Gate | Fresh | Later | Constant folds |
|---|---|---|---|---|---|
| 1 | +0.051, t 3.25 | passes | +0.064, t 2.52 | +0.044, t 2.11 | 0 of 228 |
| 5 | +0.084, t 3.27 | passes | +0.055, t 1.41 | +0.094, t 2.72 | 0 of 228 |
| 10 | +0.053, t 1.73 | fails on t | −0.044, t −0.97 | +0.100, t 2.52 | 5 of 228 |
| 20 | +0.043, t 0.92 | fails on t | −0.004, t −0.06 | +0.037, t 0.63 | 38 of 228 |

At 10 and 20 days nearly every window holds one month-end, so the columns barely vary:
at h = 10 the two decades disagree, and at h = 20 a sixth of the folds forecast a
constant. The month-end column is the most often kept term at h = 1 (85% of folds, 99% of
them positive) and h = 5 (94%, 91% positive); jobs Friday is kept in most folds and is
negative in most, as the a-priori score said.

## The holiday calendar

The latest `h` rows' label windows run past the data's last date, so the app takes the
coming trading days from a holiday calendar. `exchange_calendars` is not a declared
dependency; `pandas_market_calendars` is, and is already used (`extraction/bloomberg_csv`
labels gaps with its NYSE calendar). Of its calendars and pandas' US federal holidays,
the US bond market's (`SIFMA_US`) matches the bond index's own dates best: 43 dates
differ over 2006–2026, against 52 for federal holidays (wrong on Good Friday), 74 for
the NYSE (wrong on Columbus and Veterans Days) and 181 for every weekday. Of those 43,
41 are rows the exports hold on bond-market holidays, nearly all with the price
unchanged: every holiday since Nov 2023, and nine before. In five months such a row is
the data's last of the month, so the month-end flag sits on it, as it did in the
research (31 May 2010, 29 Mar 2013, 30 Mar 2018, 31 May 2021, 29 Mar 2024). Where the
data runs, the model reads its dates, as the research did; only past its end does the
holiday calendar decide.

## Limits

- It carries information at h = 1 and 5 only, and the page's default horizon is 20.
- The effect is a few basis points once a month. This scores the forecast, not a
  portfolio; the research's own backtest found it never tilted the portfolio through the
  trust gate, and forced open, its tilts did not pay after costs.
- Every result over Oct 2008 – Nov 2026 was looked at in choosing it, so its score is
  optimistic, and its pre-2016 figure is not a hold-out. About 400 bond runs were tried
  across s11–s14. The protection is that the idea was fixed from the literature first,
  scored with nothing fitted, holds in each decade and every year, and is gone in the
  placebo.

## Rerunning

```
uv run --project <repo> python -I docs/evidence/2026-11-17-bond-calendar.py <folder>
```

The folder holds the 11 `.xlsx` exports. Six fits in parallel processes (four horizons,
and the research's own columns at h = 1 and 5) took about three minutes here.

## Full output

## The files

| file | SHA-256 | rows | first | last |
|---|---|---|---|---|
| DXY Curncy_Dollar_index.xlsx | `9f76384e72e7a6c672430877e1f062c39d659a3796e5f8cfb79788d4129dd7e2` | 5,254 | 31 Aug 2006 | 12 Nov 2026 |
| JPMVXYGL_Index_FX_implied_volatility.xlsx | `e0d9e81508ff1008db2c37faf41cb19b20e1c9358a71604141bb27ef611355b1` | 5,223 | 31 Aug 2006 | 12 Nov 2026 |
| LBUSTRUU_Index_10_years_daily.xlsx | `bc9ab5b116764c58e9feff47248000dbcb7ce4c5875642a0c552bea808851bce` | 5,089 | 31 Aug 2006 | 11 Nov 2026 |
| LF98OAS_Index_HY_credit _spread.xlsx | `9acc5b398e91812b0101c9e0d5c48fc63560980c17964f512e54de52040792ec` | 5,085 | 31 Aug 2006 | 11 Nov 2026 |
| LF98TRUU_Index_HY_total_return_index.xlsx | `db09b4b4fdd16e27c0769827c1d8cdfbae1863345233d3562686fb4fc13affaa` | 5,089 | 31 Aug 2006 | 11 Nov 2026 |
| LUACOAS_Index_IG_credit_spread.xlsx | `39f70f78be7a1fe9bfb89161c4103abc80eacd2195c136bb7b586e481d7a1243` | 5,085 | 31 Aug 2006 | 11 Nov 2026 |
| SPX_Index_10_years_daily.xlsx | `02eeb9406ee8c907a7ea725650408be4ffd28bd062f25bd527870fef892c388e` | 5,081 | 31 Aug 2006 | 11 Nov 2026 |
| USGG10YR_Index_10y_Treasury_yield.xlsx | `eaaf35e9a3801ae17110070361d68da51f17aa1b3fc6b1199c3021574a032259` | 5,268 | 31 Aug 2006 | 12 Nov 2026 |
| USGGBE10_Index_Inflation_breakeven_data.xlsx | `fc12c64f1e3894c45d1a98e0340ab2ae7edcaf8a62a490e8fa3c5a3fea721782` | 5,266 | 31 Aug 2006 | 12 Nov 2026 |
| USYC2Y10_Index_2s10s_curve.xlsx | `c5e5d8de9159723f94a7e4487aba839ae1844adff780bd3ab34847737780003b` | 5,246 | 31 Aug 2006 | 12 Nov 2026 |
| VIX_index_price_10_years_daily.xlsx | `843db604e145f3a5736395e6b829e9a8ac90b3e01b4c15ea2b325c66f86f783c` | 5,114 | 31 Aug 2006 | 11 Nov 2026 |

Merged: 5,271 rows, 31 Aug 2006 to 12 Nov 2026, 13 columns, content hash (`ingest.upload.frame_sha256`) `7a4b117c59d3ccd2d657b1a20e07e726492b4b19ff3b3ce62052751f150ce92c`. Empty columns dropped: none. Bond target: `LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS`.

## The calendar for the coming trading days

`HOLIDAY_CALENDAR` = `SIFMA_US`.
The bond index's 5,089 dates, 31 Aug 2006 to 11 Nov 2026, against each calendar's trading days over the same span.

| calendar | index dates it calls closed | its days the index lacks | total |
|---|---|---|---|
| SIFMA_US | 41 | 2 | 43 |
| NYSE | 41 | 33 | 74 |
| US federal holidays (pandas) | 40 | 12 | 52 |
| every weekday | 0 | 181 | 181 |

Of the 41 index dates SIFMA_US calls closed, 32 fall from Nov 2023 on, when the exports began to carry a row on every bond-market holiday with the price unchanged: 23 Nov 2023, 25 Dec 2023, 15 Jan 2024, 19 Feb 2024, 29 Mar 2024, 27 May 2024, … The rest: 02 Apr 2010, 31 May 2010, 06 Apr 2012, 29 Mar 2013, 03 Apr 2015, 30 Mar 2018, 03 Sep 2018, 02 Sep 2019, 31 May 2021.

Months whose last row in the data is not SIFMA's last trading day (5 of 243 complete months): 31 May 2010 (SIFMA: 28 May), 29 Mar 2013 (SIFMA: 28 Mar), 30 Mar 2018 (SIFMA: 29 Mar), 31 May 2021 (SIFMA: 28 May), 29 Mar 2024 (SIFMA: 28 Mar). The model reads the data's own dates wherever they run, as the research did, so there the month-end flag sits on the data's last row.

## The columns against the research's

| h | columns | rows where both have a value | rows that differ | rows only the app fills |
|---|---|---|---|---|
| 1 | CAL_MONTH_END, CAL_MONTH_START, CAL_JOBS_FRIDAY, CAL_MON, CAL_TUE, CAL_WED, CAL_THU | 5,088 | 1: 10 Nov 2026 | 1 (11 Nov 2026 – 11 Nov 2026), research none |
| 5 | CAL_MONTH_END, CAL_MONTH_START, CAL_JOBS_FRIDAY | 5,084 | 1: 04 Nov 2026 | 5 (5 Nov 2026 – 11 Nov 2026), research none |
| 10 | CAL_MONTH_END, CAL_MONTH_START, CAL_JOBS_FRIDAY | 5,079 | 1: 28 Oct 2026 | 10 (29 Oct 2026 – 11 Nov 2026), research none |
| 20 | CAL_MONTH_END, CAL_MONTH_START, CAL_JOBS_FRIDAY | 5,069 | 1: 14 Oct 2026 | 20 (15 Oct 2026 – 11 Nov 2026), research none |

The last six rows at h = 5, their windows filled past 11 Nov 2026 from SIFMA_US:

| date | CAL_MONTH_END | CAL_MONTH_START | CAL_JOBS_FRIDAY |
|---|---|---|---|
| Wed 04 Nov 2026 | 0 | 0 | 1 |
| Thu 05 Nov 2026 | 0 | 0 | 1 |
| Fri 06 Nov 2026 | 0 | 0 | 0 |
| Mon 09 Nov 2026 | 0 | 0 | 0 |
| Tue 10 Nov 2026 | 0 | 0 | 0 |
| Wed 11 Nov 2026 | 0 | 0 | 0 |

All 6 fits: 179 s.

## The model, `VERSION` 1, at the page's defaults

Train 504 rows (rolling), test 20, embargo 20, the first 504 rows a tuning period; columns screened per fold and fitted by `ShrinkagePolynomial`. Signal Rank IC and t as on the Models page; the gate is Signal Rank IC above 0.02 and t above 2. Fresh: test windows whose every label ends before 1 Aug 2016 (the page's pre-2016 figure). Later: test windows starting 31 Aug 2016 – 30 Sep 2026.

| h | whole period | gate | fresh (windows) | later (windows) | constant folds | rows | test dates | runtime |
|---|---|---|---|---|---|---|---|---|
| 1 | +0.051, t 3.25 | passes | +0.064, t 2.52 (97) | +0.044, t 2.11 (127) | 0 of 228 | 4,560 | 03 Oct 2008 – 04 Nov 2026 | 87 s |
| 5 | +0.084, t 3.27 | passes | +0.055, t 1.41 (97) | +0.094, t 2.72 (127) | 0 of 228 | 4,560 | 03 Oct 2008 – 04 Nov 2026 | 52 s |
| 10 | +0.053, t 1.73 | fails t | -0.044, t -0.97 (97) | +0.100, t 2.52 (127) | 5 of 228 | 4,555 | 03 Oct 2008 – 04 Nov 2026 | 49 s |
| 20 | +0.043, t 0.92 | fails t | -0.004, t -0.06 (96) | +0.037, t 0.63 (127) | 38 of 228 | 4,545 | 03 Oct 2008 – 04 Nov 2026 | 44 s |

**Against the research** (`cal / shrink / roll504`, main at `1bc88e0`), and this checkout's harness run on the research's own columns (`research_calendar`):

| h | period | research (its log) | research's columns, this checkout | the app | app − research |
|---|---|---|---|---|---|
| 1 | whole | +0.0510, t 3.2487 | +0.0510, t 3.2487 | +0.0510, t 3.2487 | +0.0000, -0.0000 |
| 1 | fresh | +0.0637, t 2.5164 | +0.0637, t 2.5164 | +0.0637, t 2.5164 | -0.0000, +0.0000 |
| 1 | later | +0.0443, t 2.1094 | +0.0443, t 2.1094 | +0.0443, t 2.1094 | +0.0000, -0.0000 |
| 1 | constant folds | 0 | 0 | 0 |  |
| 1 | forecasts that differ |  |  | 0: none |  |
| 5 | whole | +0.0834, t 3.2448 | +0.0834, t 3.2448 | +0.0839, t 3.2669 | +0.0005, +0.0221 |
| 5 | fresh | +0.0550, t 1.4080 | +0.0550, t 1.4080 | +0.0550, t 1.4080 | -0.0000, -0.0000 |
| 5 | later | +0.0936, t 2.7164 | +0.0936, t 2.7164 | +0.0936, t 2.7164 | -0.0000, -0.0000 |
| 5 | constant folds | 0 | 0 | 0 |  |
| 5 | forecasts that differ |  |  | 1: 04 Nov 2026 |  |

**Terms**: the share of folds keeping each column, and of those the share positive.

| h | kept | latest fold |
|---|---|---|
| 1 | CAL_MONTH_END 85% (99% +); CAL_MON 81% (46% +); CAL_MONTH_START 77% (28% +); CAL_WED 71% (27% +); CAL_JOBS_FRIDAY 71% (11% +); CAL_THU 71% (38% +); CAL_TUE 60% (57% +) | CAL_MONTH_START -0.000130, CAL_JOBS_FRIDAY -0.000152, CAL_MON +0.000122, CAL_TUE +0.000129; intercept +0.000047 |
| 5 | CAL_MONTH_END 94% (91% +); CAL_JOBS_FRIDAY 85% (16% +); CAL_MONTH_START 82% (81% +) | CAL_MONTH_END +0.000648, CAL_JOBS_FRIDAY -0.000653; intercept +0.000334 |
| 10 | CAL_MONTH_END 90% (83% +); CAL_JOBS_FRIDAY 78% (28% +); CAL_MONTH_START 77% (70% +) | CAL_MONTH_END +0.000849, CAL_MONTH_START +0.000763, CAL_JOBS_FRIDAY -0.001037; intercept +0.000739 |
| 20 | CAL_JOBS_FRIDAY 81% (22% +); CAL_MONTH_END 68% (83% +); CAL_MONTH_START 58% (49% +) | CAL_MONTH_END +0.000839, CAL_MONTH_START +0.000887, CAL_JOBS_FRIDAY -0.000908; intercept +0.001388 |
