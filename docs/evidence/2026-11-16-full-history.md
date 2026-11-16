# The full history, 16 Nov 2026

Phase 5, task 1 of the 15 Nov plan: the final rerun on the Bloomberg history from 31 Aug 2006, after Phases 1–4 (main at 02a8f7f). Produced by `docs/evidence/2026-11-16-full-history.py`. The script reads the 11 exports and merges them as the Data page does (the files' SHA-256 are listed below). It fits every model through the functions the Models page's runners call, at the page's defaults: tuning 504 rows, train 504, test 20, embargo 20.

**Active models, decided before any result was seen.** The portfolio uses the derived polynomial for both equity and bond, as in the optimiser evidence (`2026-11-16-optimiser-fixes.md`). Neither research model is used, because both scored about zero on their pre-2016 hold-out.

Section 3 is descriptive only and changed nothing.

## Summary

### Which models pass the gate (Signal Rank IC above 0.02 and t above 2)

| target | h = 1 | h = 5 | h = 10 | h = 20 (default) |
|---|---|---|---|---|
| equity | derived polynomial (+0.033, t 2.10); research (+0.033, t 2.11) | derived (+0.083, t 3.20); ML (+0.068, t 2.34); research (+0.068, t 2.10) | derived (+0.104, t 2.84); ML (+0.097, t 2.48) | **none** (derived +0.103, t 1.98) |
| bond | none | none | none | none |

- **No bond model passes at any horizon.** The best is the derived polynomial at h = 1, with t 1.84.
- **At the default 20-day horizon no model passes for either index.** The derived polynomial for equity misses by 0.02 in t.
- **Fama-French 5 fails everywhere.**
- **The research models score near zero on their pre-2016 hold-out at every horizon,** with t between −0.44 and +0.65. Their full-period passes for equity at h = 1 and 5 come from the years their inputs were chosen on.

### The portfolio (derived polynomial for both)

- **Each level against 50/50.** The t-statistics mostly reflect equity weight:
  - levels 1–3 hold more equity and beat 50/50 at every horizon except level 3 at h = 5;
  - levels 4 and 5 hold less and trail it.
- **Net Sharpe.** It rises from level 1 (0.78–0.80) to level 5 (1.03–1.05), against 0.88 for 50/50.
- **The forecasts add no value at any horizon:**
  - **h = 1 and h = 10:** the gate never opens, so the paths are identical;
  - **h = 5:** t is −1.10 to −1.38;
  - **h = 20:** t is +0.01 to +0.05.
- **Costs** are at most 0.06% a year.

### Why the 20-day gate opens 3 times here against 26 on 1 Nov

| models | Sep 2026 upload (99 rebalances) | from 31 Aug 2006 (226) |
|---|---|---|
| 1 Nov pair: derived equity, ML bond | 19 open; t vs no forecasts −1.16 to +1.07 | 92 open; −1.44 to −0.15 |
| derived for both | 2 open; −0.26 to −0.13 | 3 open; +0.01 to +0.05 |

- **The bond model accounts for most of the difference, not the longer history.** With machine learning for bond, as on 1 Nov, the gate opens about as often or more on the longer history (19 of 99 on the Sep 2026 upload, 92 of 226 on the data from 2006). With the derived polynomial for both, it opens 2 and 3 times.
- **The other differences from the 1 Nov file's 26 are small next to that:** today's code (no band, π̄ with its prior) and today's machine-learning fit.
- **In no cell do the forecasts add value.** The forecast-value t ranges from −1.44 to +1.07.

### Is 2008–09 what keeps the gate shut?

Only partly, and only for the derived pair at h = 20. Starting the trust pairs on 1 Jan 2010 instead of Oct 2008, on the data from 2006:

| models | h | gate open, pairs from Oct 2008 → from Jan 2010 | t vs no forecasts, levels 1–4 |
|---|---|---|---|
| derived for both | 20 | 3 → 21 of 226 | +0.01 to +0.05 → −0.69 to +0.46 |
| derived for both | 5 | 137 → 105 of 904 | −1.10 to −1.38 → −1.30 to −1.60 |
| 1 Nov pair | 20 | 92 → 79 of 226 | −0.15 to −1.44 → −0.18 to −1.76 |
| 1 Nov pair | 5 | 184 → 0 of 904 | −1.03 to −3.10 → identical paths |

- **The years on their own:**
  - **At h = 20,** the derived pair's 2008 slope (Oct–Dec only) is −0.52 at t −2.03; 2009's is +0.43 at t 0.53.
  - **Most later years' slopes are positive but small.** Only 2011, 2015 and 2017 reach t 2, and 2016 has t −1.93.
  - **At h = 5,** the derived pair's signal difference doesn't vary at all in 2009, so that year has no slope. Some years show very large slopes, such as +35.59 in 2024 on the Sep 2026 upload; these come from years where the signal difference barely moves.
- **So 2008 holds the 20-day gate shut for the derived pair, and without it the gate opens 21 times. Opening it doesn't help:** the forecast-value t stays between −0.69 and +0.46. Elsewhere, removing 2008–09 closes the gate rather than opening it.
- **Read with care.** One year holds about 12 independent 20-day returns, so each yearly t is noisy. Task 7's simulation also found that, with no skill, the re-checked 20-day gate opens at least once in 35% of draws.

### Runtime

All 42 fits took 1,309 s wall-clock, measured locally with 8 at a time. The per-fit seconds in section 1 are slower than a fit alone.

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

Merged: 5,271 rows, 31 Aug 2006 to 12 Nov 2026, 13 columns, content hash (`ingest.upload.frame_sha256`) `7a4b117c59d3ccd2d657b1a20e07e726492b4b19ff3b3ce62052751f150ce92c`. Empty columns dropped: none.

Sep 2026 upload (`data/last_commit.pkl`): content hash `58c06ef5d94b5736daeff9895ff8f7a75b181236f0bbfc80354836f6d49d51ba`.

All 42 fits: 1309 s wall-clock.

## 1. Every model, each target and horizon

Signal Rank IC and t (the Signal Rank IC over the larger of its two standard errors), at the page's defaults. The gate: Signal Rank IC above 0.02 and t above 2; the naive baseline is not gated. Runtime is wall-clock seconds per fit, measured locally with several fits running at once, so each is slower than alone.

### Equity

| model | h | Signal Rank IC | t | rows | gate | runtime |
|---|---|---|---|---|---|---|
| Naive (training mean) | 1 | +nan | nan | 4,540 | not gated | 4 s |
| Naive (training mean) | 5 | +nan | nan | 4,540 | not gated | 4 s |
| Naive (training mean) | 10 | +nan | nan | 4,540 | not gated | 4 s |
| Naive (training mean) | 20 | +nan | nan | 4,537 | not gated | 4 s |
| Polynomial (derived) | 1 | +0.0331 | 2.10 | 4,540 | passes | 369 s |
| Polynomial (derived) | 5 | +0.0833 | 3.20 | 4,540 | passes | 406 s |
| Polynomial (derived) | 10 | +0.1041 | 2.84 | 4,540 | passes | 408 s |
| Polynomial (derived) | 20 | +0.1032 | 1.98 | 4,537 | fails t | 420 s |
| Polynomial, research | 1 | +0.0334 | 2.11 | 4,540 | passes | 157 s |
| Polynomial, research | 5 | +0.0679 | 2.10 | 4,540 | passes | 146 s |
| Polynomial, research | 10 | +0.0499 | 1.09 | 4,540 | fails t | 138 s |
| Polynomial, research | 20 | +0.0706 | 1.10 | 4,537 | fails t | 128 s |
| Machine learning | 1 | +0.0279 | 1.71 | 4,540 | fails t | 534 s |
| Machine learning | 5 | +0.0676 | 2.34 | 4,540 | passes | 581 s |
| Machine learning | 10 | +0.0968 | 2.48 | 4,540 | passes | 560 s |
| Machine learning | 20 | +0.0822 | 1.71 | 4,537 | fails t | 529 s |
| Fama-French 5 | 1 | -0.0058 | -0.35 | 4,528 | fails signal_rank_ic and t | 7 s |
| Fama-French 5 | 5 | +0.0208 | 1.27 | 4,528 | fails t | 8 s |
| Fama-French 5 | 10 | -0.0028 | -0.17 | 4,528 | fails signal_rank_ic and t | 8 s |
| Fama-French 5 | 20 | -0.0125 | -0.83 | 4,528 | fails signal_rank_ic and t | 7 s |

### Bond

| model | h | Signal Rank IC | t | rows | gate | runtime |
|---|---|---|---|---|---|---|
| Naive (training mean) | 1 | +nan | nan | 4,560 | not gated | 4 s |
| Naive (training mean) | 5 | +nan | nan | 4,560 | not gated | 4 s |
| Naive (training mean) | 10 | +nan | nan | 4,555 | not gated | 4 s |
| Naive (training mean) | 20 | +nan | nan | 4,545 | not gated | 4 s |
| Polynomial (derived) | 1 | +0.0286 | 1.84 | 4,560 | fails t | 367 s |
| Polynomial (derived) | 5 | +0.0304 | 1.00 | 4,560 | fails t | 430 s |
| Polynomial (derived) | 10 | +0.0206 | 0.48 | 4,555 | fails t | 428 s |
| Polynomial (derived) | 20 | +0.0305 | 0.49 | 4,545 | fails t | 397 s |
| Polynomial, research | 1 | +0.0031 | 0.20 | 4,560 | fails signal_rank_ic and t | 78 s |
| Polynomial, research | 5 | +0.0201 | 0.60 | 4,560 | fails t | 77 s |
| Polynomial, research | 10 | +0.0385 | 0.78 | 4,555 | fails t | 72 s |
| Polynomial, research | 20 | +0.0661 | 0.99 | 4,545 | fails t | 70 s |
| Machine learning | 1 | +0.0189 | 1.30 | 4,560 | fails signal_rank_ic and t | 549 s |
| Machine learning | 5 | +0.0133 | 0.54 | 4,560 | fails signal_rank_ic and t | 527 s |
| Machine learning | 10 | -0.0173 | -0.46 | 4,555 | fails signal_rank_ic and t | 522 s |
| Machine learning | 20 | +0.0449 | 0.88 | 4,545 | fails t | 465 s |

### Research models' pre-2016 hold-out (windows ending before 1 Aug 2016)

| target | h | full Signal Rank IC | full t | hold-out Signal Rank IC | hold-out t | hold-out windows |
|---|---|---|---|---|---|---|
| equity | 1 | +0.0334 | 2.11 | +0.0162 | 0.65 | 98 |
| equity | 5 | +0.0679 | 2.10 | +0.0178 | 0.35 | 98 |
| equity | 10 | +0.0499 | 1.09 | -0.0305 | -0.44 | 98 |
| equity | 20 | +0.0706 | 1.10 | -0.0315 | -0.30 | 97 |
| bond | 1 | +0.0031 | 0.20 | -0.0013 | -0.06 | 97 |
| bond | 5 | +0.0201 | 0.60 | +0.0194 | 0.41 | 97 |
| bond | 10 | +0.0385 | 0.78 | +0.0376 | 0.58 | 97 |
| bond | 20 | +0.0661 | 0.99 | -0.0040 | -0.05 | 96 |

## 2. Portfolio per risk level

Derived polynomial active for both indices. Net of 3 bp equity and 5 bp bond on what each rebalance trades; turnover and costs a year leave out the first allocation.

### h = 1: 4,520 rebalances, 6 Oct 2008 to 5 Nov 2026; trust gate open at 0 of 4520

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
|  | 1 | 11.64% | 0.79 | -28.2% | 66% | 0.026% | 3.25 | — |
|  | 2 | 10.27% | 0.83 | -23.6% | 92% | 0.037% | 3.36 | — |
|  | 3 | 8.80% | 0.88 | -19.9% | 103% | 0.041% | 3.69 | — |
|  | 4 | 7.20% | 0.95 | -18.7% | 98% | 0.039% | -2.61 | — |
|  | 5 | 5.18% | 1.04 | -17.7% | 66% | 0.026% | -2.85 | — |
| 50/50 benchmark | — | 8.32% | 0.88 | -19.9% | 20% | 0.008% | — | — |

### h = 5: 904 rebalances, 6 Oct 2008 to 30 Oct 2026; trust gate open at 137 of 904

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
|  | 1 | 11.33% | 0.78 | -28.5% | 145% | 0.058% | 3.00 | -1.16 |
|  | 2 | 9.92% | 0.80 | -23.8% | 149% | 0.060% | 2.77 | -1.38 |
|  | 3 | 8.50% | 0.85 | -20.0% | 150% | 0.060% | 1.18 | -1.15 |
|  | 4 | 6.95% | 0.92 | -18.7% | 130% | 0.052% | -3.11 | -1.10 |
|  | 5 | 5.10% | 1.03 | -17.7% | 28% | 0.011% | -2.92 | — |
| 50/50 benchmark | — | 8.32% | 0.88 | -19.9% | 20% | 0.008% | — | — |

### h = 10: 452 rebalances, 6 Oct 2008 to 23 Oct 2026; trust gate open at 0 of 452

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
|  | 1 | 11.56% | 0.79 | -28.4% | 19% | 0.008% | 3.19 | — |
|  | 2 | 10.16% | 0.82 | -23.6% | 27% | 0.011% | 3.20 | — |
|  | 3 | 8.68% | 0.87 | -20.0% | 30% | 0.012% | 2.97 | — |
|  | 4 | 7.08% | 0.94 | -18.8% | 29% | 0.012% | -2.87 | — |
|  | 5 | 5.10% | 1.04 | -17.8% | 19% | 0.008% | -2.92 | — |
| 50/50 benchmark | — | 8.32% | 0.88 | -19.9% | 20% | 0.008% | — | — |

### h = 20: 226 rebalances, 6 Oct 2008 to 9 Oct 2026; trust gate open at 3 of 226

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
|  | 1 | 11.60% | 0.80 | -28.5% | 37% | 0.015% | 3.22 | 0.05 |
|  | 2 | 10.20% | 0.83 | -23.7% | 36% | 0.015% | 3.23 | 0.05 |
|  | 3 | 8.72% | 0.89 | -20.1% | 33% | 0.013% | 2.47 | 0.04 |
|  | 4 | 7.11% | 0.96 | -18.8% | 26% | 0.011% | -2.71 | 0.01 |
|  | 5 | 5.11% | 1.05 | -17.8% | 13% | 0.005% | -2.90 | — |
| 50/50 benchmark | — | 8.32% | 0.88 | -19.9% | 20% | 0.008% | — | — |

## 3. Why the 20-day gate opens 3 times here against 26 on 1 Nov

Descriptive only; nothing was changed because of it. Every cell runs today's optimiser (no band, π̄ with its prior); the 1 Nov file ran the band and the plain expanding premium, and its machine-learning fit was the app's on that day.

### h = 5: the four cells, pairs from the first out-of-sample date

| models | data | trust pairs | gate open at | t vs no forecasts, levels 1–4 |
|---|---|---|---|---|
| 1 Nov pair (derived equity, ML bond) | Sep 2026 upload | 1,988 from 2 Oct 2018 | 0 of 396 | identical paths |
| 1 Nov pair (derived equity, ML bond) | from 31 Aug 2006 | 4,508 from 3 Oct 2008 | 184 of 904 | -3.10 to -1.03 |
| derived for both | Sep 2026 upload | 1,988 from 2 Oct 2018 | 0 of 396 | identical paths |
| derived for both | from 31 Aug 2006 | 4,508 from 3 Oct 2008 | 137 of 904 | -1.38 to -1.10 |

### h = 5: pairs from 1 Jan 2010 instead

| models | data | trust pairs | gate open at | t vs no forecasts, levels 1–4 |
|---|---|---|---|---|
| 1 Nov pair (derived equity, ML bond) | Sep 2026 upload | pairs start 2 Oct 2018: unchanged | — | — |
| 1 Nov pair (derived equity, ML bond) | from 31 Aug 2006 | 4,198 | 0 of 904 | identical paths |
| derived for both | Sep 2026 upload | pairs start 2 Oct 2018: unchanged | — | — |
| derived for both | from 31 Aug 2006 | 4,198 | 105 of 904 | -1.60 to -1.30 |

### h = 5: the trust slope for each calendar year on its own

OLS of y on x with an intercept over the year's pairs (by forecast date), standard errors over 4 lags counted in full, as the gate's. One year holds about 50 independent 5-day returns, so each t is noisy.

**1 Nov pair (derived equity, ML bond), Sep 2026 upload**

| year | pairs | slope | t |
|---|---|---|---|
| 2018 (part) | 60 | +0.03 | +0.03 |
| 2019 | 250 | +0.56 | +0.66 |
| 2020 | 251 | +0.13 | +0.06 |
| 2021 | 250 | +1.28 | +1.74 |
| 2022 | 249 | -0.80 | -1.90 |
| 2023 | 249 | +1.54 | +1.05 |
| 2024 | 252 | +0.88 | +1.02 |
| 2025 | 250 | -0.95 | -1.93 |
| 2026 (part) | 177 | +1.89 | +1.43 |

**1 Nov pair (derived equity, ML bond), from 31 Aug 2006**

| year | pairs | slope | t |
|---|---|---|---|
| 2008 (part) | 60 | +4.21 | +0.55 |
| 2009 | 250 | +2.25 | +1.03 |
| 2010 | 250 | +0.63 | +0.25 |
| 2011 | 250 | +2.20 | +0.59 |
| 2012 | 248 | +0.68 | +0.62 |
| 2013 | 250 | -0.06 | -0.14 |
| 2014 | 250 | +0.45 | +0.42 |
| 2015 | 250 | +0.92 | +1.87 |
| 2016 | 250 | +0.45 | +0.60 |
| 2017 | 250 | +0.10 | +0.29 |
| 2018 | 249 | -1.00 | -0.49 |
| 2019 | 250 | +0.47 | +0.61 |
| 2020 | 251 | +0.99 | +1.57 |
| 2021 | 250 | -0.80 | -0.69 |
| 2022 | 249 | -0.02 | -0.01 |
| 2023 | 249 | +0.04 | +0.05 |
| 2024 | 252 | -0.65 | -0.51 |
| 2025 | 250 | +0.41 | +0.71 |
| 2026 (part) | 200 | +1.65 | +1.56 |

**derived for both, Sep 2026 upload**

| year | pairs | slope | t |
|---|---|---|---|
| 2018 (part) | 60 | +0.16 | +0.07 |
| 2019 | 250 | +1.53 | +3.02 |
| 2020 | 251 | -0.47 | -0.33 |
| 2021 | 250 | +1.02 | +0.82 |
| 2022 | 249 | +0.84 | +0.75 |
| 2023 | 249 | +1.21 | +1.18 |
| 2024 | 252 | +35.59 | +0.31 |
| 2025 | 250 | +0.82 | +0.51 |
| 2026 (part) | 177 | +1.94 | +1.40 |

**derived for both, from 31 Aug 2006**

| year | pairs | slope | t |
|---|---|---|---|
| 2008 (part) | 60 | +13.37 | +0.67 |
| 2009 | 250 | — | — |
| 2010 | 250 | -5.30 | -0.57 |
| 2011 | 250 | +1.99 | +0.43 |
| 2012 | 248 | -0.71 | -0.61 |
| 2013 | 250 | -0.03 | -0.01 |
| 2014 | 250 | +2.84 | +4.58 |
| 2015 | 250 | +1.40 | +2.13 |
| 2016 | 250 | +1.35 | +0.83 |
| 2017 | 250 | +0.41 | +0.72 |
| 2018 | 249 | -1.21 | -0.50 |
| 2019 | 250 | +1.55 | +2.56 |
| 2020 | 251 | +1.80 | +0.51 |
| 2021 | 250 | -0.82 | -0.72 |
| 2022 | 249 | -0.20 | -0.12 |
| 2023 | 249 | +2.11 | +1.93 |
| 2024 | 252 | -3.17 | -1.13 |
| 2025 | 250 | +2.00 | +1.12 |
| 2026 (part) | 200 | +2.30 | +1.52 |

### h = 20: the four cells, pairs from the first out-of-sample date

| models | data | trust pairs | gate open at | t vs no forecasts, levels 1–4 |
|---|---|---|---|---|
| 1 Nov pair (derived equity, ML bond) | Sep 2026 upload | 1,978 from 2 Oct 2018 | 19 of 99 | -1.16 to +1.07 |
| 1 Nov pair (derived equity, ML bond) | from 31 Aug 2006 | 4,505 from 3 Oct 2008 | 92 of 226 | -1.44 to -0.15 |
| derived for both | Sep 2026 upload | 1,978 from 2 Oct 2018 | 2 of 99 | -0.26 to -0.13 |
| derived for both | from 31 Aug 2006 | 4,505 from 3 Oct 2008 | 3 of 226 | +0.01 to +0.05 |

### h = 20: pairs from 1 Jan 2010 instead

| models | data | trust pairs | gate open at | t vs no forecasts, levels 1–4 |
|---|---|---|---|---|
| 1 Nov pair (derived equity, ML bond) | Sep 2026 upload | pairs start 2 Oct 2018: unchanged | — | — |
| 1 Nov pair (derived equity, ML bond) | from 31 Aug 2006 | 4,195 | 79 of 226 | -1.76 to -0.18 |
| derived for both | Sep 2026 upload | pairs start 2 Oct 2018: unchanged | — | — |
| derived for both | from 31 Aug 2006 | 4,195 | 21 of 226 | -0.69 to +0.46 |

### h = 20: the trust slope for each calendar year on its own

OLS of y on x with an intercept over the year's pairs (by forecast date), standard errors over 19 lags counted in full, as the gate's. One year holds about 12 independent 20-day returns, so each t is noisy.

**1 Nov pair (derived equity, ML bond), Sep 2026 upload**

| year | pairs | slope | t |
|---|---|---|---|
| 2018 (part) | 60 | +1.94 | +1.87 |
| 2019 | 250 | -1.27 | -0.85 |
| 2020 | 251 | +1.66 | +3.55 |
| 2021 | 250 | +1.17 | +1.88 |
| 2022 | 249 | +0.48 | +1.26 |
| 2023 | 249 | +1.33 | +1.63 |
| 2024 | 252 | -0.23 | -0.37 |
| 2025 | 250 | -1.26 | -1.11 |
| 2026 (part) | 167 | +2.59 | +4.44 |

**1 Nov pair (derived equity, ML bond), from 31 Aug 2006**

| year | pairs | slope | t |
|---|---|---|---|
| 2008 (part) | 60 | -0.65 | -3.21 |
| 2009 | 250 | +0.65 | +0.95 |
| 2010 | 250 | -0.02 | -0.07 |
| 2011 | 250 | +1.44 | +3.43 |
| 2012 | 248 | +0.35 | +0.57 |
| 2013 | 250 | -0.61 | -0.89 |
| 2014 | 250 | +0.43 | +0.41 |
| 2015 | 250 | +0.49 | +1.83 |
| 2016 | 250 | -1.01 | -2.53 |
| 2017 | 250 | +0.51 | +2.22 |
| 2018 | 249 | +0.17 | +0.16 |
| 2019 | 250 | -0.05 | -0.07 |
| 2020 | 251 | +0.64 | +1.16 |
| 2021 | 250 | +0.46 | +1.23 |
| 2022 | 249 | +0.51 | +1.42 |
| 2023 | 249 | +0.93 | +2.36 |
| 2024 | 252 | -0.34 | -0.69 |
| 2025 | 250 | +0.66 | +0.89 |
| 2026 (part) | 197 | +1.36 | +2.61 |

**derived for both, Sep 2026 upload**

| year | pairs | slope | t |
|---|---|---|---|
| 2018 (part) | 60 | +1.57 | +1.02 |
| 2019 | 250 | -2.57 | -4.13 |
| 2020 | 251 | +1.47 | +3.18 |
| 2021 | 250 | +0.68 | +0.57 |
| 2022 | 249 | +0.48 | +1.29 |
| 2023 | 249 | +1.14 | +1.24 |
| 2024 | 252 | -7.43 | -5.07 |
| 2025 | 250 | -0.00 | -0.00 |
| 2026 (part) | 167 | +2.70 | +3.10 |

**derived for both, from 31 Aug 2006**

| year | pairs | slope | t |
|---|---|---|---|
| 2008 (part) | 60 | -0.52 | -2.03 |
| 2009 | 250 | +0.43 | +0.53 |
| 2010 | 250 | -0.18 | -0.69 |
| 2011 | 250 | +1.57 | +3.01 |
| 2012 | 248 | +0.60 | +0.99 |
| 2013 | 250 | -0.58 | -1.07 |
| 2014 | 250 | +0.52 | +0.50 |
| 2015 | 250 | +0.60 | +2.00 |
| 2016 | 250 | -0.97 | -1.93 |
| 2017 | 250 | +0.72 | +2.65 |
| 2018 | 249 | +0.15 | +0.13 |
| 2019 | 250 | +0.41 | +0.69 |
| 2020 | 251 | +0.49 | +1.05 |
| 2021 | 250 | +0.28 | +0.74 |
| 2022 | 249 | +0.29 | +0.67 |
| 2023 | 249 | +0.59 | +1.65 |
| 2024 | 252 | -0.65 | -1.33 |
| 2025 | 250 | +1.17 | +1.15 |
| 2026 (part) | 197 | +1.40 | +1.87 |

