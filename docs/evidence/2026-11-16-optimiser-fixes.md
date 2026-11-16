# The optimiser fixes, 16 Nov 2026

Phase 4, task 3 of the 15 Nov plan. Produced by `docs/evidence/2026-11-16-optimiser-fixes.py` on `fix-claude-optimiser-audit` at 8d10d03. That commit removes the no-trade band, so every rebalance trades straight to its clipped target and pays 3 bp on equity and 5 bp on bond on what it trades. It also anchors the long-run premium π̄ to a published prior: 6.6% a year (Goetzmann & Ibbotson 2005), worth 30 years of days.

The data is the 11 Bloomberg exports from 31 Aug 2006, read and merged as the Data page does. Each file's SHA-256 is listed below.

**Active models, decided before any result was seen.** The derived polynomial is the active model for both equity and bond, fitted through the Models page's runner at its defaults, once per horizon. Neither research model is used: both scored about zero on their pre-2016 hold-out (`2026-11-15-history-2006.md`).

The window rules are unchanged:
- tuning 504 rows, train 504, test 20, embargo 20;
- trust pairs 252;
- covariance 504;
- bounds 20%–80%.

The rebalances run every h trading days from 6 Oct 2008.

The old band is rerun beside the new rule as a sensitivity only. It comes from this script's own copy of the removed `_banded` and its drift loop. The two prior settings, 3.6% and 9.6% a year, are for disclosure only. Neither changed anything in the code.

## After the checkpoint

Toby reviewed these results on 16 Nov 2026 and confirmed that the band removal stands. Nothing in the code or this file was changed because of the results. As task 5 asks, the forecasts adding no value after costs at any horizon is recorded as a known limitation in `docs/methodology.md` §17.

## Summary

### Cost drag and turnover, a year, without the band and with it

The first allocation, bought from cash, is left out.

| h | level 1 | level 2 | level 3 | level 4 | level 5 |
|---|---|---|---|---|---|
| 1 | 0.026% / 66% → 0.001% / 3% | 0.037% / 92% → 0.001% / 2% | 0.041% / 103% → 0.000% / 1% | 0.039% / 98% → 0.000% / 0% | 0.026% / 66% (no band at level 5) |
| 5 | 0.058% / 145% → 0.001% / 4% | 0.060% / 149% → 0.001% / 3% | 0.060% / 150% → 0.002% / 5% | 0.052% / 130% → 0.006% / 15% | 0.011% / 28% |
| 20 | 0.015% / 37% → 0.006% / 15% | 0.015% / 36% → 0.005% / 12% | 0.013% / 33% → 0.004% / 9% | 0.011% / 26% → 0.003% / 6% | 0.005% / 13% |

Without the band, costs are at most 0.06 points a year at any level and horizon. The band saved at most that.

### Do the forecasts add value after costs?

No, at any horizon.

| h | where the trust gate is open | t against no forecasts, levels 1–4 |
|---|---|---|
| 1 | none of the 4,520 rebalances | the paths are identical, so there is nothing to test |
| 5 | 137 of 904 rebalances | −1.10 to −1.38: forecasts cost a little, not significantly |
| 20 | 3 of 226 rebalances | +0.01 to +0.05 |

The prior's sensitivity doesn't change this:
- at h = 5, t against no forecasts is −0.86 to −1.14 with a 3.6% prior, and −1.25 to −1.44 with a 9.6% prior;
- at h = 20 it stays within ±0.13.

With the band, t against no forecasts was also negative, at −0.72 to −2.80. That comes from the band's path dependence, not from the forecasts.

### Riskier levels holding less equity

With no band, there are no rebalances at any horizon where a riskier level holds less equity than the next safer one. With the old band there were 22 such rebalances at h = 5.

### π̄

π̄ stays between 4.19% and 7.86% a year and is never at or below zero. Its lowest point is 4.19% on 9 Mar 2009.

| date | π̄ a year |
|---|---|
| end-2008 | 5.01% |
| end-March 2009 | 4.67% |
| end-May 2020 | 6.38% |

### What looks implausible, or needs a reader's care

- **The old band at short horizons.** At h = 1 the old band all but froze the portfolio, with turnover of 0–3% a year. Its width was `c·(w*_L − w_mv)/π̄`, with the cost `c` per trade but π̄ in h-day terms. At h = 1, π̄ is about 0.02%, so the band was about four times `w*_L − w_mv`, wider than any move. Level 4 under the band drifted towards equity for 18 years, which is where its +1.56 t against 50/50 at h = 1 comes from.
- **Turnover with no forecasts.** Without the band, turnover at h = 1 is up to 103% a year even though the gate never opens. It comes from trading back to `w*_L` every day: the target moves as the 504-day covariance rolls, and the weights drift. It still costs only 0.04 points a year.
- **t against 50/50.** These t-statistics (+3.7 to −2.9) measure the equity weight, not the forecasts. Levels 1–3 hold more equity than 50/50 and levels 4–5 hold less, with little tracking error.
- **|k| above 1.** `k = β̂·Δs/π̄`, so a tilt larger than `w*_L − w_mv` itself has |k| above 1. That happened at 30 of the 137 open rebalances at h = 5, and 2 of the 3 at h = 20. The bounds clip those tilts, so every level lands on the same bound. That is finding 3, and it is why no level inverts.
- **The gate at h = 20.** On this data the gate opens only 3 times at h = 20, against 26 times on the Sep 2026 upload with the 1 Nov file's models. This is why the forecasts barely move the weights at h = 20. I haven't separated how much of the difference comes from the longer history and how much from the bond model.

Runtime, local: about 6–7 minutes per horizon to fit the two models, under a minute for the optimiser and backtests. The h = 20 section matched an earlier trial run exactly.

## Output

C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 8.905765e-06, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 5.766090e-05, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.100925e-04, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.536798e-04, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.874289e-04, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.128475e-04, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.317466e-04, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.456986e-04, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.559487e-04, tolerance: 5.453e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 2.213080e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.259060e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 3.816149e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.086832e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.209301e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.258888e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.273688e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
C:\Users\User\Documents\GitHub\forecasting-engine\.venv\Lib\site-packages\sklearn\linear_model\_coordinate_descent.py:827: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 4.272355e-06, tolerance: 8.726e-07
  model = cd_fast.enet_coordinate_descent_gram(
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

Prior: 6.6% a year, worth 30 years. Costs: 3 bp equity, 5 bp bond. Bounds 20%–80%.

## h = 1

Derived polynomial, equity: Signal Rank IC +0.033, t 2.10, 4,540 rows scored.
Derived polynomial, bond: Signal Rank IC +0.029, t 1.84, 4,560 rows scored.

4,520 rebalances, 6 Oct 2008 to 5 Nov 2026.

### Per level, after costs

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
| no band | 1 | 11.64% | 0.79 | -28.2% | 66% | 0.026% | 3.25 | — |
| no band | 2 | 10.27% | 0.83 | -23.6% | 92% | 0.037% | 3.36 | — |
| no band | 3 | 8.80% | 0.88 | -19.9% | 103% | 0.041% | 3.69 | — |
| no band | 4 | 7.20% | 0.95 | -18.7% | 98% | 0.039% | -2.61 | — |
| no band | 5 | 5.18% | 1.04 | -17.7% | 66% | 0.026% | -2.85 | — |
| old band (sensitivity) | 1 | 11.31% | 0.79 | -27.7% | 3% | 0.001% | 3.11 | — |
| old band (sensitivity) | 2 | 11.05% | 0.82 | -27.2% | 2% | 0.001% | 3.26 | — |
| old band (sensitivity) | 3 | 10.36% | 0.86 | -25.4% | 1% | 0.000% | 3.05 | — |
| old band (sensitivity) | 4 | 9.18% | 0.91 | -21.7% | 0% | 0.000% | 1.56 | — |
| old band (sensitivity) | 5 | 5.18% | 1.04 | -17.7% | 66% | 0.026% | -2.85 | — |
| 50/50 benchmark | — | 8.32% | 0.88 | -19.9% | 20% | 0.008% | — | — |

### Riskier levels holding less equity than safer ones

|  | rebalances where a riskier level holds less equity | rebalances |
|---|---|---|
| no band | 0 | 4520 |
| old band (sensitivity) | 0 | 4520 |

### π̄

π̄ in annual terms: 4.19% to 7.86% over the calendar, lowest on 9 Mar 2009; 4.19% to 7.86% at the rebalances. At or below zero on 0 days.

| date | trading day | π̄ a year | π̄ over 1 days |
|---|---|---|---|
| end-2008 | 31 Dec 2008 | 5.01% | 0.020% |
| 9 Mar 2009 (the S&P 500 low) | 9 Mar 2009 | 4.19% | 0.017% |
| end-Mar 2009 | 31 Mar 2009 | 4.67% | 0.019% |
| end-May 2020 | 29 May 2020 | 6.38% | 0.025% |

### Per rebalance

Gate open (β̂ > 0) at 0 of 4,520 rebalances, shut at 4,268, too little history at 252. |k| above 1, where the tilt is larger than w*_L − w_mv itself: 0.

Only the rebalances with the gate open, where β̂ is not zero.

| rebalance | β̂ | Δs | π̄ | k |
|---|---|---|---|---|

### Prior sensitivity, no band (disclosure only)

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
| prior 3.6% | 1 | 11.64% | 0.79 | -28.2% | 66% | 0.026% | 3.25 | — |
| prior 3.6% | 2 | 10.27% | 0.83 | -23.6% | 92% | 0.037% | 3.36 | — |
| prior 3.6% | 3 | 8.80% | 0.88 | -19.9% | 103% | 0.041% | 3.69 | — |
| prior 3.6% | 4 | 7.20% | 0.95 | -18.7% | 98% | 0.039% | -2.61 | — |
| prior 3.6% | 5 | 5.18% | 1.04 | -17.7% | 66% | 0.026% | -2.85 | — |
| prior 9.6% | 1 | 11.64% | 0.79 | -28.2% | 66% | 0.026% | 3.25 | — |
| prior 9.6% | 2 | 10.27% | 0.83 | -23.6% | 92% | 0.037% | 3.36 | — |
| prior 9.6% | 3 | 8.80% | 0.88 | -19.9% | 103% | 0.041% | 3.69 | — |
| prior 9.6% | 4 | 7.20% | 0.95 | -18.7% | 98% | 0.039% | -2.61 | — |
| prior 9.6% | 5 | 5.18% | 1.04 | -17.7% | 66% | 0.026% | -2.85 | — |

Runtime here: models 363 s, optimiser and backtests 55 s.

## h = 5

Derived polynomial, equity: Signal Rank IC +0.083, t 3.20, 4,540 rows scored.
Derived polynomial, bond: Signal Rank IC +0.030, t 1.00, 4,560 rows scored.

904 rebalances, 6 Oct 2008 to 30 Oct 2026.

### Per level, after costs

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
| no band | 1 | 11.33% | 0.78 | -28.5% | 145% | 0.058% | 3.00 | -1.16 |
| no band | 2 | 9.92% | 0.80 | -23.8% | 149% | 0.060% | 2.77 | -1.38 |
| no band | 3 | 8.50% | 0.85 | -20.0% | 150% | 0.060% | 1.18 | -1.15 |
| no band | 4 | 6.95% | 0.92 | -18.7% | 130% | 0.052% | -3.11 | -1.10 |
| no band | 5 | 5.10% | 1.03 | -17.7% | 28% | 0.011% | -2.92 | — |
| old band (sensitivity) | 1 | 10.76% | 0.79 | -27.7% | 4% | 0.001% | 2.89 | -2.40 |
| old band (sensitivity) | 2 | 9.99% | 0.82 | -22.8% | 3% | 0.001% | 2.83 | -2.63 |
| old band (sensitivity) | 3 | 9.30% | 0.87 | -21.4% | 5% | 0.002% | 2.50 | -2.80 |
| old band (sensitivity) | 4 | 8.13% | 0.92 | -20.3% | 15% | 0.006% | -0.63 | -1.47 |
| old band (sensitivity) | 5 | 5.10% | 1.03 | -17.7% | 28% | 0.011% | -2.92 | — |
| 50/50 benchmark | — | 8.32% | 0.88 | -19.9% | 20% | 0.008% | — | — |

### Riskier levels holding less equity than safer ones

|  | rebalances where a riskier level holds less equity | rebalances |
|---|---|---|
| no band | 0 | 904 |
| old band (sensitivity) | 22 | 904 |

### π̄

π̄ in annual terms: 4.19% to 7.86% over the calendar, lowest on 9 Mar 2009; 4.35% to 7.85% at the rebalances. At or below zero on 0 days.

| date | trading day | π̄ a year | π̄ over 5 days |
|---|---|---|---|
| end-2008 | 31 Dec 2008 | 5.01% | 0.099% |
| 9 Mar 2009 (the S&P 500 low) | 9 Mar 2009 | 4.19% | 0.083% |
| end-Mar 2009 | 31 Mar 2009 | 4.67% | 0.093% |
| end-May 2020 | 29 May 2020 | 6.38% | 0.127% |

### Per rebalance

Gate open (β̂ > 0) at 137 of 904 rebalances, shut at 715, too little history at 52. |k| above 1, where the tilt is larger than w*_L − w_mv itself: 30.

Only the rebalances with the gate open, where β̂ is not zero.

| rebalance | β̂ | Δs | π̄ | k |
|---|---|---|---|---|
| 2 Sep 2015 | 1.000 | +0.6006% | 0.1201% | +5.002 |
| 1 Dec 2015 | 1.000 | +0.0289% | 0.1232% | +0.234 |
| 8 Dec 2015 | 1.000 | +0.0210% | 0.1224% | +0.171 |
| 15 Dec 2015 | 1.000 | +0.2096% | 0.1221% | +1.717 |
| 22 Dec 2015 | 1.000 | +0.0992% | 0.1219% | +0.814 |
| 30 Dec 2015 | 1.000 | +0.0328% | 0.1226% | +0.267 |
| 7 Jan 2016 | 1.000 | +0.1995% | 0.1192% | +1.674 |
| 10 May 2016 | 1.000 | -0.3108% | 0.1207% | -2.576 |
| 17 May 2016 | 1.000 | -0.2868% | 0.1198% | -2.394 |
| 24 May 2016 | 1.000 | -0.0084% | 0.1207% | -0.069 |
| 1 Jun 2016 | 1.000 | -0.0598% | 0.1211% | -0.494 |
| 8 Jun 2016 | 1.000 | -0.0642% | 0.1212% | -0.530 |
| 15 Jun 2016 | 1.000 | +0.1394% | 0.1198% | +1.164 |
| 22 Jun 2016 | 1.000 | +0.0341% | 0.1202% | +0.284 |
| 29 Jun 2016 | 1.000 | +0.0380% | 0.1194% | +0.319 |
| 7 Jul 2016 | 1.000 | -0.0171% | 0.1197% | -0.143 |
| 14 Jul 2016 | 1.000 | -0.0451% | 0.1214% | -0.371 |
| 21 Jul 2016 | 1.000 | -0.1370% | 0.1214% | -1.128 |
| 28 Jul 2016 | 1.000 | -0.1047% | 0.1214% | -0.863 |
| 4 Aug 2016 | 1.000 | -0.1038% | 0.1212% | -0.856 |
| 11 Aug 2016 | 1.000 | -0.1285% | 0.1218% | -1.055 |
| 18 Aug 2016 | 1.000 | -0.1394% | 0.1216% | -1.146 |
| 25 Aug 2016 | 1.000 | -0.0987% | 0.1213% | -0.814 |
| 1 Sep 2016 | 1.000 | -0.0997% | 0.1212% | -0.823 |
| 9 Sep 2016 | 1.000 | -0.1291% | 0.1204% | -1.072 |
| 16 Sep 2016 | 1.000 | -0.0034% | 0.1207% | -0.028 |
| 23 Sep 2016 | 1.000 | -0.1033% | 0.1210% | -0.854 |
| 30 Sep 2016 | 1.000 | -0.0566% | 0.1210% | -0.468 |
| 7 Oct 2016 | 1.000 | -0.0841% | 0.1209% | -0.696 |
| 17 Oct 2016 | 1.000 | -0.0057% | 0.1202% | -0.047 |
| 24 Oct 2016 | 1.000 | -0.0536% | 0.1207% | -0.444 |
| 31 Oct 2016 | 1.000 | -0.0045% | 0.1203% | -0.037 |
| 7 Nov 2016 | 1.000 | +0.0000% | 0.1203% | +0.000 |
| 15 Nov 2016 | 1.000 | +0.0000% | 0.1224% | +0.000 |
| 22 Nov 2016 | 1.000 | +0.0000% | 0.1231% | +0.000 |
| 30 Nov 2016 | 1.000 | +0.0000% | 0.1230% | +0.000 |
| 7 Dec 2016 | 1.000 | -0.1103% | 0.1238% | -0.891 |
| 14 Dec 2016 | 1.000 | -0.0902% | 0.1244% | -0.725 |
| 21 Dec 2016 | 1.000 | -0.0060% | 0.1246% | -0.048 |
| 29 Dec 2016 | 1.000 | +0.0461% | 0.1240% | +0.372 |
| 6 Jan 2017 | 1.000 | +0.0813% | 0.1244% | +0.654 |
| 13 Jan 2017 | 1.000 | +0.0680% | 0.1242% | +0.547 |
| 23 Jan 2017 | 1.000 | +0.0290% | 0.1239% | +0.234 |
| 30 Jan 2017 | 1.000 | +0.0033% | 0.1244% | +0.027 |
| 6 Feb 2017 | 1.000 | -0.2201% | 0.1244% | -1.769 |
| 13 Feb 2017 | 1.000 | +0.0461% | 0.1252% | +0.368 |
| 21 Feb 2017 | 1.000 | -0.1574% | 0.1259% | -1.250 |
| 28 Feb 2017 | 1.000 | -0.0888% | 0.1256% | -0.707 |
| 7 Mar 2017 | 1.000 | -0.0415% | 0.1260% | -0.330 |
| 14 Mar 2017 | 1.000 | -0.0504% | 0.1262% | -0.399 |
| 21 Mar 2017 | 1.000 | -0.1138% | 0.1253% | -0.909 |
| 28 Mar 2017 | 1.000 | -0.1826% | 0.1254% | -1.456 |
| 4 Apr 2017 | 1.000 | -0.6503% | 0.1252% | -5.193 |
| 11 Apr 2017 | 1.000 | -0.4954% | 0.1249% | -3.966 |
| 19 Apr 2017 | 1.000 | -0.4523% | 0.1243% | -3.639 |
| 26 Apr 2017 | 1.000 | -0.8655% | 0.1255% | -6.897 |
| 3 May 2017 | 1.000 | -0.2919% | 0.1254% | -2.327 |
| 10 May 2017 | 0.991 | -0.3581% | 0.1258% | -2.822 |
| 17 May 2017 | 0.989 | -0.3904% | 0.1244% | -3.102 |
| 24 May 2017 | 1.000 | -0.3356% | 0.1254% | -2.675 |
| 1 Jun 2017 | 0.987 | -0.2891% | 0.1258% | -2.268 |
| 8 Jun 2017 | 0.973 | -0.2986% | 0.1257% | -2.310 |
| 15 Jun 2017 | 0.970 | -0.3377% | 0.1256% | -2.608 |
| 22 Jun 2017 | 0.969 | -0.3235% | 0.1255% | -2.498 |
| 29 Jun 2017 | 0.972 | +0.0000% | 0.1253% | +0.000 |
| 7 Jul 2017 | 0.965 | +0.0000% | 0.1257% | +0.000 |
| 14 Jul 2017 | 0.965 | +0.0000% | 0.1261% | +0.000 |
| 21 Jul 2017 | 0.965 | +0.0000% | 0.1260% | +0.000 |
| 28 Jul 2017 | 0.965 | +0.0000% | 0.1261% | +0.000 |
| 4 Aug 2017 | 0.965 | +0.0000% | 0.1260% | +0.000 |
| 11 Aug 2017 | 0.965 | +0.0000% | 0.1252% | +0.000 |
| 18 Aug 2017 | 0.965 | +0.0000% | 0.1248% | +0.000 |
| 25 Aug 2017 | 0.965 | -0.2642% | 0.1250% | -2.040 |
| 1 Sep 2017 | 0.962 | -0.3091% | 0.1256% | -2.368 |
| 11 Sep 2017 | 0.957 | -0.1793% | 0.1257% | -1.365 |
| 18 Sep 2017 | 0.933 | -0.2650% | 0.1262% | -1.960 |
| 25 Sep 2017 | 0.925 | +0.0000% | 0.1259% | +0.000 |
| 2 Oct 2017 | 0.927 | +0.0000% | 0.1266% | +0.000 |
| 10 Oct 2017 | 0.927 | +0.0000% | 0.1270% | +0.000 |
| 17 Oct 2017 | 0.927 | +0.0000% | 0.1270% | +0.000 |
| 24 Oct 2017 | 0.927 | +0.0000% | 0.1273% | +0.000 |
| 31 Oct 2017 | 0.927 | +0.0000% | 0.1273% | +0.000 |
| 7 Nov 2017 | 0.927 | +0.0000% | 0.1274% | +0.000 |
| 14 Nov 2017 | 0.927 | +0.0000% | 0.1273% | +0.000 |
| 21 Nov 2017 | 0.927 | +0.0000% | 0.1276% | +0.000 |
| 29 Nov 2017 | 0.927 | +0.0000% | 0.1281% | +0.000 |
| 6 Dec 2017 | 0.927 | +0.0000% | 0.1279% | +0.000 |
| 13 Dec 2017 | 0.927 | +0.0000% | 0.1285% | +0.000 |
| 20 Dec 2017 | 0.927 | +0.0000% | 0.1291% | +0.000 |
| 28 Dec 2017 | 0.927 | +0.0000% | 0.1289% | +0.000 |
| 5 Jan 2018 | 0.927 | +0.0000% | 0.1300% | +0.000 |
| 12 Jan 2018 | 0.928 | +0.0000% | 0.1308% | +0.000 |
| 22 Jan 2018 | 0.928 | +0.0000% | 0.1318% | +0.000 |
| 29 Jan 2018 | 0.928 | +0.0000% | 0.1321% | +0.000 |
| 5 Feb 2018 | 0.928 | +0.0000% | 0.1287% | +0.000 |
| 12 Feb 2018 | 0.927 | +0.0000% | 0.1291% | +0.000 |
| 20 Feb 2018 | 0.927 | +0.0000% | 0.1303% | +0.000 |
| 27 Feb 2018 | 0.927 | +0.0000% | 0.1308% | +0.000 |
| 6 Mar 2018 | 0.927 | +0.0000% | 0.1304% | +0.000 |
| 13 Mar 2018 | 0.927 | +0.0000% | 0.1309% | +0.000 |
| 20 Mar 2018 | 0.927 | +0.0730% | 0.1301% | +0.520 |
| 27 Mar 2018 | 0.923 | +0.0972% | 0.1280% | +0.700 |
| 4 Apr 2018 | 0.907 | +0.0980% | 0.1286% | +0.692 |
| 11 Apr 2018 | 0.910 | +0.0905% | 0.1284% | +0.641 |
| 18 Apr 2018 | 0.914 | +0.0000% | 0.1297% | +0.000 |
| 25 Apr 2018 | 0.918 | +0.0000% | 0.1289% | +0.000 |
| 2 May 2018 | 0.918 | +0.0000% | 0.1286% | +0.000 |
| 9 May 2018 | 0.918 | +0.0000% | 0.1297% | +0.000 |
| 16 May 2018 | 0.918 | +0.0000% | 0.1303% | +0.000 |
| 23 May 2018 | 0.918 | +0.0000% | 0.1302% | +0.000 |
| 31 May 2018 | 0.918 | +0.0000% | 0.1293% | +0.000 |
| 7 Jun 2018 | 0.918 | +0.0000% | 0.1307% | +0.000 |
| 14 Jun 2018 | 0.918 | +0.0000% | 0.1308% | +0.000 |
| 21 Jun 2018 | 0.918 | +0.0000% | 0.1302% | +0.000 |
| 28 Jun 2018 | 0.918 | +0.0000% | 0.1294% | +0.000 |
| 6 Jul 2018 | 0.918 | +0.0000% | 0.1300% | +0.000 |
| 13 Jul 2018 | 0.918 | +0.0000% | 0.1306% | +0.000 |
| 20 Jul 2018 | 0.918 | +0.0000% | 0.1307% | +0.000 |
| 27 Jul 2018 | 0.918 | +0.0000% | 0.1310% | +0.000 |
| 3 Aug 2018 | 0.918 | +0.0000% | 0.1312% | +0.000 |
| 10 Aug 2018 | 0.918 | +0.0000% | 0.1309% | +0.000 |
| 17 Aug 2018 | 0.918 | +0.0000% | 0.1311% | +0.000 |
| 24 Aug 2018 | 0.918 | +0.0000% | 0.1314% | +0.000 |
| 31 Aug 2018 | 0.918 | +0.0000% | 0.1318% | +0.000 |
| 10 Sep 2018 | 0.918 | +0.0000% | 0.1316% | +0.000 |
| 17 Sep 2018 | 0.918 | +0.0000% | 0.1318% | +0.000 |
| 24 Sep 2018 | 0.918 | +0.0000% | 0.1324% | +0.000 |
| 1 Oct 2018 | 0.918 | +0.0000% | 0.1324% | +0.000 |
| 9 Oct 2018 | 0.918 | +0.0000% | 0.1319% | +0.000 |
| 16 Oct 2018 | 0.918 | +0.0000% | 0.1307% | +0.000 |
| 23 Oct 2018 | 0.918 | +0.0000% | 0.1295% | +0.000 |
| 30 Oct 2018 | 0.918 | +0.0000% | 0.1284% | +0.000 |
| 6 Nov 2018 | 0.918 | +0.0000% | 0.1298% | +0.000 |
| 14 Nov 2018 | 0.918 | +0.0000% | 0.1287% | +0.000 |
| 21 Nov 2018 | 0.918 | +0.0000% | 0.1277% | +0.000 |
| 29 Nov 2018 | 0.918 | +0.0000% | 0.1292% | +0.000 |
| 7 Dec 2018 | 0.918 | +0.6313% | 0.1269% | +4.565 |

### Prior sensitivity, no band (disclosure only)

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
| prior 3.6% | 1 | 11.37% | 0.78 | -28.5% | 147% | 0.059% | 3.03 | -0.90 |
| prior 3.6% | 2 | 9.95% | 0.81 | -23.8% | 151% | 0.060% | 2.80 | -1.14 |
| prior 3.6% | 3 | 8.49% | 0.85 | -20.0% | 151% | 0.060% | 1.04 | -1.08 |
| prior 3.6% | 4 | 6.96% | 0.91 | -18.7% | 136% | 0.054% | -3.07 | -0.86 |
| prior 3.6% | 5 | 5.10% | 1.03 | -17.7% | 28% | 0.011% | -2.92 | — |
| prior 9.6% | 1 | 11.33% | 0.78 | -28.5% | 139% | 0.056% | 3.00 | -1.26 |
| prior 9.6% | 2 | 9.93% | 0.80 | -23.8% | 147% | 0.059% | 2.79 | -1.44 |
| prior 9.6% | 3 | 8.51% | 0.85 | -20.0% | 145% | 0.058% | 1.29 | -1.25 |
| prior 9.6% | 4 | 6.95% | 0.92 | -18.7% | 121% | 0.049% | -3.12 | -1.30 |
| prior 9.6% | 5 | 5.10% | 1.03 | -17.7% | 28% | 0.011% | -2.92 | — |

Runtime here: models 399 s, optimiser and backtests 21 s.

## h = 20

Derived polynomial, equity: Signal Rank IC +0.103, t 1.98, 4,537 rows scored.
Derived polynomial, bond: Signal Rank IC +0.030, t 0.49, 4,545 rows scored.

226 rebalances, 6 Oct 2008 to 9 Oct 2026.

### Per level, after costs

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
| no band | 1 | 11.60% | 0.80 | -28.5% | 37% | 0.015% | 3.22 | 0.05 |
| no band | 2 | 10.20% | 0.83 | -23.7% | 36% | 0.015% | 3.23 | 0.05 |
| no band | 3 | 8.72% | 0.89 | -20.1% | 33% | 0.013% | 2.47 | 0.04 |
| no band | 4 | 7.11% | 0.96 | -18.8% | 26% | 0.011% | -2.71 | 0.01 |
| no band | 5 | 5.11% | 1.05 | -17.8% | 13% | 0.005% | -2.90 | — |
| old band (sensitivity) | 1 | 11.07% | 0.79 | -27.7% | 15% | 0.006% | 3.07 | -1.18 |
| old band (sensitivity) | 2 | 10.34% | 0.83 | -25.7% | 12% | 0.005% | 3.04 | -1.65 |
| old band (sensitivity) | 3 | 8.87% | 0.88 | -20.7% | 9% | 0.004% | 2.07 | -1.19 |
| old band (sensitivity) | 4 | 7.21% | 0.96 | -19.3% | 6% | 0.003% | -2.56 | -0.72 |
| old band (sensitivity) | 5 | 5.11% | 1.05 | -17.8% | 13% | 0.005% | -2.90 | — |
| 50/50 benchmark | — | 8.32% | 0.88 | -19.9% | 20% | 0.008% | — | — |

### Riskier levels holding less equity than safer ones

|  | rebalances where a riskier level holds less equity | rebalances |
|---|---|---|
| no band | 0 | 226 |
| old band (sensitivity) | 0 | 226 |

### π̄

π̄ in annual terms: 4.19% to 7.86% over the calendar, lowest on 9 Mar 2009; 4.35% to 7.85% at the rebalances. At or below zero on 0 days.

| date | trading day | π̄ a year | π̄ over 20 days |
|---|---|---|---|
| end-2008 | 31 Dec 2008 | 5.01% | 0.398% |
| 9 Mar 2009 (the S&P 500 low) | 9 Mar 2009 | 4.19% | 0.332% |
| end-Mar 2009 | 31 Mar 2009 | 4.67% | 0.371% |
| end-May 2020 | 29 May 2020 | 6.38% | 0.506% |

### Per rebalance

Gate open (β̂ > 0) at 3 of 226 rebalances, shut at 209, too little history at 14. |k| above 1, where the tilt is larger than w*_L − w_mv itself: 2.

| rebalance | β̂ | Δs | π̄ | k |
|---|---|---|---|---|
| 6 Oct 2008 | 0.000 | -2.2062% | 0.4315% | -0.000 |
| 4 Nov 2008 | 0.000 | -4.3714% | 0.4310% | -0.000 |
| 4 Dec 2008 | 0.000 | -7.0210% | 0.3867% | -0.000 |
| 5 Jan 2009 | 0.000 | -5.6520% | 0.4053% | -0.000 |
| 3 Feb 2009 | 0.000 | -1.4244% | 0.3822% | -0.000 |
| 4 Mar 2009 | 0.000 | -3.1580% | 0.3453% | -0.000 |
| 1 Apr 2009 | 0.000 | -1.7641% | 0.3747% | -0.000 |
| 30 Apr 2009 | 0.000 | -1.7500% | 0.3920% | -0.000 |
| 29 May 2009 | 0.000 | +3.2079% | 0.4032% | +0.000 |
| 26 Jun 2009 | 0.000 | +1.5761% | 0.4016% | +0.000 |
| 27 Jul 2009 | 0.000 | +0.1577% | 0.4160% | +0.000 |
| 24 Aug 2009 | 0.000 | -0.9630% | 0.4226% | -0.000 |
| 22 Sep 2009 | 0.000 | -1.0003% | 0.4301% | -0.000 |
| 21 Oct 2009 | 0.000 | +0.2491% | 0.4304% | +0.000 |
| 19 Nov 2009 | 0.597 | -1.3882% | 0.4307% | -1.925 |
| 18 Dec 2009 | 0.597 | -0.1329% | 0.4317% | -0.184 |
| 20 Jan 2010 | 0.616 | -1.3400% | 0.4380% | -1.886 |
| 18 Feb 2010 | 0.000 | -1.3794% | 0.4319% | -0.000 |
| 18 Mar 2010 | 0.000 | -0.5161% | 0.4409% | -0.000 |
| 16 Apr 2010 | 0.000 | -0.5445% | 0.4449% | -0.000 |
| 14 May 2010 | 0.000 | +0.5998% | 0.4307% | +0.000 |
| 14 Jun 2010 | 0.000 | +0.8593% | 0.4203% | +0.000 |
| 13 Jul 2010 | 0.000 | -0.4867% | 0.4188% | -0.000 |
| 10 Aug 2010 | 0.000 | -0.7953% | 0.4201% | -0.000 |
| 8 Sep 2010 | 0.000 | +0.0339% | 0.4142% | +0.000 |
| 6 Oct 2010 | 0.000 | -0.3202% | 0.4239% | -0.000 |
| 4 Nov 2010 | 0.000 | +0.5847% | 0.4344% | +0.000 |
| 6 Dec 2010 | 0.000 | +0.6685% | 0.4377% | +0.000 |
| 4 Jan 2011 | 0.000 | -1.9106% | 0.4477% | -0.000 |
| 2 Feb 2011 | 0.000 | -0.3774% | 0.4537% | -0.000 |
| 3 Mar 2011 | 0.000 | -0.7497% | 0.4579% | -0.000 |
| 31 Mar 2011 | 0.000 | -2.1975% | 0.4549% | -0.000 |
| 29 Apr 2011 | 0.000 | -1.6336% | 0.4577% | -0.000 |
| 27 May 2011 | 0.000 | -0.5572% | 0.4490% | -0.000 |
| 27 Jun 2011 | 0.000 | -0.4177% | 0.4386% | -0.000 |
| 26 Jul 2011 | 0.000 | -1.7336% | 0.4463% | -0.000 |
| 23 Aug 2011 | 0.000 | +2.3977% | 0.4120% | +0.000 |
| 21 Sep 2011 | 0.000 | +1.7428% | 0.4106% | +0.000 |
| 20 Oct 2011 | 0.000 | +1.0225% | 0.4221% | +0.000 |
| 18 Nov 2011 | 0.000 | +2.3642% | 0.4208% | +0.000 |
| 19 Dec 2011 | 0.000 | +1.1037% | 0.4169% | +0.000 |
| 19 Jan 2012 | 0.000 | +0.7672% | 0.4359% | +0.000 |
| 16 Feb 2012 | 0.000 | -0.1708% | 0.4418% | -0.000 |
| 16 Mar 2012 | 0.000 | -0.3567% | 0.4500% | -0.000 |
| 16 Apr 2012 | 0.000 | -0.8494% | 0.4411% | -0.000 |
| 14 May 2012 | 0.000 | -0.4129% | 0.4340% | -0.000 |
| 12 Jun 2012 | 0.000 | +0.3626% | 0.4309% | +0.000 |
| 11 Jul 2012 | 0.000 | -0.6706% | 0.4309% | -0.000 |
| 8 Aug 2012 | 0.000 | -2.0217% | 0.4401% | -0.000 |
| 6 Sep 2012 | 0.000 | -1.2244% | 0.4440% | -0.000 |
| 4 Oct 2012 | 0.000 | -1.7374% | 0.4469% | -0.000 |
| 6 Nov 2012 | 0.000 | -1.8214% | 0.4413% | -0.000 |
| 6 Dec 2012 | 0.000 | -2.2562% | 0.4379% | -0.000 |
| 7 Jan 2013 | 0.000 | -1.9912% | 0.4461% | -0.000 |
| 5 Feb 2013 | 0.000 | -1.2783% | 0.4536% | -0.000 |
| 6 Mar 2013 | 0.000 | -1.7400% | 0.4567% | -0.000 |
| 4 Apr 2013 | 0.000 | -1.6147% | 0.4571% | -0.000 |
| 2 May 2013 | 0.000 | -2.1290% | 0.4602% | -0.000 |
| 31 May 2013 | 0.000 | -1.7335% | 0.4685% | -0.000 |
| 28 Jun 2013 | 0.000 | -1.1230% | 0.4682% | -0.000 |
| 29 Jul 2013 | 0.000 | -0.7627% | 0.4775% | -0.000 |
| 26 Aug 2013 | 0.000 | -0.9817% | 0.4750% | -0.000 |
| 24 Sep 2013 | 0.000 | +0.0226% | 0.4772% | +0.000 |
| 23 Oct 2013 | 0.000 | -0.5717% | 0.4805% | -0.000 |
| 21 Nov 2013 | 0.000 | -0.9648% | 0.4874% | -0.000 |
| 20 Dec 2013 | 0.000 | -0.2296% | 0.4896% | -0.000 |
| 22 Jan 2014 | 0.000 | -0.9694% | 0.4911% | -0.000 |
| 20 Feb 2014 | 0.000 | -0.4922% | 0.4887% | -0.000 |
| 20 Mar 2014 | 0.000 | -0.6351% | 0.4917% | -0.000 |
| 17 Apr 2014 | 0.000 | -0.4155% | 0.4887% | -0.000 |
| 16 May 2014 | 0.000 | -0.4154% | 0.4870% | -0.000 |
| 16 Jun 2014 | 0.000 | -0.9138% | 0.4931% | -0.000 |
| 15 Jul 2014 | 0.000 | -1.1918% | 0.4956% | -0.000 |
| 12 Aug 2014 | 0.000 | -0.6834% | 0.4899% | -0.000 |
| 10 Sep 2014 | 0.000 | -0.8105% | 0.4962% | -0.000 |
| 8 Oct 2014 | 0.000 | +0.2979% | 0.4905% | +0.000 |
| 6 Nov 2014 | 0.000 | -0.0821% | 0.4968% | -0.000 |
| 8 Dec 2014 | 0.000 | -0.6794% | 0.4980% | -0.000 |
| 7 Jan 2015 | 0.000 | +4.1987% | 0.4914% | +0.000 |
| 5 Feb 2015 | 0.000 | +2.6013% | 0.4931% | +0.000 |
| 6 Mar 2015 | 0.000 | +0.0383% | 0.4964% | +0.000 |
| 6 Apr 2015 | 0.000 | -0.1908% | 0.4934% | -0.000 |
| 4 May 2015 | 0.000 | -1.2477% | 0.4979% | -0.000 |
| 2 Jun 2015 | 0.000 | -0.4349% | 0.4983% | -0.000 |
| 30 Jun 2015 | 0.000 | +2.2191% | 0.4937% | +0.000 |
| 29 Jul 2015 | 0.000 | -0.0976% | 0.4967% | -0.000 |
| 26 Aug 2015 | 0.000 | +4.3359% | 0.4793% | +0.000 |
| 24 Sep 2015 | 0.000 | +2.6105% | 0.4771% | +0.000 |
| 23 Oct 2015 | 0.000 | -3.1459% | 0.4902% | -0.000 |
| 23 Nov 2015 | 0.000 | -2.2610% | 0.4924% | -0.000 |
| 22 Dec 2015 | 0.000 | -0.0516% | 0.4876% | -0.000 |
| 22 Jan 2016 | 0.000 | +1.9568% | 0.4721% | +0.000 |
| 22 Feb 2016 | 0.000 | -1.9939% | 0.4739% | -0.000 |
| 21 Mar 2016 | 0.000 | -0.9785% | 0.4835% | -0.000 |
| 19 Apr 2016 | 0.000 | -0.4867% | 0.4854% | -0.000 |
| 17 May 2016 | 0.000 | +0.2562% | 0.4792% | +0.000 |
| 15 Jun 2016 | 0.000 | -0.1690% | 0.4791% | -0.000 |
| 14 Jul 2016 | 0.000 | -0.0411% | 0.4856% | -0.000 |
| 11 Aug 2016 | 0.000 | -0.0802% | 0.4871% | -0.000 |
| 9 Sep 2016 | 0.000 | -0.8076% | 0.4817% | -0.000 |
| 7 Oct 2016 | 0.000 | -0.9474% | 0.4837% | -0.000 |
| 7 Nov 2016 | 0.000 | +0.7276% | 0.4814% | +0.000 |
| 7 Dec 2016 | 0.000 | -0.4826% | 0.4953% | -0.000 |
| 6 Jan 2017 | 0.000 | +0.6370% | 0.4976% | +0.000 |
| 6 Feb 2017 | 0.000 | -1.1828% | 0.4976% | -0.000 |
| 7 Mar 2017 | 0.000 | -2.0298% | 0.5042% | -0.000 |
| 4 Apr 2017 | 0.000 | -3.0982% | 0.5009% | -0.000 |
| 3 May 2017 | 0.000 | -3.4695% | 0.5016% | -0.000 |
| 1 Jun 2017 | 0.000 | -2.6650% | 0.5031% | -0.000 |
| 29 Jun 2017 | 0.000 | -1.0293% | 0.5013% | -0.000 |
| 28 Jul 2017 | 0.000 | -1.1926% | 0.5042% | -0.000 |
| 25 Aug 2017 | 0.000 | -1.0296% | 0.5001% | -0.000 |
| 25 Sep 2017 | 0.000 | -0.8239% | 0.5035% | -0.000 |
| 24 Oct 2017 | 0.000 | -0.9348% | 0.5093% | -0.000 |
| 21 Nov 2017 | 0.000 | -0.4970% | 0.5104% | -0.000 |
| 20 Dec 2017 | 0.000 | +0.0621% | 0.5162% | +0.000 |
| 22 Jan 2018 | 0.000 | -0.3537% | 0.5271% | -0.000 |
| 20 Feb 2018 | 0.000 | +0.3636% | 0.5211% | +0.000 |
| 20 Mar 2018 | 0.000 | +0.2536% | 0.5205% | +0.000 |
| 18 Apr 2018 | 0.000 | -0.5568% | 0.5189% | -0.000 |
| 16 May 2018 | 0.000 | +0.0017% | 0.5213% | +0.000 |
| 14 Jun 2018 | 0.000 | -0.5234% | 0.5233% | -0.000 |
| 13 Jul 2018 | 0.000 | -0.2495% | 0.5223% | -0.000 |
| 10 Aug 2018 | 0.000 | +0.0637% | 0.5235% | +0.000 |
| 10 Sep 2018 | 0.000 | +0.0015% | 0.5262% | +0.000 |
| 9 Oct 2018 | 0.000 | -0.0826% | 0.5278% | -0.000 |
| 6 Nov 2018 | 0.000 | +0.2684% | 0.5193% | +0.000 |
| 7 Dec 2018 | 0.000 | +3.1674% | 0.5077% | +0.000 |
| 8 Jan 2019 | 0.000 | +1.2897% | 0.5015% | +0.000 |
| 6 Feb 2019 | 0.000 | +0.2738% | 0.5104% | +0.000 |
| 7 Mar 2019 | 0.000 | +1.3005% | 0.5102% | +0.000 |
| 4 Apr 2019 | 0.000 | +0.7194% | 0.5162% | +0.000 |
| 3 May 2019 | 0.000 | +1.0315% | 0.5190% | +0.000 |
| 3 Jun 2019 | 0.000 | +2.5641% | 0.5015% | +0.000 |
| 1 Jul 2019 | 0.000 | +0.9984% | 0.5137% | +0.000 |
| 30 Jul 2019 | 0.000 | +0.4718% | 0.5157% | +0.000 |
| 27 Aug 2019 | 0.000 | +0.3918% | 0.5015% | +0.000 |
| 25 Sep 2019 | 0.000 | +0.9996% | 0.5098% | +0.000 |
| 24 Oct 2019 | 0.000 | +0.2547% | 0.5104% | +0.000 |
| 22 Nov 2019 | 0.000 | +0.2716% | 0.5154% | +0.000 |
| 23 Dec 2019 | 0.000 | -0.1845% | 0.5216% | -0.000 |
| 23 Jan 2020 | 0.000 | -0.0509% | 0.5243% | -0.000 |
| 21 Feb 2020 | 0.000 | +0.5863% | 0.5219% | +0.000 |
| 20 Mar 2020 | 0.000 | +2.5018% | 0.4637% | +0.000 |
| 20 Apr 2020 | 0.000 | +5.4110% | 0.4940% | +0.000 |
| 18 May 2020 | 0.000 | +6.7874% | 0.5026% | +0.000 |
| 16 Jun 2020 | 0.000 | +3.4907% | 0.5108% | +0.000 |
| 15 Jul 2020 | 0.000 | +1.4473% | 0.5142% | +0.000 |
| 12 Aug 2020 | 0.000 | +2.0704% | 0.5214% | +0.000 |
| 10 Sep 2020 | 0.000 | +2.4495% | 0.5193% | +0.000 |
| 8 Oct 2020 | 0.000 | +2.4341% | 0.5252% | +0.000 |
| 6 Nov 2020 | 0.000 | +1.6757% | 0.5275% | +0.000 |
| 8 Dec 2020 | 0.000 | -1.8607% | 0.5362% | -0.000 |
| 7 Jan 2021 | 0.000 | +0.1345% | 0.5412% | +0.000 |
| 5 Feb 2021 | 0.000 | -1.6527% | 0.5449% | -0.000 |
| 8 Mar 2021 | 0.000 | -0.2692% | 0.5452% | -0.000 |
| 6 Apr 2021 | 0.000 | -1.6450% | 0.5555% | -0.000 |
| 4 May 2021 | 0.000 | -2.6806% | 0.5578% | -0.000 |
| 2 Jun 2021 | 0.000 | -1.7087% | 0.5588% | -0.000 |
| 30 Jun 2021 | 0.000 | -2.2458% | 0.5604% | -0.000 |
| 29 Jul 2021 | 0.000 | -1.6678% | 0.5629% | -0.000 |
| 26 Aug 2021 | 0.000 | -0.9677% | 0.5647% | -0.000 |
| 24 Sep 2021 | 0.000 | -0.8816% | 0.5638% | -0.000 |
| 25 Oct 2021 | 0.000 | -3.2667% | 0.5690% | -0.000 |
| 23 Nov 2021 | 0.000 | +0.2111% | 0.5737% | +0.000 |
| 22 Dec 2021 | 0.000 | +0.3355% | 0.5717% | +0.000 |
| 21 Jan 2022 | 0.000 | -1.6414% | 0.5627% | -0.000 |
| 18 Feb 2022 | 0.000 | -0.9296% | 0.5637% | -0.000 |
| 21 Mar 2022 | 0.000 | +0.3341% | 0.5721% | +0.000 |
| 19 Apr 2022 | 0.000 | +2.0364% | 0.5778% | +0.000 |
| 17 May 2022 | 0.000 | +1.2959% | 0.5638% | +0.000 |
| 15 Jun 2022 | 0.000 | +0.7552% | 0.5537% | +0.000 |
| 15 Jul 2022 | 0.000 | +1.7342% | 0.5529% | +0.000 |
| 12 Aug 2022 | 0.000 | -1.1353% | 0.5684% | -0.000 |
| 12 Sep 2022 | 0.000 | +1.0286% | 0.5663% | +0.000 |
| 11 Oct 2022 | 0.000 | -0.0950% | 0.5496% | -0.000 |
| 8 Nov 2022 | 0.000 | -3.5926% | 0.5618% | -0.000 |
| 8 Dec 2022 | 0.000 | +0.4005% | 0.5584% | +0.000 |
| 9 Jan 2023 | 0.000 | +2.5129% | 0.5545% | +0.000 |
| 7 Feb 2023 | 0.000 | -1.7819% | 0.5654% | -0.000 |
| 8 Mar 2023 | 0.000 | -1.6252% | 0.5615% | -0.000 |
| 5 Apr 2023 | 0.000 | -0.2622% | 0.5579% | -0.000 |
| 4 May 2023 | 0.000 | +0.7102% | 0.5565% | +0.000 |
| 2 Jun 2023 | 0.000 | +0.6904% | 0.5678% | +0.000 |
| 3 Jul 2023 | 0.000 | +0.0613% | 0.5742% | +0.000 |
| 1 Aug 2023 | 0.000 | -0.2036% | 0.5787% | -0.000 |
| 29 Aug 2023 | 0.000 | -0.2499% | 0.5754% | -0.000 |
| 27 Sep 2023 | 0.000 | -0.8435% | 0.5707% | -0.000 |
| 26 Oct 2023 | 0.000 | -1.7605% | 0.5663% | -0.000 |
| 24 Nov 2023 | 0.000 | -1.3844% | 0.5770% | -0.000 |
| 22 Dec 2023 | 0.000 | -2.2357% | 0.5759% | -0.000 |
| 24 Jan 2024 | 0.000 | -3.0572% | 0.5810% | -0.000 |
| 22 Feb 2024 | 0.000 | -2.7570% | 0.5887% | -0.000 |
| 21 Mar 2024 | 0.000 | -1.4455% | 0.5916% | -0.000 |
| 19 Apr 2024 | 0.000 | -1.7121% | 0.5848% | -0.000 |
| 17 May 2024 | 0.000 | -1.3291% | 0.5922% | -0.000 |
| 17 Jun 2024 | 0.000 | -0.9515% | 0.5948% | -0.000 |
| 17 Jul 2024 | 0.000 | -0.0433% | 0.5953% | -0.000 |
| 14 Aug 2024 | 0.000 | +0.2329% | 0.5872% | +0.000 |
| 12 Sep 2024 | 0.000 | -0.7465% | 0.5883% | -0.000 |
| 10 Oct 2024 | 0.000 | -1.0874% | 0.5958% | -0.000 |
| 7 Nov 2024 | 0.000 | -3.0158% | 0.6019% | -0.000 |
| 6 Dec 2024 | 0.000 | -0.5036% | 0.6022% | -0.000 |
| 7 Jan 2025 | 0.000 | -0.8175% | 0.6009% | -0.000 |
| 6 Feb 2025 | 0.000 | +0.2678% | 0.6020% | +0.000 |
| 7 Mar 2025 | 0.000 | +0.3360% | 0.5912% | +0.000 |
| 4 Apr 2025 | 0.000 | +4.3536% | 0.5676% | +0.000 |
| 5 May 2025 | 0.000 | -0.5015% | 0.5877% | -0.000 |
| 3 Jun 2025 | 0.000 | -0.3700% | 0.5963% | -0.000 |
| 2 Jul 2025 | 0.000 | -0.4606% | 0.5997% | -0.000 |
| 31 Jul 2025 | 0.000 | -1.5926% | 0.6020% | -0.000 |
| 28 Aug 2025 | 0.000 | -1.2325% | 0.6032% | -0.000 |
| 26 Sep 2025 | 0.000 | -1.9820% | 0.6047% | -0.000 |
| 24 Oct 2025 | 0.000 | -1.3496% | 0.6053% | -0.000 |
| 21 Nov 2025 | 0.000 | +0.6297% | 0.6006% | +0.000 |
| 22 Dec 2025 | 0.000 | -1.7236% | 0.6064% | -0.000 |
| 22 Jan 2026 | 0.000 | -1.9837% | 0.6061% | -0.000 |
| 20 Feb 2026 | 0.000 | -0.9510% | 0.6033% | -0.000 |
| 20 Mar 2026 | 0.000 | +1.1771% | 0.5960% | +0.000 |
| 20 Apr 2026 | 0.000 | -0.0955% | 0.6072% | -0.000 |
| 18 May 2026 | 0.000 | -2.7735% | 0.6154% | -0.000 |
| 16 Jun 2026 | 0.000 | -1.6334% | 0.6149% | -0.000 |
| 16 Jul 2026 | 0.000 | -1.1004% | 0.6154% | -0.000 |
| 13 Aug 2026 | 0.000 | -1.4449% | 0.6203% | -0.000 |
| 11 Sep 2026 | 0.000 | -2.6056% | 0.6189% | -0.000 |
| 9 Oct 2026 | 0.000 | -0.3988% | 0.6230% | -0.000 |

### Prior sensitivity, no band (disclosure only)

|  | level | net annual return | net Sharpe | max drawdown | turnover a year | costs a year | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|
| prior 3.6% | 1 | 11.57% | 0.80 | -28.5% | 34% | 0.014% | 3.20 | -0.05 |
| prior 3.6% | 2 | 10.18% | 0.83 | -23.7% | 34% | 0.014% | 3.20 | -0.06 |
| prior 3.6% | 3 | 8.70% | 0.88 | -20.1% | 31% | 0.012% | 2.35 | -0.08 |
| prior 3.6% | 4 | 7.10% | 0.96 | -18.8% | 25% | 0.010% | -2.72 | -0.13 |
| prior 3.6% | 5 | 5.11% | 1.05 | -17.8% | 13% | 0.005% | -2.90 | — |
| prior 9.6% | 1 | 11.60% | 0.80 | -28.5% | 38% | 0.015% | 3.22 | 0.08 |
| prior 9.6% | 2 | 10.20% | 0.83 | -23.7% | 37% | 0.015% | 3.24 | 0.09 |
| prior 9.6% | 3 | 8.72% | 0.89 | -20.1% | 34% | 0.013% | 2.51 | 0.08 |
| prior 9.6% | 4 | 7.11% | 0.96 | -18.8% | 27% | 0.011% | -2.70 | 0.07 |
| prior 9.6% | 5 | 5.11% | 1.05 | -17.8% | 13% | 0.005% | -2.90 | — |

Runtime here: models 421 s, optimiser and backtests 13 s.

