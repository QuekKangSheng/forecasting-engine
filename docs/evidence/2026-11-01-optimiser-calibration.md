# Optimiser calibration, 1 Nov 2026

The volatility-targeted risk levels and the earned forecast trust on the real data. Produced by `docs/evidence/2026-11-01-optimiser-calibration.py`, which reads what the app itself produced. The 11 Bloomberg CSVs were uploaded through the Data page and committed in a scratch folder (31 Aug 2016 – 30 Sep 2026; equity = SPX total return, bond = LBUSTRUU total return, 9 signals). The committed frame equals the repo's `data/last_commit.pkl` apart from column order, and nothing in the repo's `data/` was written. Both models were fitted on the Models page in that folder and set active there.

## Settings

Page defaults: 20-day horizon, 504-day train window, 20-day test window, 20-day embargo, bounds 20%–80%, equity = Polynomial (derived), bond = Machine Learning. 99 rebalances from 4 Oct 2018 to 11 Aug 2026, 1,978 forecast-and-outcome pairs.

**The t-statistic gate was added after seeing the ungated result**, so it is not an independent test on this data. Results at the default bounds are shown with and without it. The Machine Learning bond model fails the model gate on t (0.59).

## Conclusion

The forecasts cost value over this backtest at levels 1 to 3, by 0.92, 0.60 and 0.16 points of net return a year, and made no material difference at level 4 (+0.03). None of these is statistically significant: the t against no forecasts furthest from zero is −1.82, and they rest on 26 tilts in 3 separate runs. Level 5 ignores forecasts. Without the gate the sign changes at level 3 (+0.19), which also shows how little the data settles.

## Gate timeline

| gate | from | to | rebalances | raw slope | t over h−1 lags |
|---|---|---|---|---|---|
| too little history | 04 Oct 2018 | 22 Oct 2019 | 14 | — | — |
| shut | 20 Nov 2019 | 03 Apr 2023 | 43 | -0.78 to 0.96 | -0.73 to 2.00 |
| open | 02 May 2023 | 04 Feb 2025 | 23 | 0.67 to 0.81 | 2.04 to 2.33 |
| shut | 05 Mar 2025 | 16 Apr 2026 | 15 | 0.53 to 0.65 | 1.54 to 1.97 |
| open | 14 May 2026 | 14 May 2026 | 1 | 0.63 to 0.63 | 2.07 to 2.07 |
| shut | 12 Jun 2026 | 12 Jun 2026 | 1 | 0.61 to 0.61 | 2.00 to 2.00 |
| open | 14 Jul 2026 | 11 Aug 2026 | 2 | 0.62 to 0.62 | 2.02 to 2.03 |

- With the gate: `β̂` is non-zero at 26 of 99 rebalances, first on 2 May 2023, averaging 0.75 when open.
- Without it: first non-zero on 14 May 2020, averaging 0.72 after that, latest 0.62, never clipped at 1.
- Premium per 20 days: 0.32% to 1.38%, at or below zero at none of the 99 rebalances.

## Slope on all pairs and on pairs from 2023

| pairs | count | independent, about | slope | t over h−1 lags | R² |
|---|---|---|---|---|---|
| every pair | 1,942 | 97 | 0.616 | 2.02 | 0.020 |
| dated 2023 onwards | 882 | 44 | 0.339 | 0.88 | 0.007 |

## Weights

With the gate:

| level | avg w*_L | range w*_L | avg final | range final | w*_L at a bound | tilted target capped | final at a bound | held by band |
|---|---|---|---|---|---|---|---|---|
| 1 | 80.0% | 80.0%–80.0% | 74.5% | 24.9%–80.0% | 100% | 15% | 17% | 64% |
| 2 | 67.0% | 65.7%–69.6% | 65.9% | 24.0%–77.0% | 0% | 13% | 0% | 58% |
| 3 | 53.4% | 51.3%–58.3% | 54.0% | 23.2%–77.8% | 0% | 9% | 0% | 54% |
| 4 | 38.9% | 36.5%–45.0% | 39.9% | 22.2%–77.9% | 0% | 5% | 0% | 46% |
| 5 | 20.0% | 20.0%–20.0% | 20.0% | 20.0%–20.0% | 100% | 0% | 100% | 0% |

Without the gate:

| level | avg w*_L | range w*_L | avg final | range final | w*_L at a bound | tilted target capped | final at a bound | held by band |
|---|---|---|---|---|---|---|---|---|
| 1 | 80.0% | 80.0%–80.0% | 68.3% | 24.9%–80.0% | 100% | 41% | 14% | 42% |
| 2 | 67.0% | 65.7%–69.6% | 59.8% | 24.0%–77.0% | 0% | 32% | 0% | 38% |
| 3 | 53.4% | 51.3%–58.3% | 51.3% | 23.2%–77.8% | 0% | 18% | 0% | 33% |
| 4 | 38.9% | 36.5%–45.0% | 38.6% | 22.2%–77.9% | 0% | 12% | 0% | 28% |
| 5 | 20.0% | 20.0%–20.0% | 20.0% | 20.0%–20.0% | 100% | 0% | 100% | 0% |

"Tilted target capped" and "final at a bound" count every rebalance; "held by band" counts those after the first.

## Risk aversion

| level | median λ_L | range λ_L | median k = λ_L/β̂, gate open | old k |
|---|---|---|---|---|
| 1 | 5.6 | 1.5–19.1 | 7.7 | 40 |
| 2 | 6.9 | 1.9–23.4 | 9.3 | 40 |
| 3 | 8.7 | 2.5–30.7 | 11.7 | 40 |
| 4 | 12.4 | 3.7–46.0 | 16.6 | 40 |
| 5 | infinite or none | — | — | 40 |

## Performance, after costs

| portfolio | net annual return | net Sharpe | max drawdown | turnover a year | t vs 50/50 |
|---|---|---|---|---|---|
| Level 1, gated | 10.85% | 0.77 | -27.2% | 92% | 1.91 |
| Level 1, ungated | 10.68% | 0.78 | -27.2% | 213% | 1.88 |
| Level 1, no forecasts | 11.82% | 0.81 | -27.2% | 4% | 2.36 |
| Level 2, gated | 9.97% | 0.79 | -23.4% | 100% | 1.90 |
| Level 2, ungated | 9.75% | 0.80 | -23.4% | 284% | 1.62 |
| Level 2, no forecasts | 10.61% | 0.83 | -23.4% | 4% | 2.42 |
| Level 3, gated | 8.56% | 0.82 | -20.4% | 102% | 0.91 |
| Level 3, ungated | 8.91% | 0.83 | -20.3% | 295% | 0.95 |
| Level 3, no forecasts | 8.74% | 0.84 | -20.4% | 5% | 1.99 |
| Level 4, gated | 6.76% | 0.83 | -19.1% | 91% | -2.01 |
| Level 4, ungated | 7.03% | 0.84 | -19.1% | 219% | -1.47 |
| Level 4, no forecasts | 6.73% | 0.84 | -19.1% | 6% | -2.30 |
| Level 5, gated | 4.33% | 0.76 | -17.9% | 13% | -2.25 |
| Level 5, ungated | 4.33% | 0.76 | -17.9% | 13% | -2.25 |
| Level 5, no forecasts | 4.33% | 0.76 | -17.9% | 13% | -2.25 |
| 50/50 benchmark | 8.18% | 0.83 | -19.9% | 21% | — |

Turnover is a year's, leaving out the first allocation. The benchmark's is its monthly reset.

## What the forecasts changed

**With the gate**

| level | gross P&L, a year | gross P&L, summed | net P&L, a year | extra costs, summed | non-zero tilts | tilts with a different weight | separate runs of tilts | days the paths differ | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -0.88% | -7.1% | -0.92% | +0.27% | 26 | 25 | 3 | 839 of 2,038 | -1.82 |
| 2 | -0.57% | -4.6% | -0.60% | +0.30% | 26 | 26 | 3 | 619 of 2,038 | -1.30 |
| 3 | -0.12% | -1.0% | -0.16% | +0.30% | 26 | 25 | 3 | 599 of 2,038 | -0.40 |
| 4 | +0.07% | +0.5% | +0.03% | +0.26% | 26 | 25 | 3 | 599 of 2,038 | 0.12 |
| 5 | +0.00% | +0.0% | +0.00% | +0.00% | 0 | 0 | 0 | 0 of 2,038 | — |

**Without the gate**

| level | gross P&L, a year | gross P&L, summed | net P&L, a year | extra costs, summed | non-zero tilts | tilts with a different weight | separate runs of tilts | days the paths differ | t vs no forecasts |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.09% | -8.8% | -1.17% | +0.65% | 79 | 62 | 1 | 1,264 of 2,038 | -1.46 |
| 2 | -0.74% | -6.0% | -0.85% | +0.87% | 79 | 77 | 1 | 1,567 of 2,038 | -1.12 |
| 3 | +0.30% | +2.4% | +0.19% | +0.90% | 79 | 77 | 1 | 1,567 of 2,038 | 0.25 |
| 4 | +0.39% | +3.2% | +0.31% | +0.66% | 79 | 77 | 1 | 1,567 of 2,038 | 0.57 |
| 5 | +0.00% | +0.0% | +0.00% | +0.00% | 0 | 0 | 0 | 0 of 2,038 | — |

## At a 10% minimum, with the gate

| minimum | level | net annual return | net Sharpe | t vs 50/50 | t vs no forecasts |
|---|---|---|---|---|---|
| 10% | 1 | 11.86% | 0.75 | 1.96 | -1.94 |
| 10% | 2 | 10.74% | 0.78 | 1.96 | -1.32 |
| 10% | 3 | 8.94% | 0.81 | 1.28 | -0.39 |
| 10% | 4 | 6.60% | 0.81 | -2.10 | -0.05 |
| 10% | 5 | 2.94% | 0.58 | -2.32 | — |

## Notes

- **The October failure does not recur.** The original plain-Markowitz optimiser (`96b0cab`, 3 Oct) sat at a bound on 92% of rebalances, because the raw training-mean premium pinned it. At level 3, with the gate, no final weight is at a bound and the tilted target is capped at 9% of rebalances (18% without the gate). The band can stop a trade short of a bound, so the target's figure is the fairer comparison.
- **Level 1 is at the upper bound and level 5 at a bound by construction.** `w*_L` is at a bound at 100% of rebalances at both. Level 5 stays at 100% in the final weight, because it ignores forecasts and has no band. Level 1 can only tilt down.
- **Risk aversion against the old k.** The implied `k = λ_L/β̂` is 7.7 to 16.6 across levels 1 to 4 at the median, against the fixed `k = 40` it replaced, so tilts are several times larger than under the old rule.
- **Tilt and band scale with 1/π̄.** The premium runs from 0.32% to 1.38% per 20 days, so the same signal moves the weight about four times as far at the low end. `λ_L` at level 3 ranges from 2.5 to 30.7 for the same reason. A strong signal can take level 4 to 78% equity and level 1 to 25%.
- **A premium at or below zero** would make the rule ignore even a perfect forecast (methodology §17). It never happens at a rebalance on this data.
- **Early `β̂` is noisy.** At 252 pairs a 20-day horizon holds about 13 independent observations. The raw slope ran from −0.78 to 0.96 while the gate was shut before 2023. On every pair to date it is 0.62 at t = 2.02 with R² of 0.020.
- **The evidence for the slope has weakened.** On pairs dated from 2023 only, the slope is 0.34 at t = 0.88. The rule never uses a sub-period; it is shown so the full-sample figure isn't over-read.
- **The gate sits close to its threshold.** Its t has stayed between 1.54 and 2.33 since it first opened in May 2023, so it closed in March 2025 and flipped three times in 2026. 2.00 appears in a "shut" row because the gate is strict and the unrounded t is not above 2.
- **The tilt's P&L is before costs and against the same level with the same band**, so it is what the forecasts changed. Costs add little to it. They are small because the band holds most rebalances, though gated turnover is still about 100% a year at levels 1 to 4, against 4% to 6% with no forecasts.
- **The result depends on the model pairing.** With Polynomial (derived) on both sides, fitted earlier on the same data, level 3's t against no forecasts was −2.72, because that bond model's signal is often constant. These figures should not be read as a property of the forecasts in general.
- The rule was not changed in response to any of these results.
