# Bond grids chosen per window, 17 Nov 2026

Asked for: for the derived polynomial and machine learning, "make sure we are not making
selection bias", bonds only, equity unchanged for now. This file records the bias, what
the app now does for the bond target, and the check that the app reproduces the
research's figures (`s15-bond-grids-honest`, `RULE.md` part B and `part_b.txt`).

Produced by `docs/evidence/2026-11-17-bond-honest-selection.py` on this branch (main at
`1a3f937` plus the change), with `FORECASTING_WORKERS=2`, on the 11 exports from 31 Aug
2006 to 11–12 Nov 2026 (their SHA-256 are in the output below), at the Models page's
defaults: train 504 rows, test 20, embargo 20, the first 504 rows a tuning period. It
runs `run_derived_polynomial(..., role=BOND)` and `run_boosted(..., role=BOND)`, the
functions the page's runners call, once per horizon, and scores the whole-period pick
on the same candidates' fits, with nothing refitted. Periods as in the research:
"fresh" is the test windows whose every label ends before 1 Aug 2016 (97, Oct 2008 –
Jul 2016), "later" the windows starting 31 Aug 2016 – 30 Sep 2026 (127).

## The bias

`run_derived_polynomial` and `run_boosted` reported the candidate with the best Signal
Rank IC over the whole scored period (`harness.select_best_candidate`), and then that
candidate's score over the same period. The pick was made on the windows it was scored
on: of four lasso and shrinkage setups (or XGBoost and LightGBM) it takes the luckiest,
and the reported score keeps the luck.

## What the app does now, for bond

For each test window k, the candidate used is the one with the best pooled Signal Rank
IC, the measure `select_best_candidate` uses, over the earlier windows whose every label
ended before window k's first date (`harness.select_online`). Until 12 such windows exist
(`MIN_SELECTION_HISTORY`) it is a declared default (`ONLINE_SELECTION_DEFAULT`: shrinkage
for the derived polynomial, XGBoost for machine learning). Ties go to the first candidate
in the grid's order (degree 1, 2, 3 lasso, shrinkage; XGBoost, LightGBM). The row's score
is `summarize` over each window's chosen fold (`harness.summarize_online`); the equation,
SHAP, latest tune and the fit a scenario refits are the latest window's candidate's; the
page says how many windows each candidate was used in. This is `select15.py`'s `online()`
exactly. The grids do not change. Equity keeps the whole-period pick.

## Derived polynomial: the app reproduces the research

| h | Whole-period pick | Per window, earlier windows only | Windows used per window |
|---|---|---|---|
| 1 | +0.029, t 1.84 (shrinkage) | +0.029, t 1.84 | shrinkage 228 |
| 5 | +0.030, t 1.00 (degree-1 lasso) | +0.032, t 1.19 | shrinkage 151, degree-1 42, degree-3 23, degree-2 12 |
| 10 | +0.021, t 0.48 (degree-3 lasso) | −0.022, t −0.53 | degree-1 133, degree-2 61, shrinkage 25, degree-3 9 |
| 20 | +0.030, t 0.49 (degree-2 lasso) | −0.054, t −1.02 | degree-1 149, degree-2 39, degree-3 27, shrinkage 13 |

Every figure, whole period, fresh and later, both ways, and the three most used
candidates at each horizon, equal `part_b.txt`'s G0 rows to 3 decimals (the
comparison table is in the output). By decade:

| h | Selection | Fresh | Later |
|---|---|---|---|
| 1 | either | +0.015, t 0.63 | +0.034, t 1.60 |
| 5 | whole-period pick | −0.009, t −0.24 | +0.043, t 1.04 |
| 5 | per window | −0.008, t −0.20 | +0.038, t 1.07 |
| 10 | whole-period pick | −0.049, t −0.80 | +0.056, t 0.95 |
| 10 | per window | −0.113, t −1.98 | +0.053, t 0.98 |
| 20 | whole-period pick | −0.088, t −1.03 | +0.092, t 1.14 |
| 20 | per window | −0.111, t −1.44 | −0.007, t −0.09 |

At h = 1 the shrinkage fit is best on every window's history, so nothing changes. At
h = 5 the honest score is a little higher: the whole-period pick, degree-1 lasso, is not
the candidate that does best window by window. At h = 10 and 20 the whole-period pick's
positive score was the pick's luck; chosen honestly, no candidate forecasts the bond
index there. None passes the gate either way.

## Machine learning: the app's own figures

No research figure exists for it yet. These are the app's, both ways, from the same
tuned fits:

| h | Whole-period pick | Per window, earlier windows only | Windows used per window |
|---|---|---|---|
| 1 | +0.019, t 1.30 (XGBoost) | +0.019, t 1.28 | XGBoost 221, LightGBM 7 |
| 5 | +0.013, t 0.54 (XGBoost) | +0.012, t 0.48 | XGBoost 208, LightGBM 20 |
| 10 | −0.017, t −0.46 (XGBoost) | −0.022, t −0.59 | XGBoost 219, LightGBM 9 |
| 20 | +0.045, t 0.88 (XGBoost) | +0.032, t 0.64 | XGBoost 158, LightGBM 70 |

The whole-period figures equal methodology §20's machine-learning bond row. XGBoost was
the whole-period pick at every horizon, so the bias is smaller than the derived grid's,
two candidates against four; per window the score falls at three horizons out of four
(by 0.013 at h = 20). Fresh and later are in the output. None passes the gate either
way.

**These are Windows figures.** On the same code, data and `uv.lock`, bond machine
learning at h = 20 gives LightGBM +0.0267 (t 0.55) on Windows and on a Linux node, but
XGBoost +0.0449 (t 0.88) on Windows against +0.0033 (t 0.07) on Linux: with one thread
and Optuna's fixed seed, a tiny cross-platform difference in XGBoost's fits sends the
tuning down a different path, and on a near-noise target the score moves by 0.04. The
hosted app runs on Linux and will not reproduce this table for bond (methodology §9,
machine learning, reproducibility). Not fixed here.

## Not changed

Equity still uses the whole-period pick (`select_best_candidate`), so its derived
polynomial's and machine learning's scores are overstated by the same kind of luck,
by an amount not measured here; it is left as it is on Joash's instruction, and its
results are exactly what they were. The gate, its thresholds, the horizons and the
grids are unchanged.

## Rerunning

```
uv run --project <repo> python -I docs/evidence/2026-11-17-bond-honest-selection.py <folder> \
    [--models derived,ml] [--horizons 1,5,10,20]
```

On this laptop, with two worker processes, the derived polynomial took 2–3 minutes a
horizon and machine learning about 2.

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

Merged: 5,271 rows, 31 Aug 2006 to 12 Nov 2026, content hash (`ingest.upload.frame_sha256`) `7a4b117c59d3ccd2d657b1a20e07e726492b4b19ff3b3ce62052751f150ce92c`. Empty columns dropped: none. Bond target `LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS`; 9 signals.

Online selection: `MIN_SELECTION_HISTORY` = 12; defaults: derived `shrinkage`, machine learning `xgboost`.

## Derived polynomial, bond

| h | selection | whole | fresh | later | windows used | runtime |
|---|---|---|---|---|---|---|
| 1 | per window, earlier windows only | +0.029, t 1.84 | +0.015, t 0.63 | +0.034, t 1.60 | shrinkage 228 | 111 s |
| 1 | whole-period pick (shrinkage) | +0.029, t 1.84 | +0.015, t 0.63 | +0.034, t 1.60 |  |  |
| 5 | per window, earlier windows only | +0.032, t 1.19 | -0.008, t -0.20 | +0.038, t 1.07 | degree1_lasso 42, degree2_lasso 12, degree3_lasso 23, shrinkage 151 | 128 s |
| 5 | whole-period pick (degree1_lasso) | +0.030, t 1.00 | -0.009, t -0.24 | +0.043, t 1.04 |  |  |
| 10 | per window, earlier windows only | -0.022, t -0.53 | -0.113, t -1.98 | +0.053, t 0.98 | degree1_lasso 133, degree2_lasso 61, degree3_lasso 9, shrinkage 25 | 148 s |
| 10 | whole-period pick (degree3_lasso) | +0.021, t 0.48 | -0.049, t -0.80 | +0.056, t 0.95 |  |  |
| 20 | per window, earlier windows only | -0.054, t -1.02 | -0.111, t -1.44 | -0.007, t -0.09 | degree1_lasso 149, degree2_lasso 39, degree3_lasso 27, shrinkage 13 | 181 s |
| 20 | whole-period pick (degree2_lasso) | +0.030, t 0.49 | -0.088, t -1.03 | +0.092, t 1.14 |  |  |

**Against the research** (`part_b.txt`, G0), to 3 decimals:

| h | selection | period | research | app |  |
|---|---|---|---|---|---|
| 1 | online | whole | +0.029, t 1.84 | +0.029, t 1.84 | same |
| 1 | online | fresh | +0.015, t 0.63 | +0.015, t 0.63 | same |
| 1 | online | later | +0.034, t 1.60 | +0.034, t 1.60 | same |
| 1 | biased | whole | +0.029, t 1.84 | +0.029, t 1.84 | same |
| 1 | biased | fresh | +0.015, t 0.63 | +0.015, t 0.63 | same |
| 1 | biased | later | +0.034, t 1.60 | +0.034, t 1.60 | same |
| 1 | windows used (top 3) |  | shrinkage 228 | shrinkage 228 | same |
| 5 | online | whole | +0.032, t 1.19 | +0.032, t 1.19 | same |
| 5 | online | fresh | -0.008, t -0.20 | -0.008, t -0.20 | same |
| 5 | online | later | +0.038, t 1.07 | +0.038, t 1.07 | same |
| 5 | biased | whole | +0.030, t 1.00 | +0.030, t 1.00 | same |
| 5 | biased | fresh | -0.009, t -0.24 | -0.009, t -0.24 | same |
| 5 | biased | later | +0.043, t 1.04 | +0.043, t 1.04 | same |
| 5 | windows used (top 3) |  | shrinkage 151, degree1_lasso 42, degree3_lasso 23 | shrinkage 151, degree1_lasso 42, degree3_lasso 23 | same |
| 10 | online | whole | -0.022, t -0.53 | -0.022, t -0.53 | same |
| 10 | online | fresh | -0.113, t -1.98 | -0.113, t -1.98 | same |
| 10 | online | later | +0.053, t 0.98 | +0.053, t 0.98 | same |
| 10 | biased | whole | +0.021, t 0.48 | +0.021, t 0.48 | same |
| 10 | biased | fresh | -0.049, t -0.80 | -0.049, t -0.80 | same |
| 10 | biased | later | +0.056, t 0.95 | +0.056, t 0.95 | same |
| 10 | windows used (top 3) |  | degree1_lasso 133, degree2_lasso 61, shrinkage 25 | degree1_lasso 133, degree2_lasso 61, shrinkage 25 | same |
| 20 | online | whole | -0.054, t -1.02 | -0.054, t -1.02 | same |
| 20 | online | fresh | -0.111, t -1.44 | -0.111, t -1.44 | same |
| 20 | online | later | -0.007, t -0.09 | -0.007, t -0.09 | same |
| 20 | biased | whole | +0.030, t 0.49 | +0.030, t 0.49 | same |
| 20 | biased | fresh | -0.088, t -1.03 | -0.088, t -1.03 | same |
| 20 | biased | later | +0.092, t 1.14 | +0.092, t 1.14 | same |
| 20 | windows used (top 3) |  | degree1_lasso 149, degree2_lasso 39, degree3_lasso 27 | degree1_lasso 149, degree2_lasso 39, degree3_lasso 27 | same |

Largest Signal Rank IC difference at 3 decimals: 0.000.

## Machine learning, bond

| h | selection | whole | fresh | later | windows used | runtime |
|---|---|---|---|---|---|---|
| 1 | per window, earlier windows only | +0.019, t 1.28 | +0.045, t 1.90 | +0.003, t 0.14 | xgboost 221, lightgbm 7 | 141 s |
| 1 | whole-period pick (xgboost) | +0.019, t 1.30 | +0.045, t 1.90 | +0.001, t 0.07 |  |  |
| 5 | per window, earlier windows only | +0.012, t 0.48 | +0.015, t 0.41 | +0.004, t 0.12 | xgboost 208, lightgbm 20 | 115 s |
| 5 | whole-period pick (xgboost) | +0.013, t 0.54 | +0.017, t 0.48 | +0.004, t 0.12 |  |  |
| 10 | per window, earlier windows only | -0.022, t -0.59 | -0.118, t -1.90 | +0.041, t 0.89 | xgboost 219, lightgbm 9 | 117 s |
| 10 | whole-period pick (xgboost) | -0.017, t -0.46 | -0.107, t -1.73 | +0.041, t 0.89 |  |  |
| 20 | per window, earlier windows only | +0.032, t 0.64 | -0.150, t -1.98 | +0.147, t 2.50 | xgboost 158, lightgbm 70 | 108 s |
| 20 | whole-period pick (xgboost) | +0.045, t 0.88 | -0.136, t -1.74 | +0.149, t 2.48 |  |  |
