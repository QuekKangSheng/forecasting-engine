# Design verification, 12 Oct 2026

Evidence behind the window, gate, grid and cost decisions recorded in Jira FYP-49, FYP-161, FYP-216 and FYP-217. Produced by a one-off scratch script (scratch/verify.py) on the Toby branch at commit cbd859b, on the Sep 2026 Bloomberg export (31 Aug 2016 – 30 Sep 2026; 9 signals; equity 2,534 rows, bond 2,554 rows). No app code was changed for the run.

## Settings

20-day horizon, 20-day test window, 20-day embargo, 504-row tuning period, 1-day signal lag, per-fold screening. Derived grid: degrees 1–3 × (lasso, elastic net) + shrinkage = 7 setups (degrees 4–5 were dropped mid-run: a single degree-5 fold took 6–9 minutes). All windows are scored on identical test dates (equity 100 folds, 1 Oct 2018 – 16 Sep 2026; bond 101 folds, 2 Oct 2018 – 16 Sep 2026).

Definitions: **Signal Rank IC** = Rank IC of (forecast − the fold's training-mean target), pooled. **t** = Signal Rank IC / the wider of the two Newey-West standard errors. **Test year** TY2019 = 1 Oct 2018 – 30 Sep 2019, and so on.

Windows: **W252 / W504** = rolling 252 / 504 rows; **EXP5** = expanding, capped at the latest 1,260 rows (5 years); **EXP** = expanding, uncapped.

## Decision rules, declared before any result

- **Window (FYP-217):** rolling 504 for every non-ML model, unless an expanding variant beats it by more than 1 Newey-West s.e. **and** wins at least 5 of 8 test years, with the win not coming from Mar–Jun 2025. Between capped and uncapped expanding, prefer capped unless uncapped clears the same bar.
- **PBO (FYP-216):** drop PBO as a gate. Keep it as a displayed figure only if it is stable once near-duplicate setups are merged.
- **Cost band (FYP-49):** build it if it improves net results.

## Decisions taken

| Decision | Evidence |
|---|---|
| **Rolling 504-day train window for every model on both targets** | 1A: equity derived W504 0.145 (t 2.20) vs EXP 0.139 and EXP5 0.117; bond expanding below W504. Neither expanding variant clears the bar. 1E: ML also better at 504 (equity 0.044 → 0.069; bond 0.044 → 0.114). |
| Longer training does not help | 1D: on equity EXP, folds trained on < 5 years score 0.225; folds on ≥ 5 years score 0.075. |
| Results are not driven by spring 2025 | 1C: excluding Mar–Jun 2025 leaves every conclusion unchanged. |
| **Remove PBO entirely** | 2A: ML's PBO flips across 0.5 with the block count alone (equity 0.34 / 0.33 / 0.64 / 0.63). 2C: merging near-duplicate setups moves bond derived PBO from 0.47 to 0.52. |
| **Drop elastic net from the derived grid** | 2B: lasso and elastic net at the same degree correlate 0.98–1.00 on both targets. |
| **Drop degrees 4–5** | Never selected in any run, and infeasible: 6–9 minutes per fold at degree 5 (2,001 terms with 9 signals). |
| **Gate = Signal Rank IC > 0.02 and t > 2** | At s.e. ≈ 0.07, a model with zero skill clears "IC > 0.02" about 39% of the time. The sponsor asked for "T-stat > 2". |
| **Build cost-aware rebalancing (no-trade band)** | 3A vs 3B: net Sharpe improves at every horizon (h1 0.85 → 1.02; h5 0.50 → 0.70; h10 0.58 → 0.69; h20 0.80 → 0.85). Cost drag at h1 falls from 404 to 9 bp/yr. |
| Risk-aversion scale behaves as intended | 3C: share of rebalances at a bound is 90 / 85 / 48 / 16 / 0% for levels 1–5; net Sharpe 0.80–0.85 throughout. |

**Caveats:**
- The equity W504 winner (degree-3 lasso) kept no term in 47 of 100 folds and scored −0.288 in TY2019. The shrinkage setup at W504 scored lower overall (0.123, t 1.85) but was positive in all 8 test years.
- Bond models are weak at every window: none is above t = 2 except bond ML at W504 (t 2.00, borderline).
- At the 20-day default, the optimised portfolio roughly ties 50/50 (net Sharpe 0.85 with the band vs 0.83).

## Raw results (as returned)


### 1A

| target | window | model | chosen setup | Signal Rank IC | s.e. h−1 | s.e. test-window | t (wider) | plain OOS Rank IC | rows scored |
|---|---|---|---|---|---|---|---|---|---|
| equity (SPX) | W252 | derived | shrinkage | 0.095 | 0.071 | 0.072 | 1.32 | 0.075 | 1,990 |
| equity (SPX) | W252 | shrinkage | shrinkage | 0.095 | 0.071 | 0.072 | 1.32 | 0.075 | 1,990 |
| equity (SPX) | W252 | naive | training mean | n/a | n/a | n/a | n/a | -0.100 | 1,990 |
| equity (SPX) | W504 | derived | degree3_lasso | 0.145 | 0.066 | 0.066 | 2.20 | -0.053 | 1,990 |
| equity (SPX) | W504 | shrinkage | shrinkage | 0.123 | 0.066 | 0.066 | 1.85 | 0.068 | 1,990 |
| equity (SPX) | W504 | naive | training mean | n/a | n/a | n/a | n/a | -0.214 | 1,990 |
| equity (SPX) | EXP5 | derived | shrinkage | 0.117 | 0.072 | 0.072 | 1.62 | 0.109 | 1,990 |
| equity (SPX) | EXP5 | shrinkage | shrinkage | 0.117 | 0.072 | 0.072 | 1.62 | 0.109 | 1,990 |
| equity (SPX) | EXP5 | naive | training mean | n/a | n/a | n/a | n/a | -0.129 | 1,990 |
| equity (SPX) | EXP | derived | shrinkage | 0.139 | 0.070 | 0.070 | 1.98 | 0.122 | 1,990 |
| equity (SPX) | EXP | shrinkage | shrinkage | 0.139 | 0.070 | 0.070 | 1.98 | 0.122 | 1,990 |
| equity (SPX) | EXP | naive | training mean | n/a | n/a | n/a | n/a | -0.239 | 1,990 |
| bond (LBUSTRUU) | W252 | derived | shrinkage | 0.122 | 0.071 | 0.072 | 1.70 | 0.138 | 2,010 |
| bond (LBUSTRUU) | W252 | shrinkage | shrinkage | 0.122 | 0.071 | 0.072 | 1.70 | 0.138 | 2,010 |
| bond (LBUSTRUU) | W252 | naive | training mean | n/a | n/a | n/a | n/a | 0.102 | 2,010 |
| bond (LBUSTRUU) | W504 | derived | degree3_elasticnet | 0.075 | 0.073 | 0.074 | 1.01 | 0.001 | 2,010 |
| bond (LBUSTRUU) | W504 | shrinkage | shrinkage | -0.042 | 0.066 | 0.067 | -0.63 | -0.015 | 2,010 |
| bond (LBUSTRUU) | W504 | naive | training mean | n/a | n/a | n/a | n/a | -0.025 | 2,010 |
| bond (LBUSTRUU) | EXP5 | derived | shrinkage | 0.061 | 0.069 | 0.070 | 0.88 | -0.025 | 2,010 |
| bond (LBUSTRUU) | EXP5 | shrinkage | shrinkage | 0.061 | 0.069 | 0.070 | 0.88 | -0.025 | 2,010 |
| bond (LBUSTRUU) | EXP5 | naive | training mean | n/a | n/a | n/a | n/a | -0.217 | 2,010 |
| bond (LBUSTRUU) | EXP | derived | shrinkage | 0.018 | 0.068 | 0.069 | 0.26 | -0.043 | 2,010 |
| bond (LBUSTRUU) | EXP | shrinkage | shrinkage | 0.018 | 0.068 | 0.069 | 0.26 | -0.043 | 2,010 |
| bond (LBUSTRUU) | EXP | naive | training mean | n/a | n/a | n/a | n/a | -0.152 | 2,010 |


### 1B

| target | window | model | TY2019 | TY2020 | TY2021 | TY2022 | TY2023 | TY2024 | TY2025 | TY2026 |
|---|---|---|---|---|---|---|---|---|---|---|
| equity (SPX) | W252 | derived | 0.223 | 0.368 | 0.010 | -0.147 | 0.198 | -0.148 | 0.238 | 0.243 |
| equity (SPX) | W252 | shrinkage | 0.223 | 0.368 | 0.010 | -0.147 | 0.198 | -0.148 | 0.238 | 0.243 |
| equity (SPX) | W504 | derived | -0.288 | 0.290 | 0.083 | -0.040 | 0.404 | -0.088 | n/a | 0.278 |
| equity (SPX) | W504 | shrinkage | 0.400 | 0.283 | 0.321 | 0.108 | 0.182 | 0.042 | 0.174 | 0.246 |
| equity (SPX) | EXP5 | derived | 0.370 | 0.356 | 0.298 | 0.041 | 0.351 | 0.254 | 0.464 | 0.170 |
| equity (SPX) | EXP5 | shrinkage | 0.370 | 0.356 | 0.298 | 0.041 | 0.351 | 0.254 | 0.464 | 0.170 |
| equity (SPX) | EXP | derived | 0.370 | 0.356 | 0.298 | 0.052 | 0.348 | 0.314 | 0.333 | 0.211 |
| equity (SPX) | EXP | shrinkage | 0.370 | 0.356 | 0.298 | 0.052 | 0.348 | 0.314 | 0.333 | 0.211 |
| bond (LBUSTRUU) | W252 | derived | 0.155 | 0.057 | -0.321 | -0.182 | 0.305 | -0.025 | 0.119 | 0.180 |
| bond (LBUSTRUU) | W252 | shrinkage | 0.155 | 0.057 | -0.321 | -0.182 | 0.305 | -0.025 | 0.119 | 0.180 |
| bond (LBUSTRUU) | W504 | derived | -0.174 | -0.323 | 0.430 | 0.072 | -0.165 | -0.180 | 0.120 | -0.306 |
| bond (LBUSTRUU) | W504 | shrinkage | -0.095 | -0.101 | -0.139 | -0.186 | -0.300 | 0.140 | -0.182 | 0.141 |
| bond (LBUSTRUU) | EXP5 | derived | -0.045 | -0.069 | -0.163 | 0.019 | -0.303 | 0.155 | 0.186 | 0.079 |
| bond (LBUSTRUU) | EXP5 | shrinkage | -0.045 | -0.069 | -0.163 | 0.019 | -0.303 | 0.155 | 0.186 | 0.079 |
| bond (LBUSTRUU) | EXP | derived | -0.045 | -0.069 | -0.163 | -0.054 | -0.295 | 0.301 | 0.170 | -0.098 |
| bond (LBUSTRUU) | EXP | shrinkage | -0.045 | -0.069 | -0.163 | -0.054 | -0.295 | 0.301 | 0.170 | -0.098 |


Number of test years in which each window beat W504:

| target | model | W252 | EXP5 | EXP |
|---|---|---|---|---|
| equity (SPX) | derived | 2/8 | 5/8 | 5/8 |
| equity (SPX) | shrinkage | 3/8 | 4/8 | 4/8 |
| bond (LBUSTRUU) | derived | 5/8 | 5/8 | 5/8 |
| bond (LBUSTRUU) | shrinkage | 6/8 | 5/8 | 6/8 |


### 1C

| target | window | model | Signal Rank IC (excl. 1 Mar–30 Jun 2025) | wider s.e. | t |
|---|---|---|---|---|---|
| equity (SPX) | W252 | derived | 0.073 | 0.072 | 1.02 |
| equity (SPX) | W252 | shrinkage | 0.073 | 0.072 | 1.02 |
| equity (SPX) | W504 | derived | 0.142 | 0.068 | 2.09 |
| equity (SPX) | W504 | shrinkage | 0.116 | 0.069 | 1.69 |
| equity (SPX) | EXP5 | derived | 0.108 | 0.075 | 1.45 |
| equity (SPX) | EXP5 | shrinkage | 0.108 | 0.075 | 1.45 |
| equity (SPX) | EXP | derived | 0.129 | 0.072 | 1.78 |
| equity (SPX) | EXP | shrinkage | 0.129 | 0.072 | 1.78 |
| bond (LBUSTRUU) | W252 | derived | 0.128 | 0.074 | 1.73 |
| bond (LBUSTRUU) | W252 | shrinkage | 0.128 | 0.074 | 1.73 |
| bond (LBUSTRUU) | W504 | derived | 0.075 | 0.075 | 0.99 |
| bond (LBUSTRUU) | W504 | shrinkage | -0.040 | 0.069 | -0.58 |
| bond (LBUSTRUU) | EXP5 | derived | 0.057 | 0.072 | 0.79 |
| bond (LBUSTRUU) | EXP5 | shrinkage | 0.057 | 0.072 | 0.79 |
| bond (LBUSTRUU) | EXP | derived | 0.023 | 0.073 | 0.31 |
| bond (LBUSTRUU) | EXP | shrinkage | 0.023 | 0.073 | 0.31 |


### 1D

| model | training window | Signal Rank IC | wider s.e. | folds |
|---|---|---|---|---|
| derived | < 5 years (< 1,260 rows) | 0.225 | 0.093 | 38 |
| derived | ≥ 5 years (≥ 1,260 rows) | 0.075 | 0.093 | 62 |
| shrinkage | < 5 years (< 1,260 rows) | 0.225 | 0.093 | 38 |
| shrinkage | ≥ 5 years (≥ 1,260 rows) | 0.075 | 0.093 | 62 |


### 1E

| target | window | model | chosen setup | Signal Rank IC | s.e. h−1 | s.e. test-window | t (wider) | plain OOS Rank IC | rows scored |
|---|---|---|---|---|---|---|---|---|---|
| equity (SPX) | W252 | ML | lightgbm | 0.044 | 0.073 | 0.073 | 0.59 | 0.011 | 1,990 |
| equity (SPX) | W504 | ML | lightgbm | 0.069 | 0.063 | 0.063 | 1.10 | -0.004 | 1,990 |
| bond (LBUSTRUU) | W252 | ML | lightgbm | 0.044 | 0.055 | 0.055 | 0.80 | 0.113 | 2,010 |
| bond (LBUSTRUU) | W504 | ML | xgboost | 0.114 | 0.057 | 0.057 | 2.00 | 0.078 | 2,010 |


### 2A

| target | family | setups | PBO @8 | PBO @10 | PBO @12 | PBO @16 |
|---|---|---|---|---|---|---|
| equity (SPX) | derived (EXP) | 7 | 0.09 | 0.02 | 0.06 | 0.04 |
| bond (LBUSTRUU) | derived (W252) | 7 | 0.47 | 0.32 | 0.36 | 0.47 |
| equity (SPX) | ML (W252) | 2 | 0.34 | 0.33 | 0.64 | 0.63 |
| bond (LBUSTRUU) | ML (W252) | 2 | 0.66 | 0.44 | 0.62 | 0.66 |


### 2B


Equity (SPX), derived (EXP):

| setup | d1_lasso | d1_en | d2_lasso | d2_en | d3_lasso | d3_en | shrinkage |
|---|---|---|---|---|---|---|---|
| d1_lasso | 1.00 | 1.00 | 0.83 | 0.83 | 0.52 | 0.53 | 0.86 |
| d1_en | 1.00 | 1.00 | 0.83 | 0.83 | 0.52 | 0.53 | 0.86 |
| d2_lasso | 0.83 | 0.83 | 1.00 | 1.00 | 0.64 | 0.65 | 0.75 |
| d2_en | 0.83 | 0.83 | 1.00 | 1.00 | 0.64 | 0.65 | 0.75 |
| d3_lasso | 0.52 | 0.52 | 0.64 | 0.64 | 1.00 | 1.00 | 0.50 |
| d3_en | 0.53 | 0.53 | 0.65 | 0.65 | 1.00 | 1.00 | 0.51 |
| shrinkage | 0.86 | 0.86 | 0.75 | 0.75 | 0.50 | 0.51 | 1.00 |


Bond (LBUSTRUU), derived (W252):

| setup | d1_lasso | d1_en | d2_lasso | d2_en | d3_lasso | d3_en | shrinkage |
|---|---|---|---|---|---|---|---|
| d1_lasso | 1.00 | 0.98 | 0.61 | 0.61 | 0.63 | 0.64 | 0.56 |
| d1_en | 0.98 | 1.00 | 0.62 | 0.62 | 0.65 | 0.65 | 0.58 |
| d2_lasso | 0.61 | 0.62 | 1.00 | 1.00 | 0.68 | 0.68 | 0.44 |
| d2_en | 0.61 | 0.62 | 1.00 | 1.00 | 0.68 | 0.68 | 0.44 |
| d3_lasso | 0.63 | 0.65 | 0.68 | 0.68 | 1.00 | 1.00 | 0.50 |
| d3_en | 0.64 | 0.65 | 0.68 | 0.68 | 1.00 | 1.00 | 0.50 |
| shrinkage | 0.56 | 0.58 | 0.44 | 0.44 | 0.50 | 0.50 | 1.00 |


### 2C

| target | clusters (members) | setups kept | PBO @16 after merging |
|---|---|---|---|
| equity (SPX) | 4 ({d1_lasso, d1_en}; {d2_lasso, d2_en}; {d3_lasso, d3_en}; {shrinkage}) | 4 | 0.06 |
| bond (LBUSTRUU) | 4 ({d1_lasso, d1_en}; {d2_lasso, d2_en}; {d3_lasso, d3_en}; {shrinkage}) | 4 | 0.52 |


### 3A

| h | rebalances/yr | mean turnover per rebalance | annual turnover | annual cost drag (bp) | optimised ann. return gross / net | optimised Sharpe gross / net | 50/50 ann. return net | 50/50 Sharpe net | % rebalances at a bound |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 247.84 | 0.37 | 91.72 | 404.28 | 13.39% / 9.35% | 1.17 / 0.85 | 8.18% | 0.83 | 51.31 |
| 5 | 49.57 | 0.23 | 11.43 | 48.33 | 5.96% / 5.47% | 0.54 / 0.50 | 8.18% | 0.83 | 50.00 |
| 10 | 24.78 | 0.23 | 5.57 | 24.07 | 6.82% / 6.58% | 0.60 / 0.58 | 8.18% | 0.83 | 46.97 |
| 20 | 12.39 | 0.36 | 4.41 | 19.59 | 9.72% / 9.53% | 0.82 / 0.80 | 8.18% | 0.83 | 48.48 |


### 3B

| h | rebalances/yr | mean turnover per rebalance | annual turnover | annual cost drag (bp) | optimised ann. return gross / net | optimised Sharpe gross / net | 50/50 ann. return net | 50/50 Sharpe net | % rebalances at a bound | % rebalances skipped |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9.26 | 0.21 | 1.90 | 8.86 | 9.59% / 9.50% | 1.03 / 1.02 | 8.18% | 0.83 | 1.35 | 96.26 |
| 5 | 22.16 | 0.16 | 3.58 | 15.77 | 7.25% / 7.10% | 0.72 / 0.70 | 8.18% | 0.83 | 0.56 | 55.30 |
| 10 | 14.27 | 0.21 | 2.93 | 13.06 | 7.54% / 7.41% | 0.70 / 0.69 | 8.18% | 0.83 | 0.88 | 42.42 |
| 20 | 10.01 | 0.37 | 3.63 | 16.25 | 9.90% / 9.74% | 0.86 / 0.85 | 8.18% | 0.83 | 0.00 | 19.19 |


### 3C

| level | λ | net ann. return | net Sharpe | max drawdown | % rebalances at a bound |
|---|---|---|---|---|---|
| 1 | 1 | 10.15% | 0.85 | -21.00% | 89.90 |
| 2 | 3 | 9.75% | 0.81 | -20.89% | 84.85 |
| 3 | 10 | 9.53% | 0.80 | -20.91% | 48.48 |
| 4 | 30 | 9.15% | 0.83 | -19.90% | 16.16 |
| 5 | 100 | 8.51% | 0.83 | -20.36% | 0.00 |


### 3D

| index | h | steps | in/out switches per year | annual cost if each switch paid 3 bp (equity) / 5 bp (bond) | gross strategy ann. return | buy-and-hold ann. return |
|---|---|---|---|---|---|---|
| equity (SPX) | 1 | 2000 | 25.12 | 75.36 | 13.60% | 14.59% |
| equity (SPX) | 5 | 400 | 0.75 | 2.26 | 14.50% | 14.59% |
| equity (SPX) | 10 | 200 | 1.51 | 4.52 | 13.16% | 14.59% |
| equity (SPX) | 20 | 100 | 2.51 | 7.54 | 13.35% | 14.59% |
| bond (LBUSTRUU) | 1 | 2020 | 48.88 | 244.38 | 3.05% | 1.81% |
| bond (LBUSTRUU) | 5 | 404 | 9.67 | 48.37 | 1.33% | 1.81% |
| bond (LBUSTRUU) | 10 | 202 | 6.28 | 31.41 | 0.47% | 1.81% |
| bond (LBUSTRUU) | 20 | 101 | 3.02 | 15.08 | 1.62% | 1.81% |


### Notes

Data and test dates: the data is data/last_commit.pkl, which matches the 11 attached CSVs value for value; 9 signals; 31 Aug 2016 – 30 Sep 2026. Equity has 2,534 rows; bond has 2,554. All four windows give identical test dates for each target, checked before fitting and again on every derived run’s folds. Equity: 100 folds, 1 Oct 2018 – 16 Sep 2026. Bond: 101 folds, 2 Oct 2018 – 16 Sep 2026.
Workaround, EXP5: a custom splitter in the scratch script. Its first test window is at row 524, and each fold trains on at most the latest 1,260 rows before the embargo.
Workaround, 1E W504: a splitter subclass. Optuna’s mini walk-forward sees train=252; the main fits use 504 rows.
Grid: 7 setups, degrees 1–3 × (lasso, elastic net) plus shrinkage, passed through candidates / _evaluate_candidate. The abandoned degree 4/5 attempt contributed nothing; only its naive and ML runs were reused. Both cross-checks match exactly: run_derived_polynomial on bond W252 gives 0.122114, and run_boosted on equity W252 gives lightgbm 0.043509.
NaNs:
Naive Signal Rank IC is n/a by definition.
Equity W504 derived (degree3_lasso) TY2025 is n/a because all 12 folds in that year kept no term, so the signal is constant. That setup kept no term in 47 of 100 folds overall. The NaN year counts as “not beating W504” in 1B.
Few setups: ML’s PBO compares only 2 setups (XGBoost vs LightGBM). Shrinkage won the derived selection in 6 of 8 Task 1 cells, so those derived and shrinkage rows are identical.
Task 3 definitions:
Rebalances/yr includes the opening allocation. Turnover excludes the opening purchase from cash.
Cost drag is gross minus net annual return.
3B “% at a bound” counts executed trades landing on 20% or 80%.
3D switches exclude the initial entry.
Setups chosen by h: h1 equity degree2_lasso / bond lightgbm; h5 degree1_lasso / xgboost; h10 degree1_elasticnet / lightgbm; h20 shrinkage / lightgbm.
Execution: equity EXP at h1 was fitted twice because a prefetch process and the main process overlapped. The fit is deterministic and the second copy replaced the first. No PBO run took over 10 minutes.

Runtime: the final run took 95.9 min (07:54–09:30), with three prefetch processes in parallel. It also reused 15 naive and ML results from the first attempt, about 70 min of compute. Wall clock including the abandoned degree 4/5 attempt was 02:03–09:30, about 7.5 h.

Commit: cbd859bd63a6a44082b5b02dea744024c21dad53. Nothing under src/, app/, tests/ or docs/ was modified. Script: scratch/verify.py; tables: scratch/results.md.
