# Word documents: passages out of date after Phases 1–5 (Nov 2026)

Phase 5, task 4. These are the passages in the three Word documents that no longer
match the code or the results, with suggested wording. The Word files themselves
are not edited.

The documents checked are Finlytics_System_Design_Methodology_v3.docx (UAT 3, 1 Nov
2026), Finlytics_Validation_Metrics_v5.docx and Finlytics_CTQ_v5.docx (2 Nov
2026).

The changes behind these passages are:

1. **Signal Rank IC baseline:** it is now taken against the model's own level.
2. **t-statistic standard error:** it is now Hansen-Hodrick over h−1 lags, never
   below Newey-West, for both the model gate and the trust gate.
3. **Bond research model:** a new model.
4. **Data from 31 Aug 2006.**
5. **Long-run premium:** π̄ is now blended with a published prior.
6. **No-trade band:** removed.
7. **Optimiser audit fixes:**
   - directional P&L follows the signal;
   - returns are dated by the end of their holding period;
   - the "signal" and "forecast" labels are separated.

The current figures are in `docs/methodology.md` §20 and
`docs/evidence/2026-11-16-full-history.md`.

## Finlytics_System_Design_Methodology_v3.docx

| Section | Out of date | Suggested wording |
|---|---|---|
| Title block | "UAT 3 · 1 Nov 2026 · commit e694cf7 (uat branch)"; "It reflects the code on the uat branch at commit e694cf7." | The new date and the commit that merges Phase 5. |
| 1. Pipeline at a glance, step "Turn the active models' signals into an allocation" | "…within bounds, through a no-trade band, rebalanced every h days." | "…within bounds, rebalanced every h days, each rebalance trading straight to its target." (6) |
| 1, step "Backtest the allocation" | "replay each model's forecast direction against buy-and-hold" | "replay the direction of each model's signal against buy-and-hold" (7) |
| 2. Data inputs, last paragraph | "The export used for every result in this document runs from 31 Aug 2016 to 30 Sep 2026: 11 files, nine signals…, 2,534 S&P 500 rows and 2,554 bond rows." | "The export used for every current result runs from 31 Aug 2006 to 11–12 Nov 2026: 11 files, nine signals once the target securities are excluded, 5,081 S&P 500 rows and 5,089 bond rows. Results on the earlier export (31 Aug 2016 – 30 Sep 2026) are kept in the appendices as the evidence for the decisions taken then." (4) |
| 5. Walk-forward validation, Training window row | "The research model (Section 7) alone trains on every earlier row" | "The two research models (Section 7) alone train on every earlier row" (3) |
| 5, worked example | "…starting 1 Oct 2018… 100 folds… to 16 Sep 2026, and 1,990 scored days (bond: 101 folds from 2 Oct 2018, 2,010 days)." | "…starting 1 Oct 2008, the weeks after the Lehman collapse, kept as a stress test… 227 folds whose test windows run back to back to 19 Oct 2026, and 4,540 scored days (bond: 228 folds from 3 Oct 2008)." (4) |
| 6. Signal screening per fold | "The FF5 benchmark, a user-supplied polynomial and the research model are not screened" | "…and the two research models are not screened" (3) |
| 7. Models, table | One "Research model (fixed inputs)" row, equity only | Rename it "Polynomial – fixed economic inputs (research, equity)". Add a row: "Polynomial – fixed carry inputs (research, bond) · Research bond model · Degree-1 lasso on seven fixed levels (10-year yield, breakeven, 2s10s, HY OAS, IG OAS, VIX, FX volatility), each controlled for its innovation over the label window with a bias-corrected AR(1) (Amihud & Hurvich 2004), expanding window · None". (3) |
| 7, "Research model (fixed inputs)" paragraph | Describes only the equity model; "so the optimiser can use either it or the derived polynomial as the equity model" | Add a paragraph for the bond research model, as in methodology §9.2. Add: "Both research models' inputs were chosen on data from Aug 2016, so each is also scored on the test windows ending before 1 Aug 2016 alone, the 'pre-2016 hold-out'. There both score about zero (equity −0.032, t −0.30; bond −0.004, t −0.05, at 20 days)." (3, 4) |
| 8. Scoring, Signal Rank IC row | "Spearman correlation of (forecast − that fold's training-mean return) with the realised return" | "Spearman correlation of (forecast − the model's own average in-sample forecast, its level) with the realised return, pooled" (1) |
| 8, t row and "s.e." row | "divided by the larger of its two Newey-West standard errors"; "Newey-West standard errors of the Rank IC: one for overlapping h-day labels…" | "divided by the larger of its two standard errors"; "one for overlapping h-day labels, Hansen-Hodrick over h−1 lags (every lag counted in full, never below Newey-West); one Newey-West over test-window lags, for errors shared within a fold's single fit" (2) |
| 8, "Why the Signal Rank IC" | "Taking away the training mean, which is exactly what the naive baseline forecasts, scores only what a model adds to it." | "Taking away each model's own level (its average in-sample forecast) scores only what its signals add. Taking away the training mean instead counted any gap between the two as skill: a forecast of zero every day passed both gates (+0.21, t 3.0)." (1) |
| 8, "Why a t-statistic" | "the eight scored years hold only about 100 independent 20-day periods… standard error is about 0.07" | "the eighteen scored years hold only about 225 independent 20-day periods". Also add: "Newey-West's declining weights understate the overlap; counted in full, a no-skill model passes t > 2 in 2.9% of samples rather than 4.6%." (2, 4) |
| 10. Portfolio construction, "Signals, not forecasts" | "Each active model's forecast less its fold's training-mean return" | "Each active model's forecast less its own level (its average in-sample forecast for that fold)" (1) |
| 10, "Long-run premium" | "π̄, the mean daily equity-minus-bond return over every day to date…" | "π̄_t = (N0·p + Σ(r_eq − r_bond)) / (N0 + t) × h: every day's equity-minus-bond return to date, blended with a published premium p = 6.6% a year (Goetzmann & Ibbotson 2005, US large stocks over long-term government bonds, 1926–2004) that counts as N0 = 30 years of data. Without it, an average from Aug 2006 would rest on two years ending in the 2008 crash. Limitations: the published bond leg probably understates our premium by about 0.4 points a year; the prior still carries about 60% of the weight by 2026; the Data page warns of any upload whose first rebalance precedes May 2005." (5) |
| 10, "Forecast trust" | "whenever the slope's Newey-West t-statistic (h − 1 lags) is not above 2" | "…the slope's t-statistic, Hansen-Hodrick over h − 1 lags and never below Newey-West, is not above 2" (2) |
| 10, "Weights" | "If π̄ ≤ 0 there is no positive λ_L, so the tilt and the band are both zero" | "If π̄ ≤ 0 there is no positive λ_L, so the tilt is zero; with the prior this has never happened on our data" (5, 6) |
| 10, "No-trade band" paragraph | The whole paragraph | Replace with: "Trading. Every rebalance trades straight to its target and pays 3 bp (equity) and 5 bp (bond) on what it trades. The no-trade band adopted on 12 Oct was removed on 16 Nov 2026. Its width was expressed in h-day terms, so at h = 1 it was about four times the tilt range and froze the portfolio, and at h = 5 it let a riskier level hold less equity than a safer one at 22 rebalances. Without it, costs are at most 0.06% a year (docs/evidence/2026-11-16-optimiser-fixes.md)." (6) |
| 10, "On the Sep 2026 export (Appendix A.3)…" | Premium 0.32%–1.38% per 20 days; "The trust gate was open at 26 of 99 rebalances" | "On the data from 2006, π̄ stays between 4.19% and 7.86% a year (lowest on 9 Mar 2009). The trust gate is open at 3 of 226 rebalances at h = 20 and 137 of 904 at h = 5. With machine learning for bond, as at UAT 3, it opens at 92 of 226: the bond model, not the longer history, explains most of the difference from UAT 3." Add the four points of methodology §17 "Reading the weights": forecasts set the direction and the bounds set the size; strong signals put every level on the same bound; level 5 is the 20% floor by design; the latest weights trail the data by up to about 7 weeks. (4, 5, 7) |
| 10, "What the page shows" | "the drifted weight, the band and whether it held, and the share of rebalances the band held" | "the weight that gives before and after the bounds, and why the signals were not used where the gate is shut" (6) |
| 11. Portfolio evaluation, Backtest | "the same schedule with β̂ = 0 and the same no-trade band" | "the same schedule with β̂ = 0" (6) |
| 11, Directional P&L | "fully invested when the forecast is positive" | "fully invested when the model's signal (its forecast less its own average forecast) is positive… each h-day return plotted at the end of its holding period" (7) |
| 11, Deflated Sharpe ratio | "Any equity–bond mix over 2018–2026 clears that" | "Any equity–bond mix over 2008–2026 clears that" (4) |
| 13. Rigour checklist, "Overstated certainty" | "Two Newey-West standard errors; the gate uses the wider" | "Hansen-Hodrick over h−1 lags (never below Newey-West) and Newey-West over test-window lags; the gate uses the wider" (2) |
| 13, "Forecast level artefacts" | "Gate on the Signal Rank IC" | "Gate on the Signal Rank IC, against each model's own level" (1) |
| 13, "Ignoring costs" | "Net metrics at sponsor costs; no-trade band" | "Net metrics at sponsor costs on every trade; cost drag reported per level" (6) |
| 13, new row | — | "Premium collapse in a crash · A short history makes the long-run premium zero or negative, switching the forecasts off or exploding the tilts · Published prior worth 30 years · Section 10" (5) |
| 13, new row | — | "Re-checked gate · Checking the trust gate at every rebalance lets a no-skill model through at some point more often than 2.3% · Disclosed with a simulation: 11% / 18% / 35% at h = 1 / 5 / 20 · Section 17" (7) |
| 14. Fixed parameters, "Research model" row | Equity inputs only | Add "Bond research model · Inputs; fit; window · seven fixed levels; lasso on z with innovations controlled (κᶜ bias-corrected AR(1), ρ capped at 0.9999); expanding · Joash's Nov 2026 research; Amihud & Hurvich 2004". (3) |
| 14, "Trust history; trust gate" | "t > 2 (h − 1 lags)" | "t > 2, Hansen-Hodrick over h − 1 lags, never below Newey-West" (2) |
| 14, "No-trade band" row | "c·(w*_L − w_mv)/π̄, c = 8 bp" | Delete the row. Add "Long-run premium prior · 6.6% a year, worth 30 years · Goetzmann & Ibbotson 2005; n = (σ/s)² with σ ≈ 17–18%, s = 3 points" (5, 6) |
| 15. How to verify, "Automated checks" | "1,463 unit, integration and page tests" | The count on the merged branch (1,776 at the Phase 5 branch: 1,776 passed, 1 skipped). Add the evidence scripts `docs/evidence/2026-11-15-history-2006.py`, `2026-11-16-optimiser-fixes.py`, `2026-11-16-trust-gate-rechecks.py` and `2026-11-16-full-history.py`. |
| 16. Current results | The whole section: Sep 2026 export at e694cf7; research model 0.259 / 3.58 "Met"; derived equity 0.156 / 2.31 "Met"; bond ML 0.070 / 1.22; the portfolio table with ML bond | Replace with methodology §20's model table (all four horizons) and its portfolio table: the Sprint 4 pair (derived equity, ML bond) as the main result, with the derived polynomial for both beside it, and a note that the main pair was chosen after both had been run. At 20 days no model passes for either index: derived equity +0.103, t 1.98. Equity passes at 1, 5 and 10 days, and no bond model passes at any horizon. Replace the "Reading" paragraph to match: "There is an equity signal at 1–10 days; no bond model passes; the forecasts add no value in the portfolio at any horizon, for either model pair, and with the Sprint 4 pair they cost significantly at 5 days (t −3.10 and −2.84 at levels 1–2)." (1, 2, 4) |
| 17. Known limitations, "No significant forecast value in the portfolio" | "Forecasts cost 0.92, 0.60 and 0.16 points… on 26 tilts in 3 runs" | "At no horizon do the forecasts add value, for either model pair: the gate never opens at h = 1 or 10. With the Sprint 4 pair, t against no forecasts is −3.10 to −1.03 at h = 5 (significantly negative at levels 1–2) and −1.44 to −0.15 at h = 20. With the gate ignored, no level does better than without forecasts." (4) |
| 17, "No bond model passes the gate" | "Bond ML t 1.22, derived polynomial 0.71" | "On the data from 2006 no bond model passes at any horizon; the best is the derived polynomial at h = 1, t 1.84. The bond research model scores −0.004 (t −0.05) on its pre-2016 hold-out." (3, 4) |
| 17, "Trust gate chosen after seeing results" | "Its t has stayed between 1.54 and 2.33 since May 2023…" | Keep the disclosure; replace the figures with the full-history ones. Add a row: "Gate re-checked every rebalance · with no skill it opens at some point in 11%, 18% and 35% of simulations at h = 1, 5 and 20, and at h = 20 too often at a single check (5–8%) · Disclosed; no threshold changed" (2, 7) |
| 17, "Tilt scales with 1/π̄" | "π̄ ≤ 0 would switch the forecasts and band off… never occurs on this data" | "π̄ is anchored to a published prior and stays between 4.2% and 7.9% a year; the prior's limitations are in Section 10" (5, 6) |
| 17, "Research model's score is optimistic" | "only data after 30 Sep 2026 is a clean test" | "Both research models are also scored on their pre-2016 hold-out, where each scores about zero" (3, 4) |
| 17, new row | — | "Forecasts add nothing after costs · Recorded as a known limitation (Phase 4 task 5)" |
| 17, "Sponsor notification" | Lists the 1 Nov changes | Add: "Signal Rank IC baseline; t standard error; bond research model; data from 2006; premium prior; band removed; audit fixes. Notified in the Nov 2026 notice (docs/sponsor-notice-2026-11.md), with the question whether Alpha Norm prefers its own long-run premium." |
| Appendix A.3, Appendix C | Sep 2026 figures under the earlier code (band, plain premium, training-mean baseline, Newey-West t) | Keep as dated history, with a line under each heading: "Superseded: figures on the Sep 2026 export under the code of 1 Nov 2026; current results are in Section 16." Add an Appendix D summarising `docs/evidence/2026-11-16-full-history.md`, including the gate breakdown. |

## Finlytics_Validation_Metrics_v5.docx

| Section | Out of date | Suggested wording |
|---|---|---|
| Version history | Ends at v5 | Add v6: Signal Rank IC against each model's own level; Hansen-Hodrick standard errors; bond research model and the pre-2016 hold-out; data from 31 Aug 2006; long-run premium prior; no-trade band removed; directional P&L on the signal. Evidence: System Design Specification, Appendix D. |
| 3. Gates, targets and diagnostics, "Forecast trust gate" | "has a Newey-West t-statistic above 2" | "has a t-statistic above 2 (Hansen-Hodrick over h − 1 lags, never below Newey-West). Because it is re-checked at every rebalance, a no-skill model opens it at some point more often than 2.3% (11–35% in simulation)." (2, 7) |
| 4. Evaluation setup, Data period | "currently 31 Aug 2016 to 30 Sep 2026 (2,534 S&P 500 and 2,554 bond trading days)… reported results cover 1 Oct 2018 onwards" | "currently 31 Aug 2006 to 11–12 Nov 2026 (5,081 S&P 500 and 5,089 bond trading days)… reported results cover 1 Oct 2008 onwards" (4) |
| 4, Validation scheme | "(100 folds for equity, 101 for bond)… the research-derived equity model alone trains on every earlier day" | "(227 folds for equity, 228 for bond)… the two research models alone train on every earlier day" (3, 4) |
| 4, Significance | "two Newey-West standard errors, one with h − 1 lags…" | "two standard errors: Hansen-Hodrick over h − 1 lags (never below Newey-West), for overlapping labels; Newey-West over test-window lags" (2) |
| 4, Reference baselines | "forecast trust set to zero, same no-trade band" | "forecast trust set to zero" (6) |
| 4, Allocation | "rebalanced every h days through a cost-aware no-trade band… risk aversion is implied from the long-run equity-minus-bond premium… Newey-West t-statistic (h − 1 lags)" | "rebalanced every h days, each rebalance trading straight to its target… implied from a long-run premium that blends the data with a published 6.6% a year worth 30 years… t-statistic (Hansen-Hodrick, h − 1 lags)" (2, 5, 6) |
| 5. Tier 1, Signal Rank IC | "(forecast − that fold's training-mean return)… what a model adds to the naive forecast"; "two Newey-West standard errors (h−1 lags; test-window lags)" | "(forecast − the model's own average in-sample forecast)… what a model's signals add beyond its own average forecast"; "two standard errors (Hansen-Hodrick over h−1 lags, never below Newey-West; Newey-West over test-window lags)" (1, 2) |
| 5. Tier 1, Directional P&L | "100% invested on days the forecast is positive" | "100% invested on steps where the model's signal (forecast less its own average forecast) is positive; each h-day return plotted at the end of its holding period" (7) |
| 5. Tier 2, t-statistic | "How many Newey-West standard errors…"; "standard error is about 0.07" | "How many standard errors (the larger of Hansen-Hodrick over h−1 lags and Newey-West over test-window lags)…"; "about 0.07 on the 2016–2026 export" (2) |
| 5. Tier 3, Forecast trust | "Newey-West t-statistic (h − 1 lags) is above 2" | "t-statistic (Hansen-Hodrick, h − 1 lags) is above 2. Re-checked at every rebalance; the chance a no-skill model opens it at some point is disclosed (System Design Specification, Section 17)." (2, 7) |
| 5. Tier 3, Turnover | "used to compute net-of-cost performance and to sanity-check allocation stability" | Add: "every rebalance trades to its target; cost drag is at most 0.06% a year on the data from 2006" (6) |
| Appendix A, Signal Rank IC | "what the model adds to the naive forecast (its forecast minus the average return of its training window)" | "what the model's signals add beyond its own average forecast (its forecast minus that average)" (1) |
| Appendix A, t-statistic | "with only about 100 independent 20-day periods in the data" | "with only about 225 independent 20-day periods in the data from 2006" (4) |
| Appendix A, Directional P&L | "held the index only on days the model said it would rise" | "held the index only when the model's signal pointed up" (7) |
| Appendix A, Forecast trust | "(t above 2)" | Add: "Checked again at every rebalance, so an unskilled model can pass it at some point by chance; the documents say how often." (7) |

## Finlytics_CTQ_v5.docx

| Section | Out of date | Suggested wording |
|---|---|---|
| Document Control, Version History | Ends at v5 | Add v6. Module 2: Signal Rank IC against each model's own level; bond research model, with each research model's pre-2016 hold-out shown; directional P&L follows the signal. Module 3: no-trade band removed (every rebalance trades to its target, costs charged on what it trades); long-run premium blended with a published prior; Data page warns of an upload whose first rebalance precedes May 2005. Evidence: System Design Specification, Appendix D. |
| Module 2, "All model families…" | "(FF5 for the equity target, the user-supplied and derived polynomial, the research-derived equity model, the ML comparators)" | "(FF5 for the equity target, the user-supplied and derived polynomial, the research equity and bond models, the ML comparators)" (3) |
| Module 2, directional P&L | "100% invested when the forecast is positive and in cash otherwise" | "100% invested when the model's signal (forecast less its own average forecast) is positive and in cash otherwise… each return shown at the end of its holding period" (7) |
| Module 2, promotion gate | "(Signal Rank IC and its Newey-West t-statistic)" | "(Signal Rank IC and its t-statistic)", the standard error being defined in the Validation Metrics doc (2) |
| Module 2, research model | "The research-derived equity model is shown as its own row, labelled as optimistic… and can be selected as the optimiser's equity model." | "Each research model (equity and bond) is shown as its own row, labelled as optimistic because its inputs were chosen on data overlapping the test period, with its score on the pre-2016 hold-out shown beneath, and can be selected as the optimiser's model for its index." (3, 4) |
| Module 3, optimisation | "the risk aversion at each level is implied from the long-run equity premium… held through a cost-aware no-trade band" | "…implied from a long-run equity premium that blends the data with a published long-run estimate… Every rebalance trades straight to its target, with costs charged on what it trades." (5, 6) |
| Module 3, spread t-statistics | "(same weight target and no-trade band, so the forecasts are the only difference)" | "(same weight target, so the forecasts are the only difference)" (6) |
