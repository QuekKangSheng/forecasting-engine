# Validation review — 18 Sep 2026

> **Status, 1 Oct 2026.** Several findings below have since been acted on; read
> this as the record of what was measured on 18 Sep, and `docs/methodology.md`
> for how the pipeline works now.
>
> - *Open item 1 (gate inside the noise):* metrics are now pooled with two
>   Newey-West standard errors shown beside the OOS Rank IC, and the table marks
>   whether it clears two of them ("Beyond 2 s.e."). The gate itself stays at
>   pooled OOS Rank IC > 0.02, as confirmed with the sponsor, so a no-skill model
>   still meets it about a third of the time — now visibly.
> - *Pooling adds a new way to score:* a forecast whose level only shifts between
>   folds can earn pooled Rank IC without ranking any day; on the live S&P data a
>   forecast using no signal scored +0.10. "Rank IC within folds" is shown beside
>   the gate so this is visible. See methodology §10.
> - *Open item 2 (screening):* unchanged by decision — the sponsor confirmed
>   strict |Rank IC| > 0.02.
> - *Open item 3 (Fama-French in percent):* still open.
> - *The "averaged across folds" wording* in item 1 no longer describes the code.

A correctness review of the modelling and validation pipeline: the walk-forward
split, the metrics reported on the Models and Model Metrics pages, screening,
and the promotion gates. Extraction and data-quality checks were not re-examined
here.

Every claim below was measured, not read off the code. The scripts were
throwaway; the method for each is described so anyone can repeat it.

## Summary

Nothing leaks. Seven separate controls, each with a known answer, came back
correct, including on the live Bloomberg exports. The out-of-sample Rank IC of
roughly +0.15 seen on the bond target survives every control, so it looks like
real structure rather than a pipeline artefact.

Three things need a decision from the team. None of them is a crash or a wrong
number on screen, which is why none had been noticed: they are all cases where
a number means less than it appears to.

## What was verified as correct

| Check | Method | Result |
|---|---|---|
| Forward-return arithmetic | Hand-computed every row of a 12-day fixture including a day the target's market was shut | Exact match |
| Signal lag | Compared each lagged value against the previous row's raw value | Exactly one row, first row blank |
| Walk-forward split | 118 folds on 10 years of daily data with holidays | No train/test overlap; **no** training row whose label reaches into its test window |
| Leakage, synthetic | Pure-noise signals, 8 seeds | Mean OOS Rank IC −0.005 |
| Leakage, live data | Real signals scrambled in time | +0.15 collapses to ≈0.01 |
| Leakage, live data | Real signals against a random-walk target, 4 seeds | ≈0.00 |
| Leakage, live data | Real target with its dates reversed | −0.03 |
| Sensitivity | A deliberately planted, persistent edge | Found: +0.26 |
| Displayed equation | Recomputed predictions from the reported terms | Identical to 0.0e+00 |
| PBO | Six noise configurations / noise plus one real edge / two identical | 0.52 / 0.16 / 0.00 |
| Boosted tuning | Read the tuning window | First fold's train window only — no look-ahead |

A suspected feature-scaling defect was **investigated and disproved**:
standardising the expanded features before the L1 fit changes out-of-sample
results only marginally, and not consistently for the better (six comparisons
across two targets and three degrees). The model was left alone; three
invariance tests were kept.

## Open item 1 — the promotion gate sits inside the noise

`OOS_RANK_IC_GATE = 0.02` (`validation/gates.py`).

**Measured.** 200 runs of a signal constructed to have no predictive power
whatsoever — a persistent random level against an independent random-walk
price, scored exactly the way the app scores, through the real harness:

| Metric | Mean | SD | 5–95% range |
|---|---|---|---|
| IC | +0.001 | 0.053 | −0.081 to +0.081 |
| OOS Rank IC | +0.001 | 0.050 | −0.078 to +0.080 |

The gate is 0.02. A model with no skill at all therefore clears it roughly
**one time in three**.

The cause is sampling error, not bias: each fold scores a 20-day test window
whose 5-day labels overlap, leaving on the order of 15 effective observations
per fold, and the run reports the plain average across folds with no measure of
spread.

**Options**, in increasing order of effort:

1. Raise the gate to something outside the noise band — on this data roughly
   0.10 for a 5% false-pass rate. Crude, and it hides the real problem.
2. Report the spread alongside the mean: the standard error across folds, or a
   t-statistic. A reader then sees ±0.05 next to +0.04 and draws the right
   conclusion. This is the smallest change that makes the number honest.
3. Gate on the t-statistic rather than the mean, and require both horizons to
   agree.

Recommendation: 2 now, 3 when the sponsor confirms what the gate is for.

## Open item 2 — screening barely screens

`INCLUSION_THRESHOLD = 0.02` (`features/screening.py`), applied to the absolute
rank IC of one signal over one fold's 120-day training window.

**Measured.** 500 runs, pure-noise signal against a pure-noise target, scored
through `screen_signals`: the threshold kept the useless signal **84% of the
time**. On a 120-observation window the rank IC's standard error is about 0.09,
so a cut at 0.02 excludes almost nothing.

The consequence is not a wrong answer but a misleading display: the "Signal
inclusion across folds" table reads as evidence that a signal earned its place,
when nearly everything passes.

Fixing this needs a threshold derived from the window length rather than a flat
number — for example, requiring the rank IC to exceed about two standard errors
(≈0.18 on a 120-day window). That is a real modelling decision, not a tweak,
because it would exclude most signals most of the time.

## Open item 3 — Fama-French factors are in percent

Ken French's daily file quotes factors in percentage points (0.05 means 0.05%).
`ingest/fama_french.py` reads them as-is, while the target is a decimal
fraction (0.0005).

Least squares absorbs the scale, so **the fit, IC, RMSE and the gates are all
unaffected**. What is affected is the coefficient table on the FF5 path: its
coefficients are a hundredfold different in meaning from the polynomial page's,
with nothing on screen saying so.

The fix is a single division at ingest, plus updating whichever tests pin the
current values. It is deferred here only because the FF5 path belongs to
another ticket.

## Fixed in this pass

- **The displayed equation is one fold's fit.** On the live data, 40–45% of
  folds ended with every coefficient regularized to zero, and the page showed
  the most recent fold alone — so whether a reader saw a five-term equation or
  "No terms survived fitting" was close to a coin flip. `ModelRunResult` now
  carries `FoldTerms`, and the page says how many folds kept any term.
- **Deriving from a single candidate** raised a developer-facing `ValueError`
  from inside CSCV. It now reports no PBO, like any other run without a
  configuration search, and an empty candidate set gives a plain-words error.

## Repeating the measurements

Each is a short script against the installed package; none needs the app:

- **Null distribution** (item 1): build a persistent random signal and an
  independent random-walk price, `align_and_lag`, `run_user_polynomial` with the
  signal as the formula, over ~200 seeds; take the spread of `oos_rank_ic`.
- **Screening pass rate** (item 2): random signal, random target, 120 rows,
  `screen_signals`, count `included` over ~500 seeds.
- **Leakage controls**: take the live merged frame and re-run the same model
  after (a) permuting the signal rows, (b) replacing the target with a random
  walk, (c) reversing the target's dates. A pipeline without leakage scores ≈0
  on all three while scoring positively on the untouched data.
