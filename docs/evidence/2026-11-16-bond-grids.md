# Bond grids, 16 Nov 2026

Asked for after the 16 Nov audit recheck: "the bond model needs work", and to keep
testing different grids. On the data from 2006 no bond model passes the gate at any
horizon, and the bond derived polynomial's latest fit is a constant (the audit's
finding 11). This file records 124 bond runs that vary what the derived polynomial and
machine learning are offered and how they are fitted, and what they showed.

Produced by `docs/evidence/2026-11-16-bond-grids.py` (the derived-polynomial grids) and
`docs/evidence/2026-11-16-bond-grids-ml.py` (machine learning), at main `1bc88e0`. Both
read the exports as the Data page merges them and run every model through the same
walk-forward and score as the Models page: test window 20, embargo 20, the first 504
rows a tuning period, labels purged; the Signal Rank IC against the model's own level,
and t on the larger of its Hansen-Hodrick and test-window standard errors.

**Data.** Only the Sep 2026 upload (31 Aug 2016 to 30 Sep 2026) was on the machine
that ran this; the files from 2006 were not. Nothing here is tested on 2008–2016. Both
scripts take the folder of exports as their argument, so they can be rerun on the 2006
files as they are.

**Periods.** Every fit is causal, so one pass scores two periods: *dev*, the test
windows whose every label ends before 1 Oct 2024, and *held*, the windows starting on or
after it. The held period has been looked at for many models since October, so it is a
second look, not a clean test.

**The rule, written before any run.** The page's grid (lasso at degrees 1–3 and
shrinkage, a 504-row window, the best Signal Rank IC chosen) is replaced only by a grid
that is (1) usable, with its latest fold not a constant and under a quarter of its folds
constant; (2) stable, its most often kept term one sign in at least 90% of the folds
keeping it; (3) better on dev by more than 0.10, the no-skill standard deviation of a
Signal Rank IC over dev's roughly 75 independent 20-day periods; and (4) above zero on
held. Among several, the simplest wins, not the highest score.

## What was varied

| Set | Runs |
|---|---|
| Columns offered × fit × train window, h = 20 | 72 |
| The bond research fit × 7 train windows × h = 1, 5, 10, 20 | 28 |
| The page's own grid at h = 1, 5, 10 | 12 |
| Machine learning: 4 search spaces × 3 column sets, h = 20 | 12 |

- **Columns:** the page's panel (gauges as levels, rates and the curve as 1-row
  changes, prices as 1-row log returns); the same plus each rate's level; that plus every
  series' change over the last `h` rows; the gauges and rates as levels only; levels
  and `h`-row changes only. Each is a rule over every series, never a hand-picked input.
- **Fits:** the page's lasso at degrees 1, 2, 3 (screened per fold), degree 1
  unscreened, the page's shrinkage, and the bond research model's fit (levels given their
  own label-window innovations, §9.2), the last on the levels only.
- **Train windows:** the latest 504 rows (the page's); every earlier row up to 1,260;
  every earlier row; and for the research fit 252, 378, 756 and 1,008 rows too.
- **Machine learning search spaces:** the page's; shallow trees (depth 1–2, leaves of
  20–150 rows); slow (50–300 trees, learning rate 0.005–0.05, depth 2–3); and wide
  (depth 2–6, 20–300 trees, leaves of 5–200 rows).

## Results

**The page today.** The page's grid picks lasso degree 3 for bond at h = 20: Signal Rank
IC +0.049, t 0.58 (dev +0.056, held −0.057), with 55 of 101 folds forecasting a
constant, the latest among them. At h = 1, 5 and 10 its four candidates score between
−0.03 and +0.06, t below 1.5 everywhere.

**No grid meets the rule.** The four closest:

| Run (h = 20) | All | t | Dev | Held | Constant folds | Fails |
|---|---|---|---|---|---|---|
| Bond research fit, latest 504 rows | +0.128 | 1.69 | +0.156 | +0.115 | 20 of 101 | (3): dev gap +0.100, not more than 0.10 |
| Machine learning, shallow trees, page columns | +0.131 | 2.01 | +0.210 | −0.065 | 17 of 101 | (4): held below zero |
| Machine learning, shallow trees, levels + `h`-row changes | +0.143 | 1.91 | +0.111 | +0.232 | 27 of 101 | (1) and (3) |
| Shrinkage, page columns + rate levels | +0.053 | 0.61 | +0.023 | +0.241 | 0 of 101 | (2): top term one sign in 63% of folds; (3) |

**The research fit's score belongs to one window length.** Its all-period Signal Rank
IC at h = 20, by train window:

| 252 | 378 | 504 | 756 | 1,008 | 1,260 | every earlier row |
|---|---|---|---|---|---|---|
| −0.081 | +0.030 | **+0.128** | −0.001 | −0.066 | +0.032 | +0.090 |

A carry signal would not vanish at 378 and 756 rows. The same pattern holds at h = 5
and 10. On the data from 2006, with every earlier row, this model scored −0.004 on its
pre-2016 hold-out (§20).

**Machine learning.** Shallow trees scored higher than the page's search space on all
three column sets (all-period +0.131, +0.073, +0.143 against +0.054, +0.065, +0.102),
the one consistent pattern in these runs. The best of them is still the best of twelve,
and its held period goes the wrong way.

**Rate levels and `h`-row changes** gave the shrinkage fit an equation that never falls
back to a constant, but its terms change sign from fold to fold and it scores no better.

**Across all 124 runs** the all-period Signal Rank IC ranges from −0.15 to +0.16. With a
no-skill standard deviation of about 0.10 and 124 tries, that is about the spread chance
alone gives.

## What it means

No bond specification tried here, linear or tree-based, on levels or changes, over
short or long windows, at any horizon the page offers, beats chance on ten years of
data. That matches the October research, the 10 Nov round and the rerun on the data
from 2006. A bond index's next-month return is mostly the next month's move in yields,
which none of these inputs forecasts; what it earns if yields do not move (its carry) is
small and slow against that noise.

Nothing in the app was changed because of these runs. In order of value:

1. Rerun both scripts on the files from 2006. If shallow trees hold up on 2008–2016,
   offering depth 1–2 for the bond target is a change to `boosted.SEARCH_SPACE`.
2. For the sponsor's request to see a bond equation: the bond tab could show the
   shrinkage candidate's latest equation beside the one picked, since it never falls back
   to a constant, with its score beside it and not presented as passing.
3. Present the bond result as it is: no bond forecast passes; the optimiser's own trust
   gate then decides whether forecasts tilt anything (with the derived polynomial for
   both indices it opened at 3 of 226 rebalances on the data from 2006); and these runs
   are the evidence that a bond signal was looked for.

## Rerunning

```
uv run --project <repo> python -I docs/evidence/2026-11-16-bond-grids.py <folder> --out grid.jsonl
uv run --project <repo> python -I docs/evidence/2026-11-16-bond-grids.py <folder> \
    --libraries lvl --fits carry --windows roll252,roll378,roll504,roll756,roll1008,roll1260,expanding \
    --horizons 1,5,10,20 --out carry_windows.jsonl
uv run --project <repo> python -I docs/evidence/2026-11-16-bond-grids.py <folder> \
    --libraries page --fits lasso1,lasso2,lasso3,shrink --windows roll504 --horizons 1,5,10 \
    --out page_horizons.jsonl
uv run --project <repo> python -I docs/evidence/2026-11-16-bond-grids-ml.py <folder> --out ml.jsonl
```

Each writes one JSON line per run (the three scores, constant folds, the latest fold's
terms, and how often each term was kept and with one sign) and skips runs already in its
output file. On the Sep 2026 upload, on three processes, the 72-run grid took about 20
minutes and the twelve machine-learning runs about 10.
