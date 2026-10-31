# Risk level calibration, 29 Oct 2026 (updated for scaled starting weight + k=40)

The share of rebalances at a bound and net Sharpe for each of the 5 risk levels. Produced by `docs/evidence/2026-10-29-risk-level-calibration.py`, run on the real committed dataset cached in `data/last_commit.pkl` (31 Aug 2016 – 30 Sep 2026; equity = SPX total return, bond = LBUSTRUU total return, 9 Bloomberg signals). No app code was changed for the run.

Updated after two manual-testing rounds:
- The starting weight now scales with the "Minimum in each index" bounds instead of a fixed `{80, 65, 50, 35, 20}%` (`starting_weight_for`) — reproduces the original numbers exactly at the default bounds below.
- `TILT_STRENGTH` (`k`) raised from 10 to **40**, provisional pending PO sign-off: at `k = 10`, level 3 moved too far from 50/50.

## Settings

Current Models/Portfolio page defaults: 20-day horizon, 504-day train window, 20-day test window, 20-day embargo. Equity uses the derived polynomial (the current default grid, degrees 1-3 lasso); bond uses machine learning — the same pairing the Sprint 4 sponsor demo used. Global weight bounds (0.2, 0.8), narrowed per level by `bounds_for_level`; fixed tilt strength `k = 40` (provisional).

## Results

| level | starting weight | bounds | net annual return | net Sharpe | max drawdown | % rebalances at a bound |
|---|---|---|---|---|---|---|
| 1 | 80% | 50%-80% | 11.83% | 0.82 | -27.51% | 49% |
| 2 | 65% | 50%-80% | 10.83% | 0.82 | -23.95% | 16% |
| 3 | 50% | 20%-80% | 9.79% | 0.85 | -20.14% | 5% |
| 4 | 35% | 20%-50% | 7.09% | 0.84 | -18.76% | 16% |
| 5 | 20% | 20%-50% | 4.18% | 0.68 | -17.95% | 56% |

## Notes

- **Starting weights match the sponsor's original example exactly at these default bounds** (80/65/50/35/20%). Moving the "Minimum in each index" slider moves them too — e.g. at bounds (0.3, 0.7), level 1 starts at 70%, not 80%.
- **Level 3 sits closest to its 50% starting point now** (5% of rebalances hit a bound, down from 39% at `k = 10`) — the reason for raising `k`.
- **Return and Sharpe still step down from level 1 to level 5** as equity outperformed bonds over this window; levels 1-2 and 4-5 still can't cross 50% the wrong way (`bounds_for_level`), confirmed by their bound shares (49% and 56%).
- Working calibration, not sponsor-confirmed — `TILT_STRENGTH = 40` is provisional pending a PO conversation.
