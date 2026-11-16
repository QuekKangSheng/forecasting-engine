# The trust gate, checked at every rebalance, with no skill — 16 Nov 2026

Phase 4, task 7 of the 15 Nov plan. Produced by `docs/evidence/2026-11-16-trust-gate-rechecks.py` on `fix-claude-optimiser-audit`, on the 11 Bloomberg exports from 31 Aug 2006 (hashes below). The design was fixed before it was run:

- the signals are the derived polynomial for both equity and bond, the active models of `2026-11-16-optimiser-fixes.md`;
- for "no skill", the signal difference Δs is shifted in a circle against its dates, by at least 252 pair dates, so it keeps its own persistence and the returns stay real but the two no longer line up;
- there are 500 shifts per horizon, and at every rebalance the product's own gate (`optimize._trust_and_gate`) runs on the pairs it would have known.

## Summary

At one rebalance the gate should open about 2.3% of the time with no skill (t above 2, one-sided). Because it is checked again at every rebalance, the chance that it opens **at some point** in the backtest from 2008 is much higher:

| h | opens at least once | mean share of rebalances open | open at the last rebalance alone |
|---|---|---|---|
| 1 | 11.2% (± 1.4%) | 2.4% | 2.8% (± 0.7%) |
| 5 | 18.0% (± 1.7%) | 2.3% | 1.8% (± 0.6%) |
| 20 | 35.0% (± 2.1%) | 7.8% | 5.4% (± 1.0%) |

- **At h = 1 and h = 5**, the single-check rate is close to the nominal 2.3%, so the gate's standard error is about right at each check. The excess comes from checking again and again.
- **At h = 20**, the gate also opens too often at each check: 5.4% at the last rebalance and 7.8% on average, against 2.3%. With 252 pairs a 20-day horizon holds only about 13 independent observations, and the overlap-adjusted standard error is too small on samples that short. In draws where it opens, it first opens at the 15th of 226 rebalances (median), early, when the sample is shortest. This is reported, not acted on: no threshold or window was changed.
- **The actual signals**, unshifted, open the gate at 0 of 4,520 rebalances at h = 1, 137 of 904 at h = 5 and 3 of 226 at h = 20. That matches the optimiser evidence exactly. With no skill, the gate is open at 12.5% of rebalances on average at h = 5 in the draws where it opens at all, so 15.2% is within what chance alone produces. That fits the forecasts adding nothing after costs.

The shifts of one series overlap, so the draws are not independent. The ± figures treat them as if they were and are only a guide.

Runtime, measured locally (16 cores): about 6–9 minutes per horizon to fit the models, and 12 minutes, 2½ minutes and 40 seconds for the simulation at h = 1, 5 and 20.

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

No-skill draws: 500 per horizon, seed 20261116 + h, shifts of at least 252 pair dates. Gate: t above 2 on the trust slope, at least 252 pairs.

## h = 1

4,508 pairs from 3 Oct 2008; 4,520 rebalances, 4,268 of them with at least 252 pairs known, from 8 Oct 2009 to 5 Nov 2026.

|  | opens at least once | mean share of rebalances open | open at the last rebalance |
|---|---|---|---|
| no skill (500 shifted draws) | 11.2% (± 1.4%) | 2.4% | 2.8% (± 0.7%) |
| actual signals (reference) | no | 0.0% (0 of 4520) | no |

In draws where it opens, it first opens at rebalance 711 of 4520 (median), and is open at 21.4% of rebalances on average.

Runtime, measured locally: models 349 s, simulation 724 s.

## h = 5

4,508 pairs from 3 Oct 2008; 904 rebalances, 852 of them with at least 252 pairs known, from 21 Oct 2009 to 30 Oct 2026.

|  | opens at least once | mean share of rebalances open | open at the last rebalance |
|---|---|---|---|
| no skill (500 shifted draws) | 18.0% (± 1.7%) | 2.3% | 1.8% (± 0.6%) |
| actual signals (reference) | yes | 15.2% (137 of 904) | no |

In draws where it opens, it first opens at rebalance 83 of 904 (median), and is open at 12.5% of rebalances on average.

Runtime, measured locally: models 470 s, simulation 149 s.

## h = 20

4,505 pairs from 3 Oct 2008; 226 rebalances, 212 of them with at least 252 pairs known, from 19 Nov 2009 to 9 Oct 2026.

|  | opens at least once | mean share of rebalances open | open at the last rebalance |
|---|---|---|---|
| no skill (500 shifted draws) | 35.0% (± 2.1%) | 7.8% | 5.4% (± 1.0%) |
| actual signals (reference) | yes | 1.3% (3 of 226) | no |

In draws where it opens, it first opens at rebalance 15 of 226 (median), and is open at 22.4% of rebalances on average.

Runtime, measured locally: models 525 s, simulation 41 s.

