# Bond research model: the look-ahead purge (carry VERSION 1 → 2), 16 Nov 2026

Produced by `docs/evidence/2026-11-16-carry-lookahead.py` on the Bloomberg data from 31 Aug 2006 (the files' SHA-256 are listed below), through the function the Models page's runner calls, at the page's defaults: tuning 504 rows, test 20, embargo 20, expanding training. Runtime is measured locally.

**What changed and why.** The Phase 6 verification found that at h = 20 the bond research model's fit read the first test-window row. A training row's innovation uses each input's move over its label window. The level dated at the label's end sits on the next panel row, because signals are lagged one row. With the embargo equal to the horizon, the last training row's label ends the day before the test window, so its move was read from the first test-window row. That row held a level dated before the test window, so no later information entered, but the rule is that no fit reads a test-window row.

VERSION 2 uses a training row only if its move is read from no row past the latest training label's end. The splitter ends every training label before the test window, so no fit reads a test-window row at any horizon. The fit sees only its training rows, not where the test window starts, so the rule drops each fold's last training row at every horizon, including h = 1–10, where it did not reach the test window (the 20-row embargo is longer than h + 1). Passing the test window's start into the fit would make the purge exact, but it would mean changing the harness every model shares. The rule was chosen before these results were run.

`tests/unit/test_carry_research.py::test_no_test_window_row_of_the_panel_changes_the_fit` rewrites every panel row from the test window on, with the embargo equal to the horizon, and fails on VERSION 1 at h = 1, 5 and 20.

VERSION 1 here is VERSION 2 with the purge switched off. Its rows reproduce the full-history evidence (`2026-11-16-full-history.md`, section 1 and the hold-out table) exactly.

## Summary

| h | Signal Rank IC, t: VERSION 1 → 2 | pre-2016 hold-out: VERSION 1 → 2 | gate |
|---|---|---|---|
| 1 | +0.0031, 0.20 → +0.0040, 0.26 | −0.0013, −0.06 → +0.0034, 0.15 | fails, both |
| 5 | +0.0201, 0.60 → +0.0207, 0.62 | +0.0194, 0.41 → +0.0150, 0.32 | fails, both |
| 10 | +0.0385, 0.78 → +0.0389, 0.79 | +0.0376, 0.58 → +0.0358, 0.55 | fails, both |
| 20 | +0.0661, 0.99 → +0.0631, 0.94 | −0.0040, −0.05 → −0.0197, −0.24 | fails, both |

- **Whole-period scores barely move:** at most 0.003 in Signal Rank IC and 0.06 in t. The model fails the gate at every horizon, before and after.
- **The 20-day hold-out falls from −0.004 (t −0.05) to −0.020 (t −0.24).** Still about zero; the conclusion that the model has no skill on the years its inputs weren't chosen on is unchanged.
- **Nothing downstream changes.** The bond research model is not in either portfolio pair.

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

All 8 fits: 176 s wall-clock.

## Bond research model, VERSION 1 against VERSION 2

Signal Rank IC and t as on the Models page; the gate is Signal Rank IC above 0.02 and t above 2. Hold-out: test windows whose labels end before 1 Aug 2016.

| h | version | Signal Rank IC | t | rows | gate | hold-out Signal Rank IC | hold-out t | hold-out windows | runtime |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | +0.0031 | 0.20 | 4,560 | fails signal_rank_ic and t | -0.0013 | -0.06 | 97 | 86 s |
| 1 | 2 | +0.0040 | 0.26 | 4,560 | fails signal_rank_ic and t | +0.0034 | 0.15 | 97 | 86 s |
| 5 | 1 | +0.0201 | 0.60 | 4,560 | fails t | +0.0194 | 0.41 | 97 | 87 s |
| 5 | 2 | +0.0207 | 0.62 | 4,560 | fails t | +0.0150 | 0.32 | 97 | 86 s |
| 10 | 1 | +0.0385 | 0.78 | 4,555 | fails t | +0.0376 | 0.58 | 97 | 86 s |
| 10 | 2 | +0.0389 | 0.79 | 4,555 | fails t | +0.0358 | 0.55 | 97 | 82 s |
| 20 | 1 | +0.0661 | 0.99 | 4,545 | fails t | -0.0040 | -0.05 | 96 | 86 s |
| 20 | 2 | +0.0631 | 0.94 | 4,545 | fails t | -0.0197 | -0.24 | 96 | 79 s |

