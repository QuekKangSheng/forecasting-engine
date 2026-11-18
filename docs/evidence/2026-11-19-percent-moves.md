# Scoring prices on percentage moves in the extreme-move check, 19 Nov 2026

The record behind the data quality report scoring a price or a total-return index
on its percentage change, where every column used to be scored on its change in
points ([outlier detection](../outlier-detection.md), "Prices in percent, everything
else in points"). [`2026-11-19-percent-moves.py`](2026-11-19-percent-moves.py) reads
any folder of exports with the app's own readers and prints the tables below.

## What was wrong

The check takes each column's day-over-day change and scores it against that
column's typical change (a robust z-score, flagged above 8). For a price the change
was in points, and a price's move in points grows with its level. The S&P 500 is
about five times its 2008 level, so on the data from 31 Aug 2006 its typical day is
set by recent years, and its worst days of 2008 look small beside it.

Nothing a model reads was affected: models have always read prices as log returns
(`ingest.align.TICKER_TRANSFORMS`). The flags are for information and stop no run,
so no forecast, gate or portfolio figure changes.

## What changed

A column is scored on its percentage (log) change when the models read it as a log
return: any total-return field, and the price fields of `SPX`, `DXY`, `LBUSTRUU`,
`LEGATRUU` and `LF98TRUU`. Rates, spreads and volatility gauges stay in points; so
does a column whose security isn't known, unless its field is a total return. The
threshold, the rebound rule and the wording of the report are unchanged, except that
a flagged price move is shown as a percentage.

Rates were deliberately left in points. A yield that falls from 5 to 0.3 on the same
wobble in points would have its ordinary late days flagged as large percentages; the
unit tests hold both directions.

## Result on the data from 31 Aug 2006

The 11 exports of `data_2006` (5,271 rows to 12 Nov 2026, 13 columns, 6 of them
prices), on main after the rebound fix of 18 Nov. "In points" reproduces the counts
before this change.

| threshold | all columns in points | prices on percentage moves |
|---|---|---|
| 3 | 3,333 | 2,938 |
| 4 | 1,791 | 1,479 |
| 5 | 1,020 | 819 |
| 6 | 637 | 521 |
| 8 | 280 | 245 |
| 10 | 168 | 150 |
| 12 | 100 | 103 |
| 15 | 55 | 50 |

At threshold 8, by price column, with the flags split 2006–12 / 2013–19 / 2020–26:

| column | in points | by period | on percentage | by period |
|---|---|---|---|---|
| `DXY_Curncy_PX_LAST` | 0 | 0 / 0 / 0 | 0 | 0 / 0 / 0 |
| `LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS` | 4 | 0 / 0 / 4 | 2 | 0 / 0 / 2 |
| `LBUSTRUU_Index_PX_LAST` | 4 | 0 / 0 / 4 | 2 | 0 / 0 / 2 |
| `LF98TRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS` | 43 | 2 / 4 / 37 | 48 | 24 / 5 / 19 |
| `SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS` | 54 | 0 / 0 / 54 | 24 | 15 / 0 / 9 |
| `SPX_Index_PX_LAST` | 30 | 0 / 0 / 30 | 24 | 15 / 0 / 9 |

The seven other columns are scored as before and keep their flags.

The S&P 500 total-return index's ten largest percentage moves:

| date | move | in points | score on points | score on percentage |
|---|---|---|---|---|
| 2020-03-16 | −11.98% | −431.7 | 21.0 | 17.3 |
| 2008-10-13 | +11.58% | +108.7 | 5.2 | 14.7 |
| 2008-10-28 | +10.79% | +95.6 | 4.5 | 13.7 |
| 2020-03-12 | −9.49% | −345.7 | 16.8 | 13.5 |
| 2008-10-15 | −9.03% | −94.0 | 4.6 | 12.9 |
| 2008-12-01 | −8.92% | −83.8 | 4.1 | 12.7 |
| 2008-09-29 | −8.79% | −111.1 | 5.5 | 12.5 |
| 2025-04-09 | +9.52% | +681.1 | 32.8 | 12.1 |
| 2020-03-24 | +9.39% | +279.5 | 13.4 | 12.0 |
| 2020-03-13 | +9.32% | +307.3 | 14.8 | 11.9 |

In points, five of the ten were under the threshold, all of them from 2008. On
percentage moves nine are flagged; the tenth, 13 Mar 2020, is the rebound of 12 Mar
and is reported with it.

## Not decided here

The threshold of 8 is still open with the sponsor. On percentage moves a score of 3
gives 2,938 flags on this data and 4 gives 1,479, so the case for 8 over 3 stands.
