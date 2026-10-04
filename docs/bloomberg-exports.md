# Uploading Bloomberg exports

The dashboard accepts Bloomberg's original history exports, one file per
security, either as **CSV** or as **Excel (`.xlsx`)**, and joins all fields on
date. Do not rename columns, substitute fields or manually convert the files
into a separate signal-CSV format. Which format you actually get depends on
how your Bloomberg access produces the export — both are accepted so you
don't need to convert one into the other by hand.

## What to export

Export the histories needed for the analysis, as either CSV or `.xlsx` — no
need to pick one over the other for the app's sake. `PX_LAST`, `PX_BID`,
`TOT_RETURN_INDEX_GROSS_DVDS` and other numeric Bloomberg fields can coexist
in the same file.

**A CSV export** is a metadata block, a blank line, then a table whose first
column is `Date`:

```
Security,SPX Index
Start Date,1/1/2016
End Date,12/31/2025
Period,Daily

Date,PX_LAST,PX_BID
1/4/2016,2012.66,#N/A N/A
```

Trailing empty metadata cells, such as `Security,SPX Index,`, are accepted.
Placeholder-only fields are removed after merging because they carry no data.

**An `.xlsx` export** from Spreadsheet Builder is one sheet laid out exactly
like the CSV export: the `Security` / `Start Date` / `End Date` / `Period`
(optionally `Currency`) rows, a blank row, then a table headed `Date`. The
reader takes the workbook's **first sheet** and parses it with the CSV reader,
so the same export gives the same columns and values in either format. Cell A1
must read `Security`; anything else is refused with the expected layout named.

Only the values Excel **saved** are read. A sheet built from `BDH` formulas
works once Excel has loaded the data and the file was saved; dates may be
stored as dates or as Excel serial numbers. If a formula has no saved value, or
a cell still reads `#N/A Requesting Data`, the file is refused: open it in
Excel on the Bloomberg PC, let the data load, save, and upload again.

Older workbooks with a `Data` sheet (`Date` plus one column per field) and a
`Metadata` sheet still load. For those the security is read from `Metadata`,
not guessed from the filename, since a filename has been observed to disagree
with what a file actually contains.

### Target indices

Signals are open: export whatever the analysis needs. The two **targets** are
not. They are fixed, and both are forecast on a **total-return** basis
(dividends and coupons reinvested), never the plain price series:

| Role | Security | Field |
|---|---|---|
| Equity target | `SPX Index` | `TOT_RETURN_INDEX_GROSS_DVDS` |
| Bond target | `LBUSTRUU Index` | `TOT_RETURN_INDEX_GROSS_DVDS` |

**The bond target is the US Aggregate, `LBUSTRUU`** — decided 17 Sep 2026. The
earlier exports used `LEGATRUU`, the *Global* Aggregate. That is a different
index with a different calendar, not a relabelling, so it needs re-exporting
rather than renaming.

The calendar difference matters. The US Aggregate follows the US bond market,
which is closed on days the NYSE is open (Columbus Day, Veterans Day). On those
days a merged file has a row but no bond price. That blank is correct and must
stay blank: a target is never forward-filled, because a filled price on a closed
day reads as a real trading day and turns into a return that never happened.

## Converting

On the **Data** page, drop every export (CSV or `.xlsx`, mixed together is
fine) into the one multi-file uploader. The page joins them on date, labels
fields by security, validates their generic shape and reports gaps and
statistically unusual moves. Nothing needs a command line and no temporary
signal CSV is produced.

**Dates in a CSV export** are `m/d/yyyy` or `d/m/yyyy` depending on the
terminal's locale, and for any day up to the 12th the two look the same. The
reader takes the format from the file's first date, so a file whose first date
is ambiguous (`1/4/2016`) is read month first, Bloomberg's default. A later date
that doesn't fit that format (`13/01/2016` in a month-first file) is not
guessed: it reads as blank, and the file fails validation (a date may not be
blank) and is left out of the merge with that reason. If a merged file's dates
look a month out, or a file is refused for a blank date, check the terminal's
date format setting. Dates in an `.xlsx`
export are stored as dates, so the order question doesn't arise.

The merged Bloomberg CSV is downloadable directly. Fama-French factors are not
part of it: they are fetched when the Fama-French model is run on the Models
page, and cached by content hash.

### What it does, and what it deliberately still doesn't

**Signal gaps are not filled here or on the Data page.** Different indices keep
different trading calendars; across a real ten-year pull the union was 2,610
dates with all signals present on only 2,499 of them. The merge keeps every
row, and the Data page lists the rows missing a value. On the Models page each
target's panel keeps only the dates that target has a price, and each signal
takes its last value on or before each of those dates, for at most 3 rows
(`ingest/align.py`).

**Target columns are never filled, at any gap length.** See "Target indices"
above.

**Range breaches are flagged, not corrected.** If almost every value in a column
falls outside its documented range, the converter says so — that pattern means
the wrong field was exported. A handful of breaches is left alone, because a
genuine market dislocation looks a lot like an outlier.
