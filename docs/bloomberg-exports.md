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
A data row with one empty cell more than the header names
(`9/30/2026,16.3,` under `Date,PX_LAST`) is accepted too; a row with a real
value past the last named column is refused. Numbers Excel wrote with thousands
separators when the sheet was saved again (`"4,512.30"`) are read as numbers,
unless the column could as well mean a decimal comma (`"4,512"` and nothing
else).

The `Security` line is needed for a file with more than one data column: without
it nothing says which security each column is, so the file is refused. A header
ending in a comma (`Date,PX_LAST,`) with nothing under the unnamed column is
still one data column. That
includes the merged file the Data page offers for download, which is for
reading, not for uploading again.

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

On the **Data** page, drop the two target exports into the target uploader and
every other export into the signal uploader (CSV or `.xlsx`, mixed together is
fine). The page guesses each target's role from the security named in its file,
for you to confirm. It joins all of them on date, labels fields by security,
validates their generic shape and reports gaps and statistically unusual moves.
Nothing needs a command line and no temporary signal CSV is produced.

**Dates in a CSV export** are `m/d/yyyy` or `d/m/yyyy` depending on the
terminal's locale, and for any day up to the 12th the two look the same. The
reader settles the order once for the whole file, from the file itself: one
date that reads only one way (`13/01/2016`, `1/13/2016`) settles it; failing
that, the dates must fall between the `Start Date` and `End Date` lines and
step the way the `Period` line says, a daily file by days and a monthly one by
months. If nothing settles it, or the dates disagree with each other, the file
is refused with that reason rather than guessed: check the terminal's date
format setting and export again with the metadata lines left in. A CSV whose
Date column holds Excel serial numbers (`46295`) is refused too, with how to fix
it, and so is one whose dates have two-digit years (`9/30/26`, `26-09-28`):
Bloomberg writes four, and with two neither the century nor which number is the
year is certain. Dates in an `.xlsx` export are stored as dates or serial numbers, so the
order question doesn't arise; text dates in a workbook are read the same way as
a CSV's.

**A repeated date** in one file is settled before the merge. A row that only
repeats an earlier one, the same date with the same values, is dropped, and the
page says how many were. A date given two different values is not the app's to
settle: the file is left out, the page names the security and the dates, and
nothing can be committed until the file is corrected and uploaded again, or
removed. Leaving the page and coming back does not lift this.

**One security split over several files**, such as a long history exported in
two parts, is joined into one series when the parts agree on every date they
share, and the page says so. A part that is blank on a shared date, as an export
taken before that day's value was in can be, takes the other part's value. If
they give a shared date different values they are left out, and nothing can be
committed until the wrong file is removed or the security exported again as one
file. Files of one
security with different fields (SPX price and SPX total return) are kept apart,
labelled by file name.

**Size.** Each file may be up to 25 MB, and so may the merged data the files
make together; files that add up to more are refused with a message instead of
merged.

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

**An impossible value leaves its file out; nothing is corrected.** Each export is
checked on its own before the merge (`extraction/validation.py`): a total-return
field (`TOT_RETURN…`) must be above zero, a `PX_` field only has to be a number,
and any other field must lie within a loose sanity range, −300 to 10,000
(`SANE_RANGE`), wide enough for a deep 2s10s inversion quoted in basis points.
A file with a value outside its check is left out of the merge, with the
column, the check and the value named; the other files still merge. Large
day-over-day moves are only flagged on the Data page, never altered, because a
genuine market dislocation looks a lot like an outlier.
