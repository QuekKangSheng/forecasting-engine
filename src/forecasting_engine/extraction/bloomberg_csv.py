"""Parsing and merging Bloomberg CSV exports.

Each export is a metadata block (``Security``, ``Start Date``, ``End Date``,
``Period``, optionally ``Currency``), a blank line, then the data table headed
``Date,...``. The row the data starts on varies per file, so it is found by
scanning for that header rather than assumed to be a fixed offset.

Files are merged on an outer join over ``Date`` — a date missing from one
file's calendar should not drop it from the others — and each file's data
columns are renamed with its security so the merged sheet says where every
column came from.

Dates are read one way for the whole column, decided from the file itself
(``read_dates``), never guessed cell by cell or from the first row alone.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO

import pandas as pd
import pandas_market_calendars as mcal

from forecasting_engine.ingest.provenance import SourceFile

DATE_COLUMN = "Date"

#: How dates are shown and exported — day/month/year, no time component.
DATE_DISPLAY_FORMAT = "%d/%m/%Y"

#: Used only to tell a market holiday apart from a genuinely unexplained gap.
#: Not a claim that every security here trades on NYSE hours — it is a
#: reasonable default calendar, not a per-security fact.
_HOLIDAY_CALENDAR = "NYSE"

#: How many dates a message or report names outright; any more are counted.
#: A file with thousands of weekend rows should not print thousands of dates.
MAX_NAMED_DATES = 10

_HEADER_RE = re.compile(r"^\s*Date\s*,", re.IGNORECASE)
_LABEL_RE = re.compile(r"[^A-Za-z0-9]+")

#: A date written as two numbers and a year, such as 9/30/2026 or 30.09.2026,
#: perhaps with a time. For any day up to the 12th the text alone does not say
#: whether it is month first or day first.
_NUMERIC_DATE = r"^(\d{1,2})([/.-])(\d{1,2})\2(\d{4})((?:[ T]\d{2}:\d{2}(?::\d{2})?)?)$"

#: The same with a two-digit year (9/30/26, 26-09-28). Bloomberg writes four
#: digits; with two, neither the century nor which number is the year is
#: certain, so such a column is refused rather than read.
_TWO_DIGIT_YEAR_DATE = r"\d{1,2}([/.-])\d{1,2}\1\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?"

#: A five-digit number in place of a date: an Excel serial date (45293 is
#: 2 January 2024), which is what a date cell becomes when the sheet that saved
#: the CSV showed it as a plain number.
_SERIAL_DATE = r"\d{5}(?:\.\d+)?"

_MONTH_FIRST = "month first"
_DAY_FIRST = "day first"

#: How many days apart consecutive dates sit, for each periodicity an export's
#: ``Period`` line can declare (by its first letter). Loose on purpose: it only
#: has to tell a series that steps by days from one that jumps by months.
_PERIOD_STEP_DAYS: Mapping[str, tuple[int, int]] = {
    "D": (1, 7),
    "W": (5, 9),
    "M": (25, 35),
    "Q": (80, 100),
    "S": (170, 195),
    "Y": (350, 380),
    "A": (350, 380),
}

#: A number written with thousands separators ("4,512.30"), as Excel writes
#: one when it saves a sheet again, and a number written plainly.
_GROUPED_NUMBER = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
_PLAIN_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)"


class BloombergCsvError(ValueError):
    """A file that is not shaped like a Bloomberg CSV export."""


@dataclass(frozen=True)
class BloombergCsvExport:
    """One parsed export: its security, and its data columns labelled by it."""

    filename: str
    security: str
    frame: pd.DataFrame
    """``Date`` plus one column per field, each named ``{label}_{field}``."""
    notes: tuple[str, ...] = ()
    """Repairs made while loading, each to be shown to the user: rows that only
    repeated an earlier row (``validation.drop_exact_repeats``), or one
    security's files joined into one (``join_split_files``)."""
    source: SourceFile | None = None
    """The uploaded file the export was read from, once its bytes are kept
    (``ingest.upload.keep_source``): how a dataset traces back to its exports."""
    parts: tuple[BloombergCsvExport, ...] = ()
    """The exports this one was joined from, when one security came split over
    several files (``join_split_files``); empty for a file read on its own."""


def read_export(filename: str, data: bytes) -> BloombergCsvExport:
    """Parse one Bloomberg CSV export. Raises ``BloombergCsvError`` if malformed."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BloombergCsvError(f"{filename}: not a UTF-8 CSV ({exc})") from exc

    return parse_lines(filename, text.splitlines())


def parse_lines(filename: str, lines: Sequence[str]) -> BloombergCsvExport:
    """Parse an export already split into CSV lines.

    ``extraction.bloomberg_xlsx`` turns a one-sheet workbook into these lines,
    so both formats share one parser.
    """
    header_idx = next((i for i, line in enumerate(lines) if _HEADER_RE.match(line)), None)
    if header_idx is None:
        raise BloombergCsvError(f"{filename}: no 'Date,...' header row found")

    metadata = _metadata(lines[:header_idx])
    security = metadata.get("security", "")
    table = lines[header_idx:]
    _refuse_overlong_rows(filename, table, first_line=header_idx + 1)
    try:
        # index_col=False: a row ending in one empty cell more than the header
        # names ("9/30/2026,16.3," under "Date,PX_LAST") is read under the
        # header, instead of pandas taking its date for the row label and
        # shifting every value one column left. The date column is read as
        # text so read_dates sees it exactly as written.
        frame = pd.read_csv(StringIO("\n".join(table)), dtype={0: str}, index_col=False)
    except pd.errors.ParserError as exc:
        raise BloombergCsvError(f"{filename}: could not parse the data table ({exc})") from exc
    if frame.empty:
        raise BloombergCsvError(
            f"{filename}: the table under the 'Date,...' header has no rows of data. "
            "Export the security again over dates it has data for."
        )

    frame = frame.rename(columns={frame.columns[0]: DATE_COLUMN})
    fields = [c for c in frame.columns if c != DATE_COLUMN]
    named = [c for c in fields if not _is_padding(frame[c])]
    if not security and len(named) > 1:
        raise BloombergCsvError(no_security_message(filename, len(named), "no Security line"))
    frame[DATE_COLUMN] = read_dates(filename, frame[DATE_COLUMN], metadata)
    for field in fields:
        frame[field] = _ungrouped(frame[field])

    lbl = label(security, filename)
    frame = frame.rename(columns={field: f"{lbl}_{field}" for field in fields})

    return BloombergCsvExport(filename=filename, security=security, frame=frame)


def _is_padding(values: pd.Series) -> bool:
    """Whether ``values`` is the column pandas makes of a header's trailing comma
    ("Date,PX_LAST,"): unnamed, so called ``Unnamed: 2``, and empty on every row.
    It is no field, so it does not count towards a file having several."""
    return bool(re.fullmatch(r"Unnamed: \d+", str(values.name))) and values.isna().all()


def no_security_message(filename: str, columns: int, missing: str) -> str:
    """Why a file with several data columns and nothing naming its security is
    refused. Without the security there is no telling whether the columns are
    one security's fields or several securities side by side, as in the merged
    file this app offers for download, and reading them as one security would
    model them as one signal."""
    return (
        f"{filename}: the file has {columns} data columns but {missing}, so nothing says "
        "which security each column belongs to. Upload each security's own Bloomberg "
        "export as the terminal saved it; the merged file this app offers for download "
        "cannot be uploaded again."
    )


def read_dates(filename: str, values: pd.Series, metadata: Mapping[str, str]) -> pd.Series:
    """The Date column as datetimes, read one way for the whole column.

    A date such as 1/2/2020 reads month first or day first. Any one date in the
    column that reads only one way (1/13/2020, 13/01/2020) settles it for every
    row. Failing that, the file's own lines are the evidence: the dates must
    fall between ``Start Date`` and ``End Date``, and must step the way
    ``Period`` says (a daily series by days, a monthly one by months). When
    nothing settles it, or the dates disagree with each other, the file is
    refused rather than guessed. Excel serial numbers and two-digit years are
    refused too (``_refuse_uncertain_forms``).

    Any other way of writing a date (``2024-01-02``) is read by pandas as
    before. A blank or unreadable date stays blank, for the schema to refuse.

    ``extraction.bloomberg_xlsx`` reads its ``Data`` sheet's text dates here too.
    """
    text = values.astype("string").str.strip()
    written = text[text.notna() & (text != "")]
    _refuse_uncertain_forms(filename, written)
    parts = written.str.extract(_NUMERIC_DATE)
    if written.empty or parts[0].isna().any():
        return _calendar_dates(pd.to_datetime(values, errors="coerce"))

    readings = {order: _numeric_dates(parts, order) for order in (_MONTH_FIRST, _DAY_FIRST)}
    orders = [order for order, dates in readings.items() if dates.notna().all()]
    if not orders:
        raise BloombergCsvError(_unreadable_dates_message(filename, written, readings))
    if len(orders) > 1:
        orders = _settle_order(readings, metadata)
    if len(orders) > 1:
        example = written.iloc[0]
        month_first, day_first = (readings[order].iloc[0] for order in (_MONTH_FIRST, _DAY_FIRST))
        raise BloombergCsvError(
            f"{filename}: every date in the file reads month first or day first ({example} "
            f"is {_spelled(month_first)} or {_spelled(day_first)}), and the file's Start "
            "Date, End Date and Period lines do not settle which. Check the date format "
            "the Bloomberg terminal exports with, and export the file again with those "
            "lines left in."
        )
    return readings[orders[0]].reindex(values.index)


def _refuse_uncertain_forms(filename: str, written: pd.Series) -> None:
    """Refuse a Date column written in a form that does not say for certain
    which dates it means.

    Excel serial numbers: a CSV does not say which of Excel's two date systems
    they count from. Two-digit years: 26-09-28 is 28 September 2026 written
    year first or 26 September 2028, and 9/30/26 could be 1926. Bloomberg
    writes four-digit years, so an export written with two was saved again by
    something else, and is better exported again than guessed at.
    """
    if not written.empty and written.str.fullmatch(_SERIAL_DATE).all():
        raise BloombergCsvError(
            f"{filename}: the Date column holds numbers such as {written.iloc[0]} instead "
            "of dates, which is how Excel saves a date cell it shows as a plain number. "
            "In Excel, format the Date column as a date and save the file again, or "
            "upload the .xlsx workbook itself."
        )
    two_digit = written[written.str.fullmatch(_TWO_DIGIT_YEAR_DATE).astype(bool)]
    if not two_digit.empty:
        raise BloombergCsvError(
            f"{filename}: the Date column writes years with two digits ({two_digit.iloc[0]}), "
            "so neither the century nor which number is the year is certain. Export the "
            "file again with four-digit years (9/30/2026, not 9/30/26)."
        )


def _numeric_dates(parts: pd.DataFrame, order: str) -> pd.Series:
    """The dates ``parts`` (from ``_NUMERIC_DATE``) give read in ``order``; NaT
    where that is no date (a 13th month, a 30 February, a year 0)."""
    first, second = parts[0].astype(int), parts[2].astype(int)
    month, day = (first, second) if order == _MONTH_FIRST else (second, first)
    year = parts[3].astype(int)
    iso = (
        year.astype(str).str.zfill(4)
        + "-"
        + month.astype(str).str.zfill(2)
        + "-"
        + day.astype(str).str.zfill(2)
        + parts[4].str.replace("T", " ", regex=False)
    )
    return _calendar_dates(pd.to_datetime(iso, format="ISO8601", errors="coerce"))


def _calendar_dates(dates: pd.Series) -> pd.Series:
    """``dates`` with any in year 0 made blank.

    pandas counts a year 0 before year 1, so ``0000-01-13`` reads; the calendar
    has none, and such a date cannot be printed (``Timestamp.strftime`` raises),
    so the page would fail on showing it. It is read as no date at all instead.
    """
    if not pd.api.types.is_datetime64_any_dtype(dates):
        return dates
    return dates.where(dates.dt.year > 0)


def _settle_order(readings: Mapping[str, pd.Series], metadata: Mapping[str, str]) -> list[str]:
    """The readings of an all-ambiguous column that fit the file's own lines.

    Start and End Date are tried first, then the Period. Evidence that rules
    out every reading says nothing about which is right, and is passed over.
    """
    orders = list(readings)
    for fits in (_fits_start_and_end, _fits_period):
        kept = [order for order in orders if fits(order, readings[order], metadata)]
        if kept:
            orders = kept
        if len(orders) == 1:
            break
    return orders


def _fits_start_and_end(order: str, dates: pd.Series, metadata: Mapping[str, str]) -> bool:
    days = dates.dt.normalize()
    start = _line_date(metadata.get("start date", ""), order)
    end = _line_date(metadata.get("end date", ""), order)
    return (start is None or start <= days.min()) and (end is None or end >= days.max())


def _line_date(text: str, order: str) -> pd.Timestamp | None:
    """A Start Date or End Date line read in ``order``, None if it says nothing.

    Written like the data (8/31/2016), it is read the same way, and is NaT where
    that is no date, so it rules ``order`` out. Written any other way, it is
    read as ISO, and anything else is no evidence at all.
    """
    parts = pd.Series([text.strip()], dtype="string").str.extract(_NUMERIC_DATE)
    if parts[0].notna().all():
        return _numeric_dates(parts, order).dt.normalize().iloc[0]
    parsed = pd.to_datetime(text.strip(), format="ISO8601", errors="coerce")
    return None if pd.isna(parsed) else parsed.normalize()


def _fits_period(order: str, dates: pd.Series, metadata: Mapping[str, str]) -> bool:
    step = _PERIOD_STEP_DAYS.get(metadata.get("period", "").strip()[:1].upper())
    days = dates.dt.normalize().drop_duplicates().sort_values()
    if step is None or len(days) < 2:
        return True
    low, high = step
    return low <= days.diff().dropna().dt.days.median() <= high


def _unreadable_dates_message(
    filename: str, written: pd.Series, readings: Mapping[str, pd.Series]
) -> str:
    not_month_first = readings[_MONTH_FIRST].isna()
    not_day_first = readings[_DAY_FIRST].isna()
    neither = written[not_month_first & not_day_first]
    if not neither.empty:
        return (
            f"{filename}: {neither.iloc[0]} in the Date column is not a date, read month "
            "first or day first. Correct it and upload the file again."
        )
    return (
        f"{filename}: the dates are not all written the same way: "
        f"{written[not_day_first].iloc[0]} only reads month first, but "
        f"{written[not_month_first].iloc[0]} only reads day first. Export the file again "
        "with one date format."
    )


def _spelled(date: pd.Timestamp) -> str:
    return f"{date.day} {date:%B %Y}"


def _refuse_overlong_rows(filename: str, table: Sequence[str], *, first_line: int) -> None:
    """Refuse a row holding more values than the header names columns for.

    Trailing empty cells are fine and are dropped when the table is read; a
    real value past the last named column would be silently lost instead.
    """
    try:
        header = next(csv.reader(table[:1]))
        for number, row in enumerate(csv.reader(table[1:]), start=first_line + 1):
            if len(row) > len(header) and any(cell.strip() for cell in row[len(header) :]):
                raise BloombergCsvError(
                    f"{filename}: line {number} has more values than the header has "
                    f"columns ({', '.join(header)}). Each row needs one value per column; "
                    "export the file again."
                )
    except csv.Error as exc:
        # Such as a cell over the csv module's limit of 131,072 characters, far
        # past any value an export holds: the file is not a table of values.
        raise BloombergCsvError(
            f"{filename}: the data table could not be read ({exc}). Export the file again."
        ) from exc


def _ungrouped(values: pd.Series) -> pd.Series:
    """``values`` as numbers, if they were written with thousands separators.

    Read only when nothing else can be meant: every value is a number, and some
    value has a decimal point or two separators, so no comma can be a decimal
    comma. "4,512" on its own is 4512 or 4.512, and is left as text for the
    schema to refuse.
    """
    if pd.api.types.is_numeric_dtype(values):
        return values
    text = values.dropna().astype("string").str.strip()
    grouped = text.str.fullmatch(_GROUPED_NUMBER)
    if not grouped.any() or not (grouped | text.str.fullmatch(_PLAIN_NUMBER)).all():
        return values
    has_point = text.str.contains(".", regex=False).any()
    if not (has_point or text[grouped].str.count(",").gt(1).any()):
        return values
    return pd.to_numeric(values.str.replace(",", "", regex=False)).astype("float64")


@dataclass(frozen=True)
class ColumnSource:
    """The export a merged column came from: its security and its field."""

    security: str
    field: str

    @property
    def ticker(self) -> str:
        return self.security.split(" ", 1)[0]


def column_sources(
    exports: Sequence[BloombergCsvExport], columns: Collection[str]
) -> dict[str, ColumnSource]:
    """Which security and field each of ``columns`` came from.

    A column is found under its security's label or, where ``merge`` or a
    target/signal collision relabelled it, under its file's. Either way only
    exports sharing a label can collide, so the security it maps to is right.
    Earlier exports win, so pass target exports first.
    """
    sources: dict[str, ColumnSource] = {}
    for export in exports:
        lbl = label(export.security, export.filename)
        distinct = file_label(export)
        for column in export.frame.columns:
            if column == DATE_COLUMN:
                continue
            field = column[len(lbl) + 1 :]
            for name in (column, f"{distinct}_{field}"):
                if name in columns:
                    sources.setdefault(name, ColumnSource(export.security, field))
    return sources


def join_split_files(
    exports: Sequence[BloombergCsvExport],
) -> tuple[list[BloombergCsvExport], list[str]]:
    """Join the files that are one security's series split into parts.

    Files sharing a security and exactly the same fields are parts of one
    series, such as a long history exported in two goes. Left apart, ``merge``
    would give each part its own columns and a model would train on only one
    of them. They become one export, noted as joined, unless they give a date
    they share two different values (``_join_parts``). Then there is no telling
    which is right, so they are left out, and the second list says why: the
    caller must not commit without them.

    Files of one security with different fields (SPX price and SPX total
    return) are not parts of one series, and are left for ``merge`` to relabel
    by file name. A file with no Security line is never joined to another.
    """
    groups: dict[object, list[BloombergCsvExport]] = {}
    for export in exports:
        fields = frozenset(export.frame.columns) - {DATE_COLUMN}
        key = (label(export.security, export.filename), fields) if export.security else id(export)
        groups.setdefault(key, []).append(export)

    joined: list[BloombergCsvExport] = []
    problems: list[str] = []
    for parts in groups.values():
        outcome = parts[0] if len(parts) == 1 else _join_parts(parts)
        if isinstance(outcome, str):
            problems.append(outcome)
        else:
            joined.append(outcome)
    return joined, problems


def _join_parts(parts: Sequence[BloombergCsvExport]) -> BloombergCsvExport | str:
    """One security's files as one export, or why they cannot be joined.

    A date the files share must have one value in every file that has a value
    for it. A blank in one file is filled from another: an export taken before
    a day's value was in has #N/A N/A there, which a later export's value does
    not contradict.
    """
    security = parts[0].security
    names = ", ".join(part.filename for part in parts)
    by_date = pd.concat([part.frame for part in parts], ignore_index=True).groupby(DATE_COLUMN)
    clashing = by_date.nunique().gt(1).any(axis=1)
    if clashing.any():
        return (
            f"{security} came in {len(parts)} files ({names}) that give different values "
            f"on {int(clashing.sum())} of the dates they share: "
            f"{named_dates(clashing.index[clashing])}. Remove the wrong file, or export "
            f"{security} again as one file; until then this data cannot be committed."
        )
    frame = by_date.first().reset_index()
    first, last = (d.strftime(DATE_DISPLAY_FORMAT) for d in frame[DATE_COLUMN].agg(["min", "max"]))
    note = (
        f"{security} came in {len(parts)} files ({names}), joined into one series "
        f"from {first} to {last}."
    )
    return BloombergCsvExport(
        filename=" + ".join(part.filename for part in parts),
        security=security,
        frame=frame,
        notes=(*(n for part in parts for n in part.notes), note),
        parts=tuple(parts),
    )


def named_dates(dates: Iterable[pd.Timestamp]) -> str:
    """``dates`` as dd/mm/yyyy, the first ``MAX_NAMED_DATES`` named and the rest
    counted, for a message."""
    dates = list(dates)
    named = dates[:MAX_NAMED_DATES]
    shown = ", ".join(pd.Timestamp(d).strftime(DATE_DISPLAY_FORMAT) for d in named)
    more = len(dates) - MAX_NAMED_DATES
    return shown + (f" and {more:,} more" if more > 0 else "")


def merge(exports: Sequence[BloombergCsvExport]) -> pd.DataFrame:
    """Outer-join every export's data on ``Date``, sorted ascending.

    Two files can share a security (e.g. SPX price and SPX total return are
    both "SPX Index"), which would otherwise collide on the same column
    names. Exports whose label collides are relabelled by filename instead
    (``file_label``), since that is what actually distinguishes them in that case.
    """
    if not exports:
        return pd.DataFrame(columns=[DATE_COLUMN])

    labels = [label(e.security, e.filename) for e in exports]
    counts = Counter(labels)

    frames = []
    for export, lbl in zip(exports, labels, strict=True):
        if counts[lbl] > 1:
            distinct = file_label(export)
            frame = export.frame.rename(
                columns={
                    c: distinct + c[len(lbl):]
                    for c in export.frame.columns
                    if c != DATE_COLUMN and c.startswith(lbl)
                }
            )
        else:
            frame = export.frame
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=DATE_COLUMN, how="outer")
    return merged.sort_values(DATE_COLUMN).reset_index(drop=True)


def drop_empty_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop data columns with no values at all, and name what was dropped.

    A Bloomberg export can carry a field the security simply has no data
    for — VIX has no PX_VOLUME, a global vol index may have no PX_BID — and
    every row reads ``#N/A N/A``. That is not a gap worth reporting per row;
    the column carries zero information, so it is removed outright.
    """
    empty = [c for c in frame.columns if c != DATE_COLUMN and frame[c].isna().all()]
    return frame.drop(columns=empty), empty


def with_display_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``frame`` with ``Date`` as dd/mm/yyyy text, no time component.

    For display and export only. Every internal computation — sorting,
    weekday and holiday checks, day-over-day change — needs a real datetime
    and must already be done before this is called.
    """
    out = frame.copy()
    out[DATE_COLUMN] = out[DATE_COLUMN].dt.strftime(DATE_DISPLAY_FORMAT)
    return out


def missing_row_report(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per date that has at least one missing value, with a likely reason.

    A gap is usually not a fault: a US-only bond index has nothing on a US
    market holiday, even when a global index does. The reason given is a
    best-effort classification against a general market calendar, not a
    per-security fact — genuinely unexplained gaps are labelled as such
    rather than guessed at, so a real data problem still stands out.
    """
    data_columns = [c for c in frame.columns if c != DATE_COLUMN]
    if not data_columns:
        return pd.DataFrame(columns=["Date", "Missing columns", "Likely reason"])

    missing_mask = frame[data_columns].isna().any(axis=1)
    if not missing_mask.any():
        return pd.DataFrame(columns=["Date", "Missing columns", "Likely reason"])

    gap_dates = frame.loc[missing_mask, DATE_COLUMN]
    readable_dates = gap_dates.dropna()
    holidays = (
        _holidays_between(readable_dates.min(), readable_dates.max())
        if not readable_dates.empty
        else set()
    )

    rows = []
    for idx in frame.index[missing_mask]:
        date = frame.loc[idx, DATE_COLUMN]
        missing = [c for c in data_columns if pd.isna(frame.loc[idx, c])]
        if pd.isna(date):
            reason = "Unreadable date"
        elif date.weekday() >= 5:
            reason = "Weekend"
        elif date.normalize() in holidays:
            reason = f"Likely {_HOLIDAY_CALENDAR} market holiday"
        else:
            reason = "Unexplained gap"
        rows.append(
            {"Date": date, "Missing columns": ", ".join(missing), "Likely reason": reason}
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=32)
def _holidays_between(start: pd.Timestamp, end: pd.Timestamp) -> set[pd.Timestamp]:
    """Weekdays in ``[start, end]`` that ``_HOLIDAY_CALENDAR`` was closed for."""
    calendar = mcal.get_calendar(_HOLIDAY_CALENDAR)
    trading_days = set(calendar.valid_days(start_date=start, end_date=end).tz_localize(None))
    all_days = pd.date_range(start, end, freq="D")
    weekdays = set(all_days[all_days.weekday < 5])
    return weekdays - trading_days


def _metadata(meta_lines: Sequence[str]) -> dict[str, str]:
    """The metadata block's ``key,value`` lines, keys lower-cased, first one kept."""
    found: dict[str, str] = {}
    for line in meta_lines:
        key, _, value = line.partition(",")
        # Real exports pad the row with a trailing empty field
        # ("Security,SPX Index,") — take only the value itself.
        found.setdefault(key.strip().lower(), value.split(",")[0].strip())
    return found


def label(security: str, filename: str) -> str:
    """A column-name-safe label identifying the file's security.

    Public (not ``_``-prefixed): ``extraction.bloomberg_xlsx`` reuses this so
    an xlsx export and a CSV export sharing a security label the same way.
    """
    base = security or filename.rsplit(".", 1)[0]
    return _LABEL_RE.sub("_", base).strip("_")


def file_label(export: BloombergCsvExport) -> str:
    """The label ``merge`` gives ``export``'s columns in place of its security's,
    when another export shares that: its file's name.

    A security joined from several files (``join_split_files``) takes its first
    part's name, so its columns read as that file's would on its own, not as
    every part's name run together (``spx_a1_csv_spx_a2_PX_LAST``).
    """
    return label("", (export.parts or (export,))[0].filename)
