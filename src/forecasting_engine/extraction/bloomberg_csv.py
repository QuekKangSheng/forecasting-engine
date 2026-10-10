"""Parsing and merging Bloomberg CSV exports.

Each export is a metadata block (``Security``, ``Start Date``, ``End Date``,
``Period``, optionally ``Currency``), a blank line, then the data table headed
``Date,...``. The row the data starts on varies per file, so it is found by
scanning for that header rather than assumed to be a fixed offset.

Files are merged on an outer join over ``Date`` — a date missing from one
file's calendar should not drop it from the others — and each file's data
columns are renamed with its security so the merged sheet says where every
column came from.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO

import pandas as pd
import pandas_market_calendars as mcal

DATE_COLUMN = "Date"

#: How dates are shown and exported — day/month/year, no time component.
DATE_DISPLAY_FORMAT = "%d/%m/%Y"

#: Used only to tell a market holiday apart from a genuinely unexplained gap.
#: Not a claim that every security here trades on NYSE hours — it is a
#: reasonable default calendar, not a per-security fact.
_HOLIDAY_CALENDAR = "NYSE"

_HEADER_RE = re.compile(r"^\s*Date\s*,", re.IGNORECASE)
_LABEL_RE = re.compile(r"[^A-Za-z0-9]+")


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
    """Repairs the reader made and reported, such as a repeated date kept
    last (see ``extraction.bloomberg_xlsx``, which is the only reader that
    populates this today)."""


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

    security = _security(list(lines[:header_idx]))
    try:
        frame = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    except pd.errors.ParserError as exc:
        raise BloombergCsvError(f"{filename}: could not parse the data table ({exc})") from exc

    frame = frame.rename(columns={frame.columns[0]: DATE_COLUMN})
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")

    lbl = label(security, filename)
    fields = [c for c in frame.columns if c != DATE_COLUMN]
    frame = frame.rename(columns={field: f"{lbl}_{field}" for field in fields})

    return BloombergCsvExport(filename=filename, security=security, frame=frame)


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
        distinct = label("", export.filename)
        for column in export.frame.columns:
            if column == DATE_COLUMN:
                continue
            field = column[len(lbl) + 1 :]
            for name in (column, f"{distinct}_{field}"):
                if name in columns:
                    sources.setdefault(name, ColumnSource(export.security, field))
    return sources


def merge(exports: Sequence[BloombergCsvExport]) -> pd.DataFrame:
    """Outer-join every export's data on ``Date``, sorted ascending.

    Two files can share a security (e.g. SPX price and SPX total return are
    both "SPX Index"), which would otherwise collide on the same column
    names. Exports whose label collides are relabelled by filename instead,
    since that is what actually distinguishes them in that case.
    """
    if not exports:
        return pd.DataFrame(columns=[DATE_COLUMN])

    labels = [label(e.security, e.filename) for e in exports]
    counts = Counter(labels)

    frames = []
    for export, lbl in zip(exports, labels, strict=True):
        if counts[lbl] > 1:
            distinct = label("", export.filename)
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


def _security(meta_lines: list[str]) -> str:
    for line in meta_lines:
        key, _, value = line.partition(",")
        if key.strip().lower() == "security":
            # Real exports pad the row with a trailing empty field
            # ("Security,SPX Index,") — take only the value itself.
            return value.split(",")[0].strip()
    return ""


def label(security: str, filename: str) -> str:
    """A column-name-safe label identifying the file's security.

    Public (not ``_``-prefixed): ``extraction.bloomberg_xlsx`` reuses this so
    an xlsx export and a CSV export sharing a security label the same way.
    """
    base = security or filename.rsplit(".", 1)[0]
    return _LABEL_RE.sub("_", base).strip("_")
