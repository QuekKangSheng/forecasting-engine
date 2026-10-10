"""Parsing Bloomberg ``.xlsx`` workbook exports into the same shape
``bloomberg_csv`` produces, for the open-schema (freeform) pipeline.

A Spreadsheet Builder export is one sheet laid out like the CSV export: a
``Security`` / ``Start Date`` / ``End Date`` / ``Period`` block, a blank row,
then a table headed ``Date``. Its rows are written out as CSV lines and parsed
by ``bloomberg_csv.parse_lines``, so the two formats give the same columns and
values. Only saved values are read: a ``BDH`` formula Excel never recalculated
and saved has nothing to read, and the file is refused with how to fix it.

The older two-sheet layout, ``Data`` (``Date`` plus one column per field) and
``Metadata`` (naming the security), is still read when a ``Data`` sheet exists.
The security comes from ``Metadata`` rather than the filename: a real export
was named for one security and contained another.
"""

from __future__ import annotations

import csv
import zipfile
from datetime import date, datetime
from io import BytesIO, StringIO

import openpyxl
import pandas as pd
from openpyxl.utils.datetime import from_excel

from forecasting_engine.extraction.bloomberg_csv import (
    DATE_COLUMN,
    BloombergCsvError,
    BloombergCsvExport,
    label,
    parse_lines,
)

DATA_SHEET = "Data"
METADATA_SHEET = "Metadata"

EXPECTED_LAYOUT = (
    "a Security / Start Date / End Date / Period block, a blank row, then a table "
    "headed Date"
)

UNSAVED_MESSAGE = (
    "Open this file in Excel on the Bloomberg PC, let the data load, save, and upload again."
)

#: What Bloomberg's add-in leaves in a cell while it is still fetching.
_REQUESTING = "#N/A Requesting"


class BloombergXlsxError(ValueError):
    """A file that is not shaped like a Bloomberg workbook export."""


def read_export(filename: str, data: bytes) -> BloombergCsvExport:
    """Parse one Bloomberg ``.xlsx`` workbook export.

    Raises ``BloombergXlsxError`` if malformed or if a formula has no saved
    value. Returns the same ``BloombergCsvExport`` shape
    ``bloomberg_csv.read_export`` does, so both readers feed the same ``merge()``.
    """
    try:
        book = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
    except (zipfile.BadZipFile, OSError) as exc:
        raise BloombergXlsxError(f"{filename}: not a readable .xlsx file ({exc})") from exc

    two_sheet = DATA_SHEET in book.sheetnames
    try:
        if two_sheet:
            rows = list(book[DATA_SHEET].iter_rows(values_only=True))
            security = _security(book)
        else:
            rows = list(book.worksheets[0].iter_rows(values_only=True))
    finally:
        book.close()

    if not two_sheet:
        _refuse_unsaved(filename, data, rows)
        return _first_sheet(filename, rows)

    if not rows:
        raise BloombergXlsxError(f"{filename}: the {DATA_SHEET!r} sheet is empty")

    header = [str(cell) for cell in rows[0]]
    frame = pd.DataFrame(rows[1:], columns=header)
    frame = frame.rename(columns={frame.columns[0]: DATE_COLUMN})
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")

    # Read straight from cell values rather than through pandas' CSV parser,
    # so Bloomberg's "#N/A N/A" placeholder (and anything else non-numeric)
    # needs coercing to NaN explicitly here — pd.read_csv does this for the
    # CSV reader automatically via its default NA-string list, this doesn't.
    for col in frame.columns:
        if col != DATE_COLUMN:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame, notes = _dedupe_dates(frame, filename)

    lbl = label(security, filename)
    fields = [c for c in frame.columns if c != DATE_COLUMN]
    frame = frame.rename(columns={field: f"{lbl}_{field}" for field in fields})

    return BloombergCsvExport(filename=filename, security=security, frame=frame, notes=notes)


def _first_sheet(filename: str, rows: list[tuple]) -> BloombergCsvExport:
    """A one-sheet export, parsed exactly as the same export saved as CSV."""
    if not rows or _text(rows[0][0] if rows[0] else None).strip() != "Security":
        raise BloombergXlsxError(
            f"{filename}: the first sheet should be laid out as {EXPECTED_LAYOUT} "
            "(cell A1 should read 'Security')."
        )
    try:
        return parse_lines(filename, _csv_lines(rows))
    except BloombergCsvError as exc:
        raise BloombergXlsxError(str(exc)) from exc


def _csv_lines(rows: list[tuple]) -> list[str]:
    """The sheet's rows as the lines a CSV export of it would have.

    Table rows are cut to the header's width, so trailing empty cells don't
    become unnamed columns, and the date cell is written as ISO text whether
    Excel stored a datetime or a serial number.
    """
    header_idx = next(
        (i for i, row in enumerate(rows) if row and _text(row[0]).strip() == DATE_COLUMN), None
    )
    width = None
    if header_idx is not None:
        header = rows[header_idx]
        width = max(i for i, cell in enumerate(header) if cell is not None) + 1
    lines = []
    for i, row in enumerate(rows):
        cells = list(row)
        if header_idx is not None and i >= header_idx:
            cells = cells[:width]
            if i > header_idx and cells:
                cells[0] = _date_text(cells[0])
        if all(cell is None for cell in cells):
            lines.append("")
            continue
        buffer = StringIO()
        csv.writer(buffer, lineterminator="").writerow([_text(cell) for cell in cells])
        lines.append(buffer.getvalue())
    return lines


def _date_text(cell: object) -> object:
    if isinstance(cell, bool):
        return cell
    if isinstance(cell, int | float):
        cell = from_excel(cell)
    if isinstance(cell, datetime | date):
        return cell.isoformat()
    return cell


def _text(cell: object) -> str:
    if cell is None:
        return ""
    if isinstance(cell, datetime | date):
        return cell.isoformat()
    return str(cell)


def _refuse_unsaved(filename: str, data: bytes, rows: list[tuple]) -> None:
    """Refuse a sheet whose formulas have no saved value to read.

    Read with ``data_only=True``, a formula Excel never recalculated and saved
    reads as empty, so the formulas are read once more to tell that apart from
    a cell that is simply blank.
    """
    if any(isinstance(cell, str) and cell.startswith(_REQUESTING) for row in rows for cell in row):
        raise BloombergXlsxError(f"{filename}: the data never loaded. {UNSAVED_MESSAGE}")
    book = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=False)
    try:
        formulas = list(book.worksheets[0].iter_rows(values_only=True))
    finally:
        book.close()
    for value_row, formula_row in zip(rows, formulas, strict=False):
        for value, formula in zip(value_row, formula_row, strict=False):
            if value is None and isinstance(formula, str) and formula.startswith("="):
                raise BloombergXlsxError(
                    f"{filename}: a formula has no saved value. {UNSAVED_MESSAGE}"
                )


def _security(book: openpyxl.Workbook) -> str:
    """The ``Security`` value from the ``Metadata`` sheet, or "" if absent.

    Read from metadata rather than trusted from the filename — the real
    export that motivated this (recorded in ``docs/ingestion-consolidation.md``) was named
    for one security and contained another.
    """
    if METADATA_SHEET not in book.sheetnames:
        return ""
    for row in book[METADATA_SHEET].iter_rows(values_only=True):
        if row and str(row[0]).strip() == "Security":
            return str(row[1]).strip()
    return ""


def _dedupe_dates(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Keep the last row for any date the sheet repeats, and say so.

    A workbook's ``Data`` sheet has every field for a date on one row, so a
    repeated date is resolved once for the whole row, not per field — unlike
    ``ingest.bloomberg.dedupe_dates``, which dedupes one field's series at a
    time. The message shape mirrors that function's.
    """
    repeated = frame[DATE_COLUMN].duplicated(keep="last")
    if not repeated.any():
        return frame, ()
    dates = sorted({d.date().isoformat() for d in frame.loc[repeated, DATE_COLUMN].dropna()})
    shown = ", ".join(dates[:5]) + (f" and {len(dates) - 5} more" if len(dates) > 5 else "")
    plural = "s" if len(dates) != 1 else ""
    note = f"{name}: {len(dates)} repeated date{plural} ({shown}); the last value on each was kept"
    return frame[~repeated].reset_index(drop=True), (note,)
