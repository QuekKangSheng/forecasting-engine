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
    no_security_message,
    parse_lines,
    read_dates,
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

#: What openpyxl raises for bytes that are not a workbook: not a zip at all, a
#: zip with no workbook inside (a KeyError for the missing part), or a part
#: whose XML is broken (a SyntaxError, from either XML parser it may use).
_UNREADABLE = (zipfile.BadZipFile, KeyError, OSError, SyntaxError)


class BloombergXlsxError(ValueError):
    """A file that is not shaped like a Bloomberg workbook export."""


def read_export(filename: str, data: bytes) -> BloombergCsvExport:
    """Parse one Bloomberg ``.xlsx`` workbook export.

    Raises ``BloombergXlsxError`` if malformed or if a formula has no saved
    value. Returns the same ``BloombergCsvExport`` shape
    ``bloomberg_csv.read_export`` does, so both readers feed the same ``merge()``.
    """
    # Opening and reading both stay inside the try: read-only mode parses a
    # sheet's XML only as its rows are read.
    try:
        book = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
        try:
            two_sheet = DATA_SHEET in book.sheetnames
            if two_sheet:
                rows = list(book[DATA_SHEET].iter_rows(values_only=True))
                security = _security(book)
                metadata = _metadata(book)
            else:
                rows = list(book.worksheets[0].iter_rows(values_only=True))
        finally:
            book.close()
    except _UNREADABLE as exc:
        raise BloombergXlsxError(f"{filename}: not a readable .xlsx file ({exc})") from exc

    if not two_sheet:
        _refuse_unsaved(filename, data, rows)
        return _first_sheet(filename, rows)

    width = _named_width(rows[0]) if rows else 0
    if not width:
        raise BloombergXlsxError(f"{filename}: the {DATA_SHEET!r} sheet has no header row")

    # Cut to the header's width: one stray cell to the right of the table makes
    # every row that wide, and the empty header cells it leaves name no field.
    rows = [row[:width] for row in rows]
    header = [str(cell) for cell in rows[0]]
    repeated = sorted({name for name in header if header.count(name) > 1})
    if repeated:
        raise BloombergXlsxError(
            f"{filename}: the {DATA_SHEET!r} sheet has more than one column headed "
            f"{', '.join(repr(name) for name in repeated)}. Keep one column per field and "
            "upload it again."
        )
    if len(rows) < 2:
        raise BloombergXlsxError(
            f"{filename}: the {DATA_SHEET!r} sheet has a header but no rows of data."
        )
    frame = pd.DataFrame(rows[1:], columns=header)
    frame = frame.rename(columns={frame.columns[0]: DATE_COLUMN})
    fields = [c for c in frame.columns if c != DATE_COLUMN]
    if not security and len(fields) > 1:
        raise BloombergXlsxError(
            no_security_message(filename, len(fields), f"no Security in a {METADATA_SHEET!r} sheet")
        )
    # A date cell holds a datetime, or an Excel serial number when it is shown
    # as a plain number; text dates go through the same reading as a CSV's.
    try:
        dates = frame[DATE_COLUMN].map(_date_text).astype("string")
        frame[DATE_COLUMN] = read_dates(filename, dates, metadata)
    except BloombergCsvError as exc:
        raise BloombergXlsxError(str(exc)) from exc

    # Read straight from cell values rather than through pandas' CSV parser,
    # so Bloomberg's "#N/A N/A" placeholder (and anything else non-numeric)
    # needs coercing to NaN explicitly here — pd.read_csv does this for the
    # CSV reader automatically via its default NA-string list, this doesn't.
    for col in fields:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    lbl = label(security, filename)
    frame = frame.rename(columns={field: f"{lbl}_{field}" for field in fields})

    return BloombergCsvExport(filename=filename, security=security, frame=frame)


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
        width = _named_width(rows[header_idx])
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


def _named_width(header: tuple) -> int:
    """How many cells of a header row there are up to its last non-empty one."""
    return max((i + 1 for i, cell in enumerate(header) if cell is not None), default=0)


def _date_text(cell: object) -> object:
    if isinstance(cell, bool):
        return cell
    if isinstance(cell, int | float):
        try:
            cell = from_excel(cell)
        except (OverflowError, ValueError):
            # Too large to be a serial date (20240102, say): left as written,
            # for the date reader to make sense of or refuse.
            return cell
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


def _metadata(book: openpyxl.Workbook) -> dict[str, str]:
    """The ``Metadata`` sheet's rows as text, keys lower-cased: the Start Date,
    End Date and Period lines the date reader weighs."""
    if METADATA_SHEET not in book.sheetnames:
        return {}
    found: dict[str, str] = {}
    for row in book[METADATA_SHEET].iter_rows(values_only=True):
        if row and row[0] is not None:
            value = row[1] if len(row) > 1 else None
            found.setdefault(str(row[0]).strip().lower(), _text(value).strip())
    return found
