"""File-level validation of an uploaded CSV or XLSX file.

This is the gate between a browser upload and the rest of the engine. It
answers one question — is this a file we can work with at all? — checking the
extension and the size limit before anything tries to parse it. Parsing
itself is per-format: ``extraction.bloomberg_csv``/``extraction.bloomberg_xlsx``
for a raw Bloomberg export, or ``parse_csv`` below for an already-merged CSV.

It deliberately says nothing about *columns*. Whether the file carries the
signals the contract requires is schema validation's job (FYP-8), which runs on
the frame this module returns. Keeping the two apart means a portfolio manager
who uploads a holiday photo gets "that is not a CSV" rather than a list of
eight missing columns.

Nothing here imports Streamlit; the dashboard is a caller, not a dependency.
See ``docs/data-specification.md`` for the contract these rules enforce.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

from forecasting_engine.ingest.provenance import SourceFile

#: Maximum accepted upload. Roughly six times the largest file the data
#: specification can plausibly produce — see the "File size" section there
#: before changing it, and change it in both places.
#:
#: Decimal megabytes, not binary. Streamlit labels file sizes in decimal MB, so
#: computing in MiB would have the uploader call a file 26.5MB while our error
#: called the same file 25.2 MB.
MAX_UPLOAD_BYTES: int = 25 * 1_000_000

#: Where accepted uploads are stored, keyed by content hash. Gitignored.
DEFAULT_UPLOADS_DIR: Path = Path("data/uploads")

DATE_COLUMN = "date"

_ACCEPTED_SUFFIXES = (".csv", ".xlsx")
_EXPORT_HINT = "Re-export it as .csv or .xlsx."


class UploadError(Exception):
    """An upload we are refusing.

    ``message`` is written for a portfolio manager and is safe to render
    directly in the dashboard.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class FileTypeError(UploadError):
    """The file is not a CSV."""


class FileSizeError(UploadError):
    """The file is larger than the documented limit."""


class CsvParseError(UploadError):
    """The file ends in .csv but its contents are not delimited text."""


@dataclass(frozen=True)
class AcceptedUpload:
    """A file that passed every file-level check."""

    frame: pd.DataFrame
    source: SourceFile

    @property
    def row_count(self) -> int:
        return len(self.frame)


def check_extension(filename: str) -> None:
    """Raise ``FileTypeError`` unless ``filename`` ends in .csv or .xlsx."""
    suffix = Path(filename).suffix
    if suffix.lower() in _ACCEPTED_SUFFIXES:
        return
    got = f"a {suffix} file" if suffix else "a file with no extension"
    accepted = " or ".join(_ACCEPTED_SUFFIXES)
    raise FileTypeError(
        f"Only {accepted} files are accepted. {filename!r} is {got}. {_EXPORT_HINT}"
    )


def check_size(size_bytes: int, *, filename: str) -> None:
    """Raise ``FileSizeError`` if ``size_bytes`` exceeds the documented limit."""
    if size_bytes <= MAX_UPLOAD_BYTES:
        return

    got, limit = _megabytes(size_bytes), _megabytes(MAX_UPLOAD_BYTES)
    if got == limit:
        # Barely over. Rounded megabytes would read "25.0 MB is over the
        # 25.0 MB limit", so be exact instead of merely consistent.
        raise FileSizeError(
            f"{filename!r} is {size_bytes:,} bytes, just over the "
            f"{limit} limit of {MAX_UPLOAD_BYTES:,} bytes."
        )
    raise FileSizeError(f"{filename!r} is {got}, which is over the {limit} limit.")


def parse_csv(data: bytes) -> pd.DataFrame:
    """Parse ``data`` as a UTF-8 CSV, or raise ``CsvParseError``."""
    try:
        return pd.read_csv(BytesIO(data), encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CsvParseError(
            f"The file could not be read as a CSV ({type(exc).__name__}). "
            f"Renaming a file does not change its format. {_EXPORT_HINT}"
        ) from exc


def accept_upload(
    filename: str,
    data: bytes,
    *,
    uploads_dir: Path | None = DEFAULT_UPLOADS_DIR,
) -> AcceptedUpload:
    """Validate an upload and, unless ``uploads_dir`` is None, persist it.

    Checks run cheapest-first, so an oversized file is refused before we spend
    anything trying to parse it. Accepted bytes are written to
    ``uploads_dir/<sha256>.csv``; an identical re-upload resolves to the same
    path and is not rewritten.
    """
    check_extension(filename)
    check_size(len(data), filename=filename)
    frame = parse_csv(data)

    source = SourceFile.of(filename, data)
    if uploads_dir is None:
        return AcceptedUpload(frame=frame, source=source)

    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / f"{source.sha256}.csv"
    if not path.exists():
        path.write_bytes(data)

    return AcceptedUpload(frame=frame, source=SourceFile.of(filename, data, path=path))


def keep_source(
    filename: str, data: bytes, *, uploads_dir: Path = DEFAULT_UPLOADS_DIR
) -> SourceFile:
    """Keep an uploaded file's bytes as they arrived, under their content hash and
    the file's own extension (``<sha256>.xlsx``), and return where they are. The
    same bytes resolve to the same path and are not rewritten, so keeping a file on
    every rerun of a page costs one existence check."""
    source = SourceFile.of(filename, data)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = uploads_dir / f"{source.sha256}{Path(filename).suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return SourceFile.of(filename, data, path=path)


def frame_csv_bytes(frame: pd.DataFrame) -> bytes:
    """``frame`` as canonical CSV: no index, ISO dates, ``\n`` line endings, UTF-8.
    Fixed choices, so the same data gives the same bytes on every machine."""
    return frame.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode("utf-8")


def frame_sha256(frame: pd.DataFrame) -> str:
    """The content hash of ``frame``'s canonical CSV: the hash a committed dataset
    is stored and logged under, and the one a run records as its data."""
    return SourceFile.of("", frame_csv_bytes(frame)).sha256


def date_range(frame: pd.DataFrame) -> tuple[str, str] | None:
    """First and last date as ISO strings, or None if they cannot be read.

    Best effort, for the upload confirmation message only. A file with no
    usable date column is still a valid *file*; it is schema validation that
    decides whether it is a valid dataset.

    Reads ISO dates only, as the contract and every other date reader do.
    Letting pandas infer a format here would quote a guessed range for
    ``01/02/2024`` in the confirmation, one line above the validator refusing
    that same date as ambiguous.
    """
    if DATE_COLUMN not in frame.columns:
        return None
    dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce", format="ISO8601").dropna()
    if dates.empty:
        return None
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _megabytes(size_bytes: int) -> str:
    return f"{size_bytes / 1_000_000:.1f} MB"
