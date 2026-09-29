"""The Fama-French five-factor daily file: download, parse, and keep a copy.

Ken French publishes the factors as a zip around a CSV with explanatory prose
before and after the table and no declared row count. The only reliable
marker of a data row is a first field of eight digits, ``YYYYMMDD``, so rows
are kept or dropped on that basis rather than by counting header lines.

The download is never used straight from the wire. Every copy is written
under ``data/fama_french`` named by its content hash and handed back as a
``SourceFile``, for two reasons. The library is updated monthly, so the same
URL returns different rows across a sprint and a run has to be able to name
the exact file it used. And the sponsor's brief is manual data with no
automated feeds: a fetch only ever happens because a person pressed Run on
the Models page, and what it fetched is kept like any other upload.

Nothing here imports Streamlit; the dashboard is a caller, not a dependency.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from forecasting_engine.ingest.provenance import SourceFile

URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

DATE_COLUMN = "Date"
FILE_NAME = "fama_french_5_factors_daily.csv"

#: Where downloaded copies live, named by content hash. Gitignored.
DEFAULT_CACHE_DIR = Path("data/fama_french")

#: A saved copy younger than this is used as is; an older one prompts a download.
MAX_AGE = timedelta(days=30)

_DATE_RE = re.compile(r"^\d{8}$")
_TIMEOUT_SECONDS = 30


class FactorFetchError(Exception):
    """The factor file could not be downloaded or read. The message says why."""


@dataclass(frozen=True)
class FactorFile:
    """A parsed factor file and the stored copy it came from."""

    frame: pd.DataFrame
    source: SourceFile


def fetch() -> pd.DataFrame:
    """Download and parse the factor file. Raises ``FactorFetchError`` on failure."""
    try:
        with urlopen(URL, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed URL
            return parse(response.read())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise FactorFetchError(f"{type(exc).__name__}: {exc}") from exc


def parse(zip_bytes: bytes) -> pd.DataFrame:
    """Parse the factor CSV out of the downloaded zip's bytes."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("Fama-French zip contained no CSV file")
        raw = archive.read(names[0]).decode("utf-8")

    lines = raw.splitlines()
    header = next((line for line in lines if "Mkt-RF" in line), None)
    if header is None:
        raise ValueError("Fama-French CSV has no 'Mkt-RF' header row")
    factor_columns = [c.strip() for c in header.split(",") if c.strip()]

    data_rows = [line.split(",") for line in lines if _DATE_RE.match(line.split(",", 1)[0].strip())]
    frame = pd.DataFrame(data_rows, columns=[DATE_COLUMN, *factor_columns])
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], format="%Y%m%d")
    for column in factor_columns:
        frame[column] = pd.to_numeric(frame[column])
    return frame


def restrict_to(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """Factors from ``start`` through ``end`` inclusive.

    The full series runs back to 1963; a caller only wants the span the
    signal data covers. An ``end`` past the file's last row just returns what
    exists, since the published file lags the calendar by about a month.
    """
    mask = (frame[DATE_COLUMN] >= start) & (frame[DATE_COLUMN] <= end)
    return frame.loc[mask].reset_index(drop=True)


def save(frame: pd.DataFrame, cache_dir: Path = DEFAULT_CACHE_DIR) -> FactorFile:
    """Write ``frame`` under its content hash and return it with its provenance.

    An identical file already stored is not rewritten, so re-downloading an
    unchanged month costs nothing and resolves to the same hash.
    """
    data = frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
    source = SourceFile.of(FILE_NAME, data)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{source.sha256}.csv"
    if path.exists():
        path.touch()  # its age is when it was last fetched, not first
    else:
        path.write_bytes(data)
    return FactorFile(frame=frame, source=SourceFile.of(FILE_NAME, data, path=path))


def load_latest(cache_dir: Path = DEFAULT_CACHE_DIR) -> FactorFile | None:
    """The most recently stored copy, or None when nothing has been downloaded."""
    if not cache_dir.is_dir():
        return None
    stored = sorted(cache_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not stored:
        return None
    path = stored[-1]
    data = path.read_bytes()
    frame = pd.read_csv(io.BytesIO(data))
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], format="ISO8601")
    return FactorFile(frame=frame, source=SourceFile.of(FILE_NAME, data, path=path))


def download(cache_dir: Path = DEFAULT_CACHE_DIR) -> FactorFile:
    """Fetch today's file and keep it. Raises ``FactorFetchError`` on failure."""
    return save(fetch(), cache_dir)


@dataclass(frozen=True)
class ResolvedFactors:
    file: FactorFile
    warning: str | None = None
    """Set when a download failed and an older saved copy is used instead."""


def resolve(cache_dir: Path = DEFAULT_CACHE_DIR, *, now: datetime | None = None) -> ResolvedFactors:
    """The factor file a run should use.

    The saved copy if it is younger than ``MAX_AGE``; otherwise a fresh
    download, falling back to the saved copy with a warning if that fails.
    Raises ``FactorFetchError`` when there is neither.
    """
    saved = load_latest(cache_dir)
    if saved is not None and _age(saved, now) < MAX_AGE:
        return ResolvedFactors(saved)
    try:
        return ResolvedFactors(download(cache_dir))
    except FactorFetchError as exc:
        if saved is None:
            raise
        fetched = datetime.fromtimestamp(saved.source.path.stat().st_mtime)
        return ResolvedFactors(
            saved,
            warning=(
                f"Could not download a newer factor file ({exc}), so the copy saved on "
                f"{fetched:%d/%m/%Y} is used instead."
            ),
        )


def _age(factors: FactorFile, now: datetime | None) -> timedelta:
    fetched = datetime.fromtimestamp(factors.source.path.stat().st_mtime)
    return (now or datetime.now()) - fetched
