"""Validating merged Bloomberg data: a pandera schema plus a plain report.

Pandera enforces structure — the date column is non-null and sorted; every
data column is numeric and, for total-return index columns, positive. A repeated
date is settled per file before that: a row that only repeats an earlier one is
dropped and the drop reported, and a date given two different values stops the
data being committed, since a model needs one value per date and nothing in the
file says which is right. Everything else a real market can legitimately produce
— a weekend row, a statistically unusual move — is reported rather than rejected,
so a human sees it before the file is used instead of it being silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import pandera.pandas as pa

from forecasting_engine.extraction.bloomberg_csv import (
    DATE_COLUMN,
    DATE_DISPLAY_FORMAT,
    MAX_NAMED_DATES,
    BloombergCsvExport,
    named_dates,
)
from forecasting_engine.ingest.align import TICKER_TRANSFORMS, Transform

#: Bloomberg field mnemonics that name a total-return index level, which must
#: be positive. This is a rough classifier by field name, not a real
#: per-security contract — there is no fixed set of securities here to build
#: one against.
POSITIVE_FIELD_MARKERS: tuple[str, ...] = ("TOT_RETURN",)

#: Field mnemonics checked only for being numeric. ``PX_LAST`` is Bloomberg's
#: last value for any series, not only prices: a curve such as ``USYC2Y10`` is
#: negative for long stretches. Everything else (spreads, vol, rates, ...)
#: falls back to SANE_RANGE below. Matched where a field name starts, so SPX's
#: own ticker does not make every SPX field one of these.
NUMERIC_FIELD_MARKERS: tuple[str, ...] = ("PX_",)

#: Robust z-score of a day-over-day change worth flagging for review. Financial
#: returns are fat-tailed; calibration on the real ten-year exports put a useful
#: review volume at eight rather than the textbook three.
MAD_THRESHOLD = 8.0
_MAD_TO_SIGMA = 0.6745

#: Generic bound for every other column. Loose on purpose — it is
#: a sanity net against a badly wrong export, not a documented per-signal range.
#: The lower bound allows for a deep yield-curve inversion quoted in basis
#: points (2s10s has gone below -100 bps), not just small negative spreads.
SANE_RANGE: tuple[float, float] = (-300.0, 10_000.0)

#: How many offending values a schema message names outright; the rest are
#: counted. One bad column in a ten-year file must not print every value in it.
MAX_NAMED_VALUES = 5


@dataclass(frozen=True)
class ValidationReport:
    """What the merged file looks like. Nothing here is auto-corrected."""

    duplicate_dates: int
    duplicate_date_values: tuple[str, ...]
    """The repeated dates themselves, capped at ``MAX_NAMED_DATES``. Exact
    counts stay in ``duplicate_dates`` and ``weekend_rows``."""
    weekend_rows: int
    weekend_date_values: tuple[str, ...]
    """The weekend dates themselves, capped at ``MAX_NAMED_DATES``."""
    missing_pct: dict[str, float]
    big_moves: pd.DataFrame
    """One row per (column, date) whose day-over-day change exceeded the threshold."""
    schema_errors: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.duplicate_dates
            or self.weekend_rows
            or not self.big_moves.empty
            or self.schema_errors
        )


def is_positive_column(name: str) -> bool:
    return any(marker in name for marker in POSITIVE_FIELD_MARKERS)


def is_numeric_only_column(name: str) -> bool:
    # A column is "{label}_{field}", so a field name starts just after an
    # underscore: "SPX_Index_PX_LAST" has a PX_ field, "SPX_Index_CUR_MKT_CAP"
    # only has "PX_" inside its ticker.
    return not is_positive_column(name) and any(
        f"_{marker}" in name for marker in NUMERIC_FIELD_MARKERS
    )


def build_schema(columns: list[str]) -> pa.DataFrameSchema:
    """A pandera schema for ``columns``, classifying each by its field name.

    The date column is not checked for uniqueness here: a repeated date is
    settled per file before the schema runs (``drop_exact_repeats``,
    ``repeated_date_error``), where the message can name the file and its
    security instead of a column.
    """
    schema_columns: dict[str, pa.Column] = {
        DATE_COLUMN: pa.Column(
            "datetime64[ns]",
            nullable=False,
            checks=pa.Check(lambda s: s.is_monotonic_increasing, error="dates must be ascending"),
        )
    }
    low, high = SANE_RANGE
    for name in columns:
        if name == DATE_COLUMN:
            continue
        if is_positive_column(name):
            checks = [pa.Check.gt(0)]
        elif is_numeric_only_column(name):
            checks = []
        else:
            checks = [pa.Check.in_range(low, high)]
        schema_columns[name] = pa.Column(float, checks=checks, nullable=True, coerce=True)
    return pa.DataFrameSchema(schema_columns)


def schema_errors(frame: pd.DataFrame) -> list[str]:
    """Pandera's schema check on ``frame`` alone, as plain messages.

    Meant to be called per source file, before merging: a file that fails
    here should be excluded the same way a file that fails to parse is,
    rather than let a bad value ride into the merged dataset.

    One message per column and check, naming the first few offending values
    and counting the rest, however many rows fail.
    """
    try:
        build_schema(list(frame.columns)).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases.groupby(["column", "check"], sort=False, dropna=False)
        return [
            f"{column}: {check} (got {_named_failures(group['failure_case'])})"
            for (column, check), group in cases
        ]
    return []


def drop_exact_repeats(export: BloombergCsvExport) -> BloombergCsvExport:
    """``export`` without the rows that repeat an earlier row exactly, with a
    note saying how many went.

    The same date with the same values is one observation written twice (a
    file appended to a copy of itself, a row pasted in again), so dropping the
    copy loses nothing. A date repeated with different values is left alone,
    for ``repeated_date_error``.
    """
    repeats = export.frame.duplicated()
    if not repeats.any():
        return export
    count = int(repeats.sum())
    note = (
        f"{export.filename}: {count} row{'s' if count != 1 else ''} repeated an earlier "
        f"row exactly, the same date with the same values, and "
        f"{'were' if count != 1 else 'was'} dropped."
    )
    frame = export.frame[~repeats].reset_index(drop=True)
    return replace(export, frame=frame, notes=(*export.notes, note))


def repeated_date_error(export: BloombergCsvExport) -> str | None:
    """Why ``export`` cannot be committed, if it gives one date different
    values, else None.

    Every model needs one value per date, and nothing in a file tells a
    revision from the figure it replaced, so this stops the commit rather than
    picking one. Run it after ``drop_exact_repeats``, so only real
    disagreements are left.
    """
    dates = export.frame[DATE_COLUMN]
    clashing = dates[dates.duplicated(keep=False) & dates.notna()].drop_duplicates()
    if clashing.empty:
        return None
    count = len(clashing)
    return (
        f"{export.filename}: {export.security or 'this file'} has different values on "
        f"the same date, on {count} date{'s' if count != 1 else ''}: "
        f"{named_dates(clashing.sort_values())}. Keep one row for each date in the file "
        "and upload it again; until then this data cannot be committed."
    )


def validate(frame: pd.DataFrame) -> ValidationReport:
    """Run the schema and the plain-pandas report against ``frame``."""
    dates = frame[DATE_COLUMN]
    dup_mask = dates.duplicated()
    weekend_mask = dates.dt.dayofweek.isin([5, 6])
    return ValidationReport(
        duplicate_dates=int(dup_mask.sum()),
        duplicate_date_values=_named_dates(dates[dup_mask]),
        weekend_rows=int(weekend_mask.sum()),
        weekend_date_values=_named_dates(dates[weekend_mask]),
        missing_pct={
            col: float(frame[col].isna().mean() * 100)
            for col in frame.columns
            if col != DATE_COLUMN
        },
        big_moves=_big_moves(frame),
        schema_errors=schema_errors(frame),
    )


def _format_failure(value: object) -> str:
    """A schema failure's offending value, as the report shows dates elsewhere.

    A date failure gets dd/mm/yyyy with no time, matching every other date in
    this tool; anything else is shown as pandas/pandera would print it.
    """
    if isinstance(value, pd.Timestamp):
        return value.strftime(DATE_DISPLAY_FORMAT)
    return repr(value)


def _named_failures(values: pd.Series) -> str:
    """The first few distinct offending values, and how many failed in all."""
    distinct = values.drop_duplicates()
    shown = ", ".join(_format_failure(v) for v in distinct.head(MAX_NAMED_VALUES))
    if len(values) > min(len(distinct), MAX_NAMED_VALUES):
        shown += f"; {len(values):,} values in all"
    return shown


def _named_dates(dates: pd.Series) -> tuple[str, ...]:
    return tuple(d.date().isoformat() for d in dates.head(MAX_NAMED_DATES))


def _big_moves(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        if col == DATE_COLUMN:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        changes = values.diff()
        scored = _scored_changes(col, values)
        scores = _robust_z(scored.dropna()).abs()
        flagged = _drop_rebounds(scores[scores > MAD_THRESHOLD], scored)
        for idx, score in flagged.items():
            rows.append(
                {
                    "column": col,
                    "date": frame.loc[idx, DATE_COLUMN],
                    "change": changes.loc[idx],
                    "robust_score": score,
                }
            )
    return pd.DataFrame(rows, columns=["column", "date", "change", "robust_score"])


def _compounds(name: str) -> bool:
    """Whether column ``name`` is a price or total-return index, whose moves grow
    with its level.

    The same series the models take as log returns (``ingest.align``): any
    total-return index field, and the price of a ticker classified as one. A
    column is "{label}_{field}" and a label starts with its ticker, so "SPX_"
    starts every SPX column. Rates, curve points, spreads and volatility are
    not, and keep their plain change.
    """
    return any(marker in name for marker in POSITIVE_FIELD_MARKERS) or any(
        name.startswith(f"{ticker}_")
        for ticker, transform in TICKER_TRANSFORMS.items()
        if transform is Transform.LOG_RETURN
    )


def _scored_changes(name: str, values: pd.Series) -> pd.Series:
    """The day-over-day changes column ``name`` is scored on.

    A price that rose fivefold moves five times as many points on an ordinary
    day, so scored in points its early crashes look small and its late ordinary
    days look extreme. A compounding series is scored on its log change instead,
    the size of a move relative to where it was. A column with a value at or
    below zero cannot be, and falls back to its plain change so a bad value is
    still scored.
    """
    if _compounds(name) and (values.dropna() > 0).all():
        return np.log(values).diff()
    return values.diff()


def _robust_z(values: pd.Series) -> pd.Series:
    """Median-absolute-deviation score, with a zero-MAD fallback."""
    median = values.median()
    if np.isnan(median):
        return pd.Series(0.0, index=values.index)
    spread = (values - median).abs().median()
    if not spread or np.isnan(spread):
        spread = (values - median).abs().mean()
    if not spread or np.isnan(spread):
        return pd.Series(0.0, index=values.index)
    return _MAD_TO_SIGMA * (values - median) / spread


def _drop_rebounds(flagged: pd.Series, changes: pd.Series) -> pd.Series:
    """Collapse an anomalous value's opposite-direction rebound into its first flag.

    A day is a rebound only of the last day that was reported. Compared with the
    day before it instead, a dropped rebound passed its own status on: in March
    2020 VIX rose 21.6, fell 17.6, then rose 24.9 to a new high, and the last
    move, the largest in twenty years, was dropped as the rebound of a rebound."""
    rebounds = []
    reported = None
    for position in flagged.index:
        if reported is not None:
            adjacent = changes.index.get_loc(position) - changes.index.get_loc(reported) == 1
            if adjacent and changes.loc[reported] * changes.loc[position] < 0:
                rebounds.append(position)
                continue
        reported = position
    return flagged.drop(index=rebounds)
