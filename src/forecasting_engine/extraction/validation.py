"""Validating merged Bloomberg data: a pandera schema plus a plain report.

Pandera enforces structure — the date column is unique, non-null and sorted;
every data column is numeric and, for total-return index columns, positive. Everything
a real market can legitimately produce — a duplicate date from a revision, a
weekend row, a statistically unusual move — is reported rather than rejected, so a
human sees it before the file is used instead of it being silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pandera.pandas as pa

from forecasting_engine.extraction.bloomberg_csv import DATE_COLUMN, DATE_DISPLAY_FORMAT

#: Bloomberg field mnemonics that name a total-return index level, which must
#: be positive. This is a rough classifier by field name, not a real
#: per-security contract — there is no fixed set of securities here to build
#: one against.
POSITIVE_FIELD_MARKERS: tuple[str, ...] = ("TOT_RETURN",)

#: Field mnemonics checked only for being numeric. ``PX_LAST`` is Bloomberg's
#: last value for any series, not only prices: a curve such as ``USYC2Y10`` is
#: negative for long stretches. Everything else (spreads, vol, rates, ...)
#: falls back to SANE_RANGE below.
NUMERIC_FIELD_MARKERS: tuple[str, ...] = ("PX_",)

#: Robust z-score of a day-over-day change worth flagging for review. Financial
#: returns are fat-tailed; calibration on the real ten-year exports put a useful
#: review volume at eight rather than the textbook three.
MAD_THRESHOLD = 8.0
_MAD_TO_SIGMA = 0.6745

#: Generic bound for every other column. Loose on purpose — it is
#: a sanity net against a badly wrong export, not a documented per-signal range.
SANE_RANGE: tuple[float, float] = (-100.0, 10_000.0)

#: How many offending dates a report names outright. ``duplicate_dates`` and
#: ``weekend_rows`` stay exact counts; a file with thousands of weekend rows
#: should not print thousands of dates.
MAX_NAMED_DATES = 10


@dataclass(frozen=True)
class ValidationReport:
    """What the merged file looks like. Nothing here is auto-corrected."""

    duplicate_dates: int
    duplicate_date_values: tuple[str, ...]
    """The repeated dates themselves, capped at ``MAX_NAMED_DATES``."""
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
    return not is_positive_column(name) and any(
        marker in name for marker in NUMERIC_FIELD_MARKERS
    )


def build_schema(columns: list[str]) -> pa.DataFrameSchema:
    """A pandera schema for ``columns``, classifying each by its field name.

    The date column is not checked for uniqueness here — a repeated date is
    common (a revision republished under the same date) and is reported via
    ``duplicate_dates`` instead, the same "reported, not blocking" treatment
    every other gap in this file gets. Blocking on it would refuse a file for
    the one kind of duplicate this tool is explicitly designed to tolerate.
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
    """
    try:
        build_schema(list(frame.columns)).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        return [
            f"{row['column']}: {row['check']} (got {_format_failure(row['failure_case'])})"
            for _, row in exc.failure_cases.iterrows()
        ]
    return []


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


def _named_dates(dates: pd.Series) -> tuple[str, ...]:
    return tuple(d.date().isoformat() for d in dates.head(MAX_NAMED_DATES))


def _big_moves(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        if col == DATE_COLUMN:
            continue
        changes = pd.to_numeric(frame[col], errors="coerce").diff()
        scores = _robust_z(changes.dropna()).abs()
        flagged = _drop_rebounds(scores[scores > MAD_THRESHOLD], changes)
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
    """Collapse an anomalous value's opposite-direction rebound into its first flag."""
    positions = list(flagged.index)
    rebounds = set()
    for earlier, later in zip(positions, positions[1:], strict=False):
        adjacent = changes.index.get_loc(later) - changes.index.get_loc(earlier) == 1
        if adjacent and changes.loc[earlier] * changes.loc[later] < 0:
            rebounds.add(later)
    return flagged.drop(index=list(rebounds))
