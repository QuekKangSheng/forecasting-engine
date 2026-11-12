"""Plain-language labels for open-schema column names.

A committed column is named ``{security}_{field}`` by
``extraction.bloomberg_csv.label()``, e.g. ``VIX_Index_PX_LAST``. A Bloomberg
export carries only the ticker in its metadata (``Security,VIX Index``), never a
long name, so the labels come from the lookup tables below.

**The wording in both tables is a working default, not team-agreed.** Change it
here and every display that labels a factor follows.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping

#: Ticker, as it appears in an export's ``Security`` metadata, to its label.
SECURITY_LABELS: Mapping[str, str] = {
    "SPX Index": "S&P 500",
    "LBUSTRUU Index": "US Aggregate Bond",
    "LEGATRUU Index": "Global Aggregate Bond",
    "VIX Index": "VIX",
    "LUACOAS Index": "US IG credit spread",
    "LF98OAS Index": "US HY credit spread",
    "USGGBE10 Index": "US 10y breakeven",
    "JPMVXYG7 Index": "G7 FX implied vol",
    # The rest of the 30 Sep 2026 Bloomberg pull, which showed as bare tickers such as
    # "JPMVXYGL Index (last price)" beside named signals in the same equation.
    "JPMVXYGL Index": "Global FX implied vol",
    "LF98TRUU Index": "US high yield bond",
    "DXY Curncy": "US dollar index",
    "USGG10YR Index": "US 10y Treasury yield",
    "USYC2Y10 Index": "US 2s10s curve",
}

#: Bloomberg field mnemonic to its label.
FIELD_LABELS: Mapping[str, str] = {
    "PX_LAST": "last price",
    "PX_BID": "bid",
    "TOT_RETURN_INDEX_GROSS_DVDS": "total return",
}

#: Bloomberg's market-sector "yellow keys". A ticker ends in one, so the first of
#: them in a column name marks where the security ends and the field begins.
#: Matched by case: Bloomberg writes ``Index``, and the ``INDEX`` inside
#: ``TOT_RETURN_INDEX_GROSS_DVDS`` is part of a field, not a key.
YELLOW_KEYS: frozenset[str] = frozenset({"Index", "Comdty", "Curncy", "Govt", "Corp", "Equity"})


def labeller(columns: Iterable[str]) -> Callable[[str], str]:
    """Return a function that labels a column, given every committed column.

    It needs the whole set because a field is only worth naming when two columns
    share a security: "S&P 500 (last price)" beside "S&P 500 (bid)", but just
    "VIX" when VIX is the only VIX column.
    """
    shared = Counter(
        security for security, _ in (_split(column) for column in columns) if security is not None
    )

    def label(column: str) -> str:
        security, field = _split(column)
        if security is None:
            return column.replace("_", " ")
        known = SECURITY_LABELS.get(security)
        if known is None:
            return f"{security} ({_field_label(field)})" if field else security
        if field and shared[security] > 1:
            return f"{known} ({_field_label(field)})"
        return known

    return label


def _split(column: str) -> tuple[str | None, str]:
    """``VIX_Index_PX_LAST`` → ``("VIX Index", "PX_LAST")``, or ``(None, "")``
    when no yellow key is present."""
    parts = column.split("_")
    for i, part in enumerate(parts):
        if part in YELLOW_KEYS and i > 0:
            return " ".join(parts[: i + 1]), "_".join(parts[i + 1 :])
    return None, ""


def _field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " "))
