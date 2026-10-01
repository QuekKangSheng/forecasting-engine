"""docs/methodology.md's parameters table must match the code, row for row."""

import importlib
import re
from datetime import timedelta
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[2] / "docs" / "methodology.md"
_ROW = re.compile(r"^\| `(?P<name>\w+)` \| `(?P<module>[\w.]+)` \| `(?P<value>.+)` \|$")

#: Every constant the table must document. A new methodology constant belongs
#: here and in the doc's table.
DOCUMENTED = {
    ("forecasting_engine.ingest.upload", "MAX_UPLOAD_BYTES"),
    ("forecasting_engine.extraction.validation", "PRICE_FIELD_MARKERS"),
    ("forecasting_engine.extraction.validation", "SANE_RANGE"),
    ("forecasting_engine.extraction.validation", "MAD_THRESHOLD"),
    ("forecasting_engine.extraction.targets", "TARGET_TICKERS"),
    ("forecasting_engine.extraction.targets", "PREFERRED_FIELD"),
    ("forecasting_engine.ingest.align", "MAX_STALENESS"),
    ("forecasting_engine.ingest.align", "PRODUCTION_LAG_DAYS"),
    ("forecasting_engine.ingest.align", "TICKER_TRANSFORMS"),
    ("forecasting_engine.ingest.align", "UNCLASSIFIED_TRANSFORM"),
    ("forecasting_engine.ingest.align", "_TOTAL_RETURN_FIELD"),
    ("forecasting_engine.ingest.align", "_PRICE_FIELD"),
    ("forecasting_engine.ingest.align", "_QUOTE_FIELDS"),
    ("model_settings", "HORIZONS"),
    ("model_settings", "DEFAULT_HORIZON"),
    ("model_settings", "EMBARGO_DAYS"),
    ("model_settings", "DEFAULT_TRAIN_WINDOW"),
    ("model_settings", "DEFAULT_TEST_WINDOW"),
    ("model_settings", "DEFAULT_MAX_TERMS"),
    ("forecasting_engine.validation.splitters", "TUNING_ROWS"),
    ("forecasting_engine.features.screening", "INCLUSION_THRESHOLD"),
    ("forecasting_engine.models.famafrench", "FACTOR_COLUMNS"),
    ("forecasting_engine.models.famafrench", "_MIN_TRAINING_ROWS"),
    ("forecasting_engine.ingest.fama_french", "MAX_AGE"),
    ("forecasting_engine.ingest.fama_french", "_TIMEOUT_SECONDS"),
    ("forecasting_engine.models.polynomial", "MAX_DEGREE"),
    ("forecasting_engine.models.polynomial", "CANDIDATE_DEGREES"),
    ("forecasting_engine.models.polynomial", "CANDIDATE_REGULARIZERS"),
    ("forecasting_engine.models.polynomial", "CLIP_SD"),
    ("forecasting_engine.models.polynomial", "INNER_CV_SPLITS"),
    ("forecasting_engine.models.polynomial", "_REGULARIZERS"),
    ("forecasting_engine.models.polynomial", "_N_ALPHAS"),
    ("forecasting_engine.models.polynomial", "_ALPHA_EPS"),
    ("forecasting_engine.models.polynomial", "_MIN_TRAINING_ROWS"),
    ("forecasting_engine.models.boosted", "_FIXED_PARAMS"),
    ("forecasting_engine.models.boosted", "_LEAF_KEYS"),
    ("forecasting_engine.models.boosted", "SEARCH_SPACE"),
    ("forecasting_engine.models.boosted", "LEAF_CAP_SHARE"),
    ("forecasting_engine.models.boosted", "N_TRIALS"),
    ("forecasting_engine.models.boosted", "RETUNE_TRIALS"),
    ("forecasting_engine.models.boosted", "RETUNE_EVERY"),
    ("forecasting_engine.models.boosted", "MIN_TUNING_FOLDS"),
    ("forecasting_engine.models.boosted", "_MIN_TRAINING_ROWS"),
    ("forecasting_engine.validation.crash", "FLAG_PERCENTILE"),
    ("forecasting_engine.validation.crash", "TAIL_STD_MULTIPLE"),
    ("forecasting_engine.validation.pbo", "N_BLOCKS"),
    ("forecasting_engine.validation.gates", "OOS_RANK_IC_GATE"),
    ("forecasting_engine.validation.gates", "PBO_GATE"),
    ("forecasting_engine.reporting.model_metrics", "SIGNIFICANCE_SE_MULTIPLE"),
    ("forecasting_engine.portfolio.performance", "TRADING_DAYS_PER_YEAR"),
    ("forecasting_engine.portfolio.performance", "RISK_FREE_RATE"),
    ("forecasting_engine.portfolio.backtest", "BASELINE_WEIGHTS"),
    ("forecasting_engine.portfolio.backtest", "DEFAULT_COSTS_BPS"),
    ("forecasting_engine.portfolio.backtest", "REBALANCE_FREQUENCY"),
}


def _table() -> dict[tuple[str, str], str]:
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Parameters", 1)[1]
    rows = {}
    for line in section.splitlines():
        match = _ROW.match(line.strip())
        if match:
            key = (match["module"], match["name"])
            assert key not in rows, f"{key} is listed twice"
            rows[key] = match["value"]
    return rows


def test_the_table_lists_exactly_the_documented_constants():
    assert set(_table()) == DOCUMENTED


@pytest.mark.parametrize(("module", "name"), sorted(DOCUMENTED))
def test_each_documented_value_matches_the_code(module, name):
    documented = eval(_table()[(module, name)], {"timedelta": timedelta})  # noqa: S307 - our own doc
    actual = getattr(importlib.import_module(module), name)
    assert actual == documented, (
        f"{module}.{name} is {actual!r} but docs/methodology.md says {documented!r}"
    )
