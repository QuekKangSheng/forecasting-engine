"""Plain-language labels for open-schema column names."""

import pytest

from forecasting_engine.reporting.factor_labels import labeller


def test_a_known_security_uses_its_label():
    label = labeller(["VIX_Index_PX_LAST", "LUACOAS_Index_PX_LAST"])
    assert label("VIX_Index_PX_LAST") == "VIX"
    assert label("LUACOAS_Index_PX_LAST") == "US IG credit spread"


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("SPX_Index_PX_LAST", "S&P 500"),
        ("LBUSTRUU_Index_PX_LAST", "US Aggregate Bond"),
        ("LEGATRUU_Index_PX_LAST", "Global Aggregate Bond"),
        ("LF98OAS_Index_PX_LAST", "US HY credit spread"),
        ("USGGBE10_Index_PX_LAST", "US 10y breakeven"),
        ("JPMVXYG7_Index_PX_LAST", "G7 FX implied vol"),
    ],
)
def test_every_security_in_the_table_maps(column, expected):
    assert labeller([column])(column) == expected


def test_the_field_is_named_only_when_two_columns_share_a_security():
    label = labeller(["SPX_Index_PX_LAST", "SPX_Index_PX_BID", "VIX_Index_PX_LAST"])
    assert label("SPX_Index_PX_LAST") == "S&P 500 (last price)"
    assert label("SPX_Index_PX_BID") == "S&P 500 (bid)"
    assert label("VIX_Index_PX_LAST") == "VIX"


def test_total_return_is_a_known_field():
    label = labeller(["SPX_Index_PX_LAST", "SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS"])
    assert label("SPX_Index_TOT_RETURN_INDEX_GROSS_DVDS") == "S&P 500 (total return)"


def test_an_unknown_security_gets_a_tidied_label_not_the_raw_code():
    assert labeller(["ABC_Index_PX_LAST"])("ABC_Index_PX_LAST") == "ABC Index (last price)"


def test_an_unknown_field_is_shown_with_spaces_for_underscores():
    label = labeller(["SPX_Index_PX_LAST", "SPX_Index_PX_OPEN"])
    assert label("SPX_Index_PX_OPEN") == "S&P 500 (PX OPEN)"


def test_an_unknown_security_with_an_unknown_field_is_tidied_both_ways():
    assert labeller(["ABC_Comdty_PX_OPEN"])("ABC_Comdty_PX_OPEN") == "ABC Comdty (PX OPEN)"


@pytest.mark.parametrize("key", ["Index", "Comdty", "Curncy", "Govt", "Corp", "Equity"])
def test_every_yellow_key_splits_the_name(key):
    column = f"XYZ_{key}_PX_LAST"
    assert labeller([column])(column) == f"XYZ {key} (last price)"


def test_a_multi_word_security_keeps_every_word_before_the_yellow_key():
    column = "EUR_USD_Curncy_PX_LAST"
    assert labeller([column])(column) == "EUR USD Curncy (last price)"


def test_a_name_with_no_yellow_key_falls_back_to_spaces():
    assert labeller(["spx_close"])("spx_close") == "spx close"


def test_the_yellow_key_is_matched_by_case():
    # Bloomberg writes "Index". A filename-derived label is lower case, and the
    # uppercase INDEX inside a field name is not a yellow key either.
    column = "spx_index_total_return_TOT_RETURN_INDEX_GROSS_DVDS"
    assert labeller([column])(column) == "spx index total return TOT RETURN INDEX GROSS DVDS"


def test_a_column_the_labeller_was_not_built_with_still_gets_a_label():
    label = labeller(["VIX_Index_PX_LAST"])
    assert label("SPX_Index_PX_LAST") == "S&P 500"


def test_a_bare_security_with_no_field_is_labelled():
    assert labeller(["VIX_Index"])("VIX_Index") == "VIX"


def test_the_calendar_models_columns_read_as_words():
    # Built by the app, not read from an export, so there is no committed column to
    # build the labeller from.
    label = labeller(["LBUSTRUU_Index_TOT_RETURN_INDEX_GROSS_DVDS"])
    assert label("CAL_MONTH_END") == "Month-ends ahead"
    assert label("CAL_JOBS_FRIDAY") == "Jobs Fridays ahead"
    assert label("CAL_MON") == "Next day a Monday"
