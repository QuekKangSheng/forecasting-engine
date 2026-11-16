"""The Data page warns when the portfolio's first rebalance would fall before the
long-run premium's prior was published."""

from streamlit.testing.v1 import AppTest


def _from_2000():
    import pandas as pd

    import bloomberg_extraction_panel

    bloomberg_extraction_panel._warn_of_an_unpublished_premium(
        pd.Series(pd.bdate_range("2000-01-03", periods=1500))
    )


def _from_2006():
    import pandas as pd

    import bloomberg_extraction_panel

    bloomberg_extraction_panel._warn_of_an_unpublished_premium(
        pd.Series(pd.bdate_range("2006-08-31", periods=1500))
    )


def test_data_whose_first_rebalance_precedes_the_prior_is_warned_of():
    app = AppTest.from_function(_from_2000).run(timeout=60)

    assert not app.exception
    (warning,) = app.warning
    assert "before May 2005" in warning.value


def test_data_from_2006_is_not_warned_of():
    app = AppTest.from_function(_from_2006).run(timeout=60)

    assert not app.exception
    assert not app.warning
