"""The Data page warns when the portfolio's first rebalance would fall before the
long-run premium's prior was published."""

from streamlit.testing.v1 import AppTest


def _from_2000():
    import pandas as pd

    import bloomberg_extraction_panel
    from forecasting_engine.extraction.targets import TargetRole

    prices = pd.Series(1.0, index=pd.bdate_range("2000-01-03", periods=1500))
    bloomberg_extraction_panel._warn_of_an_unpublished_premium(
        {TargetRole.EQUITY: prices, TargetRole.BOND: prices}
    )


def _from_2006():
    import pandas as pd

    import bloomberg_extraction_panel
    from forecasting_engine.extraction.targets import TargetRole

    prices = pd.Series(1.0, index=pd.bdate_range("2006-08-31", periods=1500))
    bloomberg_extraction_panel._warn_of_an_unpublished_premium(
        {TargetRole.EQUITY: prices, TargetRole.BOND: prices}
    )


def _one_target_from_2000():
    import pandas as pd

    import bloomberg_extraction_panel
    from forecasting_engine.extraction.targets import TargetRole

    prices = pd.Series(1.0, index=pd.bdate_range("2000-01-03", periods=1500))
    bloomberg_extraction_panel._warn_of_an_unpublished_premium({TargetRole.EQUITY: prices})


def test_data_whose_first_rebalance_precedes_the_prior_is_warned_of():
    app = AppTest.from_function(_from_2000).run(timeout=60)

    assert not app.exception
    (warning,) = app.warning
    assert "before May 2005" in warning.value


def test_data_from_2006_is_not_warned_of():
    app = AppTest.from_function(_from_2006).run(timeout=60)

    assert not app.exception
    assert not app.warning


def test_data_with_one_target_picked_is_not_warned_of_as_no_portfolio_rebalances_on_it():
    app = AppTest.from_function(_one_target_from_2000).run(timeout=60)

    assert not app.exception
    assert not app.warning
