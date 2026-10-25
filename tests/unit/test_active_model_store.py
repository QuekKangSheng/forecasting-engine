"""The DuckDB active-model-per-role log."""

from dataclasses import replace
from datetime import datetime

import duckdb
import pandas as pd
import pytest

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.ingest.align import Transform
from forecasting_engine.models.base import ModelDescription
from forecasting_engine.reporting.model_metrics import ModelRunResult
from forecasting_engine.reporting.polynomial_function import Origin
from forecasting_engine.store.active_model import (
    ActiveFunction,
    get_active_function,
    get_active_model,
    get_active_model_forecast,
    get_active_model_signal,
    set_active_model,
    settings_match,
)
from forecasting_engine.validation.crash import CrashDiagnostics

SETTINGS = {"horizon": 5, "train_window": 120, "test_window": 20, "dataset_fingerprint": "fp-1"}


def _result(ic=0.05, oos_rank_ic=0.03, rmse=0.01, forecast=None):
    return ModelRunResult(
        ic=ic,
        oos_rank_ic=oos_rank_ic,
        rmse=rmse,
        crash=CrashDiagnostics(recall=0.5, precision=0.5, f1=0.5, n_true_tail_days=10),
        forecast=forecast,
    )


@pytest.fixture
def db(tmp_path):
    return tmp_path / "history" / "test.duckdb"


def test_reading_an_absent_database_creates_it_and_returns_nothing(db):
    assert get_active_model(TargetRole.EQUITY, db_path=db) is None
    assert db.exists()


def test_setting_an_active_model_round_trips_every_field(db):
    at = datetime(2026, 8, 31, 9, 30, 0)
    set_active_model(
        TargetRole.EQUITY,
        "Machine Learning",
        _result(),
        high_risk=False,
        db_path=db,
        set_at=at,
        **SETTINGS,
    )

    row = get_active_model(TargetRole.EQUITY, db_path=db)
    assert row.role == TargetRole.EQUITY
    assert row.model_name == "Machine Learning"
    assert row.ic == 0.05
    assert row.oos_rank_ic == 0.03
    assert row.rmse == 0.01
    assert row.high_risk is False
    assert row.set_at == at
    assert row.horizon == 5
    assert row.train_window == 120
    assert row.test_window == 20
    assert row.dataset_fingerprint == "fp-1"


def test_setting_a_new_active_model_supersedes_the_prior_one(db):
    set_active_model(
        TargetRole.EQUITY,
        "Polynomial",
        _result(),
        high_risk=False,
        db_path=db,
        set_at=datetime(2026, 8, 30),
        **SETTINGS,
    )
    set_active_model(
        TargetRole.EQUITY,
        "Machine Learning",
        _result(),
        high_risk=False,
        db_path=db,
        set_at=datetime(2026, 8, 31),
        **SETTINGS,
    )

    assert get_active_model(TargetRole.EQUITY, db_path=db).model_name == "Machine Learning"


def test_each_role_has_its_own_active_model(db):
    set_active_model(
        TargetRole.EQUITY, "Machine Learning", _result(), high_risk=False, db_path=db, **SETTINGS
    )
    set_active_model(
        TargetRole.BOND, "Polynomial", _result(), high_risk=False, db_path=db, **SETTINGS
    )

    assert get_active_model(TargetRole.EQUITY, db_path=db).model_name == "Machine Learning"
    assert get_active_model(TargetRole.BOND, db_path=db).model_name == "Polynomial"


def test_a_high_risk_selection_is_recorded_as_such(db):
    set_active_model(
        TargetRole.BOND,
        "Polynomial",
        _result(oos_rank_ic=0.01),
        high_risk=True,
        db_path=db,
        **SETTINGS,
    )

    assert get_active_model(TargetRole.BOND, db_path=db).high_risk is True


def test_a_database_made_when_active_models_recorded_pbo_still_works(db):
    # Older databases have a nullable pbo column; a new row simply leaves it empty.
    db.parent.mkdir(parents=True)
    with duckdb.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE active_models (role TEXT NOT NULL, model_name TEXT NOT NULL, "
            "ic DOUBLE NOT NULL, oos_rank_ic DOUBLE NOT NULL, rmse DOUBLE NOT NULL, "
            "pbo DOUBLE, high_risk BOOLEAN NOT NULL, set_at TIMESTAMP NOT NULL)"
        )
    set_active_model(
        TargetRole.EQUITY, "FF5 Benchmark", _result(), high_risk=False, db_path=db, **SETTINGS
    )

    assert get_active_model(TargetRole.EQUITY, db_path=db).model_name == "FF5 Benchmark"


# --- forecast persistence -----------------------------------------------------------


def test_setting_an_active_model_saves_its_forecast_series(db):
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    forecast = pd.Series([0.01, -0.02, 0.03], index=dates)
    set_active_model(
        TargetRole.EQUITY,
        "Polynomial",
        _result(forecast=forecast),
        high_risk=False,
        db_path=db,
        **SETTINGS,
    )

    saved = get_active_model_forecast(TargetRole.EQUITY, db_path=db)
    pd.testing.assert_series_equal(saved, forecast, check_names=False, check_freq=False)


def test_a_forecasts_nan_rows_are_kept_so_fold_spacing_stays_intact(db):
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    forecast = pd.Series([0.01, float("nan"), 0.03], index=dates)
    set_active_model(
        TargetRole.EQUITY,
        "Polynomial",
        _result(forecast=forecast),
        high_risk=False,
        db_path=db,
        **SETTINGS,
    )

    saved = get_active_model_forecast(TargetRole.EQUITY, db_path=db)
    assert len(saved) == 3
    assert pd.isna(saved.iloc[1])


def test_with_no_result_forecast_nothing_is_saved_and_none_is_returned(db):
    set_active_model(
        TargetRole.EQUITY, "Polynomial", _result(), high_risk=False, db_path=db, **SETTINGS
    )

    assert get_active_model_forecast(TargetRole.EQUITY, db_path=db) is None


def test_a_new_selections_forecast_replaces_the_readable_one(db):
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    set_active_model(
        TargetRole.EQUITY,
        "Polynomial",
        _result(forecast=pd.Series([0.01, 0.02], index=dates)),
        high_risk=False,
        db_path=db,
        set_at=datetime(2026, 8, 30),
        **SETTINGS,
    )
    set_active_model(
        TargetRole.EQUITY,
        "Machine Learning",
        _result(forecast=pd.Series([0.05, 0.06], index=dates)),
        high_risk=False,
        db_path=db,
        set_at=datetime(2026, 8, 31),
        **SETTINGS,
    )

    saved = get_active_model_forecast(TargetRole.EQUITY, db_path=db)
    assert list(saved) == [0.05, 0.06]


def test_with_no_active_model_the_forecast_is_none(db):
    assert get_active_model_forecast(TargetRole.EQUITY, db_path=db) is None


# --- settings_match -------------------------------------------------------------


def test_matching_settings_are_reported_as_matching(db):
    set_active_model(
        TargetRole.EQUITY, "Polynomial", _result(), high_risk=False, db_path=db, **SETTINGS
    )
    set_active_model(TargetRole.BOND, "ML", _result(), high_risk=False, db_path=db, **SETTINGS)

    equity = get_active_model(TargetRole.EQUITY, db_path=db)
    bond = get_active_model(TargetRole.BOND, db_path=db)
    assert settings_match(equity, bond) is True


@pytest.mark.parametrize(
    "overridden",
    ["horizon", "test_window", "dataset_fingerprint"],
)
def test_a_differing_setting_is_reported_as_a_mismatch(db, overridden):
    bond_settings = dict(SETTINGS)
    bond_settings[overridden] = 999 if overridden != "dataset_fingerprint" else "fp-2"
    set_active_model(
        TargetRole.EQUITY, "Polynomial", _result(), high_risk=False, db_path=db, **SETTINGS
    )
    set_active_model(TargetRole.BOND, "ML", _result(), high_risk=False, db_path=db, **bond_settings)

    equity = get_active_model(TargetRole.EQUITY, db_path=db)
    bond = get_active_model(TargetRole.BOND, db_path=db)
    assert settings_match(equity, bond) is False


def test_differing_train_windows_still_match(db):
    # Equity's derived polynomial trains on every earlier date (no train window)
    # while bond trains on the latest 252: both still forecast the same days.
    set_active_model(
        TargetRole.EQUITY,
        "Polynomial",
        _result(),
        high_risk=False,
        db_path=db,
        **{**SETTINGS, "train_window": None},
    )
    set_active_model(TargetRole.BOND, "ML", _result(), high_risk=False, db_path=db, **SETTINGS)

    equity = get_active_model(TargetRole.EQUITY, db_path=db)
    bond = get_active_model(TargetRole.BOND, db_path=db)
    assert equity.train_window is None
    assert settings_match(equity, bond) is True


# --- the signal: each forecast less its fold's training mean ----------------------


def test_each_forecasts_baseline_is_saved_so_its_signal_can_be_read_back(db):
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    forecast = pd.Series([0.012, -0.004, 0.03], index=dates)
    baseline = pd.Series([0.002, 0.002, 0.01], index=dates)
    result = replace(_result(forecast=forecast), baseline=baseline, signal_rank_ic=0.05)
    set_active_model(
        TargetRole.EQUITY, "Polynomial", result, high_risk=False, db_path=db, **SETTINGS
    )

    signal = get_active_model_signal(TargetRole.EQUITY, db_path=db)
    assert list(signal) == pytest.approx([0.01, -0.006, 0.02])
    assert get_active_model(TargetRole.EQUITY, db_path=db).signal_rank_ic == 0.05


def test_a_model_saved_without_baselines_has_no_signal(db):
    dates = pd.date_range("2026-01-01", periods=2, freq="D")
    set_active_model(
        TargetRole.EQUITY,
        "Polynomial",
        _result(forecast=pd.Series([0.01, 0.02], index=dates)),
        high_risk=False,
        db_path=db,
        **SETTINGS,
    )

    assert get_active_model_signal(TargetRole.EQUITY, db_path=db) is None
    assert get_active_model_forecast(TargetRole.EQUITY, db_path=db) is not None


def test_with_no_active_model_the_signal_is_none(db):
    assert get_active_model_signal(TargetRole.EQUITY, db_path=db) is None


# --- FYP-32: an active polynomial's fitted function -----------------------------------

DERIVED = ActiveFunction(
    Origin.DERIVED,
    ModelDescription(
        name="DerivedPolynomial",
        terms=("VIX^2", "USGG10YR VIX"),
        coefficients=(-0.0012, 0.0004),
        intercept=0.008,
        input_bounds={"VIX": (5.0, 41.5), "USGG10YR": (-0.2, 0.2)},
        standardisation={"VIX": (19.0, 6.0), "USGG10YR": (0.0, 0.05)},
    ),
    {"VIX": Transform.LEVEL, "USGG10YR": Transform.DIFFERENCE},
)
USER = ActiveFunction(
    Origin.USER_SUPPLIED,
    ModelDescription(
        name="UserPolynomial", terms=("VIX / LUACOAS",), coefficients=(0.3,), intercept=-0.01
    ),
    {"VIX": Transform.LEVEL, "LUACOAS": Transform.LEVEL},
)


@pytest.mark.parametrize("function", [DERIVED, USER], ids=["derived", "user"])
def test_a_function_set_with_the_model_reads_back_as_it_was_stored(db, function):
    set_active_model(
        TargetRole.EQUITY, "Polynomial", _result(), high_risk=False, function=function,
        db_path=db, **SETTINGS,
    )

    assert get_active_function(TargetRole.EQUITY, db_path=db) == function


def test_a_model_set_without_a_function_has_none_even_after_one_that_had(db):
    common = {"high_risk": False, "db_path": db, **SETTINGS}
    set_active_model(TargetRole.EQUITY, "Polynomial", _result(), function=DERIVED,
                     set_at=datetime(2026, 10, 1), **common)
    set_active_model(TargetRole.EQUITY, "Machine Learning", _result(),
                     set_at=datetime(2026, 10, 2), **common)

    assert get_active_function(TargetRole.EQUITY, db_path=db) is None


def test_each_role_reads_its_own_function(db):
    common = {"high_risk": False, "db_path": db, **SETTINGS}
    set_active_model(TargetRole.EQUITY, "Polynomial", _result(), function=DERIVED, **common)
    set_active_model(TargetRole.BOND, "Polynomial", _result(), function=USER, **common)

    assert get_active_function(TargetRole.EQUITY, db_path=db) == DERIVED
    assert get_active_function(TargetRole.BOND, db_path=db) == USER


def test_no_active_model_has_no_function(db):
    assert get_active_function(TargetRole.BOND, db_path=db) is None


def test_a_setting_that_fails_part_way_leaves_nothing_behind(db):
    broken = ActiveFunction(Origin.DERIVED, DERIVED.description, {"VIX": "not a transform"})

    with pytest.raises(AttributeError):
        set_active_model(
            TargetRole.EQUITY, "Polynomial", _result(), high_risk=False, function=broken,
            db_path=db, **SETTINGS,
        )

    # The model row was written first, in the same transaction, and rolled back too.
    assert get_active_model(TargetRole.EQUITY, db_path=db) is None
