"""Parsing the Fama-French five-factor daily CSV, and keeping a copy of it.

The real file wraps the data table in explanatory prose with no fixed line
count, so these fixtures vary the amount of surrounding text to prove the
parser finds the table by content, not by position.
"""

import io
import os
import zipfile
from datetime import datetime

import pandas as pd
import pytest

from forecasting_engine.ingest import fama_french
from forecasting_engine.ingest.fama_french import (
    MAX_AGE,
    FactorFetchError,
    download,
    load_latest,
    parse,
    resolve,
    restrict_to,
    save,
)

CSV_BODY = """This file was created by CMPT_ME_BEME_OP_INV_RETS.
Copyright 2024 Kenneth R. French

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,-0.67,0.02,-0.35,0.03,0.13,0.01
19630702,0.79,-0.28,0.28,-0.08,-0.20,0.01

Copyright 2024 Kenneth R. French. All rights reserved.
"""


def zip_bytes(body: str = CSV_BODY) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("F-F_Research_Data_5_Factors_2x3_daily.CSV", body)
    return buffer.getvalue()


# --- parsing -----------------------------------------------------------------


def test_preamble_and_footer_prose_are_stripped():
    assert len(parse(zip_bytes())) == 2


def test_only_rows_with_an_8_digit_date_are_kept():
    frame = parse(zip_bytes())
    assert list(frame["Date"]) == list(pd.to_datetime(["1963-07-01", "1963-07-02"]))


def test_columns_are_the_five_factors_plus_rf():
    assert list(parse(zip_bytes()).columns) == ["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def test_values_are_numeric():
    assert parse(zip_bytes())["Mkt-RF"].iloc[0] == -0.67


def test_a_longer_preamble_does_not_shift_the_result():
    longer = "extra line one\nextra line two\nextra line three\n" + CSV_BODY
    assert len(parse(zip_bytes(longer))) == 2


def test_a_zip_without_a_csv_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "nothing here")
    with pytest.raises(ValueError, match="no CSV"):
        parse(buffer.getvalue())


# --- restricting to a date range ---------------------------------------------


def test_restrict_to_keeps_only_the_requested_range():
    frame = parse(zip_bytes())
    restricted = restrict_to(frame, pd.Timestamp("1963-07-02"), pd.Timestamp("1963-07-02"))
    assert list(restricted["Date"]) == [pd.Timestamp("1963-07-02")]


def test_restrict_to_an_end_past_the_data_just_returns_what_exists():
    frame = parse(zip_bytes())
    restricted = restrict_to(frame, pd.Timestamp("1963-07-01"), pd.Timestamp("2030-01-01"))
    assert len(restricted) == 2


# --- keeping a copy, named by content ----------------------------------------


def test_save_writes_the_file_under_its_content_hash(tmp_path):
    stored = save(parse(zip_bytes()), tmp_path)

    path = tmp_path / f"{stored.source.sha256}.csv"
    assert path.exists()
    assert stored.source.path == path
    assert stored.source.name == "fama_french_5_factors_daily.csv"


def test_saving_the_same_data_twice_resolves_to_one_file(tmp_path):
    first = save(parse(zip_bytes()), tmp_path)
    second = save(parse(zip_bytes()), tmp_path)

    assert first.source.sha256 == second.source.sha256
    assert len(list(tmp_path.iterdir())) == 1


def test_load_latest_reads_back_what_was_saved(tmp_path):
    saved = save(parse(zip_bytes()), tmp_path)
    loaded = load_latest(tmp_path)

    assert loaded is not None
    assert loaded.source.sha256 == saved.source.sha256
    assert list(loaded.frame["Date"]) == list(saved.frame["Date"])
    assert loaded.frame["Mkt-RF"].tolist() == saved.frame["Mkt-RF"].tolist()


def test_load_latest_is_none_when_nothing_was_downloaded(tmp_path):
    assert load_latest(tmp_path / "absent") is None
    assert load_latest(tmp_path) is None


def test_download_fetches_and_keeps_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fama_french, "fetch", lambda: parse(zip_bytes()))
    stored = download(tmp_path)
    assert stored.source.path is not None and stored.source.path.exists()


def test_a_failed_fetch_is_one_named_error(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(fama_french, "urlopen", broken)
    with pytest.raises(FactorFetchError, match="no route to host"):
        fama_french.fetch()


# --- resolving the file a run uses --------------------------------------------


def _age_saved_copy(tmp_path, days: int) -> None:
    (path,) = tmp_path.glob("*.csv")
    then = datetime.now().timestamp() - days * 86_400
    os.utime(path, (then, then))


def _fail_fetch(monkeypatch):
    def broken():
        raise FactorFetchError("OSError: no route to host")

    monkeypatch.setattr(fama_french, "fetch", broken)


def test_a_recent_saved_copy_is_used_without_downloading(tmp_path, monkeypatch):
    saved = save(parse(zip_bytes()), tmp_path)
    _fail_fetch(monkeypatch)

    resolved = resolve(tmp_path)

    assert resolved.file.source.sha256 == saved.source.sha256
    assert resolved.warning is None


def test_an_old_saved_copy_prompts_a_download(tmp_path, monkeypatch):
    save(parse(zip_bytes()), tmp_path)
    _age_saved_copy(tmp_path, MAX_AGE.days + 1)
    fetched = []
    monkeypatch.setattr(fama_french, "fetch", lambda: fetched.append(1) or parse(zip_bytes()))

    resolved = resolve(tmp_path)

    assert fetched == [1]
    assert resolved.warning is None


def test_redownloading_an_unchanged_file_resets_its_age(tmp_path, monkeypatch):
    save(parse(zip_bytes()), tmp_path)
    _age_saved_copy(tmp_path, MAX_AGE.days + 1)
    monkeypatch.setattr(fama_french, "fetch", lambda: parse(zip_bytes()))
    resolve(tmp_path)
    _fail_fetch(monkeypatch)

    assert resolve(tmp_path).warning is None


def test_a_failed_download_falls_back_to_the_saved_copy_with_a_warning(tmp_path, monkeypatch):
    save(parse(zip_bytes()), tmp_path)
    _age_saved_copy(tmp_path, MAX_AGE.days + 1)
    _fail_fetch(monkeypatch)

    resolved = resolve(tmp_path)

    assert resolved.warning is not None and "no route to host" in resolved.warning


def test_no_saved_copy_and_a_failed_download_raises(tmp_path, monkeypatch):
    _fail_fetch(monkeypatch)
    with pytest.raises(FactorFetchError):
        resolve(tmp_path)
