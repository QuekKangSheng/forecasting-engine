"""The pieces the pages use to say more with less: a share of a whole as a bar, a
figure's gap to what it is compared with, a long page's sections, a folded note, and
the links from one page to the next."""

import re

from streamlit.testing.v1 import AppTest

import forecasting_engine
import shell
import ui

# --- a share of a whole -------------------------------------------------------------------


def _widths(bar: str) -> list[float]:
    return [float(share) for share in re.findall(r"flex:([\d.]+) 1 0", bar)]


def test_each_part_of_a_split_bar_is_as_wide_as_its_share():
    bar = ui.split_bar([("Equity", 0.23, "#111111"), ("Bond", 0.77, "#222222")])

    assert _widths(bar) == [0.23, 0.77]
    assert "background:#111111" in bar and "background:#222222" in bar


def test_a_split_bar_says_in_words_what_its_colours_show():
    bar = ui.split_bar([("Equity", 0.23, "#111111"), ("Bond", 0.77, "#222222")])

    assert 'aria-label="Equity 23%, Bond 77%"' in bar
    assert "<span>Equity 23%</span><span>Bond 77%</span>" in bar


def test_a_split_bar_beside_its_own_figures_keeps_its_description_but_not_its_key():
    bar = ui.split_bar([("Equity", 0.5, "#111111"), ("Bond", 0.5, "#222222")], labelled=False)

    assert "fe-split-key" not in bar
    assert 'aria-label="Equity 50%, Bond 50%"' in bar


def test_the_limits_a_share_is_held_within_are_ticked_on_the_bar():
    without = ui.split_bar([("Equity", 0.2, "#111111"), ("Bond", 0.8, "#222222")])
    within = ui.split_bar(
        [("Equity", 0.2, "#111111"), ("Bond", 0.8, "#222222")], limits=(0.2, 0.8)
    )

    assert "fe-split-tick" not in without
    assert re.findall(r'fe-split-tick" style="left:([\d.]+%)', within) == ["20.00%", "80.00%"]


def test_a_split_bar_escapes_the_names_it_is_given():
    """An index's name comes from the uploaded data."""
    bar = ui.split_bar([('S&P <b>"500"</b>', 1.0, "#111111")])

    assert "<b>" not in bar and "S&amp;P" in bar


# --- a figure's gap to what it is compared with -------------------------------------------


def test_a_rise_in_something_wanted_is_an_up_arrow_in_the_good_colour():
    assert ui.delta(0.08, "0.08") == '<span class="fe-delta fe-delta-good">▲ 0.08</span>'


def test_a_fall_in_something_wanted_is_a_down_arrow_in_the_bad_colour():
    assert ui.delta(-0.08, "0.08") == '<span class="fe-delta fe-delta-bad">▼ 0.08</span>'


def test_a_rise_in_a_loss_points_up_and_is_still_bad():
    """The arrow is the direction and the colour is whether that is good: for a
    value-at-risk, up is the bad way."""
    more_risk = ui.delta(0.0003, "0.03 pts", higher_is_better=False)
    less_risk = ui.delta(-0.0003, "0.03 pts", higher_is_better=False)

    assert "▲" in more_risk and "fe-delta-bad" in more_risk
    assert "▼" in less_risk and "fe-delta-good" in less_risk


def test_no_change_and_nothing_to_compare_take_no_arrow_and_no_verdict():
    for change in (0.0, float("nan")):
        flat = ui.delta(change, "0.00")

        assert "fe-delta-flat" in flat
        assert "▲" not in flat and "▼" not in flat


# --- a long page's sections ---------------------------------------------------------------


def _sections():
    import ui

    ui.section_nav([("Allocation", "allocation"), ("Tail risk", "tail-risk")])
    ui.section("Allocation", anchor="allocation")
    ui.section("Tail risk", "Losses on the worst days.", anchor="tail-risk")
    ui.section("Not linked to")


def test_a_pages_section_line_links_to_the_sections_that_were_given_anchors():
    app = AppTest.from_function(_sections).run()

    marks = [m.value for m in app.markdown]
    (nav,) = [m for m in marks if "fe-subnav" in m]
    titles = [m for m in marks if "fe-section-title" in m]
    assert re.findall(r'href="#([\w-]+)"', nav) == ["allocation", "tail-risk"]
    assert [re.findall(r' id="([\w-]+)"', m) for m in titles] == [
        ["allocation"],
        ["tail-risk"],
        [],
    ]
    # The links stay in this window: a new tab would be a new session with nothing in it.
    assert nav.count('target="_self"') == 2


# --- a note kept one click away -----------------------------------------------------------


def _noted():
    import streamlit as st

    import ui

    with ui.note("Method", "sharpe"):
        st.caption("Annualised Sharpe of the previous 252 trading days.")


def test_what_is_folded_into_a_note_is_still_on_the_page():
    app = AppTest.from_function(_noted).run()

    assert [e.label for e in app.expander] == ["Method"]
    assert [c.value for c in app.caption] == ["Annualised Sharpe of the previous 252 trading days."]


# --- the links from one page to the next --------------------------------------------------


def _linked():
    import ui

    ui.page_link("models", "Next · 02 Models")


def test_a_page_run_on_its_own_draws_no_link_and_does_not_fail(monkeypatch):
    monkeypatch.setattr(ui, "_PAGES", {})

    app = AppTest.from_function(_linked).run()

    assert not app.exception
    assert len(app.main.children) == 0


def test_registering_the_pages_replaces_whatever_was_registered_before(monkeypatch):
    monkeypatch.setattr(ui, "_PAGES", {"gone": object()})
    models = object()

    ui.register_pages({"models": models})

    assert ui._PAGES == {"models": models}


# --- the sidebar's foot -------------------------------------------------------------------


def test_the_sidebar_says_which_version_and_build_is_running(monkeypatch):
    monkeypatch.setattr(shell, "code_version", lambda: "abc1234-dirty")

    foot = shell.sidebar_foot()

    assert forecasting_engine.__version__ in foot
    assert "abc1234-dirty" in foot


def test_outside_a_git_checkout_the_sidebar_says_so_rather_than_nothing(monkeypatch):
    monkeypatch.setattr(shell, "code_version", lambda: None)

    assert "not a git checkout" in shell.sidebar_foot()
