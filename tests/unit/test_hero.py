"""The Home page's opening: the typed name and its words, the status line, the tape."""

import numpy as np
import pandas as pd

import hero
import tokens


def _frame(**columns) -> pd.DataFrame:
    length = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.bdate_range("2024-01-01", periods=length))


def _tick(change: float) -> hero.Tick:
    return hero.Tick("VIX", "17.32", change, f"{change:+.2f}", (1.0, 2.0, 3.0))


def test_a_prices_change_is_a_percentage_over_the_span():
    frame = _frame(index=np.linspace(100, 110, 21))

    (tick,) = hero.ticks(frame, ["index"], ["S&P 500"], [True], span=20)

    assert (tick.name, tick.last, tick.change_text) == ("S&P 500", "110.00", "+10.0%")


def test_a_rates_change_is_in_its_own_units():
    frame = _frame(yield_10y=np.linspace(4.0, 4.4, 21))

    (tick,) = hero.ticks(frame, ["yield_10y"], ["10y yield"], [False], span=20)

    assert tick.change_text == "+0.40"


def test_the_change_is_over_the_series_own_rows_not_the_calendars():
    # Half the dates have no value: twenty of its own rows back is forty dates back.
    values = np.full(60, np.nan)
    values[::2] = np.arange(30, dtype=float)
    frame = _frame(sparse=values)

    (tick,) = hero.ticks(frame, ["sparse"], ["Sparse"], [False], span=20)

    assert tick.change_text == "+20.00"


def test_a_series_with_too_little_history_is_left_off_the_tape():
    frame = _frame(short=np.arange(20, dtype=float))

    assert hero.ticks(frame, ["short"], ["Short"], [False], span=20) == ()


def test_a_sparkline_ends_on_the_latest_value():
    frame = _frame(index=np.arange(500, dtype=float))

    (tick,) = hero.ticks(frame, ["index"], ["Index"], [True], span=20)

    assert tick.spark[-1] == 499
    assert len(tick.spark) <= hero.SPARK_POINTS + 1


def test_the_tape_is_written_twice_and_the_copy_is_hidden_from_a_screen_reader():
    tape = hero.tape([_tick(1.2), _tick(-0.4)], span=20)

    assert tape.count('class="fe-tick"') == 4
    assert tape.count('<span class="fe-tape-run" aria-hidden="true">') == 1
    assert "change over 20 trading days" in tape


def test_a_rise_a_fall_and_no_change_are_each_marked_by_shape_and_colour():
    tape = hero.tape([_tick(1.2), _tick(-0.4), _tick(0.0)], span=20)

    assert '<span class="fe-tick-up">▲ 1.20</span>' in tape
    assert '<span class="fe-tick-down">▼ 0.40</span>' in tape
    assert '<span class="fe-tick-flat">■ 0.00</span>' in tape
    assert f'stroke="{tokens.RISE}"' in tape and f'stroke="{tokens.FALL}"' in tape


def test_a_series_name_on_the_tape_is_escaped():
    tick = hero.Tick("<b>x</b>", "1.00", 0.1, "+0.10", (1.0, 2.0))

    assert "<b>x</b>" not in hero.tape([tick], span=20)


def test_with_nothing_to_show_there_is_no_tape():
    assert hero.tape([], span=20) == ""


def test_the_typed_line_holds_its_whole_text_from_the_start():
    line = hero.typed("9 signals · 2 of 2 models active")

    assert ">9 signals · 2 of 2 models active</span>" in line
    assert "steps(32)" in line and "--n:32" in line


def test_the_typed_lines_text_is_escaped():
    assert "<i>" not in hero.typed("<i>x</i>")


def test_the_name_is_the_heading_and_is_read_once():
    lead = hero.lead("Finlytics", "Next step", "Run models.", "", "")

    assert 'role="heading" aria-level="1" aria-label="Finlytics"' in lead
    assert '<span class="fe-hero-letters" aria-hidden="true">' in lead


def test_every_letter_flickers_on_by_its_own_keyframes_and_settles_solid():
    keyframes, letters = hero.flicker("Finlytics")

    assert letters.count("<span") == 9
    for i in range(9):
        assert f"animation:fe-flicker-{i} " in letters
        frames = keyframes.split(f"@keyframes fe-flicker-{i}{{")[1].split("}}")[0]
        # Starts dark, steps (never fades), ends solid.
        assert frames.startswith("0.00%{opacity:0;")
        assert "steps(1,end)" in frames and frames.endswith("100%{opacity:1")


def test_the_flicker_is_the_same_on_every_load_and_differs_by_letter():
    first, second = hero.flicker("Finlytics"), hero.flicker("Finlytics")

    assert first == second
    durations = {part.split("ms")[0] for part in first[1].split("fe-flicker-")[1:]}
    assert len(durations) > 1
    for duration in durations:
        assert 900 <= int(duration.split(" ")[1]) <= 1300


def test_the_statement_under_the_name_is_escaped():
    lead = hero.lead("<i>N</i>", "Next <b>", "Run <b> models", "", "")

    assert '<div class="fe-hero-state">Run &lt;b&gt; models</div>' in lead
    assert 'class="fe-hero-label">Next &lt;b&gt;</div>' in lead
    assert "<i>" not in lead


def test_a_statement_of_a_whole_sentence_is_set_a_size_down():
    assert "fe-hero-state-long" in hero.lead("F", "Next step", "A sentence.", "", "", long=True)
    assert "fe-hero-state-long" not in hero.lead("F", "Latest weights", "60% equity.", "", "")


def test_every_piece_is_one_line_so_markdown_cannot_break_it_up():
    pieces = [
        hero.typed("status"),
        hero.tape([_tick(1.0)], span=20),
        hero.lead("Finlytics", "Label", "Statement", "<b>below</b>", ""),
    ]

    assert all("\n" not in piece for piece in pieces)


def test_a_viewer_who_asked_for_less_motion_gets_none():
    reduced = hero.CSS.split("@media (prefers-reduced-motion: reduce)")[1]

    for moving in (".fe-tape-track", ".fe-typed-text", ".fe-hero-letters span"):
        assert moving in reduced
    assert "animation: none" in reduced
