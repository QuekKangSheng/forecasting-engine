"""The shared pieces of markup the pages are built from."""

import ui


def test_a_kicker_dims_the_number_and_lights_the_stage():
    line = ui.kicker_line("03 / Allocate")

    assert line == '<div class="fe-kicker"><b>03 /</b> Allocate</div>'


def test_a_kicker_with_no_number_is_one_plain_line():
    assert ui.kicker_line("Overview") == '<div class="fe-kicker">Overview</div>'


def test_a_panels_kicker_is_the_same_line_a_size_down():
    assert 'class="fe-kicker fe-kicker-small"' in ui.kicker_line("01 / Ingest", small=True)


def test_a_kicker_escapes_its_text():
    assert "<script>" not in ui.kicker_line("01 / <script>")


def test_an_unknown_tone_is_drawn_neutral():
    assert "var(--fe-neutral-fg)" in ui.lozenge("Held", "not a tone")


def test_every_tone_a_badge_can_take_has_its_colours_in_the_stylesheet():
    sheet = ui.stylesheet()

    for tone in ui.TONES:
        assert f"--fe-{tone}-fg:" in sheet and f"--fe-{tone}-bg:" in sheet


def test_the_stylesheet_has_no_placeholder_left_unfilled():
    """It is a template filled from ``tokens``; a missed name would reach the browser
    as a literal ``$NAME`` and silently drop that rule."""
    assert "$" not in ui.stylesheet()


def test_motion_is_dropped_for_viewers_who_ask_for_less():
    assert "prefers-reduced-motion: reduce" in ui.stylesheet()
