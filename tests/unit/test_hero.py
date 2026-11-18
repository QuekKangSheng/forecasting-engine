"""The Home page's opening: the typed name and its words, the status line."""

import hero


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
        hero.lead("Finlytics", "Label", "Statement", "<b>below</b>", ""),
    ]

    assert all("\n" not in piece for piece in pieces)


def test_a_viewer_who_asked_for_less_motion_gets_none():
    reduced = hero.CSS.split("@media (prefers-reduced-motion: reduce)")[1]

    for moving in (".fe-typed-text", ".fe-hero-letters span"):
        assert moving in reduced
    assert "animation: none" in reduced
