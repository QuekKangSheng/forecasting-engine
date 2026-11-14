"""The console's colours are stated twice: in ``app/tokens.py``, which the stylesheet
and the charts draw from, and in ``.streamlit/config.toml``, which Streamlit's own
widgets read. These hold the two together, and hold the colours to what makes them
readable on a dark page."""

import tomllib
from pathlib import Path

import pytest

import tokens

REPO = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
THEME = CONFIG["theme"]


def _luminance(colour: str) -> float:
    channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    lighter, darker = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize(
    ("option", "token"),
    [
        ("backgroundColor", tokens.BG),
        ("secondaryBackgroundColor", tokens.RAISED),
        ("textColor", tokens.TEXT),
        ("primaryColor", tokens.ACCENT),
        ("linkColor", tokens.ACCENT_BRIGHT),
        ("borderColor", tokens.BORDER),
        ("font", tokens.SANS),
        ("codeFont", tokens.MONO),
    ],
)
def test_streamlits_theme_uses_the_same_colours_and_faces_as_the_stylesheet(option, token):
    assert THEME[option] == token


def test_the_sidebar_is_the_darkest_surface():
    assert THEME["sidebar"]["backgroundColor"] == tokens.SUNKEN


def test_streamlits_own_charts_would_start_from_the_apps_series_colours():
    assert THEME["chartCategoricalColors"][:4] == [
        tokens.SERIES_OURS,
        tokens.SERIES_REFERENCE,
        tokens.SERIES_EQUITY,
        tokens.SERIES_BOND,
    ]


def test_every_typeface_the_theme_names_is_served_by_the_app_itself():
    assert CONFIG["server"]["enableStaticServing"] is True
    families = {face["family"] for face in THEME["fontFaces"]}
    assert families == {"IBM Plex Sans", "IBM Plex Mono"}
    for face in THEME["fontFaces"]:
        # ``app/static/<file>`` is where Streamlit serves the entry point's static folder.
        assert face["url"].startswith("app/static/")
        assert (REPO / face["url"]).is_file(), face["url"]
    # The typefaces' licence travels with them.
    assert (REPO / "app" / "static" / "OFL.txt").is_file()


def test_text_is_readable_on_the_page():
    assert _contrast(tokens.TEXT, tokens.BG) >= 12
    assert _contrast(tokens.TEXT_SECONDARY, tokens.BG) >= 7
    # The quietest text the app sets: captions and labels.
    assert _contrast(tokens.MUTED, tokens.BG) >= 4.5
    assert _contrast(tokens.ACCENT_BRIGHT, tokens.BG) >= 7


def test_every_state_colour_is_readable_as_text_on_the_page():
    for tone, (text, _) in tokens.STATE.items():
        assert _contrast(text, tokens.BG) >= 4.5, tone


def test_every_series_colour_stands_out_from_the_page_and_from_the_others():
    series = [
        tokens.SERIES_OURS,
        tokens.SERIES_REFERENCE,
        tokens.SERIES_EQUITY,
        tokens.SERIES_BOND,
        tokens.SERIES_BASELINE,
        *tokens.SERIES_OTHERS,
    ]

    assert len(set(series)) == len(series)
    for colour in series:
        assert _contrast(colour, tokens.BG) >= 3, colour
