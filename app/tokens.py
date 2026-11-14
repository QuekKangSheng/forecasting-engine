"""The console's colours and typefaces, in one place.

The stylesheet (``ui``) and the charts (``charts``) both draw from here, so a chart's
gridline is the same grey as a table's rule and a status badge the same green as a
status line. ``.streamlit/config.toml`` repeats the few that Streamlit's own widgets
need; ``tests/unit/test_tokens.py`` holds the two together.

The look is an operations console: a near-black field, hairlines carrying the
structure, small spaced capitals in a monospace face for labels, and one blue for
whatever can be acted on. Light type on dark, so every text colour here is measured
against ``BG``: body text clears 12:1 and the quietest label 6:1.
"""

from __future__ import annotations

# --- surfaces, darkest to lightest ---------------------------------------------------
SUNKEN = "#07090C"  # the sidebar and the status bar
BG = "#0B0D11"  # the page
SURFACE = "#10141A"  # panels, tables' sticky column
RAISED = "#171C24"  # inputs, hover

# --- lines ---------------------------------------------------------------------------
BORDER = "#212833"
BORDER_STRONG = "#36404D"

# --- text ----------------------------------------------------------------------------
TEXT = "#E9ECF1"
TEXT_SECONDARY = "#B7BFCA"
MUTED = "#8B95A4"

# --- the one accent: what can be acted on, and the app's own series -------------------
ACCENT = "#4C90F0"
ACCENT_BRIGHT = "#8ABBFF"  # as text on the dark field, where ACCENT is under 7:1

# --- state, never used for a data series ---------------------------------------------
#: tone -> (text, tinted background)
STATE = {
    "neutral": ("#B7BFCA", "rgba(143, 153, 168, 0.14)"),
    "info": ("#8ABBFF", "rgba(76, 144, 240, 0.16)"),
    "success": ("#72CA9B", "rgba(50, 164, 103, 0.16)"),
    "warning": ("#FBB360", "rgba(200, 118, 25, 0.20)"),
    "danger": ("#FA999C", "rgba(205, 66, 70, 0.20)"),
}

# --- data series ---------------------------------------------------------------------
# Chosen by measurement on the dark surface, not by eye: every pair of the three hues
# stays apart under simulated red-green colour blindness (worst pair 8.6, target 8)
# and for full colour vision (20.6, floor 15), each sits in the lightness band that
# reads on a dark field, and each clears 3:1 against it. Grey is kept for whatever a
# chart is read against: a benchmark, or what really happened.
SERIES_OURS = ACCENT
SERIES_REFERENCE = "#8F99A8"
SERIES_EQUITY = "#CF7A1E"
SERIES_BOND = "#32A467"
SERIES_BASELINE = "#D4609A"
SERIES_OTHERS = ("#B07CE8", "#B8A038", "#14A89B", "#E76A6E")

# --- type ----------------------------------------------------------------------------
# Both faces are served by the app itself (``app/static``), so they load with no
# network beyond the app's own.
SANS = "'IBM Plex Sans', -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"
