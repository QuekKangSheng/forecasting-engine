"""Shared presentation for the dashboard.

The look is an operations console: a near-black field, hairlines carrying the
structure, small spaced capitals in a monospace face for every label, figures in the
same face so columns of them line up, and one blue for whatever can be acted on.
Semantic colour is kept for state. The colours and typefaces are in ``tokens``;
``.streamlit/config.toml`` repeats the ones Streamlit's own widgets need, and this
module holds what Streamlit has no component for and restyles what it has.

**Three sizes of heading.** A page opens with a numbered kicker, its title set large
and light, and a one-line summary, then a hairline the full width of the content
area. A section starts with the same hairline and a mid-sized title marked by a short
blue bar. Everything smaller (a chart's name, a column heading, a control's label, a
headline figure's caption) is a spaced capital label. So the three levels differ in
size, weight and face at once, and a page can be scanned by its headings alone.

**Status lozenges.** A short outlined badge with a dot in its state's colour: it
reads at a glance in a way a coloured word does not, and sits quietly on a dark page
where a filled pastel block would glare.

**Status rows.** Name on the left, state on the right, a hairline between.

One theme, dark. The app no longer follows the viewer's light or dark preference:
the console is designed as it is, not flipped from a light original.

No emoji anywhere: an internal analytical tool should read as a tool.
"""

from __future__ import annotations

import html
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from string import Template

import streamlit as st

import tokens

TONES: tuple[str, ...] = ("neutral", "info", "success", "warning", "danger")

#: The content area's side padding. Full-width rules reach past it by the same
#: amount, so they run from the sidebar's edge to the window's.
_GUTTER = "2.25rem"

_TONE_VARIABLES = "\n".join(
    f"    --fe-{tone}-fg: {fg}; --fe-{tone}-bg: {bg};" for tone, (fg, bg) in tokens.STATE.items()
)

_CSS = Template(
    """
<style>
  :root {
$TONES
    --fe-bg:      $BG;
    --fe-sunken:  $SUNKEN;
    --fe-surface: $SURFACE;
    --fe-raised:  $RAISED;
    --fe-border:  $BORDER;
    --fe-border-strong: $BORDER_STRONG;
    --fe-text:    $TEXT;
    --fe-text-2:  $TEXT_SECONDARY;
    --fe-muted:   $MUTED;
    --fe-accent:  $ACCENT;
    --fe-accent-bright: $ACCENT_BRIGHT;
    --fe-shadow:  rgba(0, 0, 0, 0.6);
    --fe-sans:    $SANS;
    --fe-mono:    $MONO;
    --fe-gutter:  $GUTTER;
    --fe-bar:     2.25rem;
  }

  /* ---- The field ------------------------------------------------------------------
     A faint dot grid and a cool glow from the top right, on the frame that doesn't
     scroll, so it costs nothing as the page moves over it. */
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(1100px 480px at 78% -8%, rgba(76, 144, 240, 0.085), transparent 62%),
      radial-gradient(circle at 1px 1px, rgba(233, 236, 241, 0.05) 1px, transparent 0)
        0 0 / 22px 22px,
      var(--fe-bg);
  }
  [data-testid="stMain"], [data-testid="stHeader"] { background: transparent; }
  ::selection { background: rgba(76, 144, 240, 0.35); }
  * { scrollbar-width: thin; scrollbar-color: var(--fe-border-strong) transparent; }

  /* Wide pages with a fixed gutter, so the full-width rules below can reach exactly
     to both edges of the content area. No padding above: the status bar sits there. */
  [data-testid="stMainBlockContainer"] {
    padding: 0 var(--fe-gutter) 5rem !important;
    max-width: none !important;
  }

  /* ---- The status bar ---------------------------------------------------------------
     What the app is working from, on every page: the data, the two active models
     and the last allocation. It stays at the top of the window as the page scrolls. */
  [data-testid="stElementContainer"]:has(.fe-statusbar) {
    position: sticky;
    top: 0;
    z-index: 100;
    margin: 0 calc(-1 * var(--fe-gutter));
    width: calc(100% + 2 * var(--fe-gutter));
    max-width: none;
  }
  .fe-statusbar {
    display: flex;
    align-items: stretch;
    min-height: var(--fe-bar);
    padding: 0 4.5rem 0 var(--fe-gutter);
    background: rgba(7, 9, 12, 0.88);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--fe-border);
    font-family: var(--fe-mono);
    font-size: 10.5px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    overflow: hidden;
    white-space: nowrap;
    position: relative;
  }
  /* Where the bar is short of room the model names give way first (their full text
     is in the cell's tooltip); the data's dates never do. */
  .fe-status-cell {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0 1rem;
    border-left: 1px solid var(--fe-border);
    min-width: 5rem;
    flex: 0 1 auto;
  }
  .fe-status-cell:first-child { border-left: none; padding-left: 0; flex-shrink: 0; }
  .fe-status-cell:last-child { flex-shrink: 0; }
  /* With the sidebar folded away its button sits over the bar's first cell. */
  [data-testid="stApp"]:has([data-testid="stExpandSidebarButton"]) .fe-statusbar {
    padding-left: 3.5rem;
  }
  .fe-status-key { color: var(--fe-muted); }
  .fe-status-value {
    color: var(--fe-text);
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .fe-status-dot {
    width: 6px; height: 6px; border-radius: 50%; flex: none;
    background: var(--fe-muted);
  }

  /* ---- The sidebar: the pipeline, in order ------------------------------------------ */
  [data-testid="stSidebar"] { background: var(--fe-sunken); }
  [data-testid="stSidebarHeader"] {
    height: auto;
    padding: 1.15rem 1.25rem 1rem;
    align-items: center;
  }
  [data-testid="stSidebarHeader"]::before {
    content: "Forecasting Engine";
    flex: 1;
    padding-left: 1.15rem;
    font-family: var(--fe-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    white-space: nowrap;
    color: var(--fe-text);
    /* The mark: one small square in the accent. */
    background: linear-gradient(var(--fe-accent), var(--fe-accent)) 0 50% / 7px 7px no-repeat;
  }
  [data-testid="stLogoSpacer"] { display: none; }
  [data-testid="stSidebarNav"] { padding-top: 0.5rem; }
  [data-testid="stSidebarNavItems"] {
    counter-reset: stage -1;
    padding: 0 0.75rem;
    gap: 2px;
  }
  [data-testid="stSidebarNavItems"] li { counter-increment: stage; }
  [data-testid="stSidebarNavLink"] {
    gap: 0.9rem;
    padding: 0.55rem 0.75rem;
    border-radius: 2px;
    border-left: 2px solid transparent;
    background: transparent;
    transition: background-color 120ms ease, border-color 120ms ease;
  }
  [data-testid="stSidebarNavLink"]::before {
    content: counter(stage, decimal-leading-zero);
    font-family: var(--fe-mono);
    font-size: 10.5px;
    letter-spacing: 0.08em;
    color: var(--fe-muted);
  }
  [data-testid="stSidebarNavLink"] span {
    font-family: var(--fe-mono);
    font-size: 11.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fe-text-2);
  }
  [data-testid="stSidebarNavLink"]:hover { background: var(--fe-surface); }
  [data-testid="stSidebarNavLink"]:hover span { color: var(--fe-text); }
  [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: var(--fe-raised);
    border-left-color: var(--fe-accent);
  }
  [data-testid="stSidebarNavLink"][aria-current="page"] span { color: var(--fe-text); }
  [data-testid="stSidebarNavLink"][aria-current="page"]::before {
    color: var(--fe-accent-bright);
  }
  [data-testid="stSidebarNavLink"]:focus-visible {
    outline: 2px solid var(--fe-accent);
    outline-offset: 1px;
  }

  /* Streamlit prints its own server.maxUploadSize inside the dropzone. Ours is
     deliberately looser than the documented limit, so showing it would
     contradict the caption above the control. */
  [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

  /* ---- Headings ---------------------------------------------------------------------- */

  /* A hairline from the sidebar's edge to the window's. */
  .fe-rule {
    border-top: 1px solid var(--fe-border);
    margin: 0 calc(-1 * var(--fe-gutter));
  }

  .fe-page-head { padding: 2.25rem 0 0.35rem; }
  /* The rule under a page's heading, with a little air before the page begins. The
     air is on the row the rule sits in: a margin on the rule itself is swallowed by it. */
  [data-testid="stElementContainer"]:has(.fe-rule-head) { margin-bottom: 0.75rem; }
  .fe-kicker {
    font-family: var(--fe-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--fe-accent-bright);
    margin: 0 0 0.9rem;
  }
  .fe-kicker b { color: var(--fe-muted); font-weight: 400; }
  /* The same line at the head of a panel, a size down. */
  .fe-kicker-small { font-size: 10.5px; margin: 0.15rem 0 -0.35rem; }
  .fe-page-title {
    font-family: var(--fe-sans);
    font-size: clamp(1.9rem, 1.35rem + 1.7vw, 2.75rem);
    font-weight: 300;
    line-height: 1.08;
    letter-spacing: -0.028em;
    color: var(--fe-text);
    margin: 0;
  }
  .fe-page-summary {
    color: var(--fe-text-2);
    font-size: 15px;
    line-height: 1.55;
    margin: 1rem 0 0.4rem;
    max-width: 78ch;
  }
  /* Streamlit's own headings (a results heading, an opened run's name), at the size
     of a section title. */
  [data-testid="stHeading"] h3 {
    font-size: 18px;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: var(--fe-text);
  }
  .fe-section-title {
    position: relative;
    padding-left: 0.85rem;
    font-size: 18px;
    font-weight: 500;
    line-height: 1.3;
    letter-spacing: -0.01em;
    color: var(--fe-text);
    margin: 0.6rem 0 0;
  }
  .fe-section-title::before {
    content: "";
    position: absolute;
    left: 0; top: 0.2em; bottom: 0.2em;
    width: 2px;
    background: var(--fe-accent);
  }

  /* A label above a chart or a group of figures. */
  .fe-eyebrow {
    font-family: var(--fe-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fe-text-2);
    margin-bottom: 4px;
  }

  /* Streamlit's own help tooltip isn't available on raw markdown, so a
     section label explains itself through the browser's title tooltip. */
  .fe-eyebrow-help {
    border: 1px solid var(--fe-border-strong);
    border-radius: 50%;
    color: var(--fe-muted);
    cursor: help;
    display: inline-block;
    font-family: var(--fe-sans);
    font-size: 9px;
    font-weight: 600;
    font-style: normal;
    letter-spacing: 0;
    text-transform: none;
    height: 13px;
    line-height: 12px;
    margin-left: 6px;
    text-align: center;
    width: 13px;
    vertical-align: 1px;
    transition: color 120ms ease, border-color 120ms ease;
  }
  .fe-eyebrow-help:hover { color: var(--fe-text); border-color: var(--fe-text-2); }

  /* ---- State ------------------------------------------------------------------------- */
  .fe-lozenge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 1px 7px 1px 6px;
    border: 1px solid color-mix(in srgb, currentColor 34%, transparent);
    border-radius: 2px;
    font-family: var(--fe-mono);
    font-size: 10.5px;
    font-weight: 500;
    line-height: 16px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    white-space: nowrap;
    vertical-align: middle;
  }
  .fe-lozenge::before {
    content: "";
    width: 5px; height: 5px; border-radius: 50%; flex: none;
    background: currentColor;
  }

  .fe-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 10px 0;
    border-bottom: 1px solid var(--fe-border);
  }
  .fe-row:last-child { border-bottom: none; }
  .fe-row-name { font-size: 14px; color: var(--fe-text); }
  .fe-row-meta {
    color: var(--fe-muted);
    font-family: var(--fe-mono);
    font-size: 11.5px;
  }

  /* One finding per line. The message is prose of unpredictable length, so it
     gets the full width and wraps, rather than being clipped inside a table
     cell. Location and value sit above it in a muted meta line. */
  .fe-finding {
    padding: 10px 0;
    border-bottom: 1px solid var(--fe-border);
  }
  .fe-finding:last-child { border-bottom: none; }
  .fe-finding-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 3px;
  }
  .fe-finding-where {
    color: var(--fe-muted);
    font-family: var(--fe-mono);
    font-size: 11.5px;
  }
  .fe-finding-detail { font-size: 14px; line-height: 20px; color: var(--fe-text-2); }

  /* ---- Tables ------------------------------------------------------------------------
     A plain table for cells needing an inline lozenge badge — st.dataframe
     can't render those. Rows are divided by hairlines and the header by a
     heavier rule, with no outer box. Every figure is nowrap and set in the
     monospace face, so a column of them lines up digit under digit, and a wide
     comparison scrolls inside its own box rather than running off the page.

     A table that does scroll used to give no sign of it: its last columns were
     simply cut off at the edge. So a shadow down the right edge says there is
     more that way and is covered once the end is reached (the first gradient
     scrolls with the table, the second stays put), and the first column, which
     names each row, stays in view while the rest scrolls under it. Headings may
     wrap, so a two-word heading over a four-digit figure doesn't set the
     column's width and fewer tables need to scroll at all. */
  .fe-table-wrap {
    overflow-x: auto;
    background:
      linear-gradient(to left, var(--fe-bg) 40%, transparent) right / 48px 100%
        no-repeat local,
      linear-gradient(to left, var(--fe-shadow), transparent) right / 16px 100%
        no-repeat scroll;
  }
  .fe-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
    font-variant-numeric: tabular-nums;
  }
  .fe-table th, .fe-table td {
    padding: 9px 0 9px 14px;
    border: none;
    border-bottom: 1px solid var(--fe-border);
    text-align: left;
    white-space: nowrap;
    background: transparent;
  }
  .fe-table td { font-size: 13.5px; color: var(--fe-text); }
  .fe-table th {
    font-family: var(--fe-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--fe-muted);
    border-bottom: 1px solid var(--fe-border-strong);
    white-space: normal;
    vertical-align: bottom;
  }
  .fe-table th:last-child, .fe-table td:last-child { padding-right: 4px; }
  .fe-table tbody tr:hover td { background: var(--fe-surface); }
  /* The row's name may wrap where the table is short of room, and keeps a line down
     its right side while the other columns pass under it (a collapsed table's own
     cell border scrolls away with them). It is prose, so it is set as prose. */
  .fe-table th:first-child, .fe-table td:first-child {
    position: sticky;
    left: 0;
    z-index: 1;
    background: var(--fe-bg);
    box-shadow: inset -1px 0 0 var(--fe-border);
    white-space: normal;
    min-width: 10.5rem;
    padding: 9px 14px 9px 2px;
  }
  .fe-table td:first-child {
    font-family: var(--fe-sans);
    font-size: 13.5px;
    color: var(--fe-text-2);
  }
  .fe-table tbody tr:hover td:first-child {
    background: var(--fe-surface);
    color: var(--fe-text);
  }

  /* ---- Headline figures ----------------------------------------------------------------
     A strip under the page's heading: one cell each, divided by vertical hairlines,
     between two rules that run the full width. It draws its own top rule and the
     heading's is dropped when a strip follows it, so the two never have to be lined
     up across the gap between rows. */
  [data-testid="stElementContainer"]:has(.fe-rule-head):has(
      + [data-testid="stElementContainer"] .fe-strip) {
    display: none;
  }
  .fe-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    border-top: 1px solid var(--fe-border);
    border-bottom: 1px solid var(--fe-border);
    margin: 0 calc(-1 * var(--fe-gutter)) 0.5rem;
    padding: 0 var(--fe-gutter);
    background: linear-gradient(180deg, rgba(16, 20, 26, 0.55), rgba(16, 20, 26, 0));
  }
  .fe-strip-cell {
    padding: 18px 18px 18px 0;
    min-width: 0;
  }
  .fe-strip-cell + .fe-strip-cell {
    border-left: 1px solid var(--fe-border);
    padding-left: 18px;
  }
  /* Narrower cells than the full-width strip: beside the weights it has about two
     thirds of the page, and at 170px its fourth figure dropped to a line of its own
     on a 1280px window. */
  .fe-strip-inline {
    margin: 0;
    padding: 0;
    border: none;
    background: none;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  }
  .fe-strip-inline .fe-strip-cell { padding-top: 0; padding-bottom: 4px; }
  .fe-strip-inline .fe-strip-cell:first-child {
    border-left: 1px solid var(--fe-border);
    padding-left: 18px;
  }
  .fe-strip-label, [data-testid="stMetricLabel"] p {
    font-family: var(--fe-mono);
    font-size: 10.5px !important;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fe-muted);
  }
  .fe-strip-value, [data-testid="stMetricValue"] {
    font-family: var(--fe-sans);
    font-size: 26px;
    font-weight: 400;
    line-height: 34px;
    letter-spacing: -0.02em;
    color: var(--fe-text);
    font-variant-numeric: tabular-nums;
  }
  .fe-strip .fe-lozenge { white-space: normal; letter-spacing: 0.05em; }
  .fe-strip-note {
    font-size: 12px;
    line-height: 1.45;
    color: var(--fe-muted);
    margin-top: 3px;
  }

  /* A long fitted equation scrolls sideways inside its own box instead of being
     cut off at the column's edge. */
  .katex-display { overflow-x: auto; overflow-y: hidden; padding: 4px 0; }

  /* ---- Streamlit's own controls, in the console's dress ------------------------------- */

  [data-testid="stWidgetLabel"] p {
    font-family: var(--fe-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fe-text-2);
  }
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: var(--fe-muted);
  }
  [data-testid="stCaptionContainer"] p { max-width: 110ch; line-height: 1.55; }

  /* Buttons: spaced capitals in a hairline box. The one that starts work is the
     page's only solid white block, so it is found without looking for it. */
  [data-testid^="stBaseButton"] {
    border-radius: 2px;
    transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  [data-testid^="stBaseButton"] p {
    font-family: var(--fe-mono);
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  [data-testid="stBaseButton-secondary"] {
    background: transparent;
    border: 1px solid var(--fe-border-strong);
    color: var(--fe-text-2);
  }
  [data-testid="stBaseButton-secondary"]:hover {
    background: var(--fe-raised);
    border-color: var(--fe-text-2);
    color: var(--fe-text);
  }
  [data-testid="stBaseButton-secondary"]:active { background: var(--fe-border); }
  [data-testid="stBaseButton-primary"] {
    background: var(--fe-text);
    border: 1px solid var(--fe-text);
    color: var(--fe-bg);
  }
  [data-testid="stBaseButton-primary"]:hover {
    background: var(--fe-accent-bright);
    border-color: var(--fe-accent-bright);
    color: var(--fe-sunken);
  }
  [data-testid="stBaseButton-primary"]:active {
    background: var(--fe-accent);
    border-color: var(--fe-accent);
  }
  [data-testid^="stBaseButton"]:focus-visible {
    outline: 2px solid var(--fe-accent);
    outline-offset: 2px;
    box-shadow: none;
  }
  [data-testid^="stBaseButton"]:disabled { opacity: 0.45; }

  /* A choice between a few: joined cells, the chosen one lit. */
  .stButtonGroup button[data-variant] {
    background: transparent;
    border-color: var(--fe-border-strong);
    border-radius: 2px;
    color: var(--fe-text-2);
    transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease;
  }
  .stButtonGroup button[data-variant] p {
    font-family: var(--fe-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .stButtonGroup button[data-variant]:hover {
    background: var(--fe-raised);
    color: var(--fe-text);
  }
  .stButtonGroup button[data-variant][aria-checked="true"],
  .stButtonGroup button[data-variant][aria-pressed="true"],
  .stButtonGroup button[data-variant][data-selected="true"] {
    background: rgba(76, 144, 240, 0.16);
    border-color: var(--fe-accent);
    color: var(--fe-text);
  }
  .stButtonGroup button[data-variant]:focus-visible {
    outline: 2px solid var(--fe-accent);
    outline-offset: 2px;
  }

  /* Tabs: capitals on the page's hairline, the open one underlined in the accent. */
  [data-testid="stTab"] p {
    font-family: var(--fe-mono);
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fe-muted);
    transition: color 120ms ease;
  }
  [data-testid="stTab"]:hover p,
  [data-testid="stTab"][aria-selected="true"] p,
  [data-testid="stTab"][data-selected="true"] p { color: var(--fe-text); }

  /* Panels that open: a hairline box a shade above the page. */
  [data-testid="stExpander"] details {
    border: 1px solid var(--fe-border);
    border-radius: 2px;
    background: rgba(16, 20, 26, 0.6);
  }
  [data-testid="stExpander"] summary { transition: background-color 120ms ease; }
  [data-testid="stExpander"] summary:hover { background: var(--fe-raised); }
  [data-testid="stExpander"] summary p { font-weight: 500; }

  /* Notices: a tinted block with its state's colour down the left edge. */
  [data-testid="stAlertContainer"] {
    border-radius: 2px;
    border-left: 2px solid currentColor;
  }

  /* Inputs sit a shade above the page. */
  [data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background: var(--fe-surface);
    border-radius: 2px;
  }
  [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBar"] {
    font-family: var(--fe-mono);
    font-size: 11px;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: rgba(16, 20, 26, 0.6);
    border: 1px dashed var(--fe-border-strong);
    border-radius: 2px;
  }

  /* A chart's hover read-out: the same dark panel as everything else, figures in
     the monospace face. */
  #vg-tooltip-element.vg-tooltip {
    background: var(--fe-raised) !important;
    border: 1px solid var(--fe-border-strong) !important;
    border-radius: 2px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
    color: var(--fe-text) !important;
    font-family: var(--fe-mono) !important;
    font-size: 11.5px !important;
    padding: 8px 10px !important;
  }
  #vg-tooltip-element.vg-tooltip td.key { color: var(--fe-muted) !important; }
  #vg-tooltip-element.vg-tooltip td.value { color: var(--fe-text) !important; }

  /* ---- Work under way ------------------------------------------------------------------
     A light runs along the status bar's lower edge while the app is computing, so a
     click is seen to have been taken. It waits a quarter of a second before showing:
     most clicks are answered sooner, and a flash on every one would read as lag. */
  @keyframes fe-working {
    from { transform: translateX(-100%); }
    to   { transform: translateX(500%); }
  }
  .fe-statusbar::after {
    content: "";
    position: absolute;
    left: 0; bottom: 0;
    height: 1px; width: 20%;
    background: linear-gradient(90deg, transparent, var(--fe-accent-bright), transparent);
    transform: translateX(-100%);
  }
  [data-testid="stApp"][data-test-script-state="running"] .fe-statusbar::after {
    animation: fe-working 1.1s linear 250ms infinite;
  }

  /* ---- The sidebar's groups and foot ---------------------------------------------------- */
  [data-testid="stNavSectionHeader"] {
    font-family: var(--fe-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--fe-muted);
    padding: 1.1rem 0.75rem 0.35rem;
  }
  .fe-side-foot {
    padding: 0.2rem 0.5rem 0;
    font-family: var(--fe-mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    line-height: 1.8;
    color: var(--fe-muted);
  }
  .fe-side-foot b { color: var(--fe-text-2); font-weight: 500; }

  /* ---- Where a page's sections are ------------------------------------------------------
     A long page keeps a line of its sections under the status bar: where you are
     going is one click away however far down you are. */
  [data-testid="stElementContainer"]:has(.fe-subnav) {
    position: sticky;
    top: var(--fe-bar);
    z-index: 99;
    margin: 0 calc(-1 * var(--fe-gutter));
    width: calc(100% + 2 * var(--fe-gutter));
    max-width: none;
  }
  .fe-subnav {
    display: flex;
    align-items: center;
    gap: 1.6rem;
    height: 2.2rem;
    padding: 0 var(--fe-gutter);
    background: rgba(11, 13, 17, 0.9);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--fe-border);
    overflow-x: auto;
    white-space: nowrap;
  }
  .fe-subnav a {
    font-family: var(--fe-mono);
    font-size: 10.5px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-decoration: none;
    color: var(--fe-muted);
    transition: color 120ms ease;
  }
  .fe-subnav a:hover, .fe-subnav a:focus-visible { color: var(--fe-text); }
  .fe-subnav a:focus-visible { outline: 2px solid var(--fe-accent); outline-offset: 3px; }
  /* A section jumped to stops clear of the two bars above it. */
  .fe-section-title[id] { scroll-margin-top: 6rem; }

  /* ---- A share of a whole ---------------------------------------------------------------
     Two weights that sum to one are a split bar before they are two numbers: the
     proportion is seen at once, and the ticks show the limits it is held within. */
  .fe-split { position: relative; display: flex; gap: 2px; height: 8px; margin: 10px 0 6px; }
  .fe-split-part { height: 100%; min-width: 2px; border-radius: 1px; }
  .fe-split-tick {
    position: absolute;
    top: -4px; bottom: -4px;
    width: 1px;
    background: var(--fe-text-2);
  }
  .fe-split-key {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    font-family: var(--fe-mono);
    font-size: 10.5px;
    letter-spacing: 0.06em;
    color: var(--fe-muted);
  }

  /* A change against what a figure is compared with: an arrow for its direction, and
     green or red for whether that direction is the good one, which is not always up. */
  .fe-delta {
    font-family: var(--fe-mono);
    font-size: 11px;
    font-weight: 500;
    white-space: nowrap;
  }
  .fe-delta-good { color: var(--fe-success-fg); }
  .fe-delta-bad { color: var(--fe-danger-fg); }
  .fe-delta-flat { color: var(--fe-muted); }

  /* ---- A chart's key, with each line's latest figure -------------------------------------- */
  .fe-readout {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 20px;
    font-size: 12.5px;
    color: var(--fe-text-2);
  }
  .fe-readout-item { display: inline-flex; align-items: center; gap: 8px; white-space: nowrap; }
  .fe-readout-swatch { width: 14px; height: 2px; border-radius: 1px; flex: none; }
  .fe-readout-band { width: 9px; height: 9px; }
  .fe-readout-value {
    color: var(--fe-text);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }
  .fe-readout-asof {
    margin-left: auto;
    font-family: var(--fe-mono);
    font-size: 10.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--fe-muted);
  }

  /* ---- Tables of figures -------------------------------------------------------------------
     Figures are set against the right edge of their column, so digits of the same
     weight sit under one another; ``fe-num`` does it from the second column and
     ``fe-num3`` from the third (a table whose second column is words). ``fe-ours``
     marks the column that is the app's own result. */
  .fe-num .fe-table th:not(:first-child), .fe-num .fe-table td:not(:first-child),
  .fe-num3 .fe-table th:nth-child(n + 3), .fe-num3 .fe-table td:nth-child(n + 3) {
    text-align: right;
  }
  .fe-num .fe-table th:last-child, .fe-num .fe-table td:last-child,
  .fe-num3 .fe-table th:last-child, .fe-num3 .fe-table td:last-child { padding-right: 12px; }
  .fe-ours2 .fe-table td:nth-child(2), .fe-ours3 .fe-table td:nth-child(3) {
    background: rgba(76, 144, 240, 0.07);
    font-weight: 500;
  }
  .fe-ours2 .fe-table th:nth-child(2), .fe-ours3 .fe-table th:nth-child(3) {
    background: rgba(76, 144, 240, 0.07);
    color: var(--fe-accent-bright);
  }
  .fe-ours2 .fe-table td:nth-child(2), .fe-ours2 .fe-table th:nth-child(2),
  .fe-ours3 .fe-table td:nth-child(3), .fe-ours3 .fe-table th:nth-child(3) { padding-right: 12px; }
  .fe-table td.fe-pos { color: var(--fe-success-fg); }
  .fe-table td.fe-neg { color: var(--fe-danger-fg); }

  /* ---- A note kept one click away ------------------------------------------------------------
     How a chart was computed is wanted once, not on every visit. It folds into a
     small label under the chart instead of standing between it and the next one. */
  [class*="st-key-fe-note"] [data-testid="stExpander"] details {
    border: none;
    background: none;
  }
  [class*="st-key-fe-note"] [data-testid="stExpander"] summary {
    width: fit-content;
    min-height: 0;
    padding: 2px 6px 2px 0;
    color: var(--fe-muted);
  }
  [class*="st-key-fe-note"] [data-testid="stExpander"] summary:hover {
    background: none;
    color: var(--fe-text);
  }
  [class*="st-key-fe-note"] [data-testid="stExpander"] summary p {
    font-family: var(--fe-mono);
    font-size: 10.5px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  [class*="st-key-fe-note"] [data-testid="stExpanderDetails"] { padding: 4px 0 0; }

  /* ---- The way on --------------------------------------------------------------------------- */
  [data-testid="stPageLink"] a {
    border: 1px solid var(--fe-border-strong);
    border-radius: 2px;
    padding: 0.5rem 0.9rem;
    background: transparent;
    transition: background-color 120ms ease, border-color 120ms ease;
  }
  [data-testid="stPageLink"] a:hover {
    background: var(--fe-raised);
    border-color: var(--fe-text-2);
  }
  [data-testid="stPageLink"] a:focus-visible {
    outline: 2px solid var(--fe-accent);
    outline-offset: 2px;
  }
  [data-testid="stPageLink"] p {
    font-family: var(--fe-mono);
    font-size: 11.5px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fe-text);
  }

  /* The page's opening lines arrive once, on entering the page; nothing else moves
     by itself. Only opacity and position, which the compositor handles alone. */
  @keyframes fe-arrive {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: none; }
  }
  .fe-page-head { animation: fe-arrive 320ms cubic-bezier(0.16, 1, 0.3, 1) both; }

  @media (prefers-reduced-motion: reduce) {
    .fe-page-head { animation: none; }
    [data-testid="stApp"][data-test-script-state="running"] .fe-statusbar::after {
      animation: none;
      transform: none;
      width: 100%;
    }
    * { transition: none !important; }
  }

  @media (max-width: 760px) {
    :root { --fe-gutter: 1rem; }
    .fe-statusbar { padding-right: var(--fe-gutter); }
    .fe-status-cell:nth-child(n + 2) { display: none; }
    /* A row's name gets the line to itself and its state goes underneath, so
       neither is squeezed into a column a word wide. */
    .fe-row { flex-wrap: wrap; gap: 0.35rem 1rem; }
    .fe-row-name { flex: 1 1 100%; }
    .fe-lozenge { white-space: normal; }
  }
</style>
"""
).substitute(
    TONES=_TONE_VARIABLES,
    BG=tokens.BG,
    SUNKEN=tokens.SUNKEN,
    SURFACE=tokens.SURFACE,
    RAISED=tokens.RAISED,
    BORDER=tokens.BORDER,
    BORDER_STRONG=tokens.BORDER_STRONG,
    TEXT=tokens.TEXT,
    TEXT_SECONDARY=tokens.TEXT_SECONDARY,
    MUTED=tokens.MUTED,
    ACCENT=tokens.ACCENT,
    ACCENT_BRIGHT=tokens.ACCENT_BRIGHT,
    SANS=tokens.SANS,
    MONO=tokens.MONO,
    GUTTER=_GUTTER,
)


def stylesheet() -> str:
    """The shared stylesheet, as markup to put in the same element as something seen:
    in an element of its own it is an empty row on the page, with a row's gap around it.

    It goes out on every run rather than once per session. Streamlit renders each page
    from scratch, so a stylesheet sent with the Data page is not there on Home, and a
    session-scoped guard would silently leave the second page unstyled. Two identical
    style elements in one run (the status bar's and the page header's) are inert."""
    return _CSS


def lozenge(text: str, tone: str = "neutral") -> str:
    """A status badge, as HTML to embed in a markdown block."""
    if tone not in TONES:
        tone = "neutral"
    return (
        f'<span class="fe-lozenge" style="background:var(--fe-{tone}-bg);'
        f'color:var(--fe-{tone}-fg)">{text}</span>'
    )


def status_row(name: str, badge: str, meta: str = "") -> str:
    """One line of a status list: name, optional detail, then the badge."""
    detail = f'<span class="fe-row-meta">{meta}</span>' if meta else ""
    return (
        f'<div class="fe-row"><span class="fe-row-name">{name}</span>'
        f"<span>{detail}&nbsp;&nbsp;{badge}</span></div>"
    )


def finding_row(badge: str, where: str, detail: str) -> str:
    """One finding: badge and location on top, the message in full beneath.

    A table clips the message, and the message is the part that explains the
    finding. Here it wraps instead.
    """
    location = f'<span class="fe-finding-where">{where}</span>' if where else ""
    return (
        f'<div class="fe-finding"><div class="fe-finding-head">{badge}{location}</div>'
        f'<div class="fe-finding-detail">{detail}</div></div>'
    )


def eyebrow(text: str, help: str | None = None) -> str:
    """A small spaced-capital label, optionally with a hover explanation.

    ``help`` renders as an ⓘ carrying the browser's own title tooltip, since a
    markdown block has no access to Streamlit's ``help=``. A blank line in
    ``help`` (a multi-paragraph glossary entry) would otherwise split the
    ``title`` attribute across two markdown blocks and leak the tag's tail as
    visible text, so whitespace is collapsed to single spaces first.
    """
    title = html.escape(" ".join(help.split()), quote=True) if help else ""
    hint = f'<span class="fe-eyebrow-help" title="{title}">i</span>' if help else ""
    return f'<div class="fe-eyebrow">{text}{hint}</div>'


def page_header(title: str, summary: str | None = None, *, kicker: str | None = None) -> None:
    """The page's kicker, title and one-line summary, then a rule the full width of
    the content area. Every page opens with this, so they all start the same way.

    ``kicker`` is the small line above the title: the page's place in the pipeline
    and what happens there, such as ``"03 / Allocate"``. The number is dimmed and the
    rest takes the accent."""
    above = kicker_line(kicker) if kicker else ""
    text = f'<div class="fe-page-summary">{summary}</div>' if summary else ""
    # The stylesheet travels with the heading, in one element: an element of its own
    # would be an empty row above the title, with a row's gap around it.
    st.markdown(
        f'{_CSS}<div class="fe-page-head">{above}'
        f'<div class="fe-page-title" role="heading" aria-level="1">{html.escape(title)}</div>'
        f"{text}</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fe-rule fe-rule-head"></div>', unsafe_allow_html=True)


def kicker_line(kicker: str, *, small: bool = False) -> str:
    """A numbered stage line such as ``"03 / Allocate"``, as HTML: the number dimmed,
    the stage in the accent. ``small`` is the size used at the head of a panel."""
    number, slash, stage = kicker.partition("/")
    kind = "fe-kicker fe-kicker-small" if small else "fe-kicker"
    if not slash:
        return f'<div class="{kind}">{html.escape(kicker)}</div>'
    return (
        f'<div class="{kind}"><b>{html.escape(number.strip())} /</b> '
        f"{html.escape(stage.strip())}</div>"
    )


def rule() -> None:
    """A hairline from the sidebar's edge to the window's."""
    st.markdown('<div class="fe-rule"></div>', unsafe_allow_html=True)


def section(
    title: str, help: str | None = None, *, note: str | None = None, anchor: str | None = None
) -> None:
    """A full-width rule, then a section title: how a page is divided into parts.
    ``note`` is a line of HTML under the title, such as a row of lozenges. ``anchor``
    names the section so ``section_nav`` can jump to it."""
    rule()
    hint = ""
    if help:
        tip = html.escape(" ".join(help.split()), quote=True)
        hint = f'<span class="fe-eyebrow-help" title="{tip}">i</span>'
    below = f'<div class="fe-row-meta" style="margin-top:4px">{note}</div>' if note else ""
    name = f' id="{html.escape(anchor, quote=True)}"' if anchor else ""
    st.markdown(
        f'<div class="fe-section-title" role="heading" aria-level="2"{name}>'
        f"{html.escape(title)}{hint}</div>{below}",
        unsafe_allow_html=True,
    )


def section_nav(sections: Sequence[tuple[str, str]]) -> None:
    """A line of a long page's sections, each ``(label, anchor)``, that stays under the
    status bar as the page scrolls. The anchors are the ones given to ``section``."""
    links = "".join(
        f'<a href="#{html.escape(anchor, quote=True)}" target="_self">{html.escape(label)}</a>'
        for label, anchor in sections
    )
    st.markdown(
        f'<nav class="fe-subnav" aria-label="Sections of this page">{links}</nav>',
        unsafe_allow_html=True,
    )


def strip(cells: list[tuple[str, str, str]], *, inline: bool = False) -> str:
    """A row of headline figures, each ``(label, value, note)``, as HTML. Values
    and notes may hold markup such as a lozenge; labels are escaped. ``inline``
    sits the row inside a column rather than across the full width."""
    body = "".join(
        f'<div class="fe-strip-cell"><div class="fe-strip-label">{html.escape(label)}</div>'
        f'<div class="fe-strip-value">{value}</div>'
        + (f'<div class="fe-strip-note">{note}</div>' if note else "")
        + "</div>"
        for label, value, note in cells
    )
    kind = "fe-strip fe-strip-inline" if inline else "fe-strip"
    return f'<div class="{kind}">{body}</div>'


def split_bar(
    parts: Sequence[tuple[str, float, str]],
    *,
    limits: tuple[float, float] | None = None,
    labelled: bool = True,
) -> str:
    """Shares of a whole as one bar, each part ``(name, share, colour)`` drawn as wide
    as its share, as HTML. ``limits`` are the lowest and highest share the first part
    may take, marked with a tick each, so a share pinned at a limit is seen to be.

    The names and figures are written under the bar as well, since the colours alone
    would not say which part is which. Pass ``labelled=False`` only where they are
    already written beside it, in the same order."""
    total = sum(share for _, share, _ in parts) or 1.0
    segments = "".join(
        f'<span class="fe-split-part" style="flex:{max(share, 0) / total:.4f} 1 0;'
        f'background:{colour}"></span>'
        for _, share, colour in parts
    )
    ticks = "".join(
        f'<span class="fe-split-tick" style="left:{limit:.2%}"></span>' for limit in limits or ()
    )
    key = "".join(
        f"<span>{html.escape(name)} {share:.0%}</span>" for name, share, _ in parts
    )
    described = html.escape(", ".join(f"{name} {share:.0%}" for name, share, _ in parts))
    bar = f'<div class="fe-split" role="img" aria-label="{described}">{segments}{ticks}</div>'
    return bar + (f'<div class="fe-split-key">{key}</div>' if labelled else "")


def delta(change: float, text: str, *, higher_is_better: bool = True) -> str:
    """A change against what a figure is compared with, as HTML: an arrow for its
    direction and ``text`` for its size. Green or red says whether the direction is
    the good one, which for a loss or a risk is down, so the arrow and the colour are
    two different facts and neither is left to the other."""
    if change == 0 or change != change:  # nothing moved, or nothing to compare
        return f'<span class="fe-delta fe-delta-flat">{html.escape(text)}</span>'
    arrow = "▲" if change > 0 else "▼"
    good = (change > 0) == higher_is_better
    kind = "fe-delta-good" if good else "fe-delta-bad"
    return f'<span class="fe-delta {kind}">{arrow} {html.escape(text)}</span>'


@contextmanager
def note(label: str, key: str) -> Iterator[None]:
    """A note kept one click away: what is written inside is folded under a small
    label. For how a figure was computed, which is wanted once and not on every
    visit. ``key`` tells one note on a page from another."""
    with st.container(key=f"fe-note-{key}"), st.expander(label):
        yield


#: Page name -> the page, as the entry point registered it (``register_pages``).
_PAGES: dict[str, object] = {}


def register_pages(pages: Mapping[str, object]) -> None:
    """Tell the pages where each other are, so one can link to the next. Called by the
    entry point on every run; a page run on its own (a test) has no others to link to."""
    _PAGES.clear()
    _PAGES.update(pages)


def page_link(name: str, label: str) -> None:
    """A link to the page registered as ``name``, or nothing where the pages were not
    registered: a page run on its own has nowhere to go."""
    page = _PAGES.get(name)
    if page is not None:
        st.page_link(page, label=label, icon=":material/arrow_forward:", icon_position="right")
