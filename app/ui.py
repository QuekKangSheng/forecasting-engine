"""Shared presentation for the dashboard.

The visual language is the Atlassian Design System's: navy text, one blue for
actions, neutral greys carrying the structure, semantic colour reserved for
state, and 3px corners throughout. The palette lives in
``.streamlit/config.toml``; this module holds what Streamlit has no component
for.

**Page headers and rules.** A page opens with its title and a one-line summary,
then a hairline that runs the full width of the content area, edge to edge, as
under a Jira or Confluence page header. Sections are separated by the same
full-width rule rather than by floating cards.

**Status lozenges.** A short uppercase badge is how Atlassian shows state, and
it reads at a glance in a way a coloured word does not.

**Status rows.** Name on the left, state on the right, a hairline between.

Colours are duplicated here as custom properties rather than read from
Streamlit, which renders through hashed emotion classes and exposes no variables
to borrow. They are keyed off ``prefers-color-scheme``, which is what Streamlit
itself follows when choosing between the config's light and dark palettes, so
the two stay in step. A viewer who overrides the theme in Streamlit's own
settings menu is the one case where they can diverge.

No emoji anywhere: an internal analytical tool should read as a tool.
"""

from __future__ import annotations

import html

import streamlit as st

TONES: tuple[str, ...] = ("neutral", "info", "success", "warning", "danger")

#: The content area's side padding. Full-width rules reach past it by the same
#: amount, so they run from the sidebar's edge to the window's.
_GUTTER = "2rem"

_CSS = """
<style>
  :root {
    --fe-neutral-bg: #F1F2F4; --fe-neutral-fg: #44546F;
    --fe-info-bg:    #E9F2FF; --fe-info-fg:    #0055CC;
    --fe-success-bg: #DCFFF1; --fe-success-fg: #216E4E;
    --fe-warning-bg: #FFF7D6; --fe-warning-fg: #7F5F01;
    --fe-danger-bg:  #FFECEB; --fe-danger-fg:  #AE2E24;
    --fe-border:     #DCDFE4;
    --fe-surface:    #FFFFFF;
    --fe-sunken:     #F7F8F9;
    --fe-text:       #172B4D;
    --fe-muted:      #626F86;
    --fe-gutter:     GUTTER;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --fe-neutral-bg: #2C333A; --fe-neutral-fg: #9FADBC;
      --fe-info-bg:    #09326C; --fe-info-fg:    #85B8FF;
      --fe-success-bg: #164B35; --fe-success-fg: #7EE2B8;
      --fe-warning-bg: #533F04; --fe-warning-fg: #F5CD47;
      --fe-danger-bg:  #5D1F1A; --fe-danger-fg:  #FD9891;
      --fe-border:     #38414A;
      --fe-surface:    #1D2125;
      --fe-sunken:     #161A1D;
      --fe-text:       #B6C2CF;
      --fe-muted:      #8C9BAB;
    }
  }

  /* Wide pages with a fixed gutter, so the full-width rules below can reach
     exactly to both edges of the content area. */
  [data-testid="stMainBlockContainer"] {
    padding: 3rem var(--fe-gutter) 4rem !important;
    max-width: none !important;
  }

  /* Streamlit prints its own server.maxUploadSize inside the dropzone. Ours is
     deliberately looser than the documented limit, so showing it would
     contradict the caption above the control. */
  [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

  /* A hairline from the sidebar's edge to the window's. */
  .fe-rule {
    border-top: 1px solid var(--fe-border);
    margin: 0 calc(-1 * var(--fe-gutter));
  }

  .fe-page-title {
    font-size: 24px;
    font-weight: 600;
    line-height: 28px;
    color: var(--fe-text);
    margin: 0;
  }
  .fe-page-summary {
    color: var(--fe-muted);
    font-size: 14px;
    line-height: 20px;
    margin: 4px 0 0;
    max-width: 90ch;
  }
  .fe-section-title {
    font-size: 16px;
    font-weight: 600;
    line-height: 20px;
    color: var(--fe-text);
    margin: 0;
  }

  .fe-lozenge {
    display: inline-block;
    padding: 0 4px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 700;
    line-height: 16px;
    text-transform: uppercase;
    white-space: nowrap;
    vertical-align: middle;
  }

  .fe-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 8px 0;
    border-bottom: 1px solid var(--fe-border);
  }
  .fe-row:last-child { border-bottom: none; }
  .fe-row-name { font-size: 14px; }
  .fe-row-meta { color: var(--fe-muted); font-size: 12px; }

  /* One finding per line. The message is prose of unpredictable length, so it
     gets the full width and wraps, rather than being clipped inside a table
     cell. Location and value sit above it in a muted meta line. */
  .fe-finding {
    padding: 8px 0;
    border-bottom: 1px solid var(--fe-border);
  }
  .fe-finding:last-child { border-bottom: none; }
  .fe-finding-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 2px;
  }
  .fe-finding-where { color: var(--fe-muted); font-size: 12px; }
  .fe-finding-detail { font-size: 14px; line-height: 20px; }

  /* A plain table for cells needing an inline lozenge badge — st.dataframe
     can't render those. Rows are divided by hairlines and the header by a
     heavier rule, with no outer box, as Atlassian's tables are. Every cell is
     nowrap, so a wide comparison scrolls inside its own box rather than
     running off the page. */
  .fe-table-wrap { overflow-x: auto; }
  .fe-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    font-variant-numeric: tabular-nums;
  }
  .fe-table th, .fe-table td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--fe-border);
    text-align: left;
    white-space: nowrap;
  }
  .fe-table th {
    font-size: 12px;
    font-weight: 600;
    color: var(--fe-muted);
    border-bottom: 2px solid var(--fe-border);
  }

  /* A label above a group of figures: a small heading in the text colour,
     not a spaced-out capital eyebrow. */
  .fe-eyebrow {
    font-size: 14px;
    font-weight: 600;
    color: var(--fe-text);
    margin-bottom: 2px;
  }

  /* Streamlit's own help tooltip isn't available on raw markdown, so a
     section label explains itself through the browser's title tooltip. */
  .fe-eyebrow-help {
    border: 1px solid var(--fe-muted);
    border-radius: 50%;
    color: var(--fe-muted);
    cursor: help;
    display: inline-block;
    font-size: 9px;
    font-weight: 600;
    height: 12px;
    line-height: 11px;
    margin-left: 5px;
    text-align: center;
    width: 12px;
    vertical-align: 1px;
  }

  /* A strip of headline figures: one cell each, divided by vertical hairlines,
     with a rule above and below that runs the full width. */
  .fe-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    border-top: 1px solid var(--fe-border);
    border-bottom: 1px solid var(--fe-border);
    margin: 0 calc(-1 * var(--fe-gutter));
    padding: 0 var(--fe-gutter);
  }
  .fe-strip-cell {
    padding: 12px 16px 12px 0;
    min-width: 0;
  }
  .fe-strip-cell + .fe-strip-cell {
    border-left: 1px solid var(--fe-border);
    padding-left: 16px;
  }
  .fe-strip-inline {
    margin: 0;
    padding: 0;
    border: none;
  }
  .fe-strip-inline .fe-strip-cell:first-child {
    border-left: 1px solid var(--fe-border);
    padding-left: 16px;
  }
  .fe-strip-label { font-size: 12px; font-weight: 600; color: var(--fe-muted); }
  .fe-strip-value {
    font-size: 20px;
    font-weight: 600;
    line-height: 28px;
    color: var(--fe-text);
    font-variant-numeric: tabular-nums;
  }
  .fe-strip .fe-lozenge { white-space: normal; }
  .fe-strip-note { font-size: 12px; color: var(--fe-muted); margin-top: 2px; }

  /* Streamlit's metrics at the size of a strip value, so the two sit level. */
  [data-testid="stMetricValue"] { font-size: 20px; font-weight: 600; line-height: 28px; }
  [data-testid="stMetricLabel"] p { font-size: 12px; font-weight: 600; color: var(--fe-muted); }
</style>
""".replace("GUTTER", _GUTTER)


def inject() -> None:
    """Add the shared stylesheet to the current page.

    Called on every run rather than guarded by a session flag. Streamlit renders
    each page from scratch, so a stylesheet injected on the Data page is not
    present on Home — a session-scoped guard silently leaves the second page
    unstyled. Two identical style elements in one run are inert, so the
    unconditional call is the safe one.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


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
    """A small uppercase section label, optionally with a hover explanation.

    ``help`` renders as an ⓘ carrying the browser's own title tooltip, since a
    markdown block has no access to Streamlit's ``help=``. A blank line in
    ``help`` (a multi-paragraph glossary entry) would otherwise split the
    ``title`` attribute across two markdown blocks and leak the tag's tail as
    visible text, so whitespace is collapsed to single spaces first.
    """
    title = html.escape(" ".join(help.split()), quote=True) if help else ""
    hint = f'<span class="fe-eyebrow-help" title="{title}">i</span>' if help else ""
    return f'<div class="fe-eyebrow">{text}{hint}</div>'


def page_header(title: str, summary: str | None = None) -> None:
    """The page's title and one-line summary, then a rule the full width of the
    content area. Every page opens with this, so they all start the same way."""
    inject()
    text = f'<p class="fe-page-summary">{summary}</p>' if summary else ""
    st.markdown(
        f'<div class="fe-page-title" role="heading" aria-level="1">{html.escape(title)}</div>'
        f"{text}", unsafe_allow_html=True
    )
    rule()


def rule() -> None:
    """A hairline from the sidebar's edge to the window's."""
    st.markdown('<div class="fe-rule"></div>', unsafe_allow_html=True)


def section(title: str, help: str | None = None, *, note: str | None = None) -> None:
    """A full-width rule, then a section title: how a page is divided into parts.
    ``note`` is a line of HTML under the title, such as a row of lozenges."""
    rule()
    hint = ""
    if help:
        tip = html.escape(" ".join(help.split()), quote=True)
        hint = f'<span class="fe-eyebrow-help" title="{tip}">i</span>'
    below = f'<div class="fe-row-meta" style="margin-top:4px">{note}</div>' if note else ""
    st.markdown(
        f'<div class="fe-section-title" role="heading" aria-level="2">'
        f"{html.escape(title)}{hint}</div>{below}",
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
