"""Shared presentation for the dashboard.

The visual language is GitHub Primer's, which Atlassian's shares its bones with:
an enterprise blue accent, neutral greys carrying the structure, and semantic
colour reserved for state. The palette lives in ``.streamlit/config.toml``; this
module holds the two patterns Streamlit has no component for.

**Status lozenges.** A short uppercase badge is how both systems show state, and
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

_CSS = """
<style>
  :root {
    --fe-neutral-bg: #EFF2F5; --fe-neutral-fg: #59636E;
    --fe-info-bg:    #DDF4FF; --fe-info-fg:    #0550AE;
    --fe-success-bg: #DAFBE1; --fe-success-fg: #1A7F37;
    --fe-warning-bg: #FFF8C5; --fe-warning-fg: #7D4E00;
    --fe-danger-bg:  #FFEBE9; --fe-danger-fg:  #CF222E;
    --fe-border:     #D1D9E0;
    --fe-muted:      #59636E;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --fe-neutral-bg: #21262D; --fe-neutral-fg: #9198A1;
      --fe-info-bg:    #121D2F; --fe-info-fg:    #4493F8;
      --fe-success-bg: #0F2417; --fe-success-fg: #3FB950;
      --fe-warning-bg: #272115; --fe-warning-fg: #D29922;
      --fe-danger-bg:  #2A1618; --fe-danger-fg:  #F85149;
      --fe-border:     #3D444D;
      --fe-muted:      #9198A1;
    }
  }

  /* Streamlit prints its own server.maxUploadSize inside the dropzone. Ours is
     deliberately looser than the documented limit, so showing it would
     contradict the caption above the control. */
  [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

  .fe-lozenge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    line-height: 16px;
    letter-spacing: .04em;
    text-transform: uppercase;
    white-space: nowrap;
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
  .fe-row-meta { color: var(--fe-muted); font-size: 12.5px; }

  /* One finding per line. The message is prose of unpredictable length, so it
     gets the full width and wraps, rather than being clipped inside a table
     cell. Location and value sit above it in a muted meta line. */
  .fe-finding {
    padding: 9px 0;
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
  .fe-finding-where { color: var(--fe-muted); font-size: 12.5px; }
  .fe-finding-detail { font-size: 14px; line-height: 1.45; }

  /* A plain table for cells needing an inline lozenge badge — st.dataframe
     can't render those. */
  /* Every cell is nowrap, so a wide comparison scrolls inside its own box
     rather than running off the page. */
  .fe-table-wrap { overflow-x: auto; }
  .fe-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
  }
  .fe-table th, .fe-table td {
    padding: 8px 12px;
    border-bottom: 1px solid var(--fe-border);
    text-align: left;
    white-space: nowrap;
  }
  .fe-table th {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: var(--fe-muted);
  }

  .fe-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: var(--fe-muted);
    margin-bottom: 2px;
  }

  /* Streamlit's own help tooltip isn't available on raw markdown, so a
     section label explains itself through the browser's title tooltip. */
  .fe-eyebrow-help {
    border: 1px solid var(--fe-border);
    border-radius: 50%;
    color: var(--fe-muted);
    cursor: help;
    display: inline-block;
    font-size: 9px;
    font-weight: 600;
    height: 12px;
    line-height: 12px;
    margin-left: 5px;
    text-align: center;
    text-transform: none;
    width: 12px;
  }
</style>
"""


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
    hint = (
        f'<span class="fe-eyebrow-help" title="{html.escape(" ".join(help.split()), quote=True)}">i</span>'
        if help
        else ""
    )
    return f'<div class="fe-eyebrow">{text}{hint}</div>'
