"""The Home page's opening, around the market field (``market_field``).

Each piece is a function that returns HTML for ``st.markdown``:

- ``lead``: the maker's name as the page's heading, typed out a letter at a time
  behind a caret, then where things stand (the latest weights, or the next step).
- ``typed``: one line of status, typed out once in the monospace face.
- ``tape``: the latest value of every committed series and its change over one
  horizon, moving slowly like a ticker and stopping under the pointer.

``CSS`` styles these and the field. Motion runs once (the tape, the caret and the
field's two rings excepted) and is switched off for a viewer who asked for less.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

import tokens

#: Days of history in each tape sparkline, and the points it is drawn from.
SPARK_DAYS: int = 120
SPARK_POINTS: int = 40


@dataclass(frozen=True)
class Tick:
    """One series on the tape."""

    name: str
    last: str
    change: float
    change_text: str
    spark: tuple[float, ...]


def ticks(
    frame: pd.DataFrame,
    columns: Sequence[str],
    names: Sequence[str],
    as_return: Sequence[bool],
    span: int,
) -> tuple[Tick, ...]:
    """A tick per column of ``frame`` (indexed by date) that has ``span`` rows of
    history: its last value and its change over the last ``span`` of its own rows.

    A price's change is a percentage (``as_return``); a rate's, a spread's or a
    gauge's is in its own units, since a yield moving from 4.0 to 4.4 is 0.4
    points, not 10%.
    """
    out = []
    for column, name, returns in zip(columns, names, as_return, strict=True):
        series = frame[column].dropna()
        if len(series) <= span:
            continue
        last, before = float(series.iloc[-1]), float(series.iloc[-1 - span])
        if returns and before != 0:
            change = last / before - 1
            text = f"{change:+.1%}"
        else:
            change = last - before
            text = f"{change:+.2f}"
        recent = series.iloc[-SPARK_DAYS:]
        step = max(len(recent) // SPARK_POINTS, 1)
        # Thinned from the newest value backwards, so the line always ends on it.
        spark = tuple(float(v) for v in recent.iloc[::-1].iloc[::step].iloc[::-1])
        out.append(Tick(name, f"{last:,.2f}", change, text, spark))
    return tuple(out)


# ── The words ────────────────────────────────────────────────────────────────


def typed(text: str) -> str:
    """``text`` typed out once, a character at a time, with a block cursor that stays.

    The whole line is in the page from the start and only uncovered, so a screen
    reader, a search and a viewer who asked for less motion all get it at once.
    """
    count = len(text)
    timing = f"animation-duration:{count * 26}ms;animation-timing-function:steps({count})"
    return (
        f'<div class="fe-typed" style="--n:{count}">'
        f'<span class="fe-typed-text" style="{timing}">{html.escape(text)}</span>'
        f'<span class="fe-typed-caret" style="{timing}" aria-hidden="true"></span></div>'
    )


def lead(
    name: str,
    label: str,
    statement: str,
    below: str,
    typed_html: str,
    *,
    long: bool = False,
) -> str:
    """The opening's words: ``name`` as the page's heading, typed out a letter at a
    time behind a caret; under it a small label, the statement of where things
    stand, what goes under that (HTML its caller has escaped) and the typed status
    line. ``long`` sets a statement of a whole sentence a size down.

    The heading is read once, as ``name``: its letters, there only to be typed, are
    hidden from a screen reader."""
    letters = "".join(
        f'<span style="--i:{i}">{html.escape(letter)}</span>' for i, letter in enumerate(name)
    )
    size = " fe-hero-state-long" if long else ""
    return (
        f'<div class="fe-hero-copy">'
        f'<div class="fe-hero-name" role="heading" aria-level="1" '
        f'aria-label="{html.escape(name, quote=True)}">'
        f'<span class="fe-hero-letters" aria-hidden="true">{letters}</span>'
        f'<span class="fe-hero-caret" aria-hidden="true"></span></div>'
        f'<div class="fe-hero-label">{html.escape(label)}</div>'
        f'<div class="fe-hero-state{size}">{html.escape(statement)}</div>'
        f'<div class="fe-hero-below">{below}</div>{typed_html}</div>'
    )


# ── The tape ─────────────────────────────────────────────────────────────────


def _spark(values: Sequence[float], colour: str) -> str:
    if len(values) < 2:
        return ""
    low, high = min(values), max(values)
    spread = (high - low) or 1.0
    step = 56 / (len(values) - 1)
    points = " ".join(
        f"{i * step:.1f},{15 - 13 * (value - low) / spread:.1f}" for i, value in enumerate(values)
    )
    return (
        '<svg class="fe-tick-spark" viewBox="0 0 56 16" aria-hidden="true">'
        f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="1.2" '
        'stroke-linejoin="round"/></svg>'
    )


def _tick(tick: Tick) -> str:
    if tick.change > 0:
        kind, arrow, colour = "fe-tick-up", "▲", tokens.RISE
    elif tick.change < 0:
        kind, arrow, colour = "fe-tick-down", "▼", tokens.FALL
    else:
        kind, arrow, colour = "fe-tick-flat", "■", tokens.MUTED
    return (
        f'<span class="fe-tick"><b>{html.escape(tick.name)}</b>'
        f'<span class="fe-tick-last">{html.escape(tick.last)}</span>'
        f'<span class="{kind}">{arrow} {html.escape(tick.change_text.lstrip("+-"))}</span>'
        f"{_spark(tick.spark, colour)}</span>"
    )


def tape(items: Sequence[Tick], span: int) -> str:
    """The committed series as a ticker. The items are written twice so the strip can
    loop without a gap; the second copy is hidden from a screen reader."""
    if not items:
        return ""
    cells = "".join(_tick(item) for item in items)
    # About five seconds a series: slow enough to read a figure as it passes.
    seconds = max(len(items) * 5, 30)
    return (
        f'<div class="fe-tape" role="group" aria-label="Latest value of each committed '
        f'series and its change over {span} trading days">'
        f'<div class="fe-tape-head">Latest · {span}-day change</div>'
        f'<div class="fe-tape-window"><div class="fe-tape-track" '
        f'style="animation-duration:{seconds}s">'
        f'<span class="fe-tape-run">{cells}</span>'
        f'<span class="fe-tape-run" aria-hidden="true">{cells}</span></div></div></div>'
    )


CSS = f"""
<style>
  :root {{ --fe-rise: {tokens.RISE}; --fe-fall: {tokens.FALL}; }}

  /* ---- The hero ----------------------------------------------------------------------
     Words on the left, the field on the right, meeting at the bottom so the candles
     stand on the same line the words end on. */
  .fe-hero-copy {{
    padding: 3.4rem 0 0.4rem;
    animation: fe-arrive 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
  }}
  .fe-hero-label {{
    margin: 0 0 0.9rem;
    font-family: var(--fe-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--fe-muted);
  }}
  /* The name, typed: each letter has no size until its turn comes, so the caret
     after them moves along as they arrive. Nine one-off steps, then still. */
  .fe-hero-name {{
    margin: 0 0 1.7rem;
    font-family: var(--fe-sans);
    font-size: clamp(3.4rem, 1.4rem + 4.8vw, 6.6rem);
    font-weight: 600;
    line-height: 0.95;
    letter-spacing: -0.045em;
    white-space: nowrap;
    color: var(--fe-text);
  }}
  .fe-hero-letters span {{
    font-size: 0;
    animation: fe-key 1ms steps(1) forwards;
    animation-delay: calc(var(--i) * 115ms + 250ms);
  }}
  @keyframes fe-key {{ to {{ font-size: 1em; }} }}
  .fe-hero-caret {{
    display: inline-block;
    width: 0.07em;
    height: 0.74em;
    margin-left: 0.07em;
    background: var(--fe-accent);
    animation: fe-blink 1.05s steps(1) infinite;
  }}
  .fe-hero-state {{
    font-family: var(--fe-sans);
    font-size: clamp(1.45rem, 1rem + 1vw, 2rem);
    font-weight: 500;
    line-height: 1.18;
    letter-spacing: -0.022em;
    color: var(--fe-text);
    max-width: 24ch;
  }}
  .fe-hero-state-long {{
    font-size: clamp(1.15rem, 0.95rem + 0.5vw, 1.4rem);
    line-height: 1.35;
    color: var(--fe-text-2);
    max-width: 34ch;
  }}
  /* Under the weights: the split as a bar, then the two figures that say how it
     has done. The strip's own rule on its left is not wanted here. */
  .fe-hero-below {{ margin: 1.1rem 0 1.3rem; max-width: 36rem; }}
  .fe-hero-below:empty {{ margin: 0.9rem 0 0; }}
  .fe-hero-below .fe-split {{ margin-bottom: 1.1rem; }}
  .fe-hero-below .fe-strip-inline .fe-strip-cell:first-child {{
    border-left: none;
    padding-left: 0;
  }}

  /* The typed line. The text is uncovered a character at a time by a clip, and the
     cursor steps along with it; neither moves anything else on the page. */
  .fe-typed {{
    position: relative;
    display: block;
    max-width: 100%;
    overflow: hidden;
    font-family: var(--fe-mono);
    font-size: 12px;
    line-height: 1.7;
    letter-spacing: 0;
    color: var(--fe-accent-bright);
    white-space: nowrap;
  }}
  .fe-typed::before {{ content: "> "; color: var(--fe-muted); }}
  .fe-typed-text {{
    display: inline-block;
    clip-path: inset(0 100% 0 0);
    animation-name: fe-type;
    animation-delay: 1350ms;
    animation-fill-mode: forwards;
  }}
  .fe-typed-caret {{
    position: absolute;
    left: 2ch;
    top: 0.22em;
    width: 0.62ch;
    height: 1.15em;
    background: var(--fe-accent);
    animation-name: fe-caret, fe-blink;
    animation-delay: 1350ms, 0ms;
    animation-fill-mode: forwards, none;
    animation-duration: inherit, 1.05s;
    animation-timing-function: inherit, steps(1);
    animation-iteration-count: 1, infinite;
  }}
  @keyframes fe-type {{ to {{ clip-path: inset(0 0 0 0); }} }}
  @keyframes fe-caret {{ to {{ transform: translateX(calc(var(--n) * 1ch)); }} }}
  @keyframes fe-blink {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}

  /* The two links under the words, the first of them filled. */
  .st-key-fe-hero-go {{ margin-top: 1.1rem; }}
  .st-key-fe-hero-go [data-testid="stPageLink"] a {{
    display: inline-flex;
    padding: 0.5rem 0.95rem;
    border: 1px solid var(--fe-border-strong);
    border-radius: 2px;
    background: rgba(7, 9, 12, 0.45);
    transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
  }}
  .st-key-fe-hero-go [data-testid="stPageLink"] a:hover {{
    border-color: var(--fe-accent);
    background: rgba(76, 144, 240, 0.12);
  }}
  .st-key-fe-hero-go [data-testid="stPageLink"] a:active {{ transform: translateY(1px); }}
  .st-key-fe-hero-go > :first-child [data-testid="stPageLink"] a {{
    background: var(--fe-text);
    border-color: var(--fe-text);
  }}
  .st-key-fe-hero-go > :first-child [data-testid="stPageLink"] a p,
  .st-key-fe-hero-go > :first-child [data-testid="stPageLink"] a span {{
    color: var(--fe-bg) !important;
    font-weight: 500;
  }}
  .st-key-fe-hero-go > :first-child [data-testid="stPageLink"] a:hover {{
    background: #fff;
    border-color: #fff;
  }}

  /* ---- The field --------------------------------------------------------------------- */
  .fe-field {{
    display: block;
    width: 100%;
    height: auto;
    margin: 0.4rem 0 -0.4rem;
    overflow: hidden;
  }}
  .fe-field-label text, text.fe-field-label {{
    font-family: var(--fe-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    stroke: none;
  }}
  .fe-field text.fe-field-figure {{
    font-family: var(--fe-sans);
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    text-transform: none;
  }}
  .fe-candle {{
    transform-box: fill-box;
    transform-origin: 50% 100%;
    animation: fe-rise 640ms cubic-bezier(0.16, 1, 0.3, 1) both;
    animation-delay: calc(var(--i) * 15ms + 180ms);
  }}
  .fe-field-floor {{ animation: fe-appear 900ms ease-out both; }}
  .fe-field-arc {{
    stroke-dasharray: 1;
    stroke-dashoffset: 1;
    animation: fe-draw 900ms cubic-bezier(0.65, 0, 0.35, 1) forwards;
    animation-delay: calc(var(--i) * 220ms + 1250ms);
  }}
  .fe-field-late {{ animation: fe-appear 500ms ease-out 1500ms both; }}
  .fe-field-ring {{
    transform-box: fill-box;
    transform-origin: center;
    animation: fe-ping 3.2s cubic-bezier(0.16, 1, 0.3, 1) 1400ms infinite both;
  }}
  .fe-field-ring-2 {{ animation-delay: 3000ms; }}
  @keyframes fe-rise {{ from {{ transform: scaleY(0); }} to {{ transform: none; }} }}
  @keyframes fe-appear {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  @keyframes fe-draw {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes fe-ping {{
    0% {{ transform: scale(0.3); opacity: 0.9; }}
    70%, 100% {{ transform: scale(2.6); opacity: 0; }}
  }}

  /* ---- The tape ----------------------------------------------------------------------
     From the sidebar's edge to the window's, like the rules. */
  [data-testid="stElementContainer"]:has(.fe-tape) {{
    margin: 0 calc(-1 * var(--fe-gutter));
    width: calc(100% + 2 * var(--fe-gutter));
    max-width: none;
  }}
  .fe-tape {{
    display: flex;
    align-items: stretch;
    border-top: 1px solid var(--fe-border);
    border-bottom: 1px solid var(--fe-border);
    background: rgba(7, 9, 12, 0.55);
    margin-bottom: 1.4rem;
    font-family: var(--fe-mono);
    font-size: 11.5px;
    line-height: 2.6rem;
    white-space: nowrap;
  }}
  .fe-tape-head {{
    flex: none;
    padding: 0 1.1rem 0 var(--fe-gutter);
    border-right: 1px solid var(--fe-border);
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fe-muted);
  }}
  .fe-tape-window {{
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    mask-image: linear-gradient(
      90deg, transparent, #000 2.5rem, #000 calc(100% - 4rem), transparent
    );
  }}
  .fe-tape-track {{
    display: inline-flex;
    animation: fe-tape linear infinite;
  }}
  .fe-tape:hover .fe-tape-track {{ animation-play-state: paused; }}
  @keyframes fe-tape {{ to {{ transform: translateX(-50%); }} }}
  .fe-tape-run {{ display: inline-flex; }}
  .fe-tick {{
    display: inline-flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0 1.25rem;
    border-right: 1px solid var(--fe-border);
  }}
  .fe-tick b {{
    font-size: 10.5px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--fe-text-2);
  }}
  .fe-tick-last {{ color: var(--fe-text); font-variant-numeric: tabular-nums; }}
  .fe-tick-up {{ color: var(--fe-success-fg); }}
  .fe-tick-down {{ color: var(--fe-danger-fg); }}
  .fe-tick-flat {{ color: var(--fe-muted); }}
  .fe-tick-spark {{ width: 56px; height: 16px; flex: none; }}

  /* Where the words' column is too narrow to hold the status on one line it is set
     as ordinary text, wrapping, with nothing typed. */
  @media (max-width: 1180px) {{
    .fe-typed {{ white-space: normal; }}
    .fe-typed-text {{ clip-path: none; animation: none; display: inline; }}
    .fe-typed-caret {{ display: none; }}
  }}
  /* Below a wide window the field goes under the words at the full width, rather
     than shrinking beside a headline set over four lines. */
  @media (max-width: 1200px) {{
    [data-testid="stHorizontalBlock"]:has(.fe-hero-copy) {{ flex-wrap: wrap; }}
    [data-testid="stHorizontalBlock"]:has(.fe-hero-copy) > [data-testid="stColumn"] {{
      flex: 1 1 100% !important;
      width: 100% !important;
    }}
    .fe-hero-state {{ max-width: 30ch; }}
  }}
  @media (max-width: 900px) {{
    .fe-hero-copy {{ padding-top: 1.6rem; }}
    .fe-tape-head {{ display: none; }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    .fe-hero-copy, .fe-candle, .fe-field-floor, .fe-field-late, .fe-hero-caret {{
      animation: none;
    }}
    .fe-hero-letters span {{ animation: none; font-size: 1em; }}
    .fe-field-arc {{ animation: none; stroke-dashoffset: 0; }}
    .fe-field-ring {{ animation: none; opacity: 0.5; }}
    .fe-field-ring-2 {{ display: none; }}
    .fe-typed-text {{ animation: none; clip-path: none; }}
    .fe-typed-caret {{ animation: none; transform: translateX(calc(var(--n) * 1ch)); }}
    .fe-tape-window {{ overflow-x: auto; }}
    .fe-tape-track {{ animation: none; }}
    .fe-tape-run[aria-hidden="true"] {{ display: none; }}
  }}
</style>
"""
