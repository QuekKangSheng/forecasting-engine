"""The Home page's opening, beside the market chart (``market_chart``).

Each piece is a function that returns HTML for ``st.markdown``:

- ``lead``: the maker's name as the page's heading, flickering on letter by letter
  like a screen powering up, then where things stand (the latest weights, or the
  next step).
- ``typed``: one line of status, typed out once in the monospace face.

``CSS`` styles these. Motion runs once and is switched off for a viewer who asked
for less.
"""

from __future__ import annotations

import hashlib
import html

import tokens

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
    """The opening's words: ``name`` as the page's heading, flickering on (``flicker``);
    under it a small label, the statement of where things stand, what goes under that
    (HTML its caller has escaped) and the typed status line. ``long`` sets a
    statement of a whole sentence a size down.

    The heading is read once, as ``name``: its letters, there to flicker, are hidden
    from a screen reader."""
    keyframes, letters = flicker(name)
    size = " fe-hero-state-long" if long else ""
    return (
        f"<style>{keyframes}</style>"
        f'<div class="fe-hero-copy">'
        f'<div class="fe-hero-name" role="heading" aria-level="1" '
        f'aria-label="{html.escape(name, quote=True)}">'
        f'<span class="fe-hero-letters" aria-hidden="true">{letters}</span></div>'
        f'<div class="fe-hero-label">{html.escape(label)}</div>'
        f'<div class="fe-hero-state{size}">{html.escape(statement)}</div>'
        f'<div class="fe-hero-below">{below}</div>{typed_html}</div>'
    )


def _unit(key: str) -> float:
    """A number in [0, 1) fixed by ``key``: the same on every load and machine."""
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") / 2**32


def flicker(name: str) -> tuple[str, str]:
    """Each letter of ``name`` as a span, and the keyframes that switch it on.

    As the onerecord wordmark does it (github.com/ethantyh03/onerecord,
    ``landing-hero.tsx``): every letter blinks on and off a couple of times at its
    own irregular moments, like a screen powering on, then settles solid, within
    0.9 to 1.3 s. Opacity only, in hard steps, so the word's width never changes; no
    loop, no colour, no glow. The moments are drawn from a hash of the letter's
    place, so the word flickers the same way every time."""
    frames, spans = [], []
    for i, letter in enumerate(name):

        def r(k: int, i: int = i) -> float:
            return _unit(f"flicker-{i}-{k}")

        duration = 900 + 400 * r(0)
        on1 = 0.05 + 0.3 * r(1)
        off1 = on1 + 0.05 + 0.1 * r(2)
        on2 = off1 + 0.04 + 0.1 * r(3)
        twice = r(4) > 0.35
        off2 = on2 + 0.08 + 0.12 * r(5)
        on3 = off2 + 0.03 + 0.08 * r(6)
        # Stretched so each letter settles late in its own time (0.86-0.96 of it).
        scale = (0.86 + 0.1 * r(7)) / (on3 if twice else on2)
        steps = [(0, 0.0), (1, on1), (0.08, off1), (1, on2)]
        if twice:
            steps += [(0.3, off2), (1, on3)]
        body = "".join(
            f"{offset * scale * 100:.2f}%{{opacity:{opacity};"
            "animation-timing-function:steps(1,end)}"
            for opacity, offset in steps
        )
        frames.append(f"@keyframes fe-flicker-{i}{{{body}100%{{opacity:1}}}}")
        spans.append(
            f'<span style="animation:fe-flicker-{i} {duration:.0f}ms linear both">'
            f"{html.escape(letter)}</span>"
        )
    return "".join(frames), "".join(spans)


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
  /* The name. Each letter's flicker is its own keyframes, written with the heading. */
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
  }}

  @media (prefers-reduced-motion: reduce) {{
    .fe-hero-copy {{ animation: none; }}
    .fe-hero-letters span {{ animation: none !important; }}
    .fe-typed-text {{ animation: none; clip-path: none; }}
    .fe-typed-caret {{ animation: none; transform: translateX(calc(var(--n) * 1ch)); }}
  }}
</style>
"""
