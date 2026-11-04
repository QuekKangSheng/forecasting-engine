"""FYP-266: a completed portfolio run as one HTML file to read offline.

Written only from what ``store/portfolio_runs`` saved for the run, so generating it
solves nothing, and its figures are the ones the Portfolio page showed: the tables are
that page's own rows and the charts are drawn from the same daily returns by the same
functions. HTML because it needs nothing beyond the Python this app already has, opens
in any browser with no connection, and prints to PDF. The charts are inline SVG for the
same reason: the page's interactive charts need scripts from the internet.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from forecasting_engine.extraction.targets import TargetRole
from forecasting_engine.portfolio.backtest import BacktestResult
from forecasting_engine.reporting.portfolio_comparison import (
    NOT_RUN,
    PORTFOLIO_LABELS,
    drawdown_paths,
    rolling_sharpe_paths,
    rolling_volatility_paths,
)
from forecasting_engine.reporting.run_results import BASIS_LABELS, tail_with_simulation
from forecasting_engine.store.portfolio_runs import SavedRun
from forecasting_engine.validation.gates import SIGNAL_RANK_IC_GATE, SIGNIFICANCE_T_GATE

#: Enough to draw a line's shape; every day of a ten-year backtest would add a megabyte
#: to a file nobody can hover over. Each line's highest and lowest days are always kept.
MAX_CHART_POINTS: int = 400

_LINE_COLOURS = ("#2F6FDE", "#D9822B", "#2E8B57", "#8E5BB5")
_GATE_NAMES = {"signal_rank_ic": "Signal Rank IC", "t": "t"}
_NOT_SAVED = "Not saved for this run."

_STYLE = """
body { font: 15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: #1c2024; background: #fff; max-width: 900px; margin: 0 auto; padding: 32px 24px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 18px; margin: 32px 0 8px; padding-top: 12px; border-top: 1px solid #d9dde1; }
h3 { font-size: 15px; margin: 20px 0 6px; }
p { margin: 6px 0; }
.meta, .note { color: #5c6670; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 14px; }
th, td { text-align: left; padding: 5px 10px 5px 0; border-bottom: 1px solid #e6e9ec;
         vertical-align: top; }
th { color: #5c6670; font-weight: 600; }
svg { width: 100%; height: auto; display: block; margin: 4px 0; }
.legend span { margin-right: 16px; font-size: 13px; white-space: nowrap; }
.legend i { display: inline-block; width: 12px; height: 3px; margin: 0 5px 3px 0; }
section { break-inside: avoid-page; }
"""


def report_html(
    run: SavedRun, paths: BacktestResult, simulation: Mapping[str, object] | None = None
) -> str:
    """The whole report. ``simulation`` is the one to report, if any: Monte Carlo reads
    "Not run" without it, since a missing figure is not a zero risk."""
    results = run.results
    title = f"Run {run.run_number}" + (f": {run.name}" if run.name else "")
    sections = [
        _data_quality(results.get("data_quality")),
        _models(results.get("models")),
        _optimiser(results),
        _backtest(results),
        _significance(results),
        _charts(results, paths),
        _tail_risk(results, simulation),
        _monte_carlo(simulation),
    ]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{_esc(title)}</h1>{_header(run)}{''.join(sections)}</body></html>"
    )


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _section(heading: str, body: str) -> str:
    return f"<section><h2>{_esc(heading)}</h2>{body}</section>"


def _table(rows: Sequence[Mapping[str, object]], columns: Mapping[str, str]) -> str:
    head = "".join(f"<th>{_esc(shown)}</th>" for shown in columns.values())
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(row[field])}</td>" for field in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _header(run: SavedRun) -> str:
    backtest = run.results["backtest"]
    return (
        '<p class="meta">'
        f"Run number {run.run_number}. First shown {run.recorded_at:%d %b %Y %H:%M}. "
        f"Commit {_esc(run.code_version or 'not recorded')}.<br>"
        f"Backtest {_date(backtest['start'])} to {_date(backtest['end'])}. "
        f"Dataset SHA-256 {_esc(run.dataset_sha256)}.</p>"
    )


def _date(iso: str) -> str:
    return f"{pd.Timestamp(iso):%d %b %Y}"


def _data_quality(quality: Mapping[str, object] | None) -> str:
    if quality is None:
        return _section("Data quality", f"<p>{_NOT_SAVED}</p>")
    rows = [
        ("Rows", f"{quality['rows']:,}"),
        ("Signals", quality["signals"]),
        ("Dates", _date_range(quality)),
        ("Schema issues", len(quality["schema_errors"])),
        ("Duplicate dates", quality["duplicate_dates"]),
        ("Weekend rows", quality["weekend_rows"]),
        ("Flagged large moves", quality["big_moves"]),
        ("Columns with missing values", len(quality["incomplete"])),
        ("Columns dropped at ingest", len(quality["dropped"])),
    ]
    body = _table([{"k": k, "v": v} for k, v in rows], {"k": "Check", "v": "Result"})
    body += "".join(f"<p>{_esc(message)}</p>" for message in quality["schema_errors"])
    if quality["incomplete"]:
        worst = sorted(quality["incomplete"].items(), key=lambda kv: -kv[1])
        body += _table(
            [{"c": column, "p": f"{pct:.1f}%"} for column, pct in worst],
            {"c": "Column", "p": "Missing"},
        )
    if quality["dropped"]:
        body += f'<p class="note">Dropped: {_esc(", ".join(quality["dropped"]))}.</p>'
    body += '<p class="note">Flags are for information. Nothing here was auto-corrected.</p>'
    return _section("Data quality", body)


def _date_range(quality: Mapping[str, object]) -> str:
    if quality["first_date"] is None:
        return "None"
    return f"{_date(quality['first_date'])} to {_date(quality['last_date'])}"


def _models(models: Sequence[Mapping[str, object]] | None) -> str:
    if not models:
        return _section("Active models", f"<p>{_NOT_SAVED}</p>")
    return _section("Active models", "".join(_model(model) for model in models))


def _model(model: Mapping[str, object]) -> str:
    body = (
        f"<h3>{_esc(model['name'])}: {_esc(model['model'])}</h3>"
        f'<p class="meta">Set active {pd.Timestamp(model["set_at"]):%d %b %Y %H:%M}.</p>'
        f"<p>{_esc(_gate_text(model))}</p>"
    )
    if model["terms"] is None:
        body += "<p>Polynomial function: none, since this model is not a polynomial.</p>"
    else:
        if model["formula"]:
            body += f"<p>Function: <code>{_esc(model['formula'])}</code></p>"
        body += (
            _table(model["terms"], {k: k for k in ("Factor", "Exponent", "Coefficient")})
            if model["terms"]
            else "<p>No term survived fitting.</p>"
        )
        body += "".join(f'<p class="note">{_esc(line)}</p>' for line in model["standardisation"])
    if model["attribution"] is None:
        body += (
            "<p>Per-signal SHAP: not saved, either because this function has none or because "
            "the model was set active before SHAP was kept.</p>"
        )
    else:
        body += _table(
            [{"s": signal, "v": f"{value:.6f}"} for signal, value in model["attribution"].items()],
            {"s": "Signal", "v": "Mean |SHAP value|"},
        )
    return body


def _gate_text(model: Mapping[str, object]) -> str:
    ic, t = model["signal_rank_ic"], model["signal_t"]
    figures = (
        f"Signal Rank IC {_figure(ic, '+.3f')} (gate: above {SIGNAL_RANK_IC_GATE}), "
        f"t {_figure(t, '.2f')} (gate: above {SIGNIFICANCE_T_GATE:g})."
    )
    if model["gate_met"] is None:
        verdict = "Gate result not saved for this model."
    elif model["gate_met"]:
        verdict = "Gate met."
    else:
        failed = " and ".join(_GATE_NAMES[g] for g in model["gates_failed"])
        verdict = f"Gate failed on {failed}."
    high_risk = " Set active despite failing both gates." if model["high_risk"] else ""
    return f"{verdict} {figures}{high_risk}"


def _figure(value: float | None, spec: str) -> str:
    return "not saved" if value is None else format(value, spec)


def weights_explanation(settings: Mapping[str, object], names: Mapping[str, str]) -> str:
    """FYP-272: how the optimiser's settings turn into weights, in plain language."""
    lower, upper = settings["bounds"]
    costs = settings["costs_bps"]
    level = settings["risk_level"]
    equity, bond = names[TargetRole.EQUITY.value], names[TargetRole.BOND.value]
    forecasts = (
        "Level 5 holds that weight whatever the forecasts say."
        if level == 5
        else "The active models' signals then tilt that weight towards whichever index they "
        "favour, but only as far as their past forecasts have earned trust: until the record "
        "is statistically reliable, there is no tilt at all."
    )
    return (
        f"The portfolio holds {equity} and {bond}, long only, with at least {lower:.0%} and at "
        f"most {upper:.0%} in each. Every {settings['horizon']} trading days, the horizon the "
        f"forecasts look ahead, the weights are set again. Risk level {level} of 5 (1 is the "
        "most risk-loving, 5 the most risk-averse) sets a target volatility, evenly spaced "
        "between the lowest-risk and highest-risk mixes those limits allow, using how the "
        f"two indices moved over {settings['train_window']} trading days ending "
        f"{settings['embargo']} days before the rebalance. {forecasts} Trading costs "
        f"{costs[TargetRole.EQUITY.value]:g} bp on "
        f"{equity} and {costs[TargetRole.BOND.value]:g} bp on {bond}, so the portfolio trades "
        "only when it has drifted outside a no-trade band those costs set, and only as far as "
        "the band's edge."
    )


def _optimiser(results: Mapping[str, object]) -> str:
    settings, names = results["settings"], results["names"]
    lower, upper = settings["bounds"]
    costs = settings["costs_bps"]
    latest = results["latest"]
    rows = [
        ("Risk level", f"{settings['risk_level']} of 5"),
        ("Bounds", f"{lower:.0%} to {upper:.0%} in each index"),
        (
            "Trading cost",
            f"{costs[TargetRole.EQUITY.value]:g} bp {names[TargetRole.EQUITY.value]}, "
            f"{costs[TargetRole.BOND.value]:g} bp {names[TargetRole.BOND.value]}",
        ),
        ("Rebalance", f"every {settings['horizon']} trading days"),
        (
            f"Latest weights, {_date(latest['date'])}",
            ", ".join(
                f"{latest['weights'][role.value]:.0%} {names[role.value]}" for role in TargetRole
            ),
        ),
    ]
    return _section(
        "Optimiser settings",
        _table([{"k": k, "v": v} for k, v in rows], {"k": "Setting", "v": "Value"})
        + "<h3>How the weights were set</h3>"
        + f"<p>{_esc(weights_explanation(settings, names))}</p>"
        + "".join(f"<p>{_esc(line)}</p>" for line in results["how_set"]),
    )


def _backtest(results: Mapping[str, object]) -> str:
    level = results["settings"]["risk_level"]
    columns = {
        "metric": "Metric",
        "optimised": PORTFOLIO_LABELS["optimised"],
        "baseline": PORTFOLIO_LABELS["baseline"],
        "no_forecast": f"Level {level}, no forecasts",
        "difference": "Difference",
    }
    body = "".join(
        f"<h3>{label}</h3>{_table(results['bases'][basis]['comparison'], columns)}"
        for basis, label in BASIS_LABELS.items()
    )
    return _section(
        "Backtest",
        f"<p>{_esc('; '.join(results['active_models']))}.</p>{body}"
        '<p class="note">Difference is optimised minus benchmark: positive favours the '
        "optimised portfolio on every row, drawdown included.</p>",
    )


def _significance(results: Mapping[str, object]) -> str:
    level = results["settings"]["risk_level"]
    rows = []
    for basis, label in BASIS_LABELS.items():
        shown = results["bases"][basis]
        tests = (
            ("Against the equal-weight benchmark", shown["significance"]),
            (f"Did the forecasts add value? Against level {level}, no forecasts",
             shown["forecast_value"]),
        )
        for test, summary in tests:
            rows.append(
                {"basis": label, "test": test, "t": summary["t"], "verdict": summary["label"]}
            )
    return _section(
        "Spread t-statistics",
        _table(rows, {"basis": "Returns", "test": "Test", "t": "t", "verdict": "Verdict"})
        + f'<p class="note">{_esc(results["bases"]["net"]["significance"]["detail"])}</p>',
    )


def _charts(results: Mapping[str, object], paths: BacktestResult) -> str:
    names = {role: results["names"][role.value] for role in TargetRole}
    windows = results["windows"]
    volatility = windows["rolling_volatility"]
    sharpe = windows["rolling_sharpe"]
    charts = [
        (
            f"Rolling volatility, {volatility}-day window",
            rolling_volatility_paths(paths, "net", names, window=volatility),
            True,
        ),
        ("Drawdown", drawdown_paths(paths, "net", names), True),
        (
            f"Rolling Sharpe, {sharpe}-day window",
            rolling_sharpe_paths(paths, "net", names, window=sharpe),
            False,
        ),
    ]
    body = "".join(
        f"<h3>{_esc(title)}</h3>{line_chart_svg(frame, percent=percent, title=title)}"
        for title, frame, percent in charts
    )
    return _section(
        "Risk over the backtest",
        body
        + '<p class="note">Each index is held alone, with no trading costs; the optimised '
        "portfolio is after costs.</p>",
    )


def _tail_risk(results: Mapping[str, object], simulation: Mapping[str, object] | None) -> str:
    columns = {
        "method": "Method",
        "metric": "Measure",
        "optimised": PORTFOLIO_LABELS["optimised"],
        "baseline": PORTFOLIO_LABELS["baseline"],
    }
    body = "".join(
        f"<h3>{label}</h3>"
        + _table(tail_with_simulation(results["bases"][basis]["tail"], simulation), columns)
        for basis, label in BASIS_LABELS.items()
    )
    return _section(
        "VaR and CVaR",
        body
        + '<p class="note">VaR and CVaR are one-day losses, shown as positive numbers: lower '
        'is better. "Not run" means not computed for this run, not zero risk.</p>',
    )


def _monte_carlo(simulation: Mapping[str, object] | None) -> str:
    if simulation is None:
        return _section("Monte Carlo", f"<p>{NOT_RUN}.</p>")
    columns = {
        "metric": "Measure",
        "optimised": PORTFOLIO_LABELS["optimised"],
        "baseline": PORTFOLIO_LABELS["baseline"],
    }
    return _section(
        "Monte Carlo",
        f"<p>{simulation['paths']:,} paths of the next {simulation['horizon']} trading days "
        f"from {_date(simulation['as_of'])}, {_esc(simulation['method'])} sampling of "
        f"{simulation['history_days']} trading days from {_date(simulation['history_start'])}."
        f"</p>{_table(simulation['rows'], columns)}",
    )


def thinned(frame: pd.DataFrame, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    """At most about ``max_points`` evenly spaced days of ``frame``, plus each column's
    highest and lowest day, so a thinned line still reaches its true extremes."""
    if len(frame) <= max_points:
        return frame
    keep = set(np.linspace(0, len(frame) - 1, max_points).astype(int).tolist())
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        if np.isfinite(values).any():
            keep.update((int(np.nanargmax(values)), int(np.nanargmin(values))))
    return frame.iloc[sorted(keep)]


def line_chart_svg(frame: pd.DataFrame, *, percent: bool, title: str) -> str:
    """One line per column of ``frame`` on a shared time axis, as inline SVG with a
    legend. Empty frames say why there is no chart."""
    frame = frame.dropna(how="all")
    if len(frame) < 2:
        return '<p class="note">This backtest is too short for this chart.</p>'
    frame = thinned(frame)
    width, height = 860, 280
    left, right, top, bottom = 56, 12, 10, 26
    numbers = frame.to_numpy(dtype=float)
    low, high = float(np.nanmin(numbers)), float(np.nanmax(numbers))
    if low == high:
        low, high = low - 1, high + 1
    ticks = _ticks(low, high)
    low, high = min(low, ticks[0]), max(high, ticks[-1])
    dates = frame.index.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)

    def x(values):
        return left + (values - dates[0]) / (dates[-1] - dates[0]) * (width - left - right)

    def y(values):
        return top + (high - values) / (high - low) * (height - top - bottom)

    tick_format = ".0%" if percent else ".1f"
    grid = "".join(
        f'<line x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
        f'stroke="#e6e9ec"/><text x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end" '
        f'font-size="11" fill="#5c6670">{t:{tick_format}}</text>'
        for t in ticks
    )
    years = "".join(
        f'<text x="{x(float(pd.Timestamp(year=year, month=1, day=1).value)):.1f}" '
        f'y="{height - 8}" text-anchor="middle" font-size="11" fill="#5c6670">{year}</text>'
        for year in range(frame.index[0].year + 1, frame.index[-1].year + 1)
    )
    lines = "".join(
        _polyline(x(dates), y(frame[column].to_numpy(dtype=float)), _colour(i))
        for i, column in enumerate(frame.columns)
    )
    legend = "".join(
        f'<span><i style="background:{_colour(i)}"></i>{_esc(column)}</span>'
        for i, column in enumerate(frame.columns)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">'
        f"{grid}{years}{lines}</svg><div class=\"legend\">{legend}</div>"
    )


def _colour(position: int) -> str:
    return _LINE_COLOURS[position % len(_LINE_COLOURS)]


def _polyline(xs: np.ndarray, ys: np.ndarray, colour: str) -> str:
    """A day with no figure breaks the line rather than joining across the gap."""
    runs: list[list[str]] = [[]]
    for px, py in zip(xs, ys, strict=True):
        if np.isfinite(py):
            runs[-1].append(f"{px:.1f},{py:.1f}")
        elif runs[-1]:
            runs.append([])
    return "".join(
        f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" points="{" ".join(run)}"/>'
        for run in runs
        if run
    )


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round values spanning ``low`` to ``high``: steps of 1, 2 or 5 times a power of ten."""
    rough = (high - low) / count
    magnitude = 10 ** np.floor(np.log10(rough))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= rough)
    first = np.floor(low / step) * step
    return [float(v) for v in np.arange(first, high + step, step)]
