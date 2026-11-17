"""Bond grids for machine learning, 16 Nov 2026 (docs/evidence/2026-11-16-bond-grids.md):
the hyperparameter search space and
the columns it is offered, each run the Models page's own machine-learning run
(XGBoost and LightGBM, Optuna-tuned on the tuning period and re-tuned yearly, screened
per fold, the better library reported) on the page's walk-forward and score.

  space    page      the page's SEARCH_SPACE
           shallow   depth 1-2 and larger leaves: close to an additive model
           slow      more, smaller steps: 50-300 trees at a learning rate of 0.005-0.05
           wide      everything wider: depth 2-6, 20-300 trees, leaves of 5-200 rows
  library  as in 2026-11-16-bond-grids.py: page, page+lvl, lvl+mom

Run: uv run --project <repo> python -I <this file> <folder of exports> [--horizons 20]
        [--workers 3] [--spaces page,shallow,slow,wide] [--libraries page,page+lvl,lvl+mom]
        [--out results_ml.jsonl]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from forecasting_engine.models import boosted
from forecasting_engine.parallel import WORKERS_ENV
from forecasting_engine.validation import metrics

_SPEC = importlib.util.spec_from_file_location(
    "bond_grids", Path(__file__).resolve().parent / "2026-11-16-bond-grids.py"
)
grid = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(grid)

PAGE = dict(boosted.SEARCH_SPACE)
SPACES: dict[str, dict] = {
    "page": PAGE,
    "shallow": {**PAGE, "max_depth": (1, 2, False), "min_leaf": (20, 150, True)},
    "slow": {
        **PAGE,
        "n_estimators": (50, 300, False),
        "max_depth": (2, 3, False),
        "learning_rate": (0.005, 0.05, True),
    },
    "wide": {
        **PAGE,
        "n_estimators": (20, 300, False),
        "max_depth": (2, 6, False),
        "learning_rate": (0.005, 0.3, True),
        "min_leaf": (5, 200, True),
    },
}

_BUNDLE: dict = {}


def _init(bundle: dict) -> None:
    _BUNDLE.update(bundle)
    # Tunes and fits stay in this process, where the search space below is patched.
    os.environ[WORKERS_ENV] = "1"


def _score(signal: pd.Series, realised: pd.Series, dates, horizon: int) -> dict:
    s, r = signal.reindex(dates), realised.reindex(dates)
    paired = pd.concat([s, r], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2:
        return {"ic": None, "t": None, "rows": len(paired)}
    ic = metrics.rank_ic(s, r)
    errors = [
        metrics.rank_ic_se(s, r, lags=horizon - 1, overlap=True),
        metrics.rank_ic_se(s, r, lags=grid.TEST_WINDOW),
    ]
    errors = [e for e in errors if e == e and e > 0]
    t = ic / max(errors) if errors and ic == ic else None
    return {
        "ic": None if ic != ic else round(float(ic), 4),
        "t": None if t is None else round(float(t), 4),
        "rows": len(paired),
    }


def run(job: tuple[str, str, int]) -> dict:
    library, space, horizon = job
    started = time.perf_counter()
    boosted.SEARCH_SPACE = SPACES[space]
    panel = grid.build_panel(_BUNDLE, library, horizon)
    split = grid.splitter("roll504")
    result, description, log = boosted.run_boosted(panel, split)
    folds = [test for _, test in split.split(panel)]
    ends = [panel.label_end.reindex(test).max() for test in folds]
    dev = [d for test, end in zip(folds, ends, strict=True) if end < grid.HELD_FROM for d in test]
    held = [d for test in folds if test[0] >= grid.HELD_FROM for d in test]
    every = [d for test in folds for d in test]
    signal = result.forecast - result.baseline
    constant = result.constant_forecasts
    return {
        "library": library,
        "fit": f"ml:{space}",
        "window": "roll504",
        "horizon": horizon,
        "columns": len(panel.signals),
        "all": _score(signal, result.realised, every, horizon),
        "dev": _score(signal, result.realised, dev, horizon),
        "held": _score(signal, result.realised, held, horizon),
        "page_ic": round(float(result.gated_rank_ic), 4),
        "page_t": round(float(result.signal_t), 4),
        "constant_folds": constant.constant,
        "folds": constant.folds,
        "reported_library": log.library,
        "latest_params": {k: _plain(v) for k, v in log.tunes[-1].params[log.library].items()},
        "top_features": list(description.terms[:5]),
        "seconds": round(time.perf_counter() - started, 1),
    }


def _plain(value):
    return round(float(value), 5) if isinstance(value, float) else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--horizons", default="20")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--spaces", default=",".join(SPACES))
    parser.add_argument("--libraries", default="page,page+lvl,lvl+mom")
    parser.add_argument("--out", type=Path, default=Path("results_ml.jsonl"))
    args = parser.parse_args()
    bundle = grid.read_folder(args.folder)
    todo = [
        (library, space, int(h))
        for h in args.horizons.split(",")
        for space in args.spaces.split(",")
        for library in args.libraries.split(",")
    ]
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["library"], row["fit"].split(":")[1], row["horizon"]))
    todo = [j for j in todo if j not in done]
    print(f"{len(todo)} runs to do, {len(done)} already in {args.out}", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(bundle,)) as pool:
        pending = {pool.submit(run, job): job for job in todo}
        for future in as_completed(pending):
            job = pending[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - a failed run is a result too
                row = {
                    "library": job[0],
                    "fit": f"ml:{job[1]}",
                    "window": "roll504",
                    "horizon": job[2],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            shown = row.get("error") or (
                f"all {row['all']['ic']} t {row['all']['t']} | dev {row['dev']['ic']} | "
                f"held {row['held']['ic']} | {row['reported_library']} | {row['seconds']} s"
            )
            print(f"{'/'.join(map(str, job))}: {shown}", flush=True)


if __name__ == "__main__":
    main()
