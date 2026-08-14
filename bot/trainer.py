"""`run.py train` -- download history, label it, fit the filter, report honestly."""

from __future__ import annotations

import logging

from .backtest import load_history
from .config import Config
from .ml import Model, build_dataset, print_report, train

log = logging.getLogger(__name__)


async def run_training(
    cfg: Config,
    days: float = 45.0,
    symbols: list[str] | None = None,
    threshold: float | None = None,
    offset_days: float = 0.0,
) -> dict:
    # None means "derive it from the training distribution"; only an explicit
    # --threshold, or fixed mode, pins it.
    if threshold is None and cfg.ml.threshold_mode == "fixed":
        threshold = cfg.ml.min_probability

    history, _ = await load_history(
        cfg, days, symbols, need_market=True, offset_days=offset_days
    )
    log.info("labelling candidates (relax factor %.2f)...", cfg.ml.relax_factor)

    samples = build_dataset(cfg, history, cfg.ml.relax_factor)
    if not samples:
        raise RuntimeError(
            "no candidates found -- lower ml.relax_factor or raise --days"
        )

    wins = sum(s.label for s in samples)
    log.info(
        "%d labelled candidates across %d symbols, %.1f%% profitable "
        "(this is the number to beat)",
        len(samples), len(history), wins / len(samples) * 100.0,
    )

    model, report = train(
        samples,
        threshold=threshold,
        embargo_minutes=cfg.ml.embargo_minutes,
        test_fraction=cfg.ml.test_fraction,
        target_keep=cfg.ml.target_keep_fraction,
    )
    print_report(report)

    path = cfg.path(cfg.ml.model_path)
    model.save(path)
    log.info("model written to %s (metrics alongside it in %s)", path,
             path.with_suffix(".json").name)

    if report.verdict.startswith(("USELESS", "HARMFUL", "INCONCLUSIVE")):
        log.warning(
            "the model was saved but its verdict is %s -- do not set ml.enabled "
            "until a retrain says otherwise",
            report.verdict.split(" ")[0],
        )
    else:
        log.info("set ml.enabled: true in config.yaml to use it")
    return report.as_dict()


def describe_model(cfg: Config) -> None:
    """Print the stored report for the model on disk."""
    path = cfg.path(cfg.ml.model_path)
    if not path.exists():
        print(f"no model at {path}. Train one: python run.py train --days 45")
        return
    model = Model.load(path)
    meta = model.meta
    report = meta.get("report", {})
    print(f"\nModel: {path}")
    print(f"Trained on {meta.get('n_samples', '?')} candidates, "
          f"{len(meta.get('feature_names', []))} features\n")
    for key, value in report.items():
        if key == "top_features":
            continue
        print(f"  {key:<22} {value}")
    top = report.get("top_features") or []
    if top:
        print("\n  most informative features:")
        for name, score in top[:8]:
            print(f"    {name:<22} {score:+.5f}")
    print()
