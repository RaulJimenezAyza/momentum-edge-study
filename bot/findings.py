"""Where the research results live once the terminal window is closed.

Every measurement in this project prints a table and then disappears. The
dashboard, meanwhile, carries a paragraph of prose about what the research
found -- prose I wrote by hand, which means it is a claim about the numbers
rather than the numbers. It was already stale once: it still said the bounce
was "real but small" after the clustering analysis had downgraded it to
unproven.

So the commands write their headline here and the page reads it. If a finding
changes, the page changes with it, and if nobody has run a command the page
says so instead of quoting a result from a month ago as though it were current.

Deliberately a flat JSON file keyed by measurement name: one entry per kind,
overwritten by the newest run. Not a history -- a history invites picking the
run that agreed with you.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FILENAME = "findings.json"


def _path(cfg) -> Path:
    return cfg.path(f"data/{FILENAME}")


def record(cfg, kind: str, payload: dict[str, Any]) -> None:
    """Store one measurement's headline under `kind`, replacing any previous.

    Never raises: a research run that completed must not be lost to a disk
    error in the bookkeeping that happens after it.
    """
    try:
        path = _path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            store = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            store = {}
        entry = dict(payload)
        entry["recorded_at"] = time.time()
        entry["direction"] = cfg.strategy.direction
        store[kind] = entry
        path.write_text(json.dumps(store, indent=2, sort_keys=True),
                        encoding="utf-8")
        log.info("recorded finding %r", kind)
    except Exception as exc:                       # noqa: BLE001 - see docstring
        log.warning("could not record finding %r: %s", kind, exc)


def load(cfg) -> dict[str, Any]:
    try:
        return json.loads(_path(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def summary(cfg) -> dict[str, Any]:
    """What the dashboard shows: the strongest claim the data currently supports.

    The ordering matters and is the whole point of the file. A trade-level
    p-value is available and small, and it is the one a reader would quote. It
    is also the one that assumes twenty signals fired by a single market-wide
    dump are twenty independent facts. So when a between-window number exists,
    it wins -- it is weaker, wider, and the only one not resting on an
    assumption known to be false.
    """
    store = load(cfg)
    out: dict[str, Any] = {"has_data": bool(store)}
    if not store:
        return out

    best = store.get("between_window") or store.get("replicate")
    if best:
        out["edge"] = best.get("mean_edge", best.get("edge"))
        out["p"] = best.get("p")
        out["basis"] = "windows"
        out["windows"] = best.get("windows")
        out["df"] = best.get("df")
        out["positive"] = best.get("positive")
        out["n"] = best.get("n")
        out["recorded_at"] = best.get("recorded_at")
    elif "nullcheck" in store:
        nc = store["nullcheck"]
        out["edge"] = nc.get("edge")
        # Prefer the clustered p when one was computed; the trade-level p is
        # the optimistic sibling and should not be the headline.
        out["p"] = nc.get("block_p") if nc.get("block_p") is not None \
            else nc.get("p")
        out["basis"] = "blocks" if nc.get("block_p") is not None else "trades"
        out["n"] = nc.get("n")
        out["recorded_at"] = nc.get("recorded_at")

    equities = store.get("equity_windows")
    if equities:
        out["equities"] = {
            "edge": equities.get("mean_edge"), "p": equities.get("p"),
            "windows": equities.get("windows"), "n": equities.get("n"),
            "recorded_at": equities.get("recorded_at"),
        }
    elif "equities" in store:
        eq = store["equities"]
        out["equities"] = {
            "edge": eq.get("edge"), "p": eq.get("p"), "n": eq.get("n"),
            "interval": eq.get("interval"), "days": eq.get("days"),
            "recorded_at": eq.get("recorded_at"),
        }
    # The ladder is a different KIND of claim and is kept in its own key rather
    # than folded into the edge above. Everything else on this page is "does
    # this entry rule beat a random entry, per trade". The ladder asks whether
    # a whole allocation beats buy-and-hold, per year, over decades. Averaging
    # those two into one headline would be a category error, and the split
    # between timing and allocation is the only part of it worth showing.
    ladder = store.get("ladder")
    if ladder:
        out["ladder"] = {
            "symbol": ladder.get("symbol"),
            "total": ladder.get("mean_cagr_diff"),
            "timing": ladder.get("timing_cagr_diff"),
            "allocation": ladder.get("allocation_cagr_diff"),
            "windows": ladder.get("windows"),
            "window_years": ladder.get("window_years"),
            "p": ladder.get("p_adjusted"),
            "max_dd": ladder.get("max_dd"),
            "hold_dd": ladder.get("hold_dd"),
            "recorded_at": ladder.get("recorded_at"),
        }
        # The ladder rests on one vendor's century of prices. If that has been
        # checked against a second one, the page says so next to the claim
        # rather than leaving the check recorded where nobody reads it.
        vendor = store.get("vendorcheck")
        if vendor:
            out["ladder"]["vendor_bps"] = vendor.get("median_bps")
            out["ladder"]["vendor_months"] = vendor.get("months")

    # The long-horizon result answers the question the whole project ends on,
    # so it reaches the page rather than sitting in the file.
    lt = store.get("longterm")
    if lt:
        out["longterm"] = {
            "symbol": lt.get("symbol"), "best": lt.get("best"),
            "vs_hold": lt.get("vs_hold"), "timing": lt.get("timing"),
            "p": lt.get("p_adjusted"), "windows": lt.get("windows"),
            "window_years": lt.get("window_years"),
            "max_dd": lt.get("max_dd"), "hold_dd": lt.get("hold_dd"),
            "timing_null": lt.get("timing_null"),
            "beaten_by_holding": lt.get("beaten_by_holding"),
            "recorded_at": lt.get("recorded_at"),
        }

    # The multi-asset result, which is the one the project ends on. Its
    # headline is the pre-registered verdict, not the best row in the table --
    # showing the winner of a search as though it were the result is the exact
    # thing the protocol exists to prevent.
    al = store.get("allocate")
    if al:
        out["allocate"] = {
            "chosen": al.get("chosen"),
            "delta_sharpe": al.get("delta_sharpe"),
            "p": al.get("p_adjusted"),
            "cagr": al.get("holdout_cagr"), "equity_cagr": al.get("equity_cagr"),
            "dd": al.get("holdout_dd"), "equity_dd": al.get("equity_dd"),
            "months": al.get("holdout_months"),
            "recorded_at": al.get("recorded_at"),
        }

    if "toll" in store:
        out["toll"] = store["toll"].get("round_trip_pct")
    return out
