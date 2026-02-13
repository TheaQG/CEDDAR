# sbgm/evaluate/evaluate_prcp/eval_dates/evaluate_dates.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import logging

from sbgm.evaluate.evaluate_prcp.eval_dates.plot_dates import plot_dates_montages, _select_representative_dates

logger = logging.getLogger(__name__)


def run_dates(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    dates: Sequence[str] | None = None,
    include_lr: bool = True,
    include_members: bool = True,
    n_members: int = 3,
    cmap: str = "Blues",
    percentile: float = 99.5,
    land_only: bool = True,
) -> None:
    """
    Entry point for the 'eval_dates' block (pure plotting).
    """
    # --- robust date selection ---
    # Default: pick up to 4 representative dates.
    # If `dates` was provided (e.g. via eval_cfg.dates_list), intersect with available.
    try:
        all_dates = list(resolver.list_dates())
    except Exception:
        all_dates = []

    k_default = int(getattr(eval_cfg, "dates_max", 4))

    # Normalize user-provided list
    req: list[str] = []
    if dates:
        req = [str(d) for d in dates if d is not None]

    # If user requested dates, keep only those that exist
    chosen: list[str] = []
    if req:
        if all_dates:
            chosen = [d for d in req if d in all_dates]
        else:
            # If resolver can't list dates, just trust the provided ones
            chosen = req

        if not chosen:
            logger.warning(
                "[eval_dates] None of the requested dates were found in outputs. requested=%s. "
                "Falling back to representative dates.",
                req,
            )

    # If nothing chosen yet, pick representative dates (or first k as fallback)
    if not chosen:
        if not all_dates:
            logger.warning("[eval_dates] No dates available to plot (resolver.list_dates empty/failed).")
            return

        k = min(k_default, len(all_dates))
        try:
            chosen = _select_representative_dates(resolver, all_dates, k=k)
            logger.info("[eval_dates] Using %d representative dates: %s", len(chosen), chosen)
        except Exception:
            chosen = all_dates[:k]
            logger.info("[eval_dates] Using first %d dates: %s", len(chosen), chosen)

    # Final guard
    if not chosen:
        logger.warning("[eval_dates] No dates selected; skipping.")
        return

    dates = chosen

    plot_dates_montages(
        resolver=resolver,
        out_root=out_root,
        dates=list(dates),
        include_lr=bool(include_lr),
        include_members=bool(include_members),
        n_members=int(n_members),
        cmap=str(cmap),
        percentile=float(percentile),
        land_only=bool(land_only if hasattr(eval_cfg, "eval_land_only") else True),
    )