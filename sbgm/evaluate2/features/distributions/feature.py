from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sbgm.evaluate2.config import Eval2Plan, RunMode
from sbgm.evaluate2.data_resolver import EvalDataResolver
from sbgm.evaluate2.store import FeatureStore

from sbgm.evaluate2.features.distributions.compute import compute_distributions
from sbgm.evaluate2.features.distributions.plots import plot_distributions

logger = logging.getLogger(__name__)


@dataclass
class DistributionsConfig:
    """Distributions feature settings (eval2).

    Philosophy:
      - Feature-level only ("distributions" is the feature).
      - No low-level compute/plot tasks exposed.
      - All behavior is controlled by settings.

    Baseline overlays are intentionally NOT supported in eval2 (paper 2 does not need it).
    """

    # ---- compute toggles ----
    compute_pooled: bool = True
    compute_daily: bool = False
    compute_metrics: bool = True

    # ensemble distribution handling (only used if plan.use_ensemble is True)
    compute_ensemble: bool = False
    ensemble_mode: str = "member_mean"  # "member_mean" | "pool"

    # ---- binning / range policy ----
    n_bins: int = 200
    range_policy: str = "hr_percentile"  # "hr_percentile" | "fixed"
    hr_percentile: float = 99.9
    value_min: Optional[float] = 0.0
    value_max: Optional[float] = None

    # ---- plotting ----
    plot_pooled: bool = True
    plot_ci_daily: bool = False
    plot_seasonal: bool = False
    plot_metrics_box: bool = True
    plot_percentile_lines: bool = True


def parse_distributions_cfg(d: Dict[str, Any]) -> DistributionsConfig:
    """Parse per-feature config dict into a typed config.

    Permissive by design to keep transition cheap.
    """
    if not isinstance(d, dict):
        d = {}

    return DistributionsConfig(
        compute_pooled=bool(d.get("compute_pooled", True)),
        compute_daily=bool(d.get("compute_daily", False)),
        compute_metrics=bool(d.get("compute_metrics", True)),
        compute_ensemble=bool(d.get("compute_ensemble", False)),
        ensemble_mode=str(d.get("ensemble_mode", "member_mean")).strip().lower(),
        n_bins=int(d.get("n_bins", 200)),
        range_policy=str(d.get("range_policy", "hr_percentile")).strip().lower(),
        hr_percentile=float(d.get("hr_percentile", 99.9)),
        value_min=d.get("value_min", 0.0),
        value_max=d.get("value_max", None),
        plot_pooled=bool(d.get("plot_pooled", True)),
        plot_ci_daily=bool(d.get("plot_ci_daily", False)),
        plot_seasonal=bool(d.get("plot_seasonal", False)),
        plot_metrics_box=bool(d.get("plot_metrics_box", True)),
        plot_percentile_lines=bool(d.get("plot_percentile_lines", True)),
    )


class DistributionsFeature:
    """Eval2 Distributions feature runner."""

    name = "distributions"

    def run(
        self,
        plan: Eval2Plan,
        resolver: EvalDataResolver,
        store: FeatureStore,
        do_compute: bool,
        do_plot: bool,
        feature_cfg: Dict[str, Any],
    ) -> None:
        cfg = parse_distributions_cfg(feature_cfg)

        # Tight minimal-mode semantics: keep compute cheap + outputs small.
        if plan.run_mode == RunMode.MINIMAL:
            cfg.compute_daily = False
            cfg.compute_ensemble = False
            cfg.plot_ci_daily = False
            cfg.plot_seasonal = False
        # validate
        if cfg.n_bins <= 1:
            raise ValueError("distributions.n_bins must be > 1")
        if cfg.ensemble_mode not in ("member_mean", "pool"):
            raise ValueError("distributions.ensemble_mode must be 'member_mean' or 'pool'")
        if cfg.range_policy not in ("hr_percentile", "fixed"):
            raise ValueError("distributions.range_policy must be 'hr_percentile' or 'fixed'")

        logger.info(
            "[eval2:%s] start run_mode=%s do_compute=%s do_plot=%s n_dates=%d",
            self.name,
            plan.run_mode.value,
            do_compute,
            do_plot,
            len(plan.dates),
        )

        if do_compute:
            compute_distributions(plan=plan, resolver=resolver, store=store, cfg=cfg)

        if do_plot:
            plot_distributions(plan=plan, store=store, cfg=cfg)

        logger.info("[eval2:%s] done", self.name)
