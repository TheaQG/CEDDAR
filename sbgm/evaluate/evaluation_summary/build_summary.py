"""Utilities for building paper-oriented evaluation summaries.

This module is intentionally lightweight at first: it defines the data
containers, filesystem helpers, and orchestration scaffold needed to collect
metrics from the individual evaluation pillars into one summary artifact.

The actual pillar-specific extraction logic will be added incrementally.
"""

from __future__ import annotations

import csv
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Summary schema
# -----------------------------------------------------------------------------


@dataclass
class PillarSummary:
    """Container for one evaluation pillar.

    Attributes
    ----------
    name:
        Stable pillar name, e.g. ``"scale"`` or ``"extremes"``.
    metrics:
        Flat key-value store for scalar summary metrics.
    artifacts:
        References to relevant files produced by the pillar, e.g. CSV, JSON,
        PNG, or NPZ files that support the summary.
    notes:
        Optional free-form notes about missing metrics, fallbacks, caveats, etc.
    """

    name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }


@dataclass
class EvaluationSummary:
    """Top-level paper-oriented evaluation summary."""

    run_id: str
    model_key: str
    eval_root: str
    generated_root: Optional[str] = None
    config_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    pillars: Dict[str, PillarSummary] = field(default_factory=dict)

    def add_pillar(self, pillar: PillarSummary) -> None:
        self.pillars[pillar.name] = pillar

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_key": self.model_key,
            "eval_root": self.eval_root,
            "generated_root": self.generated_root,
            "config_path": self.config_path,
            "metadata": self.metadata,
            "pillars": {name: pillar.to_dict() for name, pillar in self.pillars.items()},
        }


# -----------------------------------------------------------------------------
# Pillar registry
# -----------------------------------------------------------------------------


SummaryBuilder = Callable[[Path, Mapping[str, Any], EvaluationSummary], PillarSummary]


PILLAR_REGISTRY: Dict[str, SummaryBuilder] = {}


def register_pillar(name: str) -> Callable[[SummaryBuilder], SummaryBuilder]:
    """Decorator for registering pillar summary builder functions."""

    def _decorator(func: SummaryBuilder) -> SummaryBuilder:
        PILLAR_REGISTRY[name] = func
        return func

    return _decorator


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _read_json(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _read_yaml(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        out = float(value)
        if out != out:  # NaN guard
            return None
        return out
    except Exception:
        return None

def _find_first_existing(root: Path, candidates: Iterable[str]) -> Optional[Path]:
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def _artifact_if_exists(root: Path, rel_path: str) -> Optional[str]:
    p = root / rel_path
    return _safe_relpath(p, root) if p.exists() else None


def _add_note_if_missing(pillar: PillarSummary, condition: bool, note: str) -> None:
    if condition:
        pillar.notes.append(note)


# -----------------------------------------------------------------------------
# Summary context resolution
# -----------------------------------------------------------------------------


def _infer_run_id(cfg: Mapping[str, Any], eval_root: Path) -> str:
    for key in ("run_id", "experiment_id", "exp_id"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return eval_root.name


def _infer_model_key(cfg: Mapping[str, Any], eval_root: Path) -> str:
    for key in ("model_key", "model_name", "experiment_name"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return eval_root.name


def _resolve_eval_root(cfg: Mapping[str, Any]) -> Path:
    paths = cfg.get("paths", {}) or {}

    for candidate in (
        paths.get("evaluation_dir"),
        cfg.get("eval_root"),
        cfg.get("evaluation_root"),
    ):
        p = _as_path(candidate)
        if p is not None:
            return p

    raise ValueError("Could not resolve evaluation root from config. Expected one of paths.evaluation_dir / eval_root / evaluation_root.")


def _resolve_generated_root(cfg: Mapping[str, Any]) -> Optional[Path]:
    paths = cfg.get("paths", {}) or {}
    for candidate in (
        paths.get("generation_dir"),
        paths.get("generated_samples_dir"),
        cfg.get("gen_root"),
        cfg.get("generated_root"),
    ):
        p = _as_path(candidate)
        if p is not None:
            return p
    return None


def make_summary_context(cfg: Mapping[str, Any]) -> EvaluationSummary:
    eval_root = _resolve_eval_root(cfg)
    gen_root = _resolve_generated_root(cfg)

    summary = EvaluationSummary(
        run_id=_infer_run_id(cfg, eval_root),
        model_key=_infer_model_key(cfg, eval_root),
        eval_root=str(eval_root),
        generated_root=None if gen_root is None else str(gen_root),
        config_path=str(cfg.get("config_path")) if cfg.get("config_path") is not None else None,
        metadata={
            "summary_version": 1,
            "available_pillars": sorted(PILLAR_REGISTRY.keys()),
        },
    )
    return summary


# -----------------------------------------------------------------------------
# Pillar builder placeholders
# -----------------------------------------------------------------------------

def _pick_distributional_primary_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    """Pick the most paper-relevant generated-vs-target row from dist_metrics.csv.

    Priority:
      1) ensemble pooled histogram summary
      2) ensemble member-mean summary
      3) PMM / legacy generated summary
    """
    priorities = ("gen_ens_pool", "gen_ens_mean", "gen_pmm", "gen")
    for comp in priorities:
        for row in rows:
            if str(row.get("comp", "")).strip().lower() == comp:
                return row
    return None

@register_pillar("distributional")
def build_distributional_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="distributional")

    dist_root = _find_first_existing(
        eval_root,
        [
            "prcp/distributional",
            "distributional",
            "prcp/distributions",
            "distributions",
        ],
    )
    if dist_root is None:
        pillar.notes.append("Distributional evaluation root not found under evaluation directory.")
        return pillar

    tables = dist_root / "tables"
    metrics_csv = tables / "dist_metrics.csv"

    pillar.artifacts["root"] = _safe_relpath(dist_root, eval_root)

    for key, rel in {
        "metrics_csv": "tables/dist_metrics.csv",
        "bins_csv": "tables/dist_bins.csv",
        "pooled_samples_npz": "tables/dist_pooled_samples.npz",
        "daily_npz": "tables/dist_daily.npz",
        "ensemble_hist_npz": "tables/dist_member_histograms.npz",
        "pooled_plot": "figures/dist_pooled.png",
        "seasonal_plot": "figures/dist_seasons.png",
    }.items():
        art = _artifact_if_exists(dist_root, rel)
        if art is not None:
            pillar.artifacts[key] = art

    rows = _read_csv_rows(metrics_csv)
    if not rows:
        pillar.notes.append("dist_metrics.csv missing or empty; no distributional scalar summary extracted.")
        return pillar

    primary = _pick_distributional_primary_row(rows)
    if primary is None:
        pillar.notes.append("Could not identify a generated-vs-target row in dist_metrics.csv.")
        return pillar

    comp_name = str(primary.get("comp", "")).strip()
    pillar.metrics["primary_comparison"] = comp_name

    wasserstein = _to_float_or_none(primary.get("wasserstein"))
    ks_stat = _to_float_or_none(primary.get("ks_stat"))
    ks_p = _to_float_or_none(primary.get("ks_p"))
    kl_hr_to_x = _to_float_or_none(primary.get("kl_hr_to_x"))

    if wasserstein is not None:
        pillar.metrics["wasserstein"] = wasserstein
    else:
        pillar.notes.append(f"Primary row '{comp_name}' is missing Wasserstein distance.")

    if ks_stat is not None:
        pillar.metrics["ks_stat"] = ks_stat
    if ks_p is not None:
        pillar.metrics["ks_p"] = ks_p
    if kl_hr_to_x is not None:
        pillar.metrics["kl_hr_to_x"] = kl_hr_to_x

    lr_row = None
    for row in rows:
        if str(row.get("comp", "")).strip().lower() == "lr":
            lr_row = row
            break

    if lr_row is not None:
        lr_wasserstein = _to_float_or_none(lr_row.get("wasserstein"))
        lr_ks_stat = _to_float_or_none(lr_row.get("ks_stat"))
        lr_ks_p = _to_float_or_none(lr_row.get("ks_p"))
        lr_kl = _to_float_or_none(lr_row.get("kl_hr_to_x"))

        if lr_wasserstein is not None:
            pillar.metrics["lr_wasserstein"] = lr_wasserstein
            if wasserstein is not None:
                pillar.metrics["delta_wasserstein_vs_lr"] = wasserstein - lr_wasserstein
        if lr_ks_stat is not None:
            pillar.metrics["lr_ks_stat"] = lr_ks_stat
        if lr_ks_p is not None:
            pillar.metrics["lr_ks_p"] = lr_ks_p
        if lr_kl is not None:
            pillar.metrics["lr_kl_hr_to_x"] = lr_kl
    else:
        pillar.notes.append("No LR reference row found in dist_metrics.csv.")

    pillar.metrics["n_metric_rows"] = len(rows)

    if comp_name.lower() not in ("gen_ens_pool", "gen_ens_mean", "gen_pmm", "gen"):
        pillar.notes.append(
            f"Primary distributional comparison '{comp_name}' is not one of the expected generated summary labels."
        )

    return pillar


@register_pillar("extremes")
def build_extremes_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="extremes")

    ext_root = _find_first_existing(
        eval_root,
        [
            "prcp/extremes",
            "extremes",
            "prcp/extreme",
            "extreme",
        ],
    )
    if ext_root is None:
        pillar.notes.append("Extremes evaluation root not found under evaluation directory.")
        return pillar

    tables = ext_root / "tables"
    pillar.artifacts["root"] = _safe_relpath(ext_root, eval_root)

    for key, rel in {
        "core_metrics_csv": "tables/ext_core_metrics.csv",
        "tails_csv": "tables/ext_tails.csv",
        "tails_uncertainty_csv": "tables/ext_tails_uncertainty.csv",
        "rxk_gev_csv": "tables/ext_rxk_gev.csv",
        "pot_gpd_csv": "tables/ext_pot_gpd.csv",
        "daily_series_npz": "tables/ext_daily_series.npz",
        "meta_npz": "tables/ext_meta.npz",
        "rxk_gev_ens_stats_npz": "tables/ext_rxk_gev_ens_stats.npz",
        "pot_gpd_ens_stats_npz": "tables/ext_pot_gpd_ens_stats.npz",
        "tails_ens_bands_npz": "tables/ext_tails_ens_bands.npz",
        "tails_plot": "figures/ext_tails.png",
        "gev_plot": "figures/ext_rxk_gev.png",
        "pot_plot": "figures/ext_pot_gpd.png",
    }.items():
        art = _artifact_if_exists(ext_root, rel)
        if art is not None:
            pillar.artifacts[key] = art

    core_rows = _read_csv_rows(tables / "ext_core_metrics.csv")
    if not core_rows:
        pillar.notes.append("ext_core_metrics.csv missing or empty; no direct extremes summary extracted.")
        return pillar

    by_which = {str(r.get("which", "")).strip().upper(): r for r in core_rows if str(r.get("which", "")).strip()}

    primary = by_which.get("GEN_ENS") or by_which.get("GEN")
    if primary is None:
        pillar.notes.append("Neither GEN_ENS nor GEN row was found in ext_core_metrics.csv.")
        return pillar

    primary_name = str(primary.get("which", "")).strip().upper()
    pillar.metrics["primary_comparison"] = primary_name

    for out_key, csv_key in {
        "p99": "p99",
        "p99_9": "p99_9",
        "rx1day": "rx1day",
        "rx5day": "rx5day",
        "wet_freq": "wet_freq",
        "wet_hit_rate": "wet_hit_rate",
        "p99_std": "p99_std",
        "p99_9_std": "p99_9_std",
        "rx1day_std": "rx1day_std",
        "rx5day_std": "rx5day_std",
        "wet_freq_std": "wet_freq_std",
        "wet_hit_rate_std": "wet_hit_rate_std",
    }.items():
        val = _to_float_or_none(primary.get(csv_key))
        if val is not None:
            pillar.metrics[out_key] = val

    n_val = primary.get("n")
    try:
        if n_val is not None and str(n_val).strip():
            pillar.metrics["n"] = int(float(n_val))
    except Exception:
        pass

    n_members_val = primary.get("n_members")
    try:
        if n_members_val is not None and str(n_members_val).strip():
            pillar.metrics["n_members"] = int(float(n_members_val))
    except Exception:
        pass

    hr_row = by_which.get("HR")
    if hr_row is not None:
        for out_key, csv_key in {
            "hr_p99": "p99",
            "hr_p99_9": "p99_9",
            "hr_rx1day": "rx1day",
            "hr_rx5day": "rx5day",
            "hr_wet_freq": "wet_freq",
            "hr_wet_hit_rate": "wet_hit_rate",
        }.items():
            val = _to_float_or_none(hr_row.get(csv_key))
            if val is not None:
                pillar.metrics[out_key] = val
    else:
        pillar.notes.append("No HR row found in ext_core_metrics.csv.")

    lr_row = by_which.get("LR")
    if lr_row is not None:
        for out_key, csv_key in {
            "lr_p99": "p99",
            "lr_p99_9": "p99_9",
            "lr_rx1day": "rx1day",
            "lr_rx5day": "rx5day",
            "lr_wet_freq": "wet_freq",
            "lr_wet_hit_rate": "wet_hit_rate",
        }.items():
            val = _to_float_or_none(lr_row.get(csv_key))
            if val is not None:
                pillar.metrics[out_key] = val
    else:
        pillar.notes.append("No LR row found in ext_core_metrics.csv.")

    pillar.metrics["n_metric_rows"] = len(core_rows)

    tails_rows = _read_csv_rows(tables / "ext_tails.csv")
    if not tails_rows:
        pillar.notes.append("ext_tails.csv missing or empty.")

    return pillar


@register_pillar("features")
def build_features_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="features")

    feat_root = _find_first_existing(
        eval_root,
        [
            "prcp/features",
            "features",
            "prcp/feature",
            "feature",
        ],
    )
    if feat_root is None:
        pillar.notes.append("Features evaluation root not found under evaluation directory.")
        return pillar

    tables = feat_root / "tables"
    pillar.artifacts["root"] = _safe_relpath(feat_root, eval_root)

    for key, rel in {
        "core_metrics_csv": "tables/features_core_metrics.csv",
        "all_npz": "tables/sal_all.npz",
        "djf_npz": "tables/sal_DJF.npz",
        "mam_npz": "tables/sal_MAM.npz",
        "jja_npz": "tables/sal_JJA.npz",
        "son_npz": "tables/sal_SON.npz",
        "all_plot": "figures/features_sal_ALL.png",
        "djf_plot": "figures/features_sal_DJF.png",
        "mam_plot": "figures/features_sal_MAM.png",
        "jja_plot": "figures/features_sal_JJA.png",
        "son_plot": "figures/features_sal_SON.png",
    }.items():
        art = _artifact_if_exists(feat_root, rel)
        if art is not None:
            pillar.artifacts[key] = art

    rows = _read_csv_rows(tables / "features_core_metrics.csv")
    if not rows:
        pillar.notes.append("features_core_metrics.csv missing or empty; no SAL summary extracted.")
        return pillar

    def _pick_row(preferred_groups: Sequence[str]) -> Optional[Mapping[str, Any]]:
        preferred_upper = [g.upper() for g in preferred_groups]
        for g in preferred_upper:
            for row in rows:
                if str(row.get("group", "")).strip().upper() == g:
                    return row
        return rows[0] if rows else None

    primary = _pick_row(["ALL", "all"])
    if primary is None:
        pillar.notes.append("Could not identify a primary SAL row in features_core_metrics.csv.")
        return pillar

    group_name = str(primary.get("group", "")).strip()
    pillar.metrics["primary_group"] = group_name

    for out_key, csv_key in {
        "gen_A": "GEN_vs_HR_A",
        "gen_S": "GEN_vs_HR_S",
        "gen_L": "GEN_vs_HR_L",
        "gen_SAL": "GEN_vs_HR_SAL",
        "gen_ens_A": "GEN_ENS_mean_vs_HR_A",
        "gen_ens_S": "GEN_ENS_mean_vs_HR_S",
        "gen_ens_L": "GEN_ENS_mean_vs_HR_L",
        "gen_ens_SAL": "GEN_ENS_mean_vs_HR_SAL",
        "gen_ens_A_std": "GEN_ENS_std_A",
        "gen_ens_S_std": "GEN_ENS_std_S",
        "gen_ens_L_std": "GEN_ENS_std_L",
        "gen_ens_SAL_std": "GEN_ENS_std_SAL",
        "lr_A": "LR_vs_HR_A",
        "lr_S": "LR_vs_HR_S",
        "lr_L": "LR_vs_HR_L",
        "lr_SAL": "LR_vs_HR_SAL",
    }.items():
        val = _to_float_or_none(primary.get(csv_key))
        if val is not None:
            pillar.metrics[out_key] = val

    pillar.metrics["n_metric_rows"] = len(rows)

    missing_primary = []
    for required in ("GEN_vs_HR_A", "GEN_vs_HR_S", "GEN_vs_HR_L", "GEN_vs_HR_SAL"):
        if _to_float_or_none(primary.get(required)) is None:
            missing_primary.append(required)
    if missing_primary:
        pillar.notes.append(
            "Primary SAL row is missing one or more deterministic GEN metrics: " + ", ".join(missing_primary)
        )

    if all(_to_float_or_none(primary.get(k)) is None for k in (
        "GEN_ENS_mean_vs_HR_A", "GEN_ENS_mean_vs_HR_S", "GEN_ENS_mean_vs_HR_L", "GEN_ENS_mean_vs_HR_SAL"
    )):
        pillar.notes.append("Primary SAL row has no ensemble-mean SAL metrics.")

    if all(_to_float_or_none(primary.get(k)) is None for k in (
        "LR_vs_HR_A", "LR_vs_HR_S", "LR_vs_HR_L", "LR_vs_HR_SAL"
    )):
        pillar.notes.append("Primary SAL row has no LR-vs-HR SAL reference metrics.")

    available_groups = []
    for row in rows:
        g = str(row.get("group", "")).strip()
        if g and g not in available_groups:
            available_groups.append(g)
    pillar.metrics["available_groups"] = available_groups

    return pillar



@register_pillar("scale")
def build_scale_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="scale")

    scale_root = _find_first_existing(
        eval_root,
        [
            "prcp/scale",
            "scale",
        ],
    )
    if scale_root is None:
        pillar.notes.append("Scale evaluation root not found under evaluation directory.")
        return pillar

    tables = scale_root / "tables"
    pillar.artifacts["root"] = _safe_relpath(scale_root, eval_root)

    for key, rel in {
        "psd_curves_npz": "tables/scale_psd_curves.npz",
        "psd_summary_csv": "tables/scale_psd_summary.csv",
        "psd_band_ratios_csv": "tables/scale_psd_band_ratios_avg.csv",
        "psd_slopes_csv": "tables/scale_psd_slopes_3band.csv",
        "bandpass_corr_csv": "tables/scale_bandpass_corr_avg.csv",
        "fss_summary_csv": "tables/scale_fss_summary.csv",
        "iss_summary_csv": "tables/scale_iss_summary.csv",
        "overview_csv": "tables/scale_overview.csv",
        "psd_plot": "figures/scale_psd.png",
        "psd_lowhigh_plot": "figures/scale_psd_lowhigh.png",
        "fss_plot": "figures/scale_fss.png",
        "iss_plot": "figures/scale_iss.png",
    }.items():
        art = _artifact_if_exists(scale_root, rel)
        if art is not None:
            pillar.artifacts[key] = art

    # ------------------------------------------------------------------
    # PSD slope summary (paper-facing: low / mid / high)
    # ------------------------------------------------------------------
    slope_rows = _read_csv_rows(tables / "scale_psd_slopes_3band.csv")
    if not slope_rows:
        pillar.notes.append("scale_psd_slopes_3band.csv missing or empty.")
    else:
        def _find_slope_row(which_candidates: Sequence[str], band_name: str) -> Optional[Mapping[str, Any]]:
            band_norm = str(band_name).strip().lower()
            for which in which_candidates:
                which_norm = str(which).strip().upper()
                for row in slope_rows:
                    row_which = str(row.get("which", "")).strip().upper()
                    row_band = str(row.get("band", "")).strip().lower()
                    if row_which == which_norm and row_band == band_norm:
                        return row
            return None

        primary_psd = None
        if any(str(r.get("which", "")).strip().upper() == "GEN_ENS_MEAN" for r in slope_rows):
            primary_psd = "GEN_ENS_MEAN"
        elif any(str(r.get("which", "")).strip().upper() == "GEN" for r in slope_rows):
            primary_psd = "GEN"

        if primary_psd is not None:
            pillar.metrics["primary_psd_comparison"] = primary_psd
        else:
            pillar.notes.append("Could not identify primary PSD series (GEN_ENS_MEAN or GEN) in slope summary.")

        slope_map = {
            "slope_low": ([(primary_psd or "GEN")], "low"),
            "slope_mid": ([(primary_psd or "GEN")], "mid"),
            "slope_high": ([(primary_psd or "GEN")], "high"),
            "hr_slope_low": (["HR"], "low"),
            "hr_slope_mid": (["HR"], "mid"),
            "hr_slope_high": (["HR"], "high"),
            "lr_slope_low": (["LR", "LR_HRGRID"], "low"),
            "lr_slope_mid": (["LR", "LR_HRGRID"], "mid"),
            "lr_slope_high": (["LR", "LR_HRGRID"], "high"),
        }
        undersampled_notes = []
        for out_key, (which_candidates, band_name) in slope_map.items():
            row = _find_slope_row(which_candidates, band_name)
            if row is None:
                continue
            val = _to_float_or_none(row.get("slope"))
            if val is not None:
                pillar.metrics[out_key] = val
            r2 = _to_float_or_none(row.get("r2"))
            if r2 is not None:
                pillar.metrics[f"{out_key}_r2"] = r2

            n_points = row.get("n_points")
            n_points_int = None
            try:
                if n_points is not None and str(n_points).strip():
                    n_points_int = int(float(n_points))
                    pillar.metrics[f"{out_key}_n_points"] = n_points_int
            except Exception:
                n_points_int = None

            if n_points_int is not None and n_points_int < 3:
                which_label = str(row.get("which", "")).strip() or "/".join(which_candidates)
                undersampled_notes.append(
                    f"PSD slope band '{band_name}' for '{which_label}' is undersampled (n_points={n_points_int}); interpret with caution."
                )

        for note in undersampled_notes:
            if note not in pillar.notes:
                pillar.notes.append(note)

        pillar.metrics["n_slope_rows"] = len(slope_rows)

    # ------------------------------------------------------------------
    # Band-pass GEN↔LR correlations
    # ------------------------------------------------------------------
    corr_rows = _read_csv_rows(tables / "scale_bandpass_corr_avg.csv")
    if not corr_rows:
        pillar.notes.append("scale_bandpass_corr_avg.csv missing or empty.")
    else:
        for band_name in ("low", "mid", "high"):
            row = None
            for r in corr_rows:
                pair = str(r.get("pair", "")).strip().upper()
                band = str(r.get("band", "")).strip().lower()
                if pair == "GEN_VS_LR" and band == band_name:
                    row = r
                    break
            if row is None:
                continue
            mean_v = _to_float_or_none(row.get("corr_mean"))
            std_v = _to_float_or_none(row.get("corr_std"))
            if mean_v is not None:
                pillar.metrics[f"corr_gen_lr_{band_name}"] = mean_v
            if std_v is not None:
                pillar.metrics[f"corr_gen_lr_{band_name}_std"] = std_v
            n_dates = row.get("n_dates")
            try:
                if n_dates is not None and str(n_dates).strip():
                    pillar.metrics[f"corr_gen_lr_{band_name}_n_dates"] = int(float(n_dates))
            except Exception:
                pass
        pillar.metrics["n_band_corr_rows"] = len(corr_rows)

    # ------------------------------------------------------------------
    # ISS summary (best-effort extraction; schema may evolve)
    # ------------------------------------------------------------------
    iss_rows = _read_csv_rows(tables / "scale_iss_summary.csv")
    if not iss_rows:
        pillar.notes.append("scale_iss_summary.csv missing or empty.")
    else:
        pillar.metrics["n_iss_rows"] = len(iss_rows)

        # Pick an ISS row using a threshold priority close to the common paper choices.
        preferred_thr = ("1.00", "1", "5.00", "5")
        primary_iss_row = None
        for thr in preferred_thr:
            for row in iss_rows:
                thr_raw = str(row.get("thr_mm", row.get("threshold", row.get("thr", "")))).strip()
                if thr_raw == thr:
                    primary_iss_row = row
                    break
            if primary_iss_row is not None:
                break
        if primary_iss_row is None:
            primary_iss_row = iss_rows[0]

        thr_label = str(primary_iss_row.get("thr_mm", primary_iss_row.get("threshold", primary_iss_row.get("thr", "")))).strip()
        if thr_label:
            pillar.metrics["primary_iss_threshold_mm"] = thr_label

        def _find_metric_key(row: Mapping[str, Any], prefixes: Sequence[str]) -> Optional[str]:
            for key in row.keys():
                key_norm = str(key).strip().lower()
                for pref in prefixes:
                    if key_norm.startswith(pref):
                        return key
            return None

        # Prefer the smallest available scale column as the paper-facing compact ISS summary.
        iss_cols = []
        for key in primary_iss_row.keys():
            key_str = str(key).strip().lower()
            if key_str.startswith("iss_") and key_str.endswith("km"):
                try:
                    km = float(key_str.split("_")[1].replace("km", ""))
                    iss_cols.append((km, key))
                except Exception:
                    continue
        iss_cols = sorted(iss_cols, key=lambda x: x[0])
        if iss_cols:
            km0, key0 = iss_cols[0]
            v0 = _to_float_or_none(primary_iss_row.get(key0))
            if v0 is not None:
                pillar.metrics["iss_primary"] = v0
                pillar.metrics["iss_primary_scale_km"] = km0
        else:
            pillar.notes.append("Could not identify ISS scale columns in scale_iss_summary.csv.")

    return pillar


@register_pillar("probabilistic")
def build_probabilistic_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="probabilistic")

    prob_root = _find_first_existing(
        eval_root,
        [
            "prcp/probabilistic",
            "probabilistic",
            "prcp/probability",
            "probability",
        ],
    )
    if prob_root is None:
        pillar.notes.append("Probabilistic evaluation root not found under evaluation directory.")
        return pillar

    tables = prob_root / "tables"
    pillar.artifacts["root"] = _safe_relpath(prob_root, eval_root)

    for key, rel in {
        "core_metrics_csv": "tables/prob_core_metrics.csv",
        "summary_csv": "tables/prob_summary.csv",
        "summary_txt": "tables/prob_summary.txt",
        "crps_daily_csv": "tables/prob_crps_daily.csv",
        "pit_values_npz": "tables/prob_pit_values.npz",
        "rank_hist_npz": "tables/prob_rank_histogram.npz",
        "spread_skill_csv": "tables/prob_spread_skill.csv",
        "reliability_csv": "tables/prob_reliability.csv",
        "reliability_thresh_csv": "tables/prob_reliability_thresholds.csv",
        "energy_variogram_csv": "tables/prob_energy_variogram.csv",
        "pit_plot": "figures/prob_pit.png",
        "rank_plot": "figures/prob_rank_histogram.png",
        "reliability_plot": "figures/prob_reliability.png",
        "spread_skill_plot": "figures/prob_spread_skill.png",
        "energy_variogram_plot": "figures/prob_energy_variogram.png",
    }.items():
        art = _artifact_if_exists(prob_root, rel)
        if art is not None:
            pillar.artifacts[key] = art

    rows = _read_csv_rows(tables / "prob_core_metrics.csv")
    if not rows:
        pillar.notes.append("prob_core_metrics.csv missing or empty; no probabilistic summary extracted.")
        return pillar

    by_which = {str(r.get("which", "")).strip().upper(): r for r in rows if str(r.get("which", "")).strip()}
    primary = by_which.get("GEN_ENS") or rows[0]
    if primary is None:
        pillar.notes.append("Could not identify a primary probabilistic summary row.")
        return pillar

    primary_name = str(primary.get("which", "")).strip().upper()
    pillar.metrics["primary_comparison"] = primary_name

    for out_key, csv_key in {
        "crps_mean": "crps_mean",
        "crps_std": "crps_std",
        "pit_ks_D": "pit_ks_D",
        "pit_n": "pit_n",
        "pmm_mae_mean": "pmm_mae_mean",
        "pmm_mae_std": "pmm_mae_std",
        "spread_skill_slope": "spread_skill_slope",
        "spread_skill_pearson_r": "spread_skill_pearson_r",
        "rankhist_max_abs_z": "rankhist_max_abs_z",
        "spatial_crps_land_mean": "spatial_crps_land_mean",
        "n_dates": "n_dates",
    }.items():
        if csv_key in ("pit_n", "n_dates"):
            raw = primary.get(csv_key)
            try:
                if raw is not None and str(raw).strip():
                    pillar.metrics[out_key] = int(float(raw))
            except Exception:
                pass
        else:
            val = _to_float_or_none(primary.get(csv_key))
            if val is not None:
                pillar.metrics[out_key] = val

    pillar.metrics["n_metric_rows"] = len(rows)

    missing_primary = []
    for required in ("crps_mean", "pit_ks_D", "pit_n"):
        if required not in pillar.metrics:
            missing_primary.append(required)
    if missing_primary:
        pillar.notes.append(
            "Primary probabilistic row is missing one or more paper-facing metrics: " + ", ".join(missing_primary)
        )

    if "pmm_mae_mean" not in pillar.metrics:
        pillar.notes.append("Primary probabilistic row has no PMM MAE summary (expected for ensemble-only configurations without PMM diagnostic).")

    return pillar


@register_pillar("climatological")
def build_climatological_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="climatological")

    spatial_root = _find_first_existing(
        eval_root,
        [
            "prcp/spatial",
            "spatial",
        ],
    )
    if spatial_root is None:
        pillar.notes.append("Spatial/climatological evaluation root not found under evaluation directory.")
        return pillar

    tables = spatial_root / "tables"
    pillar.artifacts["root"] = _safe_relpath(spatial_root, eval_root)

    for key, rel in {
        "summary_csv": "tables/spatial_summary.csv",
        "hr_all_npz": "tables/spatial_hr_ALL.npz",
        "ensmean_all_npz": "tables/spatial_ensmean_ALL.npz",
        "ensstd_all_npz": "tables/spatial_ensstd_ALL.npz",
        "lr_all_npz": "tables/spatial_lr_ALL.npz",
    }.items():
        art = _artifact_if_exists(spatial_root, rel)
        if art is not None:
            pillar.artifacts[key] = art

    rows = _read_csv_rows(tables / "spatial_summary.csv")
    if not rows:
        pillar.notes.append("spatial_summary.csv missing or empty; no climatological summary extracted.")
        return pillar

    source_order = ["ensmean", "pmm", "lr", "hr"]

    def _group_sort_key(g: str) -> tuple[int, str]:
        gs = str(g).strip()
        if gs.upper() == "ALL":
            return (0, gs)
        if gs.isdigit():
            return (1, gs)
        return (2, gs)

    def _find_row(source_name: str, preferred_groups: Sequence[str]) -> Optional[Mapping[str, Any]]:
        source_norm = str(source_name).strip().lower()
        preferred = [str(g).strip().upper() for g in preferred_groups]
        filtered = [r for r in rows if str(r.get("source", "")).strip().lower() == source_norm]
        for g in preferred:
            for row in filtered:
                if str(row.get("group", "")).strip().upper() == g:
                    return row
        if filtered:
            filtered = sorted(filtered, key=lambda r: _group_sort_key(str(r.get("group", ""))))
            return filtered[0]
        return None

    primary_source = None
    for s in source_order:
        if any(str(r.get("source", "")).strip().lower() == s for r in rows):
            primary_source = s
            break
    if primary_source is None:
        pillar.notes.append("Could not identify a primary climatological source row in spatial_summary.csv.")
        return pillar

    primary_row = _find_row(primary_source, ["ALL"])
    if primary_row is None:
        pillar.notes.append(f"Could not load primary climatological row for source '{primary_source}'.")
        return pillar

    pillar.metrics["primary_source"] = primary_source
    pillar.metrics["primary_group"] = str(primary_row.get("group", "")).strip()

    # Primary annual accumulation diagnostics
    for out_key, csv_key in {
        "annual_sum_mean": "sum_mean",
        "annual_sum_std": "sum_std",
        "annual_sum_total": "sum_total",
    }.items():
        val = _to_float_or_none(primary_row.get(csv_key))
        if val is not None:
            pillar.metrics[out_key] = val

    # HR reference
    hr_row = _find_row("hr", ["ALL"])
    if hr_row is not None:
        for out_key, csv_key in {
            "hr_annual_sum_mean": "sum_mean",
            "hr_annual_sum_std": "sum_std",
            "hr_annual_sum_total": "sum_total",
        }.items():
            val = _to_float_or_none(hr_row.get(csv_key))
            if val is not None:
                pillar.metrics[out_key] = val
    else:
        pillar.notes.append("No HR climatological ALL row found in spatial_summary.csv.")

    # LR reference
    lr_row = _find_row("lr", ["ALL"])
    if lr_row is not None:
        for out_key, csv_key in {
            "lr_annual_sum_mean": "sum_mean",
            "lr_annual_sum_std": "sum_std",
            "lr_annual_sum_total": "sum_total",
        }.items():
            val = _to_float_or_none(lr_row.get(csv_key))
            if val is not None:
                pillar.metrics[out_key] = val

    # Keep yearly annual accumulation summaries too
    def _collect_yearly(source_name: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for row in sorted(
            [r for r in rows if str(r.get("source", "")).strip().lower() == source_name],
            key=lambda r: _group_sort_key(str(r.get("group", ""))),
        ):
            group = str(row.get("group", "")).strip()
            if not group or not group.isdigit():
                continue
            mean_v = _to_float_or_none(row.get("sum_mean"))
            std_v = _to_float_or_none(row.get("sum_std"))
            total_v = _to_float_or_none(row.get("sum_total"))
            payload: Dict[str, float] = {}
            if mean_v is not None:
                payload["sum_mean"] = mean_v
            if std_v is not None:
                payload["sum_std"] = std_v
            if total_v is not None:
                payload["sum_total"] = total_v
            if payload:
                out[group] = payload
        return out

    yearly_primary = _collect_yearly(primary_source)
    if yearly_primary:
        pillar.metrics["annual_sum_by_year"] = yearly_primary
    else:
        pillar.notes.append(f"No yearly climatological rows found for primary source '{primary_source}'.")

    yearly_hr = _collect_yearly("hr")
    if yearly_hr:
        pillar.metrics["hr_annual_sum_by_year"] = yearly_hr

    yearly_lr = _collect_yearly("lr")
    if yearly_lr:
        pillar.metrics["lr_annual_sum_by_year"] = yearly_lr

    pillar.metrics["n_metric_rows"] = len(rows)

    missing_primary = []
    for req_key in ("annual_sum_mean", "annual_sum_std"):
        if req_key not in pillar.metrics:
            missing_primary.append(req_key)
    if missing_primary:
        pillar.notes.append(
            "Primary climatological row is missing one or more annual accumulation diagnostics: " + ", ".join(missing_primary)
        )

    return pillar


@register_pillar("temporal")
def build_temporal_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="temporal")

    tmp_root = _find_first_existing(
        eval_root,
        [
            "prcp/temporal",
            "temporal",
        ],
    )
    if tmp_root is None:
        pillar.notes.append("Temporal evaluation root not found.")
        return pillar

    pillar.artifacts["root"] = _safe_relpath(tmp_root, eval_root)

    tables = tmp_root / "tables"

    # --- find ALL group first ---
    files = sorted(tables.glob("temporal_metrics_*.npz"))
    if not files:
        pillar.notes.append("No temporal_metrics_*.npz found.")
        return pillar

    target_file = None
    for f in files:
        if "ALL" in f.name:
            target_file = f
            break
    if target_file is None:
        target_file = files[0]

    pillar.artifacts["metrics_npz"] = _safe_relpath(target_file, tmp_root)

    data = np.load(target_file)

    def _safe_get(key):
        return data[key] if key in data else None

    # -------------------------------------------------
    # AUTOCORR (lag-1)
    # -------------------------------------------------
    def _lag1(arr):
        return float(arr[0]) if arr is not None and len(arr) > 0 else None

    ac_hr = _safe_get("HR_autocorr")
    ac_gen = _safe_get("PMM_autocorr")
    ac_lr = _safe_get("LR_autocorr")
    ac_ens = _safe_get("GEN_ENS_autocorr")

    if ac_hr is not None:
        pillar.metrics["lag1_hr"] = _lag1(ac_hr)

    if ac_ens is not None:
        pillar.metrics["lag1_gen"] = _lag1(ac_ens)
    elif ac_gen is not None:
        pillar.metrics["lag1_gen"] = _lag1(ac_gen)

    if ac_lr is not None:
        pillar.metrics["lag1_lr"] = _lag1(ac_lr)

    if "lag1_hr" in pillar.metrics and "lag1_gen" in pillar.metrics:
        pillar.metrics["delta_lag1_gen_vs_hr"] = (
            pillar.metrics["lag1_gen"] - pillar.metrics["lag1_hr"]
        )

    # -------------------------------------------------
    # SPELL DISTRIBUTIONS
    # -------------------------------------------------
    def _mean_length(bins, pmf):
        if bins is None or pmf is None:
            return None
        bins = bins.astype(float)
        pmf = pmf.astype(float)
        if np.sum(pmf) <= 0:
            return None
        return float(np.sum(bins * pmf))

    # Wet
    wet_bins_hr = _safe_get("HR_wet_bins")
    wet_pmf_hr = _safe_get("HR_wet_pmf")
    wet_bins_gen = _safe_get("GEN_ENS_wet_bins")
    wet_pmf_gen = _safe_get("GEN_ENS_wet_pmf")
    wet_bins_lr = _safe_get("LR_wet_bins")
    wet_pmf_lr = _safe_get("LR_wet_pmf")

    pillar.metrics["wet_mean_length_hr"] = _mean_length(wet_bins_hr, wet_pmf_hr)
    pillar.metrics["wet_mean_length_gen"] = _mean_length(wet_bins_gen, wet_pmf_gen)
    pillar.metrics["wet_mean_length_lr"] = _mean_length(wet_bins_lr, wet_pmf_lr)

    # Dry
    dry_bins_hr = _safe_get("HR_dry_bins")
    dry_pmf_hr = _safe_get("HR_dry_pmf")
    dry_bins_gen = _safe_get("GEN_ENS_dry_bins")
    dry_pmf_gen = _safe_get("GEN_ENS_dry_pmf")
    dry_bins_lr = _safe_get("LR_dry_bins")
    dry_pmf_lr = _safe_get("LR_dry_pmf")

    pillar.metrics["dry_mean_length_hr"] = _mean_length(dry_bins_hr, dry_pmf_hr)
    pillar.metrics["dry_mean_length_gen"] = _mean_length(dry_bins_gen, dry_pmf_gen)
    pillar.metrics["dry_mean_length_lr"] = _mean_length(dry_bins_lr, dry_pmf_lr)

    # -------------------------------------------------
    # DISTANCE METRICS
    # -------------------------------------------------
    for key in [
        "pair_wet_JSD_GENENS_HR",
        "pair_wet_KS_GENENS_HR",
        "pair_dry_JSD_GENENS_HR",
        "pair_dry_KS_GENENS_HR",
        "pair_wet_JSD_LR_HR",
        "pair_dry_JSD_LR_HR",
    ]:
        val = _safe_get(key)
        if val is not None:
            pillar.metrics[key.lower()] = float(val)

    return pillar


@register_pillar("dates")
def build_dates_summary(eval_root: Path, cfg: Mapping[str, Any], summary: EvaluationSummary) -> PillarSummary:
    pillar = PillarSummary(name="dates")
    pillar.notes.append("Scaffold only: dates summary extraction not implemented yet.")
    return pillar


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def build_summary(
    cfg: Mapping[str, Any],
    pillars: Optional[Iterable[str]] = None,
) -> EvaluationSummary:
    """Build a paper-oriented evaluation summary from an evaluation root.

    Parameters
    ----------
    cfg:
        Evaluation configuration or resolved runtime configuration.
    pillars:
        Optional subset of pillars to build. When omitted, all registered
        pillars are included.
    """

    summary = make_summary_context(cfg)
    eval_root = Path(summary.eval_root)

    requested = list(pillars) if pillars is not None else list(PILLAR_REGISTRY.keys())
    summary.metadata["requested_pillars"] = requested

    for pillar_name in requested:
        builder = PILLAR_REGISTRY.get(pillar_name)
        if builder is None:
            logger.warning("[build_summary] Unknown pillar '%s'; skipping.", pillar_name)
            continue
        try:
            pillar_summary = builder(eval_root, cfg, summary)
        except Exception as e:
            logger.exception("[build_summary] Failed while building pillar '%s': %s", pillar_name, e)
            pillar_summary = PillarSummary(name=pillar_name)
            pillar_summary.notes.append(f"Failed to build pillar summary: {e}")
        summary.add_pillar(pillar_summary)

    return summary


def write_summary(
    summary: EvaluationSummary,
    out_path: Optional[Path] = None,
) -> Path:
    """Write the summary JSON to disk and return the written path."""

    eval_root = Path(summary.eval_root)
    target = out_path if out_path is not None else (eval_root / "summary" / "evaluation_summary.json")
    _write_json(target, summary.to_dict())
    logger.info("[build_summary] Wrote summary JSON → %s", target)
    return target


def build_and_write_summary(
    cfg: Mapping[str, Any],
    pillars: Optional[Iterable[str]] = None,
    out_path: Optional[Path] = None,
) -> Path:
    summary = build_summary(cfg=cfg, pillars=pillars)
    return write_summary(summary=summary, out_path=out_path)


__all__ = [
    "PillarSummary",
    "EvaluationSummary",
    "PILLAR_REGISTRY",
    "register_pillar",
    "build_summary",
    "write_summary",
    "build_and_write_summary",
]