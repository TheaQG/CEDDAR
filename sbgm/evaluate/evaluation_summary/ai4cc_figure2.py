"""Seed-aware Figure 2 analysis for the AI4CC manuscript.

The script reads only ``summary/evaluation_summary.json`` files.  It can analyse
either the original curated comparison or every automatically discovered
configuration with a complete seed-1/seed-2 pair.  It converts every metric to
an improvement score relative to the mean of the three V0 training seeds and
plots the individual training seeds together with the observed V0 min--max
envelope.  An exported inventory records configurations excluded because they
are incomplete or marked as special/test runs.

The module is deliberately usable both from a terminal and from a notebook::

    from ai4cc_figure2 import run_analysis

    result = run_analysis(
        eval_root="~/CEDDAR_evaluation",
        output_dir="~/CEDDAR_evaluation/ai4cc_figure2",
    )
    result["figure"]

Positive improvement always means better than the V0 seed mean.  This is a
target-oriented score in the metric's original units, not a dimensionless
effect size:

* target 0 (for example CRPS): ``V0_mean - value``;
* target 1 (ISS): ``value - V0_mean``;
* HR reference: mean V0 distance to HR minus the model distance to HR.

The shaded interval is descriptive: it is the observed minimum and maximum of
the three transformed V0 values.  It is not a confidence interval.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


# %% Analysis configuration -------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one target-oriented metric shown in Figure 2."""

    key: str
    label: str
    target: str  # "zero", "one", or "hr"
    reference_key: str | None = None
    unit: str = ""


METRICS: dict[str, MetricSpec] = {
    "probabilistic.crps_mean": MetricSpec(
        key="probabilistic.crps_mean",
        label="CRPS",
        target="zero",
        unit="mm day$^{-1}$",
    ),
    "scale.iss_primary": MetricSpec(
        key="scale.iss_primary",
        label="Integrated skill score",
        target="one",
    ),
    "scale.slope_mid": MetricSpec(
        key="scale.slope_mid",
        label="Mid-scale PSD slope",
        target="hr",
        reference_key="scale.hr_slope_mid",
    ),
    "extremes.p99_9": MetricSpec(
        key="extremes.p99_9",
        label="99.9th percentile",
        target="hr",
        reference_key="extremes.hr_p99_9",
        unit="mm day$^{-1}$",
    ),
    "climatological.annual_sum_mean": MetricSpec(
        key="climatological.annual_sum_mean",
        label="Mean annual precipitation",
        target="hr",
        reference_key="climatological.hr_annual_sum_mean",
        unit="mm year$^{-1}$",
    ),
    "climatological.annual_sum_std": MetricSpec(
        key="climatological.annual_sum_std",
        label="Spatial SD of annual precipitation",
        target="hr",
        reference_key="climatological.hr_annual_sum_std",
        unit="mm year$^{-1}$",
    ),
    # Additional targetable metrics used only by the robustness audit.
    "distributional.wasserstein": MetricSpec(
        "distributional.wasserstein", "Wasserstein distance", "zero"
    ),
    "distributional.ks_stat": MetricSpec(
        "distributional.ks_stat", "KS statistic", "zero"
    ),
    "probabilistic.pit_ks_D": MetricSpec(
        "probabilistic.pit_ks_D", "PIT KS statistic", "zero"
    ),
    "probabilistic.spatial_crps_land_mean": MetricSpec(
        "probabilistic.spatial_crps_land_mean",
        "Land-only spatial CRPS",
        "zero",
        unit="mm day$^{-1}$",
    ),
    "probabilistic.rankhist_max_abs_z": MetricSpec(
        "probabilistic.rankhist_max_abs_z", "Rank-histogram max |z|", "zero"
    ),
    "probabilistic.spread_skill_slope": MetricSpec(
        "probabilistic.spread_skill_slope", "Spread-skill slope", "one"
    ),
    "scale.slope_low": MetricSpec(
        "scale.slope_low",
        "Low-scale PSD slope",
        "hr",
        reference_key="scale.hr_slope_low",
    ),
    "scale.slope_high": MetricSpec(
        "scale.slope_high",
        "High-scale PSD slope",
        "hr",
        reference_key="scale.hr_slope_high",
    ),
    "extremes.p99": MetricSpec(
        "extremes.p99",
        "99th percentile",
        "hr",
        reference_key="extremes.hr_p99",
        unit="mm day$^{-1}$",
    ),
    "extremes.rx1day": MetricSpec(
        "extremes.rx1day",
        "RX1day",
        "hr",
        reference_key="extremes.hr_rx1day",
        unit="mm day$^{-1}$",
    ),
    "extremes.rx5day": MetricSpec(
        "extremes.rx5day",
        "RX5day",
        "hr",
        reference_key="extremes.hr_rx5day",
        unit="mm",
    ),
    "extremes.wet_freq": MetricSpec(
        "extremes.wet_freq",
        "Wet-day frequency",
        "hr",
        reference_key="extremes.hr_wet_freq",
    ),
    "extremes.wet_hit_rate": MetricSpec(
        "extremes.wet_hit_rate", "Wet-day hit rate", "one"
    ),
    "features.gen_ens_SAL": MetricSpec(
        "features.gen_ens_SAL", "SAL", "zero"
    ),
    "features.gen_ens_S": MetricSpec(
        "features.gen_ens_S", "SAL structure component", "zero"
    ),
    "features.gen_ens_A": MetricSpec(
        "features.gen_ens_A", "SAL amplitude component", "zero"
    ),
    "features.gen_ens_L": MetricSpec(
        "features.gen_ens_L", "SAL location component", "zero"
    ),
    "temporal.lag1_gen": MetricSpec(
        "temporal.lag1_gen",
        "Lag-1 autocorrelation",
        "hr",
        reference_key="temporal.lag1_hr",
    ),
    "temporal.wet_mean_length_gen": MetricSpec(
        "temporal.wet_mean_length_gen",
        "Mean wet-spell length",
        "hr",
        reference_key="temporal.wet_mean_length_hr",
    ),
    "temporal.dry_mean_length_gen": MetricSpec(
        "temporal.dry_mean_length_gen",
        "Mean dry-spell length",
        "hr",
        reference_key="temporal.dry_mean_length_hr",
    ),
    "temporal.pair_wet_jsd_genens_hr": MetricSpec(
        "temporal.pair_wet_jsd_genens_hr", "Wet-spell duration JSD", "zero"
    ),
    "temporal.pair_dry_jsd_genens_hr": MetricSpec(
        "temporal.pair_dry_jsd_genens_hr", "Dry-spell duration JSD", "zero"
    ),
}

DEFAULT_METRICS: tuple[str, ...] = (
    "probabilistic.crps_mean",
    "scale.iss_primary",
    "scale.slope_mid",
    "extremes.p99_9",
    "climatological.annual_sum_mean",
    "climatological.annual_sum_std",
)

CURATED_CONFIGURATION_ORDER: tuple[str, ...] = (
    "V0",
    "C_10",
    "V0_I_0",
    "V0_T_1",
    "D_MZIP_1",
    "D_IP_1",
    "D_MTP_1",
    "D_MT_1",
    "D_MTC_1",
)

# Scientific ordering for currently known configurations. Newly discovered
# complete pairs that are not listed here are appended deterministically, so
# adding evaluation folders never requires editing this tuple.
ALL_PREFERRED_CONFIGURATION_ORDER: tuple[str, ...] = (
    "V0",
    "C_10",
    "V0_C_0",
    "V0_C_1",
    "V0_I_0",
    "V0_I_1",
    "V0_M_0",
    "V0_M_1",
    "V0_T_0",
    "V0_T_1",
    "V0_Z_0",
    "V0_Z_1",
    "D_IP_1",
    "D_MZP_1",
    "D_MTP_0",
    "D_MTP_1",
    "D_MTCP_1",
    "D_MZIP_1",
    "D_MT_1",
    "D_MTC_1",
)

ANALYSIS_SCOPES: tuple[str, ...] = ("curated", "all-paired")

NO_P_CONFIGURATIONS: frozenset[str] = frozenset({"D_MT_1", "D_MTC_1"})

CURATED_EXPECTED_SEEDS: dict[str, tuple[int, ...]] = {
    configuration: ((1, 2, 3) if configuration == "V0" else (1, 2))
    for configuration in CURATED_CONFIGURATION_ORDER
}

DISPLAY_LABELS: dict[str, str] = {
    "V0": "V0 (baseline)",
    "C_10": "C_10 (context)",
    "V0_I_0": "V0_I_0",
    "V0_T_1": "V0_T_1",
    "D_MZIP_1": "D_MZIP_1",
    "D_IP_1": "D_IP_1",
    "D_MTP_1": "D_MTP_1",
    "D_MT_1": "D_MT_1 (no P)",
    "D_MTC_1": "D_MTC_1 (no P)",
}


def display_label(configuration: str) -> str:
    """Return a readable label without requiring every run to be registered."""

    if configuration in DISPLAY_LABELS:
        return DISPLAY_LABELS[configuration]
    if configuration in NO_P_CONFIGURATIONS:
        return f"{configuration} (no P)"
    return configuration


def is_special_configuration(configuration: str) -> bool:
    """Identify diagnostic/ablation runs excluded from the main comparison."""

    lowered = configuration.lower()
    return lowered.startswith("test_") or "nosdf" in lowered


def resolve_configuration_order(configurations: Iterable[str]) -> tuple[str, ...]:
    """Order known families scientifically and append future families safely."""

    available = set(map(str, configurations))
    preferred = [name for name in ALL_PREFERRED_CONFIGURATION_ORDER if name in available]
    remaining = sorted(available.difference(preferred))
    return tuple(preferred + remaining)


def configuration_group(configuration: str) -> str:
    """Return the design-based configuration family used in summaries."""

    if configuration in NO_P_CONFIGURATIONS:
        return "no-P"
    if configuration.startswith("D_"):
        return "precipitation-conditioned physics set"
    return "context/single-variable"

SEED_STYLES: dict[int, dict[str, Any]] = {
    1: {"color": "#0072B2", "marker": "o"},
    2: {"color": "#D55E00", "marker": "s"},
    3: {"color": "#009E73", "marker": "^"},
}

# Predefined before inspecting the stability result: these are the metrics used
# by the broader Paper II evaluation, grouped by scientific evaluation pillar.
STABILITY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("distributional.wasserstein", "Wasserstein distance", "Distributional"),
    ("distributional.ks_stat", "KS statistic", "Distributional"),
    ("probabilistic.crps_mean", "CRPS", "Probabilistic"),
    ("probabilistic.pit_ks_D", "PIT KS statistic", "Probabilistic"),
    ("scale.iss_primary", "Integrated skill score", "Scale"),
    ("scale.slope_low", "Low-scale PSD slope", "Scale"),
    ("scale.slope_mid", "Mid-scale PSD slope", "Scale"),
    ("scale.slope_high", "High-scale PSD slope", "Scale"),
    ("scale.corr_gen_lr_low", "Low-scale GEN–LR correlation", "Scale"),
    ("scale.corr_gen_lr_mid", "Mid-scale GEN–LR correlation", "Scale"),
    ("scale.corr_gen_lr_high", "High-scale GEN–LR correlation", "Scale"),
    ("extremes.p99", "99th percentile", "Extremes"),
    ("extremes.p99_9", "99.9th percentile", "Extremes"),
    ("extremes.rx1day", "RX1day", "Extremes"),
    ("extremes.rx5day", "RX5day", "Extremes"),
    ("features.gen_ens_SAL", "SAL", "Features"),
    ("climatological.annual_sum_mean", "Mean annual precipitation", "Climatological"),
    ("climatological.annual_sum_std", "Spatial SD of annual precipitation", "Climatological"),
    ("temporal.lag1_gen", "Lag-1 autocorrelation", "Temporal"),
    ("temporal.wet_mean_length_gen", "Mean wet-spell length", "Temporal"),
    ("temporal.dry_mean_length_gen", "Mean dry-spell length", "Temporal"),
)

CORE_STABILITY_METRICS = STABILITY_METRICS

# Scientifically distinct additions selected before inspecting their results.
# Reference values, sample counts, standard errors and duplicate annual totals
# are deliberately excluded so that they do not masquerade as independent
# outcomes.
EXPANDED_STABILITY_METRICS: tuple[tuple[str, str, str], ...] = (
    *CORE_STABILITY_METRICS,
    ("probabilistic.spatial_crps_land_mean", "Land-only spatial CRPS", "Probabilistic"),
    ("probabilistic.rankhist_max_abs_z", "Rank-histogram max |z|", "Probabilistic"),
    ("probabilistic.spread_skill_slope", "Spread-skill slope", "Probabilistic"),
    ("extremes.wet_freq", "Wet-day frequency", "Extremes"),
    ("extremes.wet_hit_rate", "Wet-day hit rate", "Extremes"),
    ("features.gen_ens_S", "SAL structure component", "Features"),
    ("features.gen_ens_A", "SAL amplitude component", "Features"),
    ("features.gen_ens_L", "SAL location component", "Features"),
    ("temporal.pair_wet_jsd_genens_hr", "Wet-spell duration JSD", "Temporal"),
    ("temporal.pair_dry_jsd_genens_hr", "Dry-spell duration JSD", "Temporal"),
)

# Metrics with defensible evaluation targets. GEN-LR correlations are retained
# in the seed-sensitivity registry but excluded here because neither higher nor
# lower correlation is unambiguously better for a stochastic downscaler.
ROBUSTNESS_ATTRIBUTION_METRICS: tuple[str, ...] = (
    "distributional.wasserstein",
    "distributional.ks_stat",
    "probabilistic.crps_mean",
    "probabilistic.pit_ks_D",
    "probabilistic.spatial_crps_land_mean",
    "probabilistic.rankhist_max_abs_z",
    "probabilistic.spread_skill_slope",
    "scale.iss_primary",
    "scale.slope_low",
    "scale.slope_mid",
    "scale.slope_high",
    "extremes.p99",
    "extremes.p99_9",
    "extremes.rx1day",
    "extremes.rx5day",
    "extremes.wet_freq",
    "extremes.wet_hit_rate",
    "features.gen_ens_SAL",
    "features.gen_ens_S",
    "features.gen_ens_A",
    "features.gen_ens_L",
    "climatological.annual_sum_mean",
    "climatological.annual_sum_std",
    "temporal.lag1_gen",
    "temporal.wet_mean_length_gen",
    "temporal.dry_mean_length_gen",
    "temporal.pair_wet_jsd_genens_hr",
    "temporal.pair_dry_jsd_genens_hr",
)

SEASONAL_CANDIDATE_PREFIXES: tuple[str, ...] = (
    "seasonal.",
    "monthly.",
    "climatological.seasonal_",
    "climatological.monthly_",
)

PILLAR_COLORS: dict[str, str] = {
    "Distributional": "#CC79A7",
    "Probabilistic": "#0072B2",
    "Scale": "#009E73",
    "Extremes": "#D55E00",
    "Features": "#E69F00",
    "Climatological": "#56B4E9",
    "Temporal": "#6A3D9A",
}


# %% Loading and run-name handling -----------------------------------------


def _flatten_dict(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_dict(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def flatten_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the scalar metric payload from one evaluation summary."""

    row: dict[str, Any] = {
        "run_id": summary.get("run_id") or summary.get("model_key"),
        "model_key": summary.get("model_key"),
    }
    for pillar_name, pillar in (summary.get("pillars", {}) or {}).items():
        metrics = (pillar or {}).get("metrics", {}) or {}
        row.update(_flatten_dict(metrics, str(pillar_name)))
    return row


def _fallback_run_id(summary_path: Path) -> str:
    if summary_path.parent.name == "summary":
        return summary_path.parent.parent.name
    if summary_path.name == "evaluation_summary.json":
        return summary_path.parent.name
    return summary_path.stem


def parse_run_identity(run_id: str) -> tuple[str, int, str]:
    """Return ``(configuration, seed, short_run_label)`` for a CEDDAR run."""

    short = str(run_id).split("__HR_", 1)[0]
    match = re.search(r"(?:__|_)seed(?P<seed>\d+)$", short, flags=re.IGNORECASE)
    if match:
        seed = int(match.group("seed"))
        configuration = short[: match.start()]
    else:
        seed = 1
        configuration = short
    configuration = configuration.rstrip("_")
    clean_label = configuration if seed == 1 else f"{configuration}_seed{seed}"
    return configuration, seed, clean_label


def find_summary_files(eval_root: str | Path) -> list[Path]:
    """Find summaries recursively, including a summary passed as a direct path."""

    root = Path(eval_root).expanduser().resolve()
    if root.is_file():
        if root.name != "evaluation_summary.json":
            raise ValueError(f"Expected evaluation_summary.json, received: {root}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {root}")
    return sorted(root.rglob("evaluation_summary.json"))


def load_summary_catalog(eval_root: str | Path) -> pd.DataFrame:
    """Load every discovered summary before applying an analysis scope."""

    rows: list[dict[str, Any]] = []
    for path in find_summary_files(eval_root):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        row = flatten_summary(summary)
        summary_run_id = row.get("run_id")
        # The directory name is authoritative: diagnostic folders can reuse an
        # internal model/run ID and would otherwise lose prefixes such as test_.
        source_run_id = _fallback_run_id(path)
        configuration, seed, clean_label = parse_run_identity(source_run_id)
        row.update(
            {
                "run_id": source_run_id,
                "summary_run_id": summary_run_id,
                "configuration": configuration,
                "seed": seed,
                "run_label": clean_label,
                "summary_path": str(path),
            }
        )
        rows.append(row)

    if not rows:
        raise ValueError(f"No evaluation summaries were found below {Path(eval_root).expanduser()}.")

    df = pd.DataFrame(rows)
    duplicate_mask = df.duplicated(["configuration", "seed"], keep=False)
    if duplicate_mask.any():
        duplicate_paths = df.loc[
            duplicate_mask, ["configuration", "seed", "summary_path"]
        ].to_string(index=False)
        raise ValueError(
            "Multiple summaries map to the same configuration and seed:\n"
            f"{duplicate_paths}"
        )

    return df.sort_values(["configuration", "seed"], kind="stable").reset_index(drop=True)


def build_configuration_inventory(catalog: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed completeness and automatic inclusion eligibility."""

    records: list[dict[str, Any]] = []
    for configuration, family in catalog.groupby("configuration", sort=True):
        seeds = tuple(sorted(map(int, family["seed"].unique())))
        special = is_special_configuration(str(configuration))
        if special:
            reason = "special/test or noSDF ablation"
        elif configuration == "V0" and len(seeds) < 2:
            reason = "baseline has fewer than two seeds"
        elif configuration != "V0" and not {1, 2}.issubset(seeds):
            reason = "incomplete seed-1/seed-2 pair"
        else:
            reason = "included"
        records.append(
            {
                "configuration": str(configuration),
                "available_seeds": ",".join(map(str, seeds)),
                "n_summaries": int(len(family)),
                "is_special": special,
                "has_complete_pair": {1, 2}.issubset(seeds),
                "all_paired_status": reason,
            }
        )
    return pd.DataFrame.from_records(records)


def selection_report(
    df: pd.DataFrame,
    *,
    scope: str,
    inventory: pd.DataFrame,
) -> str:
    """Return a compact, scope-aware report of included and excluded runs."""

    lines = [
        f"Analysis scope: {scope}",
        f"Included: {df['configuration'].nunique()} configurations, {len(df)} summaries",
    ]
    if scope == "curated":
        found = {
            (str(configuration), int(seed))
            for configuration, seed in zip(df["configuration"], df["seed"])
        }
        missing = [
            f"{configuration}_seed{seed}" if seed > 1 else configuration
            for configuration, seeds in CURATED_EXPECTED_SEEDS.items()
            for seed in seeds
            if (configuration, seed) not in found
        ]
        if missing:
            lines.append("Missing curated runs: " + ", ".join(missing))
    else:
        excluded = inventory.loc[inventory["all_paired_status"].ne("included")]
        if not excluded.empty:
            descriptions = [
                f"{row.configuration} ({row.all_paired_status}; seeds {row.available_seeds})"
                for row in excluded.itertuples()
            ]
            lines.append("Excluded from paired analysis: " + "; ".join(descriptions))
    return "\n".join(lines)


def load_analysis_summaries(
    eval_root: str | Path,
    *,
    scope: str = "all-paired",
    allow_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Load either the curated comparison or every complete two-seed family."""

    if scope not in ANALYSIS_SCOPES:
        raise ValueError(f"Unknown scope '{scope}'. Choose from {ANALYSIS_SCOPES}.")
    catalog = load_summary_catalog(eval_root)
    inventory = build_configuration_inventory(catalog)

    if scope == "curated":
        keep = catalog.apply(
            lambda row: row["configuration"] in CURATED_EXPECTED_SEEDS
            and int(row["seed"]) in CURATED_EXPECTED_SEEDS[row["configuration"]],
            axis=1,
        )
        df = catalog.loc[keep].copy()
        configuration_order = tuple(
            name for name in CURATED_CONFIGURATION_ORDER if name in set(df["configuration"])
        )
    else:
        eligible = set(
            inventory.loc[inventory["all_paired_status"].eq("included"), "configuration"]
        )
        df = catalog.loc[catalog["configuration"].isin(eligible)].copy()
        configuration_order = resolve_configuration_order(eligible)

    if df.empty:
        raise ValueError(f"No summaries qualify for the '{scope}' analysis scope.")

    order = {name: index for index, name in enumerate(configuration_order)}
    df["_configuration_order"] = df["configuration"].map(order)
    df = (
        df.sort_values(["_configuration_order", "seed"], kind="stable")
        .drop(columns="_configuration_order")
        .reset_index(drop=True)
    )

    report = selection_report(df, scope=scope, inventory=inventory)
    if "Missing curated runs:" in report and not allow_missing:
        raise ValueError(report + "\nUse --allow-missing for exploratory partial plots.")
    if "Missing curated runs:" in report:
        warnings.warn(report, stacklevel=2)
    return df, inventory, configuration_order


def load_selected_summaries(
    eval_root: str | Path,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Backward-compatible loader for the original curated comparison."""

    df, _, _ = load_analysis_summaries(
        eval_root,
        scope="curated",
        allow_missing=allow_missing,
    )
    return df


# %% Target-oriented improvement scores ------------------------------------


def _numeric(df: pd.DataFrame, key: str) -> pd.Series:
    if key not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[key], errors="coerce")


def _constant_hr_reference(df: pd.DataFrame, key: str, metric_label: str) -> float:
    references = _numeric(df, key).dropna()
    if references.empty:
        raise ValueError(
            f"{metric_label} requires HR reference metric '{key}', but it is missing."
        )
    target = float(references.mean())
    spread = float(references.max() - references.min())
    tolerance = max(1e-10, abs(target) * 1e-8)
    if spread > tolerance:
        warnings.warn(
            f"HR reference '{key}' differs across summaries (range={spread:.6g}); "
            f"using their mean ({target:.6g}).",
            stacklevel=2,
        )
    return target


def compute_improvements(
    df: pd.DataFrame,
    metric_keys: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Return one row per run and metric with positive consistently better."""

    baseline_mask = df["configuration"].eq("V0")
    if int(baseline_mask.sum()) < 2:
        raise ValueError("At least two V0 seeds are required to define the baseline spread.")

    records: list[pd.DataFrame] = []
    for metric_key in metric_keys:
        if metric_key not in METRICS:
            raise KeyError(
                f"Unknown metric '{metric_key}'. Available: {', '.join(METRICS)}"
            )
        spec = METRICS[metric_key]
        values = _numeric(df, metric_key)
        baseline_values = values.loc[baseline_mask].dropna()
        if baseline_values.empty:
            warnings.warn(f"Skipping {metric_key}: no numeric V0 values.", stacklevel=2)
            continue

        baseline_mean = float(baseline_values.mean())
        reference_value = np.nan
        if spec.target == "zero":
            target_value = 0.0
            improvement = baseline_mean - values
            target_improvement = baseline_mean
        elif spec.target == "one":
            target_value = 1.0
            improvement = values - baseline_mean
            target_improvement = 1.0 - baseline_mean
        elif spec.target == "hr":
            if not spec.reference_key:
                raise ValueError(f"No HR reference key configured for {metric_key}.")
            reference_value = _constant_hr_reference(df, spec.reference_key, spec.label)
            target_value = reference_value
            baseline_distance = float((baseline_values - reference_value).abs().mean())
            improvement = baseline_distance - (values - reference_value).abs()
            target_improvement = baseline_distance
        else:
            raise ValueError(f"Unsupported target type '{spec.target}' for {metric_key}.")

        metric_rows = df[
            ["run_id", "run_label", "configuration", "seed", "summary_path"]
        ].copy()
        metric_rows["metric"] = metric_key
        metric_rows["metric_label"] = spec.label
        metric_rows["unit"] = spec.unit
        metric_rows["target_type"] = spec.target
        metric_rows["reference_key"] = spec.reference_key
        metric_rows["reference_value"] = reference_value
        metric_rows["target_value"] = target_value
        metric_rows["target_improvement"] = target_improvement
        metric_rows["value"] = values
        metric_rows["v0_mean_value"] = baseline_mean
        metric_rows["improvement"] = improvement
        records.append(metric_rows)

    if not records:
        raise ValueError("None of the requested metrics could be computed.")

    long_df = pd.concat(records, ignore_index=True)
    band = (
        long_df.loc[long_df["configuration"].eq("V0")]
        .groupby("metric", sort=False)["improvement"]
        .agg(v0_band_min="min", v0_band_max="max")
    )
    long_df = long_df.join(band, on="metric")
    return long_df


def compute_seed_sensitivity(
    df: pd.DataFrame,
    metric_registry: Sequence[tuple[str, str, str]] = STABILITY_METRICS,
    *,
    configuration_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute descriptive two-seed gaps relative to the observed V0 range.

    The calculation deliberately uses raw metric values.  Applying the
    target-oriented absolute-distance transform first could hide differences
    when two values lie on opposite sides of an HR target.  Each non-baseline
    configuration contributes one absolute two-seed gap per available metric;
    this is not an estimate of seed variance.
    """

    records: list[dict[str, Any]] = []
    for metric_index, (metric_key, metric_label, pillar) in enumerate(metric_registry):
        values = _numeric(df, metric_key)
        baseline_values = values.loc[df["configuration"].eq("V0")].dropna()
        if len(baseline_values) < 2:
            warnings.warn(
                f"Skipping seed sensitivity for {metric_key}: fewer than two V0 values.",
                stacklevel=2,
            )
            continue
        v0_min = float(baseline_values.min())
        v0_max = float(baseline_values.max())
        v0_range = v0_max - v0_min
        if not math.isfinite(v0_range) or v0_range <= 0.0:
            warnings.warn(
                f"Skipping seed sensitivity for {metric_key}: V0 range is zero or invalid.",
                stacklevel=2,
            )
            continue

        ordered_configurations = tuple(configuration_order or resolve_configuration_order(df["configuration"]))
        for configuration in ordered_configurations:
            if configuration == "V0":
                continue
            family = df.loc[df["configuration"].eq(configuration)].sort_values("seed")
            family_values = _numeric(family, metric_key).dropna()
            if len(family_values) != 2:
                continue
            seed_1_value = float(family_values.iloc[0])
            seed_2_value = float(family_values.iloc[1])
            raw_gap = abs(seed_1_value - seed_2_value)
            records.append(
                {
                    "configuration": configuration,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "metric_index": metric_index,
                    "pillar": pillar,
                    "seed_1_value": seed_1_value,
                    "seed_2_value": seed_2_value,
                    "raw_seed_gap": raw_gap,
                    "v0_min": v0_min,
                    "v0_max": v0_max,
                    "v0_range": v0_range,
                    "normalized_seed_gap": raw_gap / v0_range,
                }
            )

    if not records:
        raise ValueError("No two-seed sensitivity values could be computed.")
    return pd.DataFrame.from_records(records)


def summarize_seed_sensitivity(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Create a descriptive configuration-level sensitivity audit table."""

    summary = (
        sensitivity.groupby("configuration", sort=False)["normalized_seed_gap"]
        .agg(
            median_normalized_gap="median",
            mean_normalized_gap="mean",
            q25_normalized_gap=lambda values: values.quantile(0.25),
            q75_normalized_gap=lambda values: values.quantile(0.75),
            maximum_normalized_gap="max",
            n_metrics="count",
        )
        .reset_index()
    )
    above = (
        sensitivity.assign(above_v0_range=sensitivity["normalized_seed_gap"].gt(1.0))
        .groupby("configuration", sort=False)["above_v0_range"]
        .mean()
        .rename("fraction_metrics_above_v0_range")
    )
    summary = summary.join(above, on="configuration")
    return summary.sort_values("median_normalized_gap", kind="stable").reset_index(drop=True)


def summarize_pillar_balanced_sensitivity(
    sensitivity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give each scientific pillar equal influence on configuration summaries."""

    pillar_summary = (
        sensitivity.groupby(["configuration", "pillar"], sort=False)[
            "normalized_seed_gap"
        ]
        .agg(
            pillar_median_gap="median",
            pillar_mean_gap="mean",
            n_metrics="count",
        )
        .reset_index()
    )
    configuration_summary = (
        pillar_summary.groupby("configuration", sort=False)["pillar_median_gap"]
        .agg(
            pillar_balanced_median_gap="median",
            pillar_balanced_mean_gap="mean",
            minimum_pillar_median_gap="min",
            maximum_pillar_median_gap="max",
            n_pillars="count",
        )
        .reset_index()
        .sort_values("pillar_balanced_median_gap", kind="stable")
        .reset_index(drop=True)
    )
    configuration_summary["pillar_balanced_rank"] = np.arange(
        1, len(configuration_summary) + 1
    )
    return pillar_summary, configuration_summary


def leave_one_pillar_out_sensitivity(
    pillar_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Re-rank configurations after omitting each evaluation pillar in turn."""

    records: list[pd.DataFrame] = []
    for omitted_pillar in sorted(pillar_summary["pillar"].unique()):
        reduced = pillar_summary.loc[pillar_summary["pillar"].ne(omitted_pillar)]
        scores = (
            reduced.groupby("configuration", sort=False)["pillar_median_gap"]
            .median()
            .rename("leave_one_out_median_gap")
            .reset_index()
            .sort_values("leave_one_out_median_gap", kind="stable")
            .reset_index(drop=True)
        )
        scores["leave_one_out_rank"] = np.arange(1, len(scores) + 1)
        scores["omitted_pillar"] = omitted_pillar
        records.append(scores)
    return pd.concat(records, ignore_index=True)


def build_stability_robustness_summary(
    core_sensitivity: pd.DataFrame,
    expanded_sensitivity: pd.DataFrame,
    expanded_pillar_balanced: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> pd.DataFrame:
    """Compare rankings across core, expanded and pillar-balanced definitions."""

    core = summarize_seed_sensitivity(core_sensitivity).rename(
        columns={"median_normalized_gap": "core_metric_median_gap"}
    )[["configuration", "core_metric_median_gap"]]
    expanded = summarize_seed_sensitivity(expanded_sensitivity).rename(
        columns={"median_normalized_gap": "expanded_metric_median_gap"}
    )[["configuration", "expanded_metric_median_gap"]]
    result = core.merge(expanded, on="configuration", how="outer").merge(
        expanded_pillar_balanced[
            ["configuration", "pillar_balanced_median_gap", "n_pillars"]
        ],
        on="configuration",
        how="outer",
    )
    for score_column, rank_column in (
        ("core_metric_median_gap", "core_metric_rank"),
        ("expanded_metric_median_gap", "expanded_metric_rank"),
        ("pillar_balanced_median_gap", "pillar_balanced_rank"),
    ):
        result[rank_column] = result[score_column].rank(method="min")

    rank_range = (
        leave_one_out.groupby("configuration")["leave_one_out_rank"]
        .agg(
            best_leave_one_pillar_out_rank="min",
            worst_leave_one_pillar_out_rank="max",
            median_leave_one_pillar_out_rank="median",
        )
        .reset_index()
    )
    result = result.merge(rank_range, on="configuration", how="left")
    result["configuration_group"] = result["configuration"].map(configuration_group)
    result["leave_one_pillar_out_rank_range"] = (
        result["worst_leave_one_pillar_out_rank"]
        - result["best_leave_one_pillar_out_rank"]
    )
    return result.sort_values("pillar_balanced_rank", kind="stable").reset_index(drop=True)


def summarize_stability_by_group(robustness: pd.DataFrame) -> pd.DataFrame:
    """Summarize configuration-level scores without treating metrics as replicates."""

    return (
        robustness.groupby("configuration_group", sort=False)
        .agg(
            n_configurations=("configuration", "count"),
            core_group_median_gap=("core_metric_median_gap", "median"),
            expanded_group_median_gap=("expanded_metric_median_gap", "median"),
            pillar_balanced_group_median_gap=(
                "pillar_balanced_median_gap",
                "median",
            ),
        )
        .reset_index()
    )


def summarize_attribution_classes(
    improvements: pd.DataFrame,
    metric_to_pillar: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify each seed and each configuration-metric pair against V0."""

    details = improvements.loc[
        improvements["configuration"].ne("V0")
        & improvements["improvement"].notna()
    ].copy()
    details["pillar"] = details["metric"].map(metric_to_pillar)
    details["seed_class"] = np.select(
        [
            details["improvement"].gt(details["v0_band_max"]),
            details["improvement"].lt(details["v0_band_min"]),
        ],
        ["favourable", "unfavourable"],
        default="inside_v0_range",
    )

    pair_records: list[dict[str, Any]] = []
    for (configuration, metric), family in details.groupby(
        ["configuration", "metric"], sort=False
    ):
        classes = set(map(str, family["seed_class"]))
        if classes == {"favourable"}:
            pair_status = "both_favourable"
        elif classes == {"unfavourable"}:
            pair_status = "both_unfavourable"
        elif classes == {"inside_v0_range"}:
            pair_status = "both_inside_v0_range"
        else:
            pair_status = "mixed"
        pair_records.append(
            {
                "configuration": configuration,
                "metric": metric,
                "metric_label": str(family["metric_label"].iloc[0]),
                "pillar": str(family["pillar"].iloc[0]),
                "pair_status": pair_status,
                "n_seeds": int(len(family)),
            }
        )
    pairs = pd.DataFrame.from_records(pair_records)

    statuses = (
        "both_favourable",
        "both_inside_v0_range",
        "both_unfavourable",
        "mixed",
    )
    counts = pd.crosstab(pairs["configuration"], pairs["pair_status"]).reindex(
        columns=statuses,
        fill_value=0,
    )
    counts["n_metrics"] = counts.sum(axis=1)
    for status in statuses:
        counts[f"fraction_{status}"] = counts[status] / counts["n_metrics"]
    return details, counts.reset_index()


def build_metric_coverage_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Record availability and intentional scope decisions for robustness metrics."""

    core_keys = {metric[0] for metric in CORE_STABILITY_METRICS}
    rows: list[dict[str, Any]] = []
    for metric_key, metric_label, pillar in EXPANDED_STABILITY_METRICS:
        values = _numeric(df, metric_key)
        rows.append(
            {
                "metric": metric_key,
                "metric_label": metric_label,
                "pillar": pillar,
                "registry": "core" if metric_key in core_keys else "expanded_addition",
                "n_numeric_runs": int(values.notna().sum()),
                "n_configurations": int(
                    values.notna().groupby(df["configuration"]).any().sum()
                ),
                "has_target_or_reference": metric_key in METRICS,
                "status": "available" if values.notna().any() else "missing",
            }
        )

    seasonal_columns = [
        column
        for column in df.columns
        if any(column.startswith(prefix) for prefix in SEASONAL_CANDIDATE_PREFIXES)
    ]
    rows.append(
        {
            "metric": "seasonal_or_monthly_precipitation",
            "metric_label": "Seasonal/monthly precipitation diagnostics",
            "pillar": "Climatological",
            "registry": "desired_future_addition",
            "n_numeric_runs": 0,
            "n_configurations": 0,
            "has_target_or_reference": False,
            "status": (
                "available_columns: " + ", ".join(seasonal_columns)
                if seasonal_columns
                else "not available in evaluation_summary.json; requires raw evaluation"
            ),
        }
    )
    return pd.DataFrame.from_records(rows)


# %% Figure -----------------------------------------------------------------


def _padded_limits(values: Iterable[float], *, include_zero: bool) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(finite.min())
    upper = float(finite.max())
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    if math.isclose(span, 0.0):
        scale = max(abs(lower), 1.0)
        return lower - 0.08 * scale, upper + 0.08 * scale
    padding = 0.07 * span
    return lower - padding, upper + padding


def plot_figure2(
    improvements: pd.DataFrame,
    metric_keys: Sequence[str] = DEFAULT_METRICS,
    *,
    configuration_order: Sequence[str] | None = None,
    include_no_p: bool = True,
    figsize: tuple[float, float] = (12.2, 9.2),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot seed points and the observed V0 min--max envelope."""

    n_metrics = len(metric_keys)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.09,
        top=0.89,
        hspace=0.42,
        wspace=0.18,
    )
    axes_flat = axes.ravel()
    allowed_configurations = set(improvements["configuration"])
    if not include_no_p:
        allowed_configurations -= NO_P_CONFIGURATIONS
    ordered_configurations = tuple(
        configuration_order or resolve_configuration_order(improvements["configuration"])
    )
    available_configurations = [
        configuration
        for configuration in ordered_configurations
        if configuration in allowed_configurations
    ]
    y_lookup = {
        configuration: len(available_configurations) - 1 - index
        for index, configuration in enumerate(available_configurations)
    }

    for panel_index, metric_key in enumerate(metric_keys):
        ax = axes_flat[panel_index]
        panel = improvements.loc[
            improvements["metric"].eq(metric_key)
            & improvements["configuration"].isin(available_configurations)
            & improvements["improvement"].notna()
        ].copy()
        spec = METRICS[metric_key]
        if panel.empty:
            ax.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue

        band_min = float(panel["v0_band_min"].iloc[0])
        band_max = float(panel["v0_band_max"].iloc[0])
        ax.axvspan(band_min, band_max, color="#BDBDBD", alpha=0.42, zorder=0)
        ax.axvline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)

        for configuration in available_configurations:
            family = panel.loc[panel["configuration"].eq(configuration)].sort_values("seed")
            if family.empty:
                continue
            y_center = y_lookup[configuration]
            offsets = np.linspace(-0.11, 0.11, len(family)) if len(family) > 1 else [0.0]
            y_values = y_center + np.asarray(offsets)
            ax.plot(
                family["improvement"],
                y_values,
                color="#6F6F6F",
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )
            for (_, point), y_value in zip(family.iterrows(), y_values):
                style = SEED_STYLES.get(int(point["seed"]), SEED_STYLES[1])
                ax.scatter(
                    float(point["improvement"]),
                    float(y_value),
                    s=35,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )

        x_values = list(panel["improvement"]) + [band_min, band_max]
        ax.set_xlim(*_padded_limits(x_values, include_zero=True))
        ax.set_ylim(-0.55, len(available_configurations) - 0.45)
        ax.set_yticks(
            [y_lookup[configuration] for configuration in available_configurations],
            [display_label(configuration) for configuration in available_configurations],
        )
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, zorder=0)
        ax.set_title(spec.label, loc="left", fontsize=10.5, fontweight="semibold")
        unit_suffix = f" [{spec.unit}]" if spec.unit else ""
        ax.set_xlabel(f"Improvement vs V0 seed mean{unit_suffix}")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    for ax in axes_flat[n_metrics:]:
        ax.set_axis_off()

    legend_handles: list[Any] = [
        Line2D(
            [0],
            [0],
            marker=SEED_STYLES[seed]["marker"],
            color="none",
            markerfacecolor=SEED_STYLES[seed]["color"],
            markeredgecolor="white",
            markersize=7,
            label=f"Training seed {seed}",
        )
        for seed in (1, 2, 3)
    ]
    legend_handles.append(
        Patch(facecolor="#BDBDBD", alpha=0.42, label="Observed V0 seed min–max")
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Conditioning effects relative to observed training-seed variability"
        + (" (including no-P models)" if include_no_p else " (no-P models excluded)"),
        fontsize=13,
        fontweight="semibold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Positive values indicate better target-oriented performance; shaded ranges are descriptive, not confidence intervals.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#555555",
    )
    return fig, axes


def _target_value(panel: pd.DataFrame, spec: MetricSpec) -> float:
    targets = pd.to_numeric(panel["target_value"], errors="coerce").dropna()
    if targets.empty:
        raise ValueError(f"No evaluation target is available for {spec.key}.")
    return float(targets.mean())


def plot_absolute_targets(
    improvements: pd.DataFrame,
    metric_keys: Sequence[str] = DEFAULT_METRICS,
    *,
    configuration_order: Sequence[str] | None = None,
    include_no_p: bool = False,
    figsize: tuple[float, float] = (12.2, 9.2),
) -> tuple[plt.Figure, np.ndarray]:
    """Plot raw metric values against their explicit evaluation targets.

    This is intended as a supplementary context figure.  The improvement plots
    remain the primary seed-aware comparison because a distant physical or
    idealized target can visually compress small between-seed differences.
    """

    n_metrics = len(metric_keys)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.19,
        right=0.985,
        bottom=0.09,
        top=0.89,
        hspace=0.42,
        wspace=0.18,
    )
    axes_flat = axes.ravel()

    allowed_configurations = set(improvements["configuration"])
    if not include_no_p:
        allowed_configurations -= NO_P_CONFIGURATIONS
    ordered_configurations = tuple(
        configuration_order or resolve_configuration_order(improvements["configuration"])
    )
    available_configurations = [
        configuration
        for configuration in ordered_configurations
        if configuration in allowed_configurations
    ]
    y_lookup = {
        configuration: len(available_configurations) - 1 - index
        for index, configuration in enumerate(available_configurations)
    }

    for panel_index, metric_key in enumerate(metric_keys):
        ax = axes_flat[panel_index]
        panel = improvements.loc[
            improvements["metric"].eq(metric_key)
            & improvements["configuration"].isin(available_configurations)
            & improvements["value"].notna()
        ].copy()
        spec = METRICS[metric_key]
        if panel.empty:
            ax.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue

        target = _target_value(panel, spec)
        baseline_values = panel.loc[panel["configuration"].eq("V0"), "value"]
        baseline_values = pd.to_numeric(baseline_values, errors="coerce").dropna()
        if not baseline_values.empty:
            ax.axvspan(
                float(baseline_values.min()),
                float(baseline_values.max()),
                color="#BDBDBD",
                alpha=0.42,
                zorder=0,
            )
        ax.axvline(
            target,
            color="#7A0177",
            linewidth=1.35,
            linestyle="-.",
            zorder=1,
        )

        for configuration in available_configurations:
            family = panel.loc[panel["configuration"].eq(configuration)].sort_values("seed")
            if family.empty:
                continue
            y_center = y_lookup[configuration]
            offsets = np.linspace(-0.11, 0.11, len(family)) if len(family) > 1 else [0.0]
            y_values = y_center + np.asarray(offsets)
            ax.plot(
                family["value"],
                y_values,
                color="#6F6F6F",
                linewidth=1.0,
                alpha=0.75,
                zorder=2,
            )
            for (_, point), y_value in zip(family.iterrows(), y_values):
                style = SEED_STYLES.get(int(point["seed"]), SEED_STYLES[1])
                ax.scatter(
                    float(point["value"]),
                    float(y_value),
                    s=35,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolor="white",
                    linewidth=0.45,
                    zorder=3,
                )

        raw_values = list(pd.to_numeric(panel["value"], errors="coerce")) + [target]
        x_limits = _padded_limits(raw_values, include_zero=False)
        ax.set_xlim(*x_limits)
        ax.set_ylim(-0.55, len(available_configurations) - 0.45)
        ax.set_yticks(
            [y_lookup[configuration] for configuration in available_configurations],
            [display_label(configuration) for configuration in available_configurations],
        )
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, zorder=0)
        direction = {
            "zero": "lower is better",
            "one": "higher is better",
            "hr": "closer to HR is better",
        }[spec.target]
        ax.set_title(
            f"{spec.label} ({direction})",
            loc="left",
            fontsize=10.5,
            fontweight="semibold",
        )
        unit_suffix = f" [{spec.unit}]" if spec.unit else ""
        ax.set_xlabel(f"Metric value{unit_suffix}")
        target_fraction = (target - x_limits[0]) / (x_limits[1] - x_limits[0])
        if target_fraction <= 0.5:
            target_text_x = min(target_fraction + 0.015, 0.98)
            target_text_alignment = "left"
        else:
            target_text_x = max(target_fraction - 0.015, 0.02)
            target_text_alignment = "right"
        ax.text(
            target_text_x,
            0.965,
            f"Target = {target:.4g}",
            transform=ax.transAxes,
            ha=target_text_alignment,
            va="top",
            fontsize=8.2,
            color="#7A0177",
        )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    for ax in axes_flat[n_metrics:]:
        ax.set_axis_off()

    legend_handles: list[Any] = [
        Line2D(
            [0],
            [0],
            marker=SEED_STYLES[seed]["marker"],
            color="none",
            markerfacecolor=SEED_STYLES[seed]["color"],
            markeredgecolor="white",
            markersize=7,
            label=f"Training seed {seed}",
        )
        for seed in (1, 2, 3)
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#7A0177",
                linestyle="-.",
                linewidth=1.35,
                label="Evaluation target",
            ),
            Patch(facecolor="#BDBDBD", alpha=0.42, label="Observed V0 value min–max"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Absolute metric values relative to evaluation targets"
        + (" (including no-P models)" if include_no_p else " (no-P models excluded)"),
        fontsize=13,
        fontweight="semibold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "This target-context figure is supplementary; the improvement figures provide the clearer training-seed comparison.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#555555",
    )
    return fig, axes


def plot_seed_sensitivity(
    sensitivity: pd.DataFrame,
    *,
    configuration_order: Sequence[str] | None = None,
    include_no_p: bool,
    figsize: tuple[float, float] = (12.2, 6.8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the metric-wise observed two-seed gaps for each configuration."""

    allowed_configurations = set(sensitivity["configuration"])
    if not include_no_p:
        allowed_configurations -= NO_P_CONFIGURATIONS
    ordered_configurations = tuple(
        configuration_order or resolve_configuration_order(sensitivity["configuration"])
    )
    configurations = [
        configuration
        for configuration in ordered_configurations
        if configuration != "V0" and configuration in allowed_configurations
    ]
    plot_df = sensitivity.loc[
        sensitivity["configuration"].isin(configurations)
        & sensitivity["normalized_seed_gap"].gt(0.0)
    ].copy()
    if plot_df.empty:
        raise ValueError("No positive seed-sensitivity values are available to plot.")

    fig, ax = plt.subplots(figsize=figsize)
    crowded = len(configurations) > 10
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.29 if crowded else 0.18,
        top=0.70,
    )
    x_lookup = {configuration: index for index, configuration in enumerate(configurations)}

    # Light grouping backgrounds support interpretation without encoding a result.
    simple_positions = [
        x_lookup[c]
        for c in configurations
        if c.startswith("C_") or c.startswith("V0_")
    ]
    physics_positions = [
        x_lookup[c]
        for c in configurations
        if c.startswith("D_") and c not in NO_P_CONFIGURATIONS
    ]
    no_p_positions = [x_lookup[c] for c in configurations if c in NO_P_CONFIGURATIONS]
    for positions, color in (
        (simple_positions, "#F5F5F5"),
        (physics_positions, "#EDF7F3"),
        (no_p_positions, "#FCEEEE"),
    ):
        if positions:
            ax.axvspan(min(positions) - 0.48, max(positions) + 0.48, color=color, zorder=0)

    metric_indices = sorted(plot_df["metric_index"].unique())
    if len(metric_indices) == 1:
        offset_lookup = {metric_indices[0]: 0.0}
    else:
        offsets = np.linspace(-0.17, 0.17, len(metric_indices))
        offset_lookup = dict(zip(metric_indices, offsets))

    for _, point in plot_df.iterrows():
        x = x_lookup[str(point["configuration"])] + offset_lookup[int(point["metric_index"])]
        ax.scatter(
            x,
            float(point["normalized_seed_gap"]),
            s=34,
            color=PILLAR_COLORS[str(point["pillar"])],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )

    medians = plot_df.groupby("configuration")["normalized_seed_gap"].median()
    for configuration in configurations:
        if configuration not in medians:
            continue
        x = x_lookup[configuration]
        median = float(medians[configuration])
        ax.plot([x - 0.23, x + 0.23], [median, median], color="black", linewidth=2.2, zorder=4)
        ax.scatter(
            x,
            median,
            s=52,
            marker="D",
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )

    ax.axhline(1.0, color="#4D4D4D", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_yscale("log")
    finite = plot_df["normalized_seed_gap"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.set_ylim(float(finite.min()) / 1.7, float(finite.max()) * 1.7)
    ax.set_xlim(-0.55, len(configurations) - 0.45)
    counts = plot_df.groupby("configuration")["metric"].nunique()
    ax.set_xticks(
        range(len(configurations)),
        [
            f"{display_label(configuration)}\n(n={int(counts.get(configuration, 0))})"
            for configuration in configurations
        ],
    )
    if crowded:
        plt.setp(ax.get_xticklabels(), rotation=52, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Observed two-seed gap / V0 seed range\n(log scale)")
    ax.set_xlabel("Model configuration")
    ax.grid(axis="y", which="both", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    for positions, label in (
        (simple_positions, "Context and single-variable configurations"),
        (physics_positions, "Precipitation-conditioned physics sets"),
        (no_p_positions, "No-P sets"),
    ):
        if positions:
            ax.text(
                float(np.mean(positions)),
                1.035,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="semibold",
            )

    pillar_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=7,
            label=pillar,
        )
        for pillar, color in PILLAR_COLORS.items()
    ]
    summary_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            markerfacecolor="white",
            markersize=6,
            linewidth=2.0,
            label="Configuration median",
        ),
        Line2D(
            [0],
            [0],
            color="#4D4D4D",
            linestyle="--",
            linewidth=1.2,
            label="Gap equals observed V0 range",
        ),
    ]
    fig.legend(
        handles=pillar_handles + summary_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.905),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Observed two-seed sensitivity across evaluation metrics"
        + (" (including no-P models)" if include_no_p else " (no-P models excluded)"),
        y=0.985,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.018,
        "Each point is one predefined metric. Metrics are correlated, medians are descriptive, and two seeds do not estimate configuration-level variance.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#555555",
    )
    return fig, ax


def plot_stability_robustness(
    robustness: pd.DataFrame,
    *,
    include_no_p: bool = True,
    figsize: tuple[float, float] = (10.8, 7.8),
) -> tuple[plt.Figure, plt.Axes]:
    """Compare configuration sensitivity under three summary definitions."""

    plot_df = robustness.copy()
    if not include_no_p:
        plot_df = plot_df.loc[~plot_df["configuration"].isin(NO_P_CONFIGURATIONS)]
    plot_df = plot_df.sort_values("pillar_balanced_rank", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.13, top=0.86)
    score_specs = (
        ("core_metric_median_gap", "Core 21-metric median", "#0072B2", "o"),
        ("expanded_metric_median_gap", "Expanded 31-metric median", "#D55E00", "s"),
        ("pillar_balanced_median_gap", "Expanded equal-pillar median", "#009E73", "D"),
    )
    for row_index, row in plot_df.iterrows():
        values = [float(row[column]) for column, _, _, _ in score_specs]
        ax.plot(values, [y[row_index]] * len(values), color="#A0A0A0", linewidth=1.0, zorder=1)
    for column, label, color, marker in score_specs:
        ax.scatter(
            plot_df[column],
            y,
            label=label,
            color=color,
            marker=marker,
            s=46,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )

    ax.axvline(1.0, color="#4D4D4D", linestyle="--", linewidth=1.1)
    ax.set_yticks(y, [display_label(c) for c in plot_df["configuration"]])
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Median observed seed-1/seed-2 gap / V0 seed range")
    ax.set_title(
        "Sensitivity ranking robustness across metric definitions",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
    )
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.025,
        "Lower is less seed-sensitive. Scores are descriptive; two seeds do not estimate configuration-level variance.",
        ha="center",
        fontsize=8.4,
        color="#555555",
    )
    return fig, ax


def plot_attribution_composition(
    attribution_summary: pd.DataFrame,
    configuration_order: Sequence[str],
    *,
    figsize: tuple[float, float] = (10.8, 7.8),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot agreement of the two seeds across targetable evaluation metrics."""

    order = [
        configuration
        for configuration in configuration_order
        if configuration != "V0"
        and configuration in set(attribution_summary["configuration"])
    ]
    plot_df = attribution_summary.set_index("configuration").loc[order].reset_index()
    y = np.arange(len(plot_df))[::-1]
    categories = (
        ("fraction_both_favourable", "Both seeds favourable", "#009E73"),
        ("fraction_both_inside_v0_range", "Both inside V0 range", "#BDBDBD"),
        ("fraction_mixed", "Seed-dependent/mixed", "#E69F00"),
        ("fraction_both_unfavourable", "Both seeds unfavourable", "#D55E00"),
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.13, top=0.86)
    left = np.zeros(len(plot_df), dtype=float)
    for column, label, color in categories:
        values = plot_df[column].to_numpy(dtype=float)
        ax.barh(y, values, left=left, height=0.66, color=color, label=label)
        left += values

    ax.set_yticks(
        y,
        [
            f"{display_label(configuration)} (n={int(n_metrics)})"
            for configuration, n_metrics in zip(
                plot_df["configuration"], plot_df["n_metrics"]
            )
        ],
    )
    ax.set_xlim(0.0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Fraction of targetable evaluation metrics")
    ax.set_title(
        "Agreement of two training seeds relative to the observed V0 range",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
    )
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    fig.text(
        0.5,
        0.025,
        "Metrics are correlated and pillars contain unequal numbers of metrics; this is a descriptive attribution audit.",
        ha="center",
        fontsize=8.4,
        color="#555555",
    )
    return fig, ax


# %% End-to-end runner and command line ------------------------------------


def run_analysis(
    eval_root: str | Path,
    output_dir: str | Path,
    *,
    scope: str = "all-paired",
    metric_keys: Sequence[str] = DEFAULT_METRICS,
    allow_missing: bool = False,
    dpi: int = 300,
    show: bool = False,
) -> dict[str, Any]:
    """Load, score, export, and plot the AI4CC Figure 2 analysis."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw, inventory, configuration_order = load_analysis_summaries(
        eval_root,
        scope=scope,
        allow_missing=allow_missing,
    )
    improvements = compute_improvements(raw, metric_keys)
    sensitivity = compute_seed_sensitivity(
        raw,
        CORE_STABILITY_METRICS,
        configuration_order=configuration_order,
    )
    sensitivity_summary = summarize_seed_sensitivity(sensitivity)
    expanded_sensitivity = compute_seed_sensitivity(
        raw,
        EXPANDED_STABILITY_METRICS,
        configuration_order=configuration_order,
    )
    core_pillar_summary, core_pillar_balanced = summarize_pillar_balanced_sensitivity(
        sensitivity
    )
    expanded_pillar_summary, expanded_pillar_balanced = (
        summarize_pillar_balanced_sensitivity(expanded_sensitivity)
    )
    leave_one_out = leave_one_pillar_out_sensitivity(expanded_pillar_summary)
    stability_robustness = build_stability_robustness_summary(
        sensitivity,
        expanded_sensitivity,
        expanded_pillar_balanced,
        leave_one_out,
    )
    stability_group_summary = summarize_stability_by_group(stability_robustness)
    robustness_improvements = compute_improvements(
        raw,
        ROBUSTNESS_ATTRIBUTION_METRICS,
    )
    metric_to_pillar = {
        metric_key: pillar
        for metric_key, _, pillar in EXPANDED_STABILITY_METRICS
        if metric_key in ROBUSTNESS_ATTRIBUTION_METRICS
    }
    attribution_details, attribution_summary = summarize_attribution_classes(
        robustness_improvements,
        metric_to_pillar,
    )
    metric_coverage = build_metric_coverage_audit(raw)
    full_count = len(configuration_order)
    focus_count = len([c for c in configuration_order if c not in NO_P_CONFIGURATIONS])
    full_panel_size = (15.5, max(9.2, 2.6 + 0.66 * full_count))
    focus_panel_size = (15.5, max(9.2, 2.6 + 0.66 * focus_count))
    sensitivity_width = max(12.2, 0.76 * max(full_count - 1, 1))
    full_figure, full_axes = plot_figure2(
        improvements,
        metric_keys,
        configuration_order=configuration_order,
        include_no_p=True,
        figsize=full_panel_size,
    )
    focus_figure, focus_axes = plot_figure2(
        improvements,
        metric_keys,
        configuration_order=configuration_order,
        include_no_p=False,
        figsize=focus_panel_size,
    )
    target_figure, target_axes = plot_absolute_targets(
        improvements,
        metric_keys,
        configuration_order=configuration_order,
        include_no_p=False,
        figsize=focus_panel_size,
    )
    full_sensitivity_figure, full_sensitivity_axes = plot_seed_sensitivity(
        sensitivity,
        configuration_order=configuration_order,
        include_no_p=True,
        figsize=(sensitivity_width, 7.6),
    )
    focus_sensitivity_figure, focus_sensitivity_axes = plot_seed_sensitivity(
        sensitivity,
        configuration_order=configuration_order,
        include_no_p=False,
        figsize=(max(12.2, sensitivity_width - 1.0), 7.6),
    )
    robustness_figure, robustness_axes = plot_stability_robustness(
        stability_robustness,
        include_no_p=True,
        figsize=(11.2, max(7.8, 3.0 + 0.34 * (full_count - 1))),
    )
    attribution_figure, attribution_axes = plot_attribution_composition(
        attribution_summary,
        configuration_order,
        figsize=(11.2, max(7.8, 3.0 + 0.34 * (full_count - 1))),
    )

    scope_tag = "all_models" if scope == "all-paired" else "curated"
    prefix = f"ai4cc_figure2_{scope_tag}"
    inventory_path = output / f"{prefix}_configuration_inventory.csv"
    raw_path = output / f"{prefix}_selected_metrics.csv"
    improvement_path = output / f"{prefix}_improvements.csv"
    sensitivity_path = output / f"{prefix}_seed_sensitivity.csv"
    sensitivity_summary_path = output / f"{prefix}_seed_sensitivity_summary.csv"
    expanded_sensitivity_path = output / f"{prefix}_expanded_seed_sensitivity.csv"
    core_pillar_summary_path = output / f"{prefix}_core_pillar_sensitivity.csv"
    core_pillar_balanced_path = output / f"{prefix}_core_pillar_balanced_summary.csv"
    expanded_pillar_summary_path = output / f"{prefix}_expanded_pillar_sensitivity.csv"
    expanded_pillar_balanced_path = output / f"{prefix}_expanded_pillar_balanced_summary.csv"
    leave_one_out_path = output / f"{prefix}_leave_one_pillar_out.csv"
    stability_robustness_path = output / f"{prefix}_stability_robustness_summary.csv"
    stability_group_summary_path = output / f"{prefix}_stability_group_summary.csv"
    robustness_improvements_path = output / f"{prefix}_robustness_improvements.csv"
    attribution_details_path = output / f"{prefix}_attribution_seed_classifications.csv"
    attribution_summary_path = output / f"{prefix}_attribution_pair_summary.csv"
    metric_coverage_path = output / f"{prefix}_metric_coverage_audit.csv"
    full_png_path = output / f"{prefix}_improvements_with_no_p.png"
    full_pdf_path = output / f"{prefix}_improvements_with_no_p.pdf"
    focus_png_path = output / f"{prefix}_improvements_without_no_p.png"
    focus_pdf_path = output / f"{prefix}_improvements_without_no_p.pdf"
    target_png_path = output / f"{prefix}_absolute_targets_without_no_p.png"
    target_pdf_path = output / f"{prefix}_absolute_targets_without_no_p.pdf"
    full_sensitivity_png_path = output / f"{prefix}_seed_sensitivity_with_no_p.png"
    full_sensitivity_pdf_path = output / f"{prefix}_seed_sensitivity_with_no_p.pdf"
    focus_sensitivity_png_path = output / f"{prefix}_seed_sensitivity_without_no_p.png"
    focus_sensitivity_pdf_path = output / f"{prefix}_seed_sensitivity_without_no_p.pdf"
    robustness_png_path = output / f"{prefix}_stability_robustness.png"
    robustness_pdf_path = output / f"{prefix}_stability_robustness.pdf"
    attribution_png_path = output / f"{prefix}_attribution_composition.png"
    attribution_pdf_path = output / f"{prefix}_attribution_composition.pdf"

    requested_columns = [
        "run_id",
        "summary_run_id",
        "run_label",
        "configuration",
        "seed",
        "summary_path",
        *dict.fromkeys(
            key
            for metric_key in metric_keys
            for key in (metric_key, METRICS[metric_key].reference_key)
            if key is not None
        ),
    ]
    selected_columns = [column for column in requested_columns if column in raw.columns]
    inventory.to_csv(inventory_path, index=False)
    raw.loc[:, selected_columns].to_csv(raw_path, index=False)
    improvements.to_csv(improvement_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    sensitivity_summary.to_csv(sensitivity_summary_path, index=False)
    expanded_sensitivity.to_csv(expanded_sensitivity_path, index=False)
    core_pillar_summary.to_csv(core_pillar_summary_path, index=False)
    core_pillar_balanced.to_csv(core_pillar_balanced_path, index=False)
    expanded_pillar_summary.to_csv(expanded_pillar_summary_path, index=False)
    expanded_pillar_balanced.to_csv(expanded_pillar_balanced_path, index=False)
    leave_one_out.to_csv(leave_one_out_path, index=False)
    stability_robustness.to_csv(stability_robustness_path, index=False)
    stability_group_summary.to_csv(stability_group_summary_path, index=False)
    robustness_improvements.to_csv(robustness_improvements_path, index=False)
    attribution_details.to_csv(attribution_details_path, index=False)
    attribution_summary.to_csv(attribution_summary_path, index=False)
    metric_coverage.to_csv(metric_coverage_path, index=False)
    full_figure.savefig(full_png_path, dpi=dpi, bbox_inches="tight")
    full_figure.savefig(full_pdf_path, bbox_inches="tight")
    focus_figure.savefig(focus_png_path, dpi=dpi, bbox_inches="tight")
    focus_figure.savefig(focus_pdf_path, bbox_inches="tight")
    target_figure.savefig(target_png_path, dpi=dpi, bbox_inches="tight")
    target_figure.savefig(target_pdf_path, bbox_inches="tight")
    full_sensitivity_figure.savefig(full_sensitivity_png_path, dpi=dpi, bbox_inches="tight")
    full_sensitivity_figure.savefig(full_sensitivity_pdf_path, bbox_inches="tight")
    focus_sensitivity_figure.savefig(focus_sensitivity_png_path, dpi=dpi, bbox_inches="tight")
    focus_sensitivity_figure.savefig(focus_sensitivity_pdf_path, bbox_inches="tight")
    robustness_figure.savefig(robustness_png_path, dpi=dpi, bbox_inches="tight")
    robustness_figure.savefig(robustness_pdf_path, bbox_inches="tight")
    attribution_figure.savefig(attribution_png_path, dpi=dpi, bbox_inches="tight")
    attribution_figure.savefig(attribution_pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(full_figure)
        plt.close(focus_figure)
        plt.close(target_figure)
        plt.close(full_sensitivity_figure)
        plt.close(focus_sensitivity_figure)
        plt.close(robustness_figure)
        plt.close(attribution_figure)

    return {
        "raw": raw,
        "configuration_inventory": inventory,
        "configuration_order": configuration_order,
        "scope": scope,
        "improvements": improvements,
        # The focused figure is the recommended main-paper version.
        "figure": focus_figure,
        "axes": focus_axes,
        "full_figure": full_figure,
        "full_axes": full_axes,
        "target_figure": target_figure,
        "target_axes": target_axes,
        "seed_sensitivity": sensitivity,
        "seed_sensitivity_summary": sensitivity_summary,
        "expanded_seed_sensitivity": expanded_sensitivity,
        "core_pillar_sensitivity": core_pillar_summary,
        "core_pillar_balanced_summary": core_pillar_balanced,
        "expanded_pillar_sensitivity": expanded_pillar_summary,
        "expanded_pillar_balanced_summary": expanded_pillar_balanced,
        "leave_one_pillar_out": leave_one_out,
        "stability_robustness_summary": stability_robustness,
        "stability_group_summary": stability_group_summary,
        "robustness_improvements": robustness_improvements,
        "attribution_seed_classifications": attribution_details,
        "attribution_pair_summary": attribution_summary,
        "metric_coverage_audit": metric_coverage,
        "full_sensitivity_figure": full_sensitivity_figure,
        "full_sensitivity_axes": full_sensitivity_axes,
        "focus_sensitivity_figure": focus_sensitivity_figure,
        "focus_sensitivity_axes": focus_sensitivity_axes,
        "robustness_figure": robustness_figure,
        "robustness_axes": robustness_axes,
        "attribution_figure": attribution_figure,
        "attribution_axes": attribution_axes,
        "raw_csv": raw_path,
        "configuration_inventory_csv": inventory_path,
        "improvements_csv": improvement_path,
        "seed_sensitivity_csv": sensitivity_path,
        "seed_sensitivity_summary_csv": sensitivity_summary_path,
        "expanded_seed_sensitivity_csv": expanded_sensitivity_path,
        "core_pillar_sensitivity_csv": core_pillar_summary_path,
        "core_pillar_balanced_summary_csv": core_pillar_balanced_path,
        "expanded_pillar_sensitivity_csv": expanded_pillar_summary_path,
        "expanded_pillar_balanced_summary_csv": expanded_pillar_balanced_path,
        "leave_one_pillar_out_csv": leave_one_out_path,
        "stability_robustness_summary_csv": stability_robustness_path,
        "stability_group_summary_csv": stability_group_summary_path,
        "robustness_improvements_csv": robustness_improvements_path,
        "attribution_seed_classifications_csv": attribution_details_path,
        "attribution_pair_summary_csv": attribution_summary_path,
        "metric_coverage_audit_csv": metric_coverage_path,
        "png": focus_png_path,
        "pdf": focus_pdf_path,
        "improvements_with_no_p_png": full_png_path,
        "improvements_with_no_p_pdf": full_pdf_path,
        "improvements_without_no_p_png": focus_png_path,
        "improvements_without_no_p_pdf": focus_pdf_path,
        "absolute_targets_png": target_png_path,
        "absolute_targets_pdf": target_pdf_path,
        "seed_sensitivity_with_no_p_png": full_sensitivity_png_path,
        "seed_sensitivity_with_no_p_pdf": full_sensitivity_pdf_path,
        "seed_sensitivity_without_no_p_png": focus_sensitivity_png_path,
        "seed_sensitivity_without_no_p_pdf": focus_sensitivity_pdf_path,
        "stability_robustness_png": robustness_png_path,
        "stability_robustness_pdf": robustness_pdf_path,
        "attribution_composition_png": attribution_png_path,
        "attribution_composition_pdf": attribution_pdf_path,
        "selection_report": selection_report(raw, scope=scope, inventory=inventory),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the seed-aware AI4CC Figure 2 from evaluation summaries."
    )
    parser.add_argument(
        "--eval-root",
        required=True,
        help="Directory containing copied evaluation_summary.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="ai4cc_figure2_output",
        help="Directory for the figure and audit CSV files.",
    )
    parser.add_argument(
        "--scope",
        choices=ANALYSIS_SCOPES,
        default="all-paired",
        help=(
            "Use every automatically discovered complete seed pair (default), "
            "or reproduce the original curated comparison."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=tuple(METRICS),
        default=list(DEFAULT_METRICS),
        help="Metric keys to plot, in panel order.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue with a partial set of the expected 19 runs.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        args.eval_root,
        args.output_dir,
        scope=args.scope,
        metric_keys=args.metrics,
        allow_missing=args.allow_missing,
        dpi=args.dpi,
        show=args.show,
    )
    print(result["selection_report"])
    print(f"Configuration inventory: {result['configuration_inventory_csv']}")
    print(f"Improvements including no-P: {result['improvements_with_no_p_png']}")
    print(f"Improvements excluding no-P: {result['improvements_without_no_p_png']}")
    print(f"Absolute values and targets: {result['absolute_targets_png']}")
    print(f"Seed sensitivity including no-P: {result['seed_sensitivity_with_no_p_png']}")
    print(f"Seed sensitivity excluding no-P: {result['seed_sensitivity_without_no_p_png']}")
    print(f"Seed-sensitivity summary: {result['seed_sensitivity_summary_csv']}")
    print(f"Robustness ranking summary: {result['stability_robustness_summary_csv']}")
    print(f"Pillar-balanced summary: {result['expanded_pillar_balanced_summary_csv']}")
    print(f"Leave-one-pillar-out audit: {result['leave_one_pillar_out_csv']}")
    print(f"Attribution pair summary: {result['attribution_pair_summary_csv']}")
    print(f"Metric coverage audit: {result['metric_coverage_audit_csv']}")
    print(f"Stability robustness figure: {result['stability_robustness_png']}")
    print(f"Attribution composition figure: {result['attribution_composition_png']}")
    print("PDF versions are saved alongside the PNG files.")
    print(f"Audit table: {result['improvements_csv']}")


if __name__ == "__main__":
    main()
