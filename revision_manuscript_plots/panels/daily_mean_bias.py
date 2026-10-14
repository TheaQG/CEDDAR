from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sbgm.variable_utils import get_color_for_model


METHOD_ORDER = (
    "bilinear_era5",
    "qm",
    "ceddar_mean",
    "ceddar_median",
    "ceddar_pmm",
)

METHOD_LABELS = {
    "bilinear_era5": "Bilinear ERA5",
    "qm": "QM",
    "ceddar_mean": "CEDDAR mean",
    "ceddar_median": "CEDDAR median",
    "ceddar_pmm": "CEDDAR PMM",
}


@dataclass
class DailyMeanBiasData:
    daily: pd.DataFrame
    summary: pd.DataFrame
    year: int
    evaluation_days: np.ndarray


def load_daily_mean_bias(revision_dir: str | Path,) -> DailyMeanBiasData:
    """
    Load the archived daily domain-mean precipitation-bias product.
    """
    revision_dir = Path(revision_dir).expanduser()

    daily_path = (revision_dir / "daily_mean_bias.csv")
    summary_path = (revision_dir / "daily_mean_bias_summary.csv")

    if not daily_path.is_file():
        raise FileNotFoundError(daily_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    daily = pd.read_csv(daily_path)
    summary = pd.read_csv(summary_path)

    required_daily = {
        "date",
        "year",
        "evaluation_day",
        "season",
        "method",
        "danra_mean_mm_day",
        "model_mean_mm_day",
        "bias_mm_day",
        "n_land_pixels",
    }

    required_summary = {
        "method",
        "n_days",
        "mean_bias_mm_day",
        "median_bias_mm_day",
        "std_bias_mm_day",
        "mae_mm_day",
        "rmse_mm_day",
        "danra_accumulation_mm",
        "model_accumulation_mm",
        "accumulation_bias_mm",
        "relative_accumulation_bias_pct",
    }

    missing_daily = (required_daily - set(daily.columns))
    missing_summary = (required_summary - set(summary.columns))

    if missing_daily:
        raise ValueError(f"{daily_path.name} missing columns: {sorted(missing_daily)}")
    if missing_summary:
        raise ValueError(f"{summary_path.name} missing columns: {sorted(missing_summary)}")

    for column in ("year", "evaluation_day", "danra_mean_mm_day", "model_mean_mm_day", "bias_mm_day", "n_land_pixels",):
        daily[column] = pd.to_numeric(daily[column], errors="coerce",)

    years = (daily["year"].dropna().astype(int).unique())

    if len(years) != 1:
        raise ValueError(f"Expected exactly one evaluation year, found {sorted(years)}")

    year = int(years[0])

    missing_methods = (set(METHOD_ORDER) - set(daily["method"].astype(str).unique()))

    if missing_methods:
        raise ValueError(f"Daily bias table missing methods: {sorted(missing_methods)}")

    # ---------------------------------------------------------
    # Validate exact paired daily coverage across all methods.
    # ---------------------------------------------------------

    date_sets = []

    for method in METHOD_ORDER:
        sub = daily.loc[daily["method"] == method]

        if sub["date"].duplicated().any():
            raise ValueError(f"{method}: duplicated dates in daily bias table.")

        date_sets.append(set(sub["date"].astype(str)))

    first = date_sets[0]

    if any(dates != first for dates in date_sets[1:]):
        raise ValueError("Methods do not contain identical date sets.")

    # DANRA reference should be identical across method rows.
    reference_counts = (daily.groupby("date")["danra_mean_mm_day"].nunique(dropna=False))

    if (reference_counts > 1).any():
        raise ValueError("DANRA reference differs across method rows for at least one date.")
    if not np.all(np.isfinite(daily["bias_mm_day"].to_numpy(float))):
        raise ValueError("Non-finite values in daily bias table.")

    evaluation_days = np.sort(daily["evaluation_day"].dropna().unique().astype(int))
    expected_days = np.arange(1, len(first) + 1,)

    if not np.array_equal(evaluation_days, expected_days,):
        raise ValueError("evaluation_day is not a contiguous 1-based sequence.")

    return DailyMeanBiasData(
        daily=daily,
        summary=summary,
        year=year,
        evaluation_days=(evaluation_days),
    )


def _method_style(method: str,) -> dict:
    """
    Method styling consistent with the other revised manuscript figures.
    """
    if method == "bilinear_era5":
        return dict(color=get_color_for_model("LR"), ls="--",)
    if method == "qm":
        return dict(color=get_color_for_model("QM"), ls="--",)
    if method == "ceddar_mean":
        return dict(color=get_color_for_model("ensemble"), ls="-",)
    if method == "ceddar_median":
        return dict(color="0.45", ls="-",)
    if method == "ceddar_pmm":
        return dict(color=get_color_for_model("PMM"), ls="-",)

    raise KeyError(method)


def _rolling_mean(values: np.ndarray, *, window: int,) -> np.ndarray:
    """
    Trailing rolling mean.

    Returns an array with the same length as the input and NaN for
    the first window-1 entries.
    """
    if window < 1:
        raise ValueError("Rolling window must be >= 1.")

    return (pd.Series(np.asarray(values, dtype=float,)).rolling(window=window, min_periods=window,).mean().to_numpy())


def plot_daily_mean_bias(ax, data: DailyMeanBiasData, *, rolling_window: int = 14,):
    """
    Plot daily Danish-land domain-mean precipitation bias.

    Faint lines show individual daily biases. Thick lines show a
    trailing rolling mean over `rolling_window` evaluation days.
    """
    all_biases = []

    for method in METHOD_ORDER:
        sub = (data.daily.loc[data.daily["method"] == method].sort_values("evaluation_day"))

        x = sub["evaluation_day"].to_numpy(dtype=float)
        bias = sub["bias_mm_day"].to_numpy(dtype=float)
        
        all_biases.append(bias)

        style = _method_style(method)

        # Daily variability.
        ax.plot(
            x,
            bias,
            color=style["color"],
            lw=0.65,
            alpha=0.16,
            ls="-",
            zorder=1,
        )

        # Smoothed systematic component.
        smooth = _rolling_mean(bias, window=rolling_window,)

        ax.plot(
            x,
            smooth,
            color=style["color"],
            ls=style["ls"],
            lw=1.8,
            alpha=0.95,
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.axhline(
        0.0,
        color="0.35",
        ls=":",
        lw=1.1,
        zorder=0,
    )

    all_bias = np.concatenate(all_biases)

    max_abs = float(np.nanmax(np.abs(all_bias)))

    # Symmetric limits make wet and dry departures directly comparable.
    limit = max(2.0, np.ceil(max_abs * 1.05),)

    # ax.set_ylim(-limit, limit,)
    ax.set_ylim(-7.5, 10,)
    ax.set_xlim(data.evaluation_days[0], data.evaluation_days[-1],)
    
    ax.set_xlabel(f"{data.year} evaluation day", fontsize=10.5,)
    ax.set_ylabel(r"Model $-$ DANRA (mm day$^{-1}$)", fontsize=10.5,)

    ax.set_title("Daily domain-mean precipitation bias", fontsize=12,)
    ax.grid(True, axis="y", ls=":", alpha=0.22,)
    ax.tick_params(axis="both", labelsize=9.5,)

    ax.spines[["top", "right"]].set_visible(False)

    return ax
