from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sbgm.variable_utils import get_color_for_model


SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")

SUBSET_LABELS = {
    "all_land": "All land pixels",
    "observed_wet": "Observed-wet land pixels",
}


@dataclass
class SeasonalCRPSData:
    daily: pd.DataFrame
    summary: pd.DataFrame


def load_seasonal_crps(revision_dir: str | Path,) -> SeasonalCRPSData:
    """
    Load and validate the revised daily and summary CRPS products.

    Expected files
    --------------
    <revision_dir>/crps.csv
        Daily CRPS values with columns:
        date, season, subset, crps, crps_sum, n_pixels

    <revision_dir>/crps_summary.csv
        Seasonal summary statistics with columns:
        season, subset, metric, mean, median, p25, p75,
        n_dates, n_valid_dates, n_pixel_days
    """
    revision_dir = Path(revision_dir).expanduser()

    daily_path = revision_dir / "crps.csv"
    summary_path = revision_dir / "crps_summary.csv"

    if not daily_path.is_file():
        raise FileNotFoundError(daily_path)

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)

    daily = pd.read_csv(daily_path)
    summary = pd.read_csv(summary_path)

    daily_required = {
        "date",
        "season",
        "subset",
        "crps",
        "crps_sum",
        "n_pixels",
    }

    summary_required = {
        "season",
        "subset",
        "metric",
        "mean",
        "median",
        "p25",
        "p75",
        "n_dates",
        "n_valid_dates",
        "n_pixel_days",
    }

    missing_daily = daily_required - set(daily.columns)
    missing_summary = summary_required - set(summary.columns)

    if missing_daily:
        raise ValueError(f"{daily_path.name} missing columns: {sorted(missing_daily)}")

    if missing_summary:
        raise ValueError(f"{summary_path.name} missing columns: {sorted(missing_summary)}")

    daily = daily.copy()
    summary = summary.copy()

    daily["date"] = daily["date"].astype(str)
    daily["season"] = daily["season"].astype(str)
    daily["subset"] = daily["subset"].astype(str)
    daily["crps"] = pd.to_numeric(daily["crps"], errors="coerce",)
    daily["crps_sum"] = pd.to_numeric(daily["crps_sum"], errors="coerce",)
    daily["n_pixels"] = pd.to_numeric(daily["n_pixels"], errors="coerce",)

    for column in ("mean", "median", "p25", "p75", "n_dates", "n_valid_dates", "n_pixel_days",):
        summary[column] = pd.to_numeric(summary[column], errors="coerce",)

    # ---------------------------------------------------------
    # Validate expected subsets and seasons
    # ---------------------------------------------------------

    expected_subsets = set(SUBSET_LABELS)

    missing_subsets = (expected_subsets - set(daily["subset"].unique()))

    if missing_subsets:
        raise ValueError(f"Daily CRPS file missing subsets: {sorted(missing_subsets)}")

    missing_seasons = (set(SEASON_ORDER) - set(daily["season"].unique()))

    if missing_seasons:
        raise ValueError(f"Daily CRPS file missing seasons: {sorted(missing_seasons)}")

    # ---------------------------------------------------------
    # Validate daily values against the saved summary
    # ---------------------------------------------------------

    seasonal_summary = summary.loc[summary["season"].isin(SEASON_ORDER) & summary["subset"].isin(expected_subsets) & (summary["metric"] == "crps")].copy()

    for subset in SUBSET_LABELS:
        for season in SEASON_ORDER:
            d = daily.loc[(daily["subset"] == subset) & (daily["season"] == season)]
            s = seasonal_summary.loc[(seasonal_summary["subset"] == subset) & (seasonal_summary["season"] == season)]

            if len(s) != 1:
                raise ValueError(f"Expected exactly one summary row for subset={subset}, season={season}; found {len(s)}")

            s = s.iloc[0]

            values = (d["crps"].to_numpy(dtype=float))
            finite = values[np.isfinite(values)]

            if len(d) != int(s["n_dates"]):
                raise ValueError(f"n_dates mismatch for {subset}/{season}: daily={len(d)}, summary={int(s['n_dates'])}")
            if len(finite) != int(s["n_valid_dates"]):
                raise ValueError(f"n_valid_dates mismatch for {subset}/{season}: daily={len(finite)}, summary={int(s['n_valid_dates'])}")
            if len(finite) == 0:
                raise ValueError(f"No finite CRPS values for {subset}/{season}")

            checks = {
                "mean": np.mean(finite),
                "median": np.median(finite),
                "p25": np.quantile(finite, 0.25),
                "p75": np.quantile(finite, 0.75),
            }

            for key, calculated in checks.items():
                saved = float(s[key])

                if not np.isclose(calculated, saved, rtol=1e-9, atol=1e-10,):
                    raise ValueError(f"{key} mismatch for {subset}/{season}: daily={calculated}, summary={saved}")

    return SeasonalCRPSData(daily=daily, summary=seasonal_summary,)


def _season_values(data: SeasonalCRPSData, *, subset: str,) -> list[np.ndarray]:
    values = []

    for season in SEASON_ORDER:
        x = (data.daily.loc[(data.daily["subset"] == subset) & (data.daily["season"] == season),"crps",].to_numpy(dtype=float))
        x = x[np.isfinite(x)]
        values.append(x)

    return values


def _season_counts(data: SeasonalCRPSData, *, subset: str,) -> list[int]:
    counts = []

    for season in SEASON_ORDER:
        row = data.summary.loc[(data.summary["subset"] == subset) & (data.summary["season"] == season)]

        if len(row) != 1:
            raise ValueError(f"Could not identify summary count for {subset}/{season}")

        counts.append(int(row.iloc[0]["n_valid_dates"]))

    return counts


def plot_seasonal_crps(
    ax,
    data: SeasonalCRPSData,
    *,
    subset: str,
    show_ylabel: bool = True,
):
    """
    Plot seasonal daily-CRPS distributions for one spatial subset.

    Boxes show the interquartile range, the horizontal line is the
    median, triangles show the mean, and points beyond 1.5 IQR are
    retained as outliers.
    """
    if subset not in SUBSET_LABELS:
        raise KeyError(f"Unknown CRPS subset: {subset}")

    values = _season_values(data, subset=subset,)
    counts = _season_counts(data, subset=subset,)

    positions = np.arange(1, len(SEASON_ORDER) + 1,)
    color = get_color_for_model("ensemble")

    ax.boxplot(
        values,
        positions=positions,
        widths=0.55,
        whis=1.5,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 1.5,},
        medianprops={"color": "0.15", "linewidth": 1.6,},
        whiskerprops={"color": color, "linewidth": 1.2,},
        capprops={"color": color, "linewidth": 1.2,},
        flierprops={
            "marker": "o",
            "markersize": 3.2,
            "markerfacecolor": "none",
            "markeredgecolor": "0.45",
            "markeredgewidth": 0.8,
            "alpha": 0.65,
        },
    )

    means = np.array([np.mean(x) for x in values], dtype=float,)

    ax.scatter(
        positions,
        means,
        marker="^",
        s=48,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )

    tick_labels = [f"{season}\n($N$ = {n})" for season, n in zip(SEASON_ORDER, counts)]
    
    ax.set_xticks(positions, tick_labels)
    ax.set_title(SUBSET_LABELS[subset], fontsize=12,)
    if show_ylabel:
        ax.set_ylabel(r"CRPS (mm day$^{-1}$)", fontsize=10.5,)

    # Keep the full observed range, including difficult-day outliers.
    finite_all = np.concatenate(values)

    ymax = float(np.ceil(np.nanmax(finite_all) * 1.05))
    ymax = max(ymax, 1.0,)

    ax.set_ylim(0.0, ymax,)
    ax.grid(True, axis="y", ls=":", alpha=0.25)
    ax.tick_params(axis="both", labelsize=9.5,)
    ax.spines[["top", "right"]].set_visible(False)

    return {
        "means": means,
        "counts": np.asarray(counts, dtype=int,),
        "ymax": ymax,
    }
