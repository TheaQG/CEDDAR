from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sbgm.variable_utils import get_color_for_model


METHOD_ORDER = (
    "ceddar_members",
    "ceddar_pmm",
    "bilinear_era5",
    "qm",
)

METHOD_LABELS = {
    "ceddar_members": "CEDDAR members",
    "ceddar_pmm": "CEDDAR PMM",
    "bilinear_era5": "Bilinear ERA5",
    "qm": "QM",
}


@dataclass
class ISSData:
    summary: pd.DataFrame
    thresholds: np.ndarray
    scales: np.ndarray
    common_n: dict[float, int]


def load_iss_data(
    revision_dir: str | Path,
) -> ISSData:
    """
    Load revision ISS results evaluated on the common,
    DANRA-defined observed-event sample.

    Expected files
    --------------
    iss_summary.csv

    with columns:
        method
        threshold_mm
        scale_km
        n_days
        iss_mean
        iss_std
        iss_sem
    """
    revision_dir = Path(revision_dir).expanduser()

    summary_path = (revision_dir / "iss_summary.csv")
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = pd.read_csv(summary_path)

    required = {
        "method",
        "threshold_mm",
        "scale_km",
        "n_days",
        "iss_mean",
        "iss_std",
        "iss_sem",
    }

    missing = (required - set(summary.columns))
    if missing:
        raise ValueError(f"{summary_path.name} missing columns: {sorted(missing)}")

    summary = summary.copy()

    for column in ("threshold_mm", "scale_km", "n_days", "iss_mean", "iss_std", "iss_sem",):
        summary[column] = pd.to_numeric(summary[column], errors="coerce",)

    # ---------------------------------------------------------
    # Validate methods
    # ---------------------------------------------------------

    methods = set(summary["method"])
    missing_methods = (set(METHOD_ORDER) - methods)

    if missing_methods:
        raise ValueError(f"ISS summary is missing methods: {sorted(missing_methods)}")

    # ---------------------------------------------------------
    # Available thresholds/scales
    # ---------------------------------------------------------

    thresholds = np.sort(summary["threshold_mm"].dropna().unique().astype(float))
    scales = np.sort(summary["scale_km"].dropna().unique().astype(float))

    # ---------------------------------------------------------
    # Verify identical N across methods/scales for each threshold
    # ---------------------------------------------------------

    common_n: dict[float, int,] = {}

    for threshold in thresholds:
        sub = summary.loc[np.isclose(summary["threshold_mm"], threshold, atol=1e-8, rtol=0,)]

        n_values = (sub["n_days"].dropna().astype(int).unique())
        if len(n_values) != 1:
            raise ValueError(f"ISS comparison does not use a single common sample size for threshold {threshold:g}: {sorted(n_values)}")

        common_n[float(threshold)] = int(n_values[0])

    return ISSData(
        summary=summary,
        thresholds=thresholds,
        scales=scales,
        common_n=common_n,
    )


def _method_style(method: str,) -> dict:
    """
    Match method styling used in the manuscript figures.
    """
    if method == "ceddar_members":
        return dict(
            color=get_color_for_model("ensemble"),
            marker="o",
            ls="-",
            lw=1.7,
            ms=4.5,
        )

    if method == "ceddar_pmm":
        return dict(
            color=get_color_for_model("PMM"),
            marker="s",
            ls="-",
            lw=1.7,
            ms=4.5,
        )

    if method == "bilinear_era5":
        return dict(
            color=get_color_for_model("LR"),
            marker="^",
            ls="--",
            lw=1.5,
            ms=4.5,
        )

    if method == "qm":
        return dict(
            color=get_color_for_model("QM"),
            marker="D",
            ls="--",
            lw=1.5,
            ms=4.2,
        )

    raise KeyError(method)


def plot_iss_threshold(
    ax,
    data: ISSData,
    threshold: float,
    *,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    """
    Plot mean ISS as a function of neighbourhood scale for one precipitation threshold.
    """
    sub = data.summary.loc[np.isclose(data.summary["threshold_mm"], threshold, atol=1e-8, rtol=0,)]

    for method in METHOD_ORDER:
        m = (sub.loc[sub["method"] == method].sort_values("scale_km"))

        if m.empty:
            continue

        ax.plot(
            m["scale_km"],
            m["iss_mean"],
            label=METHOD_LABELS[method],
            **_method_style(method),
        )

    # Metric uses strict exceedance (> threshold).
    ax.set_title(rf"Threshold $>$ {threshold:g} mm day$^{{-1}}$", fontsize=12,)

    if show_xlabel:
        ax.set_xlabel("Neighbourhood scale (km)", fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("Intensity-scale skill (ISS)", fontsize=10.5)

    ax.set_xticks(data.scales)

    # ISS = 1 is perfect; ISS = 0 is the
    # random-reference level.
    ax.set_ylim(0.35, 0.9,)
    ax.grid(True, ls=":", alpha=0.25,)
    ax.tick_params(axis="both", labelsize=9.5,)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n = data.common_n[float(threshold)]

    ax.text(
        0.98,
        0.06,
        f"N = {n} days",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="0.35",
    )