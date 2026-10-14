from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sbgm.variable_utils import get_color_for_model


@dataclass
class SigmaStarMetricsData:
    summary: pd.DataFrame
    daily: pd.DataFrame
    meta: dict


METRIC_SPECS = {
    "r_lp": {"title": "Large-scale coherence", "ylabel": "Low-pass correlation\n(PMM, ERA5)",},
    "slope_gen": {"title": "Mesoscale PSD slope", "ylabel": r"PSD slope (5--20 km)",},
    "crps": {"title": "Probabilistic skill", "ylabel": r"CRPS (mm day$^{-1}$)",},
}


def _resolve_paths(eval_dir: str | Path):
    """
    Accept either

        <sigma-run>/
            tables/
            sigma_control_meta.json

    or the tables/ directory itself.
    """
    path = Path(eval_dir).expanduser()

    if (path / "tables").is_dir():
        root = path
        tables = path / "tables"
    elif path.name == "tables":
        root = path.parent
        tables = path
    elif ((path / "agg_summary.csv").is_file() and (path / "metrics_by_sigma.csv").is_file()):
        root = path.parent
        tables = path
    else:
        raise FileNotFoundError(f"Could not identify sigma* evaluation tables below {path}")

    return root, tables


def load_sigma_star_metrics(eval_dir: str | Path,) -> SigmaStarMetricsData:
    root, tables = _resolve_paths(eval_dir)

    summary_path = tables / "agg_summary.csv"
    daily_path = tables / "metrics_by_sigma.csv"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not daily_path.is_file():
        raise FileNotFoundError(daily_path)

    summary = pd.read_csv(summary_path)
    daily = pd.read_csv(daily_path)

    required_daily = {
        "date",
        "sigma_star",
        "r_lp",
        "slope_gen",
        "slope_hr",
        "crps",
    }

    missing = required_daily - set(daily.columns)
    if missing:
        raise ValueError("metrics_by_sigma.csv is missing required columns: " + ", ".join(sorted(missing)))

    summary = (summary.sort_values("sigma_star").reset_index(drop=True))
    daily = (daily.sort_values(["sigma_star", "date"]).reset_index(drop=True))

    # Check that daily and aggregated products describe the same sweep.
    sigma_summary = np.sort(summary["sigma_star"].dropna().unique().astype(float))
    sigma_daily = np.sort(daily["sigma_star"].dropna().unique().astype(float))

    if (sigma_summary.shape != sigma_daily.shape or not np.allclose(sigma_summary, sigma_daily, atol=1e-8, rtol=0,)):
        raise ValueError("sigma* grid differs between agg_summary.csv and metrics_by_sigma.csv")

    meta = {}
    for name in ("sigma_control_meta.json", "sigma_control.json",):
        p = root / name
        if p.is_file():
            meta = json.loads(p.read_text())
            break

    return SigmaStarMetricsData(summary=summary, daily=daily, meta=meta,)


def metric_statistics(data: SigmaStarMetricsData, key: str,):
    """
    Calculate across-date statistics separately for each sigma*.

    STD reproduces the legacy sigma-control convention:
        population SD across finite dates (ddof=0).

    SEM is the nominal standard error:
        sample SD / sqrt(N), using ddof=1.

    The SEM assumes dates are independent.
    """
    if key not in METRIC_SPECS:
        raise KeyError(f"Unknown metric: {key}")

    sigma = np.sort(data.daily["sigma_star"].dropna().unique().astype(float))

    mean = np.full(sigma.shape, np.nan)
    std = np.full(sigma.shape, np.nan)
    sem = np.full(sigma.shape, np.nan)
    n = np.zeros(sigma.shape, dtype=int)

    for i, s in enumerate(sigma):
        values = data.daily.loc[np.isclose(data.daily["sigma_star"].to_numpy(float), s, atol=1e-8, rtol=0,), key,].to_numpy(float)

        values = values[np.isfinite(values)]
        n[i] = len(values)

        if len(values) == 0:
            continue

        mean[i] = values.mean()

        if len(values) >= 2:
            std[i] = values.std(ddof=0)
            sem[i] = (values.std(ddof=1) / np.sqrt(len(values)))

    return {
        "sigma": sigma,
        "mean": mean,
        "std": std,
        "sem": sem,
        "n": n,
    }


def _danra_slope_reference(data: SigmaStarMetricsData,) -> float:
    """
    DANRA slope is independent of sigma*. Average once per date
    rather than repeatedly across the sigma* sweep.
    """
    tmp = (data.daily[["date", "slope_hr"]].groupby("date", as_index=False).mean(numeric_only=True))

    values = tmp["slope_hr"].to_numpy(float)
    values = values[np.isfinite(values)]

    if not len(values):
        return np.nan

    return float(values.mean())


def _sigma_ticks(sigma: np.ndarray):
    if len(sigma) <= 7:
        return sigma

    ticks = sigma[::2]

    # Make sure reference sigma*=1 is shown.
    if np.any(np.isclose(sigma, 1.0)):
        ticks = np.unique(np.concatenate([ticks, [1.0]]))

    return np.sort(ticks)


def plot_sigma_star_metric(
    ax,
    data: SigmaStarMetricsData,
    key: str,
    *,
    error_mode: str,
    show_title: bool = True,
    show_xlabel: bool = True,
    show_legend: bool = False,
    annotate_context: bool = False,
):
    """
    Plot one sigma* metric using either across-date SD or SEM.
    """
    if error_mode not in {"std", "sem"}:
        raise ValueError("error_mode must be 'std' or 'sem'")

    spec = METRIC_SPECS[key]
    stats = metric_statistics(data, key)

    sigma = stats["sigma"]
    mean = stats["mean"]
    error = stats[error_mode]

    color = get_color_for_model("pmm")

    # Reference inference trajectory.
    ax.axvline(
        1.0,
        color="0.55",
        lw=1.0,
        ls=":",
        zorder=0,
    )

    ax.plot(
        sigma,
        mean,
        marker="o",
        ms=4.5,
        lw=1.7,
        color=color,
        zorder=3,
    )

    finite = (np.isfinite(mean) & np.isfinite(error))

    if finite.any():
        ax.errorbar(
            sigma[finite],
            mean[finite],
            yerr=error[finite],
            fmt="none",
            color=color,
            lw=1.1,
            capsize=2.5,
            zorder=2,
        )

    if key == "slope_gen":
        hr_ref = _danra_slope_reference(data)

        if np.isfinite(hr_ref):
            ax.axhline(
                hr_ref,
                color=get_color_for_model("hr"),
                ls="--",
                lw=1.2,
                label="DANRA",
                zorder=1,
            )

            if show_legend:
                ax.legend(
                    frameon=False,
                    fontsize=9,
                    loc="best",
                )

    if key == "crps" and annotate_context:
        metrics = data.meta.get("metrics", {})
        threshold = metrics.get("crps_rain_thresh")
        land_only = metrics.get("eval_land_only", False,)

        parts = []

        if threshold is not None:
            parts.append(rf"DANRA $>{float(threshold):g}$ " + r"mm day$^{-1}$")

        if land_only:
            parts.append("land")

        if parts:
            ax.text(
                0.02,
                0.97,
                ", ".join(parts),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
            )

    if show_title:
        ax.set_title(spec["title"], fontsize=12,)

    ax.set_ylabel(spec["ylabel"],fontsize=10.5,)
    if show_xlabel:
        ax.set_xlabel(r"$\sigma^*$", fontsize=11,)
    else:
        ax.set_xlabel("")

    ax.set_xticks(_sigma_ticks(sigma))

    pad = 0.025 * max(float(np.ptp(sigma)), 0.1,)
    ax.set_xlim(sigma.min() - pad, sigma.max() + pad,)

    ax.grid(True, ls=":", alpha=0.25,)
    ax.tick_params(axis="both", labelsize=9.5,)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.12)

    return stats