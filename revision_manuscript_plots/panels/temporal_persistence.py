from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sbgm.variable_utils import get_color_for_model


METHOD_ORDER = (
    "HR",
    "GEN_ENS",
    "PMM",
    "LR",
)

METHOD_LABELS = {
    "HR": "DANRA",
    "GEN_ENS": "CEDDAR members",
    "PMM": "CEDDAR PMM",
    "LR": "Bilinear ERA5",
}


@dataclass
class TemporalPersistenceData:
    year: int
    dates: np.ndarray
    autocorr: dict[str, np.ndarray]
    wet_bins: dict[str, np.ndarray]
    wet_pmf: dict[str, np.ndarray]
    dry_bins: dict[str, np.ndarray]
    dry_pmf: dict[str, np.ndarray]
    wet_threshold_mm: float


def _expected_daily_dates(year: int) -> np.ndarray:
    start = np.datetime64(f"{year:04d}-01-01", "D")
    stop = np.datetime64(f"{year + 1:04d}-01-01", "D")
    return np.arange(start, stop, dtype="datetime64[D]")


def _parse_dates(values: np.ndarray) -> np.ndarray:
    parsed = []

    for value in np.asarray(values).astype(str):
        value = value.strip()

        if len(value) == 8 and value.isdigit():
            value = (f"{value[:4]}-{value[4:6]}-{value[6:8]}")

        parsed.append(np.datetime64(value, "D"))

    return np.asarray(parsed, dtype="datetime64[D]",)


def load_temporal_persistence(
    temporal_dir: str | Path,
    *,
    year: int = 2019,
    wet_threshold_mm: float = 1.0,
    require_complete_year: bool = True,
) -> TemporalPersistenceData:
    """
    Load saved temporal diagnostics for one evaluation year.

    The legacy temporal metric implementation treats adjacent array entries
    as adjacent days. Therefore, by default, this loader requires the saved
    time series to form a complete, gap-free calendar year before the
    autocorrelation and spell diagnostics are accepted for plotting.
    """
    temporal_dir = Path(temporal_dir).expanduser()

    if (temporal_dir.name != "tables" and (temporal_dir / "tables").is_dir()):
        temporal_dir = temporal_dir / "tables"
    if not temporal_dir.is_dir():
        raise FileNotFoundError(temporal_dir)

    metrics_path = (temporal_dir / f"temporal_metrics_{year}.npz")
    series_path = (temporal_dir / f"temporal_series_{year}.npz")

    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    if not series_path.is_file():
        raise FileNotFoundError(series_path)

    with np.load(metrics_path, allow_pickle=False,) as npz:
        metrics = {key: np.asarray(npz[key]) for key in npz.files}
    with np.load(series_path,allow_pickle=False,) as npz:
        series = {key: np.asarray(npz[key]) for key in npz.files}

    if "dates" not in series:
        raise KeyError(f"{series_path}: missing 'dates'")

    dates = _parse_dates(series["dates"])

    # ---------------------------------------------------------
    # Validate chronological continuity.
    # ---------------------------------------------------------

    if dates.size == 0:
        raise ValueError(f"{series_path}: no dates")

    if not np.all(dates[1:] > dates[:-1]):
        raise ValueError(f"{series_path}: dates are not strictly increasing")

    if require_complete_year:
        expected = _expected_daily_dates(year)

        if not np.array_equal(dates, expected,):
            missing = np.setdiff1d(expected, dates,)
            extra = np.setdiff1d(dates, expected,)

            raise ValueError(f"{year} is not a complete daily calendar sequence: saved={len(dates)}, expected={len(expected)}, missing={len(missing)}, extra={len(extra)}. The legacy temporal metrics should not be interpreted across date gaps.")

    # ---------------------------------------------------------
    # Load and validate saved temporal metrics.
    # ---------------------------------------------------------

    autocorr: dict[str,np.ndarray,] = {}
    wet_bins: dict[str,np.ndarray,] = {}
    wet_pmf: dict[str,np.ndarray,] = {}
    dry_bins: dict[str,np.ndarray,] = {}
    dry_pmf: dict[str,np.ndarray,] = {}

    for method in METHOD_ORDER:
        required = (f"{method}_autocorr", f"{method}_wet_bins", f"{method}_wet_pmf", f"{method}_dry_bins", f"{method}_dry_pmf",)
        missing = [key for key in required if key not in metrics]

        if missing:
            raise KeyError(f"{metrics_path}: missing {missing}")

        ac = np.asarray(metrics[f"{method}_autocorr"], dtype=float,)
        wb = np.asarray(metrics[f"{method}_wet_bins"], dtype=float,)
        wp = np.asarray(metrics[f"{method}_wet_pmf"], dtype=float,)
        db = np.asarray(metrics[f"{method}_dry_bins"], dtype=float,)
        dp = np.asarray(metrics[f"{method}_dry_pmf"], dtype=float,)

        if ac.ndim != 1:
            raise ValueError(f"{method}: autocorrelation must be 1D, got {ac.shape}")
        if (wb.ndim != 1 or wp.ndim != 1 or wb.shape != wp.shape):
            raise ValueError(f"{method}: wet spell arrays are inconsistent")
        if (db.ndim != 1 or dp.ndim != 1 or db.shape != dp.shape):
            raise ValueError(f"{method}: dry spell arrays are inconsistent")
        if not np.isclose(np.nansum(wp), 1.0, atol=1e-8,):
            raise ValueError(f"{method}: wet PMF does not sum to one")
        if not np.isclose(np.nansum(dp), 1.0, atol=1e-8,):
            raise ValueError(f"{method}: dry PMF does not sum to one")

        autocorr[method] = ac
        wet_bins[method] = wb
        wet_pmf[method] = wp
        dry_bins[method] = db
        dry_pmf[method] = dp

    n_lags = {len(values) for values in autocorr.values()}

    if len(n_lags) != 1:
        raise ValueError("Autocorrelation arrays have different lag lengths")

    return TemporalPersistenceData(
        year=int(year),
        dates=dates,
        autocorr=autocorr,
        wet_bins=wet_bins,
        wet_pmf=wet_pmf,
        dry_bins=dry_bins,
        dry_pmf=dry_pmf,
        wet_threshold_mm=float(wet_threshold_mm),
    )


def _method_style(method: str,) -> dict:
    if method == "HR":
        return dict(
            color=get_color_for_model("HR"),
            marker="D",
            ls="-",
            lw=1.8,
            ms=4.0,
        )

    if method == "GEN_ENS":
        return dict(
            color=get_color_for_model("ensemble"),
            marker="o",
            ls="-",
            lw=1.7,
            ms=4.0,
        )

    if method == "PMM":
        return dict(
            color=get_color_for_model("PMM"),
            marker="s",
            ls="-",
            lw=1.7,
            ms=4.0,
        )

    if method == "LR":
        return dict(
            color=get_color_for_model("LR"),
            marker="^",
            ls="--",
            lw=1.5,
            ms=4.2,
        )

    raise KeyError(method)


def _finish_axis(ax, ) -> None:
    ax.grid(True, ls=":", alpha=0.25,)
    ax.tick_params(axis="both", labelsize=9.5,)
    ax.spines[["top", "right"]].set_visible(False)


def plot_autocorrelation(ax, data: TemporalPersistenceData,):
    """
    Plot autocorrelation of daily land-domain mean precipitation.

    CEDDAR members are represented by the mean of member-wise
    autocorrelation curves, as saved by the legacy evaluator.
    """
    for method in METHOD_ORDER:
        ac = data.autocorr[method]

        lags = np.arange(1, len(ac) + 1,)

        ax.plot(
            lags,
            ac,
            label=METHOD_LABELS[method],
            **_method_style(method),
        )

    ax.axhline(
        0.0,
        color="0.6",
        lw=0.8,
        ls=":",
        zorder=0,
    )

    max_abs = max(float(np.nanmax(np.abs(values))) for values in data.autocorr.values())
    upper = max(0.45, max_abs * 1.12,)

    ax.set_ylim(-0.10, upper,)
    ax.set_xticks(np.arange(1, len(next(iter(data.autocorr.values()))) + 1,))
    ax.set_xlabel("Lag (days)", fontsize=10.5,)
    ax.set_ylabel("Autocorrelation", fontsize=10.5,)
    ax.set_title("Domain-mean autocorrelation", fontsize=12,)

    _finish_axis(ax)


def plot_spell_distribution(
    ax,
    data: TemporalPersistenceData,
    *,
    kind: str,
    show_ylabel: bool = True,
):
    """
    Plot empirical wet- or dry-spell PMFs for land-domain mean
    precipitation. No geometric fits are shown.
    """
    if kind == "wet":
        bins = data.wet_bins
        pmf = data.wet_pmf
        title = "Wet-spell length"
    elif kind == "dry":
        bins = data.dry_bins
        pmf = data.dry_pmf
        title = "Dry-spell length"
    else:
        raise ValueError("kind must be 'wet' or 'dry'")

    for method in METHOD_ORDER:
        ax.plot(
            bins[method],
            pmf[method],
            label=METHOD_LABELS[method],
            **_method_style(method),
        )

    all_pmf = np.concatenate([np.asarray(pmf[method], dtype=float,) for method in METHOD_ORDER])

    ymax = float(np.nanmax(all_pmf))

    ax.set_ylim(-0.005, max(0.48, ymax * 1.10,),)

    # Use the same horizontal extent for wet and dry spells so
    # the two distributions can be compared directly. No 2019
    # method has non-zero probability beyond 20 days.
    ax.set_xlim(0.7, 20.3)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Spell length (days)", fontsize=10.5,)

    if show_ylabel:
        ax.set_ylabel("Probability", fontsize=10.5,)

    ax.set_title(title, fontsize=12,)

    _finish_axis(ax)
