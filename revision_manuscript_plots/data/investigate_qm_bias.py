from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WET_THRESHOLD = 1.0

SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


# ================================================================
# Small loading helpers
# ================================================================


def load_npz(path: Path):
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def first_matching_key(arrays, candidates, *, path,):
    for key in candidates:
        if key in arrays:
            return np.asarray(arrays[key], dtype=float,)

    raise KeyError(f"{path}: none of {candidates} found. Available keys: {list(arrays)}")


def squeeze_field(field):
    """Convert common [1,H,W] fields to [H,W]."""

    field = np.asarray(field, dtype=float,)

    field = np.squeeze(field)

    if field.ndim != 2:
        raise ValueError(f"Expected 2D field after squeeze, got {field.shape}")
    
    return field


def load_ceddar_lr_hr(path: Path):
    arrays = load_npz(path)

    lr = first_matching_key(arrays, ("lr", "lr_phys", "era5", "LR",), path=path,)
    hr = first_matching_key(arrays, ("hr", "hr_phys", "danra", "HR",), path=path,)

    return (squeeze_field(lr), squeeze_field(hr),)


def load_qm_lr_hr(path: Path):
    arrays = load_npz(path)

    hr = first_matching_key(arrays, ("hr", "hr_phys", "danra", "HR",), path=path,)
    lr = first_matching_key(arrays, ("lr_hr", "lr", "lr_phys", "era5", "LR",), path=path,)

    return (squeeze_field(lr), squeeze_field(hr),)


def load_ceddar_ensemble(path: Path):
    arrays = load_npz(path)

    ens = first_matching_key(arrays, ("ens",), path=path,)

    # Expected:
    # [32,1,H,W] or [32,H,W]
    if (ens.ndim == 4 and ens.shape[1] == 1):
        ens = ens[:, 0]

    if ens.ndim != 3:
        raise ValueError(f"{path}: expected ensemble [M,H,W], got {ens.shape}")

    return ens


def load_ceddar_pmm(path: Path):
    arrays = load_npz(path)
    field = first_matching_key(arrays,("pmm", "gen", "generated",), path=path,)

    return squeeze_field(field)


def load_qm(path: Path):
    arrays = load_npz(path)
    field = first_matching_key(arrays, ("pmm", "qm", "gen", "generated",), path=path,)

    return squeeze_field(field)


def load_land_mask(path: Path):
    arrays = load_npz(path)

    land = first_matching_key(arrays, ("lsm", "mask", "land", "lsm_hr",), path=path,)
    land = squeeze_field(land)

    return land >= 0.5


# ================================================================
# Date handling
# ================================================================


def dates_in(directory: Path):
    return {p.stem for p in directory.glob("*.npz")}


def month_from_date(date):
    return int(date[4:6])


def season_from_date(date):
    month = month_from_date(date)

    for season, months in SEASONS.items():
        if month in months:
            return season

    raise ValueError(date)


# ================================================================
# Diagnostics
# ================================================================

def rolling_mean(values, window=14):
    values = np.asarray(values, dtype=float,)
    kernel = np.ones(window, dtype=float,) / window

    return np.convolve(values, kernel, mode="valid",)

def field_metrics(field, land, *, wet_threshold=WET_THRESHOLD,):
    """Land-only precipitation diagnostics."""

    values = np.asarray(field, dtype=float,)[land]

    values = values[np.isfinite(values)]

    wet = values >= wet_threshold
    low = ((values > 0) & (values < wet_threshold))

    wet_frequency = np.mean(wet)

    if np.any(wet):
        conditional_mean_wet = np.mean(values[wet])
        p95_wet = np.percentile(values[wet], 95,)
        p99_wet = np.percentile(values[wet], 99,)
    else:
        conditional_mean_wet = np.nan
        p95_wet = np.nan
        p99_wet = np.nan

    # Exact decomposition:
    #
    # E[P] =
    #   E[P * I(0<P<1)]
    #   + P(P>=1) E[P | P>=1]
    #
    low_precip_contribution = np.mean(
        np.where(low, values, 0.0,))

    wet_contribution = (wet_frequency * conditional_mean_wet if np.isfinite(conditional_mean_wet) else np.nan)

    return {
        "mean": np.mean(values),
        "wet_frequency": wet_frequency,
        "conditional_mean_wet": (conditional_mean_wet),
        "low_precip_contribution": (low_precip_contribution),
        "wet_contribution": wet_contribution,
        "p95_wet": p95_wet,
        "p99_wet": p99_wet,
    }


def summarise_rows(rows, method):
    subset = [row for row in rows if row["method"] == method]

    result = {"method": method, "n_days": len(subset),}

    metrics = (
        "mean",
        "wet_frequency",
        "conditional_mean_wet",
        "low_precip_contribution",
        "wet_contribution",
        "p95_wet",
        "p99_wet",
    )

    for metric in metrics:
        values = np.array([row[metric] for row in subset], dtype=float,)

        result[f"{metric}_mean"] = (np.nanmean(values))
        result[f"{metric}_std"] = (np.nanstd(values))

    return result


# ================================================================
# Main analysis
# ================================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--generation-root", type=Path, required=True,)
    parser.add_argument("--output-dir", type=Path, required=True,)
    parser.add_argument("--year", default="2019",)

    args = parser.parse_args()

    root = (args.generation_root.expanduser().resolve())
    output = (args.output_dir.expanduser().resolve())
    output.mkdir(parents=True, exist_ok=True,)

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    ceddar_root = (root / ("B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56"))
    qm_root = (root / "baselines" / "qm" / "test")

    ceddar_ens_dir = (ceddar_root / "ensembles_phys")
    ceddar_lrhr_dir = (ceddar_root / "lr_hr_phys")
    ceddar_pmm_dir = (ceddar_root / "pmm_phys")

    qm_lrhr_dir = (qm_root / "lr_hr")
    qm_pmm_dir = (qm_root / "pmm")
    qm_lsm_dir = (qm_root / "lsm")

    # ------------------------------------------------------------
    # Date intersection
    # ------------------------------------------------------------

    date_sets = {
        "ceddar ensemble": dates_in(ceddar_ens_dir),
        "ceddar lr/hr": dates_in(ceddar_lrhr_dir),
        "ceddar pmm": dates_in(ceddar_pmm_dir),
        "qm lr/hr": dates_in(qm_lrhr_dir),
        "qm": dates_in(qm_pmm_dir),
    }

    print("\nAvailable files")
    print("----------------")

    for name, dates in date_sets.items():
        year_dates = sorted(d for d in dates if d.startswith(args.year))

        print(f"{name:20s}: {len(year_dates):4d}")

    common_dates = sorted(
        set.intersection(
            *[
                {
                    d
                    for d in dates
                    if d.startswith(
                        args.year
                    )
                }
                for dates
                in date_sets.values()
            ]
        )
    )

    if not common_dates:
        raise RuntimeError("No common dates found.")

    print("\nCommon dates:", len(common_dates))
    print("First:", common_dates[:3])
    print("Last :", common_dates[-3:])

    # ------------------------------------------------------------
    # Load land mask
    # ------------------------------------------------------------

    lsm_candidates = sorted(qm_lsm_dir.glob("*.npz"))

    if not lsm_candidates:
        raise FileNotFoundError(f"No land mask found in {qm_lsm_dir}")

    land = load_land_mask(lsm_candidates[0])

    print("\nLand pixels:", int(land.sum()), "/", land.size)

    # ------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------

    methods = (
        "danra",
        "era5",
        "qm",
        "ceddar_mean",
        "ceddar_median",
        "ceddar_pmm",
    )

    daily_rows = []

    spatial_sums = {
        method: np.zeros(
            land.shape,
            dtype=np.float64,
        )
        for method in methods
    }

    # ------------------------------------------------------------
    # Iterate over exact common dates
    # ------------------------------------------------------------

    pooled = {method: {"sum_precip": 0.0, "n_pixels": 0, "n_wet": 0, "sum_wet_precip": 0.0, "n_low": 0, "sum_low_precip": 0.0} for method in methods}

    for i, date in enumerate(common_dates):
        # CEDDAR's copy of LR/HR
        lr_ceddar, hr_ceddar = (load_ceddar_lr_hr(ceddar_lrhr_dir / f"{date}.npz"))

        # QM's copy of LR/HR
        lr_qm, hr_qm = (load_qm_lr_hr(qm_lrhr_dir / f"{date}.npz"))

        # --------------------------------------------------------
        # Critical provenance check:
        # Are both evaluations using identical inputs?
        # --------------------------------------------------------

        hr_diff = np.nanmax(np.abs(hr_ceddar - hr_qm))
        lr_diff = np.nanmax(np.abs(lr_ceddar - lr_qm))

        if (hr_diff > 1e-4 or lr_diff > 1e-4):
            print(f"WARNING {date}: HR max Δ={hr_diff:.3e}, LR max Δ={lr_diff:.3e}")

        ensemble = (load_ceddar_ensemble(ceddar_ens_dir / f"{date}.npz"))
        qm = load_qm(qm_pmm_dir / f"{date}.npz")
        ceddar_pmm = (load_ceddar_pmm(ceddar_pmm_dir / f"{date}.npz"))

        fields = {
            "danra": hr_ceddar,
            "era5": lr_ceddar,
            "qm": qm,
            "ceddar_mean": (np.mean(ensemble, axis=0,)),
            "ceddar_median": (np.median(ensemble, axis=0,)),
            "ceddar_pmm": (ceddar_pmm),
        }

        for method, field in (fields.items()):
            metrics = field_metrics(field, land,)

            row = {
                "date": date,
                "season": (season_from_date(date)),
                "method": method,
                **metrics,
            }

            daily_rows.append(row)

            spatial_sums[method] += np.nan_to_num(field, nan=0.0,)

            values = np.asarray(field, dtype=float)[land]
            values = values[np.isfinite(values)]

            wet = values >= WET_THRESHOLD
            low = ((values > 0.0) & (values < WET_THRESHOLD))

            p = pooled[method]

            p["sum_precip"] += values.sum()
            p["n_pixels"] += values.size

            p["n_wet"] += wet.sum()
            p["sum_wet_precip"] += values[wet].sum()

            p["n_low"] += low.sum()
            p["sum_low_precip"] += values[low].sum()

        if ((i + 1) % 50 == 0 or i == 0 or i + 1 == len(common_dates)):
            print(f"Processed {i + 1:4d}/{len(common_dates)}")


    print("\nPooled pixel-day diagnostics")
    print("-----------------------------")

    for method in methods:
        p = pooled[method]

        mean_precip = (p["sum_precip"] / p["n_pixels"])
        wet_frequency = (p["n_wet"] / p["n_pixels"])
        conditional_mean_wet = (p["sum_wet_precip"] / p["n_wet"])
        wet_contribution = (p["sum_wet_precip"] / p["n_pixels"])
        low_contribution = (p["sum_low_precip"] / p["n_pixels"])

        print()
        print(method)

        print(f"  mean precipitation    : {mean_precip:.3f}")
        print(f"  wet frequency         : {wet_frequency:.3f}")
        print(f"  conditional mean wet. : {conditional_mean_wet:.3f}")
        print(f"  wet contribution      : {wet_contribution:.3f}")
        print(f"  <1 mm contribution    : {low_contribution:.3f}")
        print(f"  closure               : {wet_contribution + low_contribution:.3f}")

    # ------------------------------------------------------------
    # Daily CSV
    # ------------------------------------------------------------

    daily_path = (output / "qm_daily_diagnostics.csv")

    fieldnames = [
        "date",
        "season",
        "method",
        "mean",
        "wet_frequency",
        "conditional_mean_wet",
        "low_precip_contribution",
        "wet_contribution",
        "p95_wet",
        "p99_wet",
    ]

    with daily_path.open("w", newline="",) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,)

        writer.writeheader()
        writer.writerows(daily_rows)

    # ------------------------------------------------------------
    # Overall summaries
    # ------------------------------------------------------------

    summaries = [summarise_rows(daily_rows, method,) for method in methods]

    print("\nOverall daily diagnostics")
    print("-------------------------")

    header = (
        f"{'method':16s}"
        f"{'P mean':>10s}"
        f"{'wet freq':>11s}"
        f"{'wet mean':>11s}"
        f"{'<1 mm':>10s}"
    )

    print(header)

    for row in summaries:
        print(
            f"{row['method']:16s}"
            f"{row['mean_mean']:10.3f}"
            f"{row['wet_frequency_mean']:11.3f}"
            f"{row['conditional_mean_wet_mean']:11.3f}"
            f"{row['low_precip_contribution_mean']:10.3f}"
        )

    # ------------------------------------------------------------
    # Exact spatial accumulation statistics
    # ------------------------------------------------------------

    print("\nSpatial accumulation")
    print("--------------------")

    danra_mean = np.mean(spatial_sums["danra"][land])

    for method in methods:
        values = (spatial_sums[method][land])

        mean_total = np.mean(values)
        std_total = np.std(values)

        bias = (100.0 * (mean_total / danra_mean - 1.0))

        print(
            f"{method:16s}"
            f"{mean_total:9.1f} mm"
            f" ± {std_total:7.1f}"
            f"   {bias:+6.1f}%"
        )

    # ------------------------------------------------------------
    # Seasonal summary
    #
    # This is useful now even if you later turn it into a
    # manuscript table.
    # ------------------------------------------------------------

    seasonal_path = (output / "qm_seasonal_diagnostics.csv")

    seasonal_rows = []

    for season in SEASONS:
        for method in methods:
            subset = [row for row in daily_rows if (row["season"] == season and row["method"] == method)]

            out = {
                "season": season,
                "method": method,
                "n_days": len(subset),
            }

            for metric in ("mean", "wet_frequency", "conditional_mean_wet", "low_precip_contribution",):
                values = np.array([row[metric] for row in subset], dtype=float,)

                out[metric] = (np.nanmean(values))

            seasonal_rows.append(out)

    with seasonal_path.open("w", newline="",) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "season",
                "method",
                "n_days",
                "mean",
                "wet_frequency",
                "conditional_mean_wet",
                "low_precip_contribution",
            ),
        )

        writer.writeheader()
        writer.writerows(seasonal_rows)

    # ------------------------------------------------------------
    # Figure 1:
    # daily domain-mean precipitation
    # ------------------------------------------------------------

    lookup = {}

    for method in methods:
        lookup[method] = {row["date"]: row for row in daily_rows if row["method"] == method}

    daily_mean = {method: np.array([lookup[method][date]["mean"] for date in common_dates], dtype=float,) for method in methods}

    danra_daily = daily_mean["danra"]
    compare_methods = ("era5", "qm", "ceddar_mean", "ceddar_median", "ceddar_pmm")
    labels = {
        "era5": "Bilinear ERA5",
        "qm": "QM",
        "ceddar_mean": "CEDDAR mean",
        "ceddar_median": "CEDDAR median",
        "ceddar_pmm": "CEDDAR PMM",
    }

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), sharex=True, sharey=True)
    axes = axes.ravel()

    all_values = [danra_daily, *[daily_mean[method] for method in compare_methods]]
    lo = min(np.nanmin(v) for v in all_values)
    hi = max(np.nanmax(v) for v in all_values)
    pad = 0.03 * (hi - lo)

    for ax, method in zip(axes, compare_methods):
        x = danra_daily
        y = daily_mean[method]

        valid = (np.isfinite(x)) & (np.isfinite(y))
        x_valid = x[valid]
        y_valid = y[valid]

        ax.scatter(x_valid, y_valid, s=13, alpha=0.40,)
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1,color="0.45", label="1:1")

        # Least-square fit
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        x_fit = np.linspace(lo, hi, 200)
        y_fit = slope * x_fit + intercept
        r = np.corrcoef(x_valid, y_valid)[0, 1]

        ax.plot(x_fit, y_fit, "-", linewidth=1.3, label="OLS fit")
        ax.set_title(labels[method])
        rel_bias = np.nanmean(y_valid - x_valid)
        ax.text(0.04, 0.95, (rf"$y={slope:.2f}x+{intercept:.2f}$" + "\n" + rf"r={r:.2f}" + "\n" + rf"bias={rel_bias:+.1f}"), transform=ax.transAxes, va="top", ha="left", fontsize=8)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    axes[-1].axis("off")
    for ax in axes [3:5]:
        ax.set_xlabel(r"DANRA domain mean (mm day$^{-1}$)")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(r"Model domain mean (mm day$^{-1}$)")
    fig.suptitle("Daily domain-mean precipitation", y=0.995)

    fig.tight_layout()
    fig.savefig(output / "daily_mean_vs_danra.pdf")
    plt.close(fig)

    # ------------------------------------------------------------
    # Figure 2:
    # daily domain-mean bias through time
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9.0, 3.8))

    x = np.arange(len(common_dates))

    # Get n colors for the different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(compare_methods))) # type: ignore

    for method in compare_methods:
        bias = (daily_mean[method] - danra_daily)

        ax.plot(x, bias, linewidth=0.55, alpha=0.18, color=colors[compare_methods.index(method)])

        smooth = rolling_mean(bias, window=14)
        x_smooth = np.arange(len(smooth)) + 13
        ax.plot(x_smooth, smooth, linewidth=1.4, alpha=0.75, color=colors[compare_methods.index(method)], label=labels[method])

    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)

    ax.set_ylabel(r"Model $-$ DANRA (mm day$^{-1}$)")
    ax.set_xlabel(f"{args.year} evaluation day")
    ax.set_title("Daily domain-mean precipitation bias")

    ax.legend(frameon=False, ncol=3,)

    fig.tight_layout()
    fig.savefig(output / "daily_mean_bias.pdf")
    plt.close(fig)

    # ------------------------------------------------------------
    # Save spatial sums so they can also be inspected later.
    # ------------------------------------------------------------

    np.savez_compressed(output / "qm_spatial_comparison.npz", dates=np.asarray(common_dates), land=land, **spatial_sums,)

    print("\nWritten:")

    for path in (
        daily_path,
        seasonal_path,
        output / "qm_spatial_comparison.npz",
        output / "qm_vs_danra_daily_mean.pdf",
        output / "qm_daily_bias.pdf",
        output / "qm_occurrence_intensity.pdf",
    ):
        print(" ", path)


if __name__ == "__main__":
    main()