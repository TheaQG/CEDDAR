from __future__ import annotations

"""
Generate daily spatial Pearson correlations for the deterministic evaluation.

For every date in the existing deterministic revision evaluation, calculate
the spatial Pearson correlation across Danish land pixels between DANRA and:

    - bilinear ERA5
    - QM
    - CEDDAR ensemble mean
    - CEDDAR ensemble median
    - CEDDAR PMM

The script deliberately uses the existing deterministic `dates.txt` rather
than constructing a new evaluation sample. This keeps the correlation
diagnostic on exactly the same dates as the other deterministic metrics.

Outputs
-------
<deterministic-dir>/
    daily_spatial_correlation.csv
    spatial_correlation_summary.csv
    spatial_correlation_metadata.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_manuscript_plots.data import legacy


DEFAULT_MODEL = ("B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56")

METHOD_ORDER = (
    "era5_bilinear",
    "qm",
    "ceddar_mean",
    "ceddar_median",
    "ceddar_pmm",
)

SEASON_ORDER = (
    "ALL",
    "DJF",
    "MAM",
    "JJA",
    "SON",
)


# =============================================================================
# Generic loading helpers
# =============================================================================


def load_npz(path: Path,) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False,) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def first_matching_key(
    arrays: dict[str, np.ndarray],
    candidates: tuple[str, ...],
    *,
    path: Path,
) -> np.ndarray:
    for key in candidates:
        if key in arrays:
            return np.asarray(arrays[key], dtype=float,)

    raise KeyError(f"{path}: none of {candidates} found. Available keys: {list(arrays)}")


def squeeze_field(
    field: np.ndarray,
    *,
    path: Path,
) -> np.ndarray:
    field = np.asarray(field, dtype=float,)
    # Accept [1,H,W], [1,1,H,W], etc.
    field = np.squeeze(field)

    if field.ndim != 2:
        raise ValueError(f"{path}: expected a 2D field after squeeze; got {field.shape}")

    return field


def load_ceddar_lr_hr(path: Path,) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_npz(path)

    lr = first_matching_key(arrays, ("lr", "lr_phys", "era5", "LR",), path=path,)
    hr = first_matching_key(arrays, ("hr", "hr_phys", "danra", "HR",), path=path,)

    return (squeeze_field(lr, path=path,), squeeze_field(hr, path=path,))


def load_qm_lr_hr(path: Path,) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the LR/HR pair saved with the QM baseline.

    These fields are used only as a provenance check: the HR reference
    should agree with the CEDDAR DANRA field, and the LR field should agree
    with the CEDDAR ERA5 condition.
    """
    arrays = load_npz(path)

    lr = first_matching_key(arrays, ("lr_hr", "lr", "lr_phys", "era5", "LR",), path=path,)
    hr = first_matching_key(arrays, ("hr", "hr_phys", "danra", "HR",), path=path,)

    return (squeeze_field(lr, path=path,), squeeze_field(hr, path=path,))


def load_ensemble(path: Path,) -> np.ndarray:
    arrays = load_npz(path)

    ensemble = first_matching_key(arrays, ("ens",), path=path,)

    # [M,1,H,W] -> [M,H,W]
    if (ensemble.ndim == 4 and ensemble.shape[1] == 1):
        ensemble = ensemble[:, 0,]

    if ensemble.ndim != 3:
        raise ValueError(f"{path}: expected ensemble [M,H,W], got {ensemble.shape}")

    return np.asarray(ensemble, dtype=float,)


def load_pmm(path: Path,) -> np.ndarray:
    arrays = load_npz(path)
    field = first_matching_key(arrays,("pmm", "gen", "generated",), path=path,)
    return squeeze_field(field, path=path,)


def load_qm(path: Path,) -> np.ndarray:
    arrays = load_npz(path)
    field = first_matching_key(arrays,("pmm", "qm", "gen", "generated",), path=path,)
    return squeeze_field(field, path=path,)


# =============================================================================
# Date / season helpers
# =============================================================================


def normalise_date(value: str,) -> str:
    value = str(value).strip()
    value = value.replace("-", "",)

    if (len(value) != 8 or not value.isdigit()):
        raise ValueError(f"Unsupported date representation: {value!r}")

    return value


def season_from_date(date: str,) -> str:
    month = int(date[4:6])

    if month in (12, 1, 2,):
        return "DJF"
    if month in (3, 4, 5,):
        return "MAM"
    if month in (6, 7, 8,):
        return "JJA"
    return "SON"


def read_dates(path: Path,) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing deterministic date list: {path}")

    dates = [normalise_date(line) for line in path.read_text().splitlines() if line.strip()]

    if not dates:
        raise ValueError(f"{path}: no dates")

    if len(dates) != len(set(dates)):
        raise ValueError(f"{path}: duplicate dates")
    if dates != sorted(dates):
        raise ValueError(f"{path}: dates are not sorted")

    return dates


# =============================================================================
# Correlation
# =============================================================================


def spatial_pearson(
    reference: np.ndarray,
    prediction: np.ndarray,
    land: np.ndarray,
    *,
    constant_tolerance: float = 1e-12,
) -> tuple[float, int, float, float,]:
    """
    Pearson correlation across Danish land pixels for one daily field.

    The same fixed land mask is used for every method. Non-finite land values
    are treated as an error rather than silently changing method-specific
    spatial support.

    Returns
    -------
    correlation, n_pixels, reference_std, prediction_std

    Correlation is NaN when either field has effectively zero spatial variance,
    because Pearson correlation is undefined in that case.
    """
    reference = np.asarray(reference, dtype=float,)
    prediction = np.asarray(prediction, dtype=float,)
    land = np.asarray(land, dtype=bool,)

    if (reference.shape != land.shape or prediction.shape != land.shape):
        raise ValueError(f"Field and land-mask shapes differ: reference={reference.shape}, prediction={prediction.shape}, land={land.shape}")

    x = reference[land]
    y = prediction[land]

    finite = (np.isfinite(x) & np.isfinite(y))

    if not np.all(finite):
        bad = int(np.sum(~finite))

        raise RuntimeError(f"Found {bad} non-finite paired values on Danish land. Refusing method-dependent spatial masking.")

    n_pixels = int(len(x))

    if n_pixels < 2:
        raise RuntimeError("Fewer than two land pixels are available.")

    x_std = float(np.std(x, ddof=0,))
    y_std = float(np.std(y, ddof=0,))

    if (x_std <= constant_tolerance or y_std <= constant_tolerance):
        return (np.nan, n_pixels, x_std, y_std,)

    correlation = float(np.corrcoef(x, y,)[0, 1,])
    # Guard against tiny floating-point excursions outside [-1, 1].
    correlation = float(np.clip(correlation, -1.0, 1.0))

    return (correlation, n_pixels, x_std, y_std,)


# =============================================================================
# Summary
# =============================================================================


def summarise(daily: pd.DataFrame,) -> pd.DataFrame:
    rows: list[dict] = []

    for season in SEASON_ORDER:
        if season == "ALL":
            seasonal = daily
        else:
            seasonal = daily.loc[daily["season"] == season]

        for method in METHOD_ORDER:
            sub = seasonal.loc[seasonal["method"] == method]
            values = sub["correlation"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]

            row = {
                "season": season,
                "method": method,
                "n_dates": int(len(values)),
                "n_valid": int(len(finite)),
                "n_undefined": int(len(values) - len(finite)),
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "p25": np.nan,
                "p75": np.nan,
            }

            if finite.size:
                row.update(
                    {
                        "mean": float(np.mean(finite)),
                        "median": float(np.median(finite)),
                        "std": float(np.std(finite, ddof=1)) if finite.size >= 2 else np.nan,
                        "p25": float(np.quantile(finite, 0.25)),
                        "p75": float(np.quantile(finite, 0.75)),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=("Generate daily land-only spatial Pearson correlations for the deterministic manuscript evaluation."))

    parser.add_argument("--generation-root", type=Path, required=True, help=("Root containing the final CEDDAR model directory and baselines/qm/test."),)
    parser.add_argument("--deterministic-dir", type=Path, required=True,help=("Existing P1_revision/deterministic directory. Its dates.txt defines the evaluation sample and the correlation CSVs are written here."),)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=("Final CEDDAR generation-directory name."),)
    parser.add_argument("--qm-subdir", type=Path, default=Path("baselines/qm/test"), help=("QM generation directory relative to --generation-root."),)
    parser.add_argument("--dates-file", type=Path, default=None, help=("Optional explicit date list. Defaults to <deterministic-dir>/dates.txt."),)
    parser.add_argument("--constant-tolerance", type=float, default=1e-12, help=("Spatial standard deviation at or below which Pearson correlation is treated as undefined."),)

    args = parser.parse_args()

    generation_root = (args.generation_root.expanduser().resolve())
    deterministic_dir = (args.deterministic_dir.expanduser().resolve())

    if not deterministic_dir.is_dir():
        raise FileNotFoundError(deterministic_dir)

    dates_file = (args.dates_file.expanduser().resolve() if args.dates_file is not None else deterministic_dir / "dates.txt")
    dates = read_dates(dates_file)

    ceddar_root = (generation_root / args.model)
    qm_root = (generation_root / args.qm_subdir)

    ceddar_lrhr_dir = (ceddar_root / "lr_hr_phys")
    ensemble_dir = (ceddar_root / "ensembles_phys")
    pmm_dir = (ceddar_root / "pmm_phys")
    qm_lrhr_dir = (qm_root / "lr_hr")
    qm_dir = (qm_root / "pmm")

    required_dirs = {
        "CEDDAR LR/HR": (ceddar_lrhr_dir),
        "CEDDAR ensemble": (ensemble_dir),
        "CEDDAR PMM": (pmm_dir),
        "QM LR/HR": (qm_lrhr_dir),
        "QM": (qm_dir),
    }

    for name, path in required_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{name}: {path}")

    print()
    print("Deterministic correlation evaluation")
    print("------------------------------------")
    print(f"Dates: {len(dates)}")
    print(f"First: {dates[:3]}")
    print(f"Last : {dates[-3:]}")

    # -------------------------------------------------------------------------
    # Static Danish land mask
    # -------------------------------------------------------------------------

    land = np.asarray(legacy.load_land_mask(ceddar_root), dtype=bool,)
    n_land = int(land.sum())

    print()
    print(f"Land pixels: {n_land} / {land.size}")

    # -------------------------------------------------------------------------
    # Stream dates
    # -------------------------------------------------------------------------

    rows: list[dict] = []
    member_counts: set[int] = set()

    max_hr_copy_difference = 0.0
    max_lr_copy_difference = 0.0

    for index, date in enumerate(dates, start=1,):
        ceddar_lrhr_path = (ceddar_lrhr_dir / f"{date}.npz")
        ensemble_path = (ensemble_dir / f"{date}.npz")
        pmm_path = (pmm_dir / f"{date}.npz")
        qm_lrhr_path = (qm_lrhr_dir / f"{date}.npz")
        qm_path = (qm_dir / f"{date}.npz")

        lr, hr = load_ceddar_lr_hr(ceddar_lrhr_path)
        ensemble = load_ensemble(ensemble_path)
        pmm = load_pmm(pmm_path)
        qm_lr, qm_hr = load_qm_lr_hr(qm_lrhr_path)
        qm = load_qm(qm_path)

        member_counts.add(int(ensemble.shape[0]))

        # -------------------------------------------------------------
        # Shape checks
        # -------------------------------------------------------------

        expected_shape = (land.shape)

        named_fields = {
            "DANRA": hr,
            "bilinear ERA5": lr,
            "CEDDAR PMM": pmm,
            "QM": qm,
            "QM DANRA copy": qm_hr,
            "QM ERA5 copy": qm_lr,
        }

        for name, field in named_fields.items():
            if field.shape != expected_shape:
                raise ValueError(f"{date}: {name} shape {field.shape} != land-mask shape {expected_shape}")

        if (ensemble.shape[1:] != expected_shape):
            raise ValueError(f"{date}: ensemble shape {ensemble.shape} is incompatible with land-mask shape {expected_shape}")

        # -------------------------------------------------------------
        # Provenance checks
        # -------------------------------------------------------------

        hr_diff = float(np.nanmax(np.abs(hr - qm_hr)))
        lr_diff = float(np.nanmax(np.abs(lr - qm_lr)))

        max_hr_copy_difference = max(max_hr_copy_difference,hr_diff,)
        max_lr_copy_difference = max(max_lr_copy_difference, lr_diff)

        if (hr_diff > 1e-4 or lr_diff > 1e-4):
            raise RuntimeError(f"{date}: CEDDAR and QM LR/HR copies disagree: HR max |Δ|={hr_diff:.3e}, LR max |Δ|={lr_diff:.3e}")

        # -------------------------------------------------------------
        # Deterministic CEDDAR summaries
        # -------------------------------------------------------------

        ceddar_mean = np.mean(ensemble, axis=0,)
        ceddar_median = np.median(ensemble, axis=0,)

        fields = {
            "era5_bilinear": lr,
            "qm": qm,
            "ceddar_mean": (ceddar_mean),
            "ceddar_median": (ceddar_median),
            "ceddar_pmm": pmm,
        }

        # -------------------------------------------------------------
        # Require finite values over Danish land for every product.
        # This guarantees a fixed spatial sample across methods.
        # -------------------------------------------------------------

        reference_land = hr[land]

        if not np.all(np.isfinite(reference_land)):
            raise RuntimeError(f"{date}: DANRA contains non-finite values on Danish land.")

        for method, field in fields.items():
            if not np.all(np.isfinite(field[land])):
                raise RuntimeError(f"{date}: {method} contains non-finite values on Danish land.")

        # -------------------------------------------------------------
        # Daily spatial correlations
        # -------------------------------------------------------------

        for method in METHOD_ORDER:
            (
                correlation,
                n_pixels,
                reference_std,
                method_std,
            ) = spatial_pearson(
                hr,
                fields[method],
                land,
                constant_tolerance=(args.constant_tolerance),
            )

            rows.append(
                {
                    "date": date,
                    "season": (season_from_date(date)),
                    "method": method,
                    "subset": ("all_land"),
                    "correlation": (correlation),
                    "n_pixels": (n_pixels),
                    "reference_std_mm_day": (reference_std),
                    "method_std_mm_day": (method_std),
                }
            )

        if (index == 1 or index % 50 == 0 or index == len(dates)):
            print(f"Processed {index}/{len(dates)}")

    daily = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    expected_rows = (len(dates) * len(METHOD_ORDER))

    if len(daily) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, generated {len(daily)}.")

    for method in METHOD_ORDER:
        sub = daily.loc[daily["method"] == method]

        method_dates = sub["date"].tolist()

        if method_dates != dates:
            raise RuntimeError(f"{method}: output dates differ from deterministic dates.txt.")

        if not np.all(sub["n_pixels"].to_numpy(dtype=int) == n_land):
            raise RuntimeError(f"{method}: inconsistent land-pixel support.")

    summary = summarise(daily)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    daily_path = (deterministic_dir / "daily_spatial_correlation.csv")
    summary_path = (deterministic_dir / "spatial_correlation_summary.csv")
    metadata_path = (deterministic_dir / "spatial_correlation_metadata.json")

    daily.to_csv(daily_path, index=False,)
    summary.to_csv(summary_path, index=False,)

    metadata = {
        "description": ("Daily spatial Pearson correlation between deterministic precipitation products and DANRA over Danish land pixels."),
        "definition": ("For each evaluation date and method, Pearson r is calculated across the fixed Danish-land pixel set between the deterministic product and DANRA."),
        "aggregation_note": ("Correlations are calculated separately for each date. Pixel-days are not flattened into a single pooled correlation."),
        "sample_source": str(dates_file),
        "n_dates": len(dates),
        "n_land_pixels": (n_land),
        "methods": list(METHOD_ORDER),
        "subset": ("all_land"),
        "constant_tolerance": float(args.constant_tolerance),
        "undefined_rule": ("Correlation is NaN when DANRA or the evaluated deterministic field has spatial standard deviation <= constant_tolerance."),
        "nonfinite_rule": ("Non-finite values on Danish land raise an error rather than changing method-specific spatial support."),
        "ceddar_mean_definition": ("Pixelwise arithmetic mean across saved ensemble members."),
        "ceddar_median_definition": ("Pixelwise median across saved ensemble members."),
        "ceddar_pmm_definition": ("Saved probability-matched mean field."),
        "era5_definition": ("Saved bilinearly regridded ERA5 conditioning field on the DANRA grid."),
        "model": (args.model),
        "generation_root": str(generation_root),
        "ceddar_root": str(ceddar_root),
        "qm_root": str(qm_root),
        "ensemble_member_counts_seen": sorted(member_counts),
        "max_abs_hr_copy_difference": (max_hr_copy_difference),
        "max_abs_lr_copy_difference": (max_lr_copy_difference),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print()
    print("Saved")
    print("-----")
    print(daily_path)
    print(summary_path)
    print(metadata_path)

    print()
    print("Overall correlation summary")
    print("---------------------------")

    overall = summary.loc[summary["season"] == "ALL"].copy()

    print(
        overall[
            [
                "method",
                "n_dates",
                "n_valid",
                "n_undefined",
                "mean",
                "median",
                "p25",
                "p75",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
