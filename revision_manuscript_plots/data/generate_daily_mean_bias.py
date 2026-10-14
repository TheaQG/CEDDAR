from __future__ import annotations

"""
Generate daily land-domain mean precipitation biases for the revised SI figure.

The product is computed directly from the archived generated fields so that all
methods use exactly the same dates and Danish land pixels.

Methods
-------
- bilinear_era5
- qm
- ceddar_mean
- ceddar_median
- ceddar_pmm

All biases are relative to DANRA:

    bias = model domain mean - DANRA domain mean

Outputs
-------
<output-dir>/
    daily_mean_bias.csv
    daily_mean_bias_summary.csv
    metadata.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from revision_manuscript_plots.data import legacy


DEFAULT_MODEL = ("B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56")

METHODS = (
    "bilinear_era5",
    "qm",
    "ceddar_mean",
    "ceddar_median",
    "ceddar_pmm",
)


# =============================================================================
# Small loading helpers
# =============================================================================


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
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


def squeeze_field(field: np.ndarray, *, path: Path, ) -> np.ndarray:
    field = np.asarray(field, dtype=float,)
    field = np.squeeze(field)

    if field.ndim != 2:
        raise ValueError(f"{path}: expected 2D field after squeeze, got {field.shape}")

    return field


def load_ceddar_lr_hr(path: Path,) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_npz(path)

    lr = first_matching_key(arrays, ("lr", "lr_phys", "era5", "LR"), path=path,)
    hr = first_matching_key(arrays, ("hr", "hr_phys", "danra", "HR"), path=path,)

    return (squeeze_field(lr, path=path), squeeze_field(hr, path=path),)


def load_qm_lr_hr(path: Path,) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_npz(path)

    lr = first_matching_key(arrays, ("lr_hr", "lr", "lr_phys", "era5", "LR"), path=path,)
    hr = first_matching_key(arrays, ("hr", "hr_phys", "danra", "HR"), path=path,)

    return (squeeze_field(lr, path=path), squeeze_field(hr, path=path),)


def load_ensemble(path: Path,) -> np.ndarray:
    arrays = load_npz(path)

    ens = first_matching_key(arrays, ("ens",), path=path,)

    # [M,1,H,W] -> [M,H,W]
    if ens.ndim == 4 and ens.shape[1] == 1:
        ens = ens[:, 0]

    if ens.ndim != 3:
        raise ValueError(f"{path}: expected ensemble [M,H,W], got {ens.shape}")

    return np.asarray(ens, dtype=float,)


def load_pmm(path: Path,) -> np.ndarray:
    arrays = load_npz(path)
    field = first_matching_key(arrays, ("pmm", "gen", "generated"), path=path,)
    return squeeze_field(field, path=path,)


def load_qm(path: Path,) -> np.ndarray:
    arrays = load_npz(path)
    field = first_matching_key(arrays, ("pmm", "qm", "gen", "generated"), path=path,)
    return squeeze_field(field, path=path,)


def dates_in(directory: Path,) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    return {path.stem for path in directory.glob("*.npz")}


def season_from_date(date: str,) -> str:
    month = int(date[4:6])

    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def expected_dates_for_year(year: str,) -> list[str]:
    start = np.datetime64(f"{year}-01-01", "D",)
    stop = np.datetime64(f"{int(year) + 1}-01-01", "D",)

    dates = np.arange(start, stop, dtype="datetime64[D]",)

    return [str(date).replace("-", "") for date in dates]


def land_mean(
    field: np.ndarray,
    land: np.ndarray,
    *,
    name: str,
    date: str,
) -> float:
    field = np.asarray(field, dtype=float,)

    if field.shape != land.shape:
        raise ValueError(f"{date}: {name} shape {field.shape} != land mask {land.shape}")

    values = field[land]

    if not np.all(np.isfinite(values)):
        bad = int(np.sum(~np.isfinite(values)))
        raise RuntimeError(f"{date}: {name} has {bad} non-finite values on Danish land. Refusing method-dependent masking.")

    return float(values.mean())


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=("Generate daily land-domain mean precipitation biases for DANRA, ERA5, QM and CEDDAR summaries."))

    parser.add_argument("--generation-root", type=Path, required=True, help=("Root containing the final CEDDAR model directory and baselines/qm/test."),)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--year", default="2019")
    parser.add_argument("--qm-subdir", type=Path, default=Path("baselines/qm/test"),)
    parser.add_argument("--allow-incomplete-year", action="store_true", help=("Allow missing calendar dates. By default the selected year must be complete and gap-free."),)

    args = parser.parse_args()

    root = (args.generation_root.expanduser().resolve())
    output = (args.output_dir.expanduser().resolve())
    output.mkdir(parents=True, exist_ok=True,)

    year = str(args.year)

    # -------------------------------------------------------------------------
    # Source paths
    # -------------------------------------------------------------------------

    ceddar_root = (root / args.model)
    qm_root = (root / args.qm_subdir)

    ens_dir = (ceddar_root / "ensembles_phys")
    ceddar_lrhr_dir = (ceddar_root / "lr_hr_phys")
    pmm_dir = (ceddar_root / "pmm_phys")
    qm_lrhr_dir = (qm_root / "lr_hr")
    qm_dir = (qm_root / "pmm")

    source_dirs = {
        "ceddar ensemble": ens_dir,
        "ceddar lr/hr": ceddar_lrhr_dir,
        "ceddar pmm": pmm_dir,
        "qm lr/hr": qm_lrhr_dir,
        "qm": qm_dir,
    }

    for name, path in source_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{name}: {path}")

    # -------------------------------------------------------------------------
    # Exact common dates
    # -------------------------------------------------------------------------

    date_sets = {name: {date for date in dates_in(path) if date.startswith(year)} for name, path in source_dirs.items()}

    print("\nAvailable dates")
    print("---------------")

    for name, dates in date_sets.items():
        print(f"{name:20s}: {len(dates):4d}")
    common_dates = sorted(set.intersection(*date_sets.values()))

    if not common_dates:
        raise RuntimeError(f"No common dates found for {year}.")
    if not args.allow_incomplete_year:
        expected = expected_dates_for_year(year)

        if common_dates != expected:
            missing = sorted(set(expected) - set(common_dates))
            extra = sorted(set(common_dates) - set(expected))

            raise RuntimeError(
                f"{year} is not a complete common calendar year: found {len(common_dates)}, expected {len(expected)}; missing={len(missing)}, extra={len(extra)}. Use --allow-incomplete-year only if this is intentional.")

    print(f"\nCommon dates: {len(common_dates)}")
    print("First:", common_dates[:3])
    print("Last :", common_dates[-3:])

    # -------------------------------------------------------------------------
    # Static Danish land mask
    # -------------------------------------------------------------------------

    land = np.asarray(legacy.load_land_mask(ceddar_root), dtype=bool,)

    print("\nLand pixels:", int(land.sum()), "/", land.size)

    # -------------------------------------------------------------------------
    # Daily domain means and biases
    # -------------------------------------------------------------------------

    rows: list[dict] = []

    max_hr_copy_difference = 0.0
    max_lr_copy_difference = 0.0
    member_counts: set[int] = set()

    for i, date in enumerate(common_dates):
        lr, hr = load_ceddar_lr_hr(ceddar_lrhr_dir / f"{date}.npz")
        lr_qm, hr_qm = load_qm_lr_hr(qm_lrhr_dir / f"{date}.npz")
        ensemble = load_ensemble(ens_dir / f"{date}.npz")
        pmm = load_pmm(pmm_dir / f"{date}.npz")
        qm = load_qm(qm_dir / f"{date}.npz")

        member_counts.add(int(ensemble.shape[0]))

        if (lr.shape != land.shape or hr.shape != land.shape or lr_qm.shape != land.shape or hr_qm.shape != land.shape
            or pmm.shape != land.shape or qm.shape != land.shape or ensemble.shape[1:] != land.shape):
            raise ValueError(f"{date}: field/mask shape mismatch.")

        # -------------------------------------------------------------
        # Provenance check:
        # CEDDAR and QM evaluations should use the same LR/HR inputs.
        # -------------------------------------------------------------

        hr_diff = float(np.nanmax(np.abs(hr - hr_qm)))
        lr_diff = float(np.nanmax(np.abs(lr - lr_qm)))

        max_hr_copy_difference = max(max_hr_copy_difference, hr_diff,)
        max_lr_copy_difference = max(max_lr_copy_difference, lr_diff,)

        if (hr_diff > 1e-4 or lr_diff > 1e-4):
            raise RuntimeError(f"{date}: CEDDAR and QM copies of LR/HR do not agree: HR max Δ={hr_diff:.3e}, LR max Δ={lr_diff:.3e}")

        ensemble_mean = np.mean(ensemble, axis=0,)
        ensemble_median = np.median(ensemble, axis=0,)
        danra_mean = land_mean(hr, land, name="DANRA", date=date,)

        fields = {
            "bilinear_era5": lr,
            "qm": qm,
            "ceddar_mean": ensemble_mean,
            "ceddar_median": ensemble_median,
            "ceddar_pmm": pmm,
        }

        evaluation_day = (i + 1)
        for method in METHODS:
            model_mean = land_mean(fields[method], land, name=method, date=date,)

            rows.append({
                    "date": date,
                    "year": int(year),
                    "evaluation_day": evaluation_day,
                    "season": season_from_date(date),
                    "method": method,
                    "danra_mean_mm_day": danra_mean,
                    "model_mean_mm_day": model_mean,
                    "bias_mm_day": (model_mean - danra_mean),
                    "n_land_pixels": int(land.sum()),
                })

        if ((i + 1) % 50 == 0 or i == 0 or i + 1 == len(common_dates)):
            print(f"Processed {i + 1}/{len(common_dates)}")

    daily = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------

    summary_rows = []

    for method in METHODS:
        sub = (daily.loc[daily["method"] == method].sort_values("evaluation_day"))
        
        bias = sub["bias_mm_day"].to_numpy(float)
        model = sub["model_mean_mm_day"].to_numpy(float)
        danra = sub["danra_mean_mm_day"].to_numpy(float)

        if not (np.all( np.isfinite(bias)) and np.all(np.isfinite(model)) and np.all(np.isfinite(danra))):
            raise RuntimeError(f"{method}: non-finite daily mean/bias after common-sample evaluation.")

        model_accumulation = float(model.sum())
        danra_accumulation = float(danra.sum())

        accumulation_bias = (model_accumulation - danra_accumulation)
        relative_accumulation_bias = (100.0 * accumulation_bias / danra_accumulation)

        summary_rows.append(
            {
                "method": method,
                "n_days": len(sub),
                "mean_bias_mm_day": float(np.mean(bias)),
                "median_bias_mm_day": float(np.median(bias)),
                "std_bias_mm_day": float(np.std(bias, ddof=1)),
                "mae_mm_day": float(np.mean(np.abs(bias))),
                "rmse_mm_day": float(np.sqrt(np.mean(bias ** 2))),
                "danra_accumulation_mm": (danra_accumulation),
                "model_accumulation_mm": (model_accumulation),
                "accumulation_bias_mm": (accumulation_bias),
                "relative_accumulation_bias_pct": (relative_accumulation_bias),
            }
        )

    summary = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------------
    # Save products
    # -------------------------------------------------------------------------

    daily_path = (output / "daily_mean_bias.csv")
    summary_path = (output / "daily_mean_bias_summary.csv")
    metadata_path = (output / "metadata.json")

    daily.to_csv(daily_path, index=False,)
    summary.to_csv(summary_path, index=False,)

    metadata = {
        "description": ("Daily Danish-land domain-mean precipitation bias relative to DANRA."),
        "bias_definition": ("model domain mean - DANRA domain mean"),
        "year": int(year),
        "n_common_dates": len(common_dates),
        "date_first": common_dates[0],
        "date_last": common_dates[-1],
        "complete_calendar_year_required": (not args.allow_incomplete_year),
        "methods": list(METHODS),
        "model": args.model,
        "generation_root": str(root),
        "ceddar_root": str(ceddar_root),
        "qm_root": str(qm_root),
        "n_land_pixels": int(land.sum()),
        "ensemble_member_counts_seen": sorted(member_counts),
        "max_abs_hr_copy_difference": (max_hr_copy_difference),
        "max_abs_lr_copy_difference": (max_lr_copy_difference),
        "sample_policy": ("All methods use the exact intersection of archived CEDDAR and QM source dates and the same static Danish land mask. Non-finite method values on land raise an error rather than changing the averaging mask."),
        "ceddar_mean_definition": ("Pixelwise arithmetic mean across ensemble members, followed by Danish-land domain mean."),
        "ceddar_median_definition": ("Pixelwise median across ensemble members, followed by Danish-land domain mean."),
        "ceddar_pmm_definition": ("Saved probability-matched mean field, followed by Danish-land domain mean."),
        "common_dates": (common_dates),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2,))

    print("\nSaved")
    print("-----")
    print(daily_path)
    print(summary_path)
    print(metadata_path)

    print("\nSummary")
    print("-------")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
