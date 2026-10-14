from __future__ import annotations

"""
Regenerate ISS on a method-independent set of observed-event days.

The evaluation sample is defined from DANRA only:

    eligible(date, threshold) =
        at least one valid Danish land grid cell has DANRA > threshold

For every eligible date, ISS is then computed for exactly the same four methods:
    - CEDDAR member-mean ISS (ISS per member, then mean across members)
    - CEDDAR PMM
    - bilinearly regridded ERA5
    - quantile mapping (QM)

This avoids the legacy behaviour where each method's summary could be averaged
over a different subset of dates because undefined ISS values were silently
excluded method by method.

Outputs
-------
<output-dir>/
    iss_daily.csv
    iss_summary.csv
    iss_event_days.csv
    iss_threshold_counts.csv
    metadata.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from revision_manuscript_plots.data import legacy
from sbgm.evaluate.evaluate_prcp.eval_scale.metrics_scale import compute_iss_at_scales


DEFAULT_MODEL = (
    "B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56"
)

METHODS = (
    "ceddar_members",
    "ceddar_pmm",
    "bilinear_era5",
    "qm",
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
            return np.asarray(arrays[key], dtype=float)

    raise KeyError(f"{path}: none of {candidates} found. Available keys: {list(arrays)}")


def squeeze_field(field: np.ndarray, *, path: Path, ) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    field = np.squeeze(field)

    if field.ndim != 2:
        raise ValueError(f"{path}: expected 2D field after squeeze, got {field.shape}")

    return field


def load_ceddar_lr_hr(path: Path,) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_npz(path)

    lr = first_matching_key(arrays, ("lr", "lr_phys", "era5", "LR"), path=path,)
    hr = first_matching_key(arrays, ("hr", "hr_phys", "danra", "HR"), path=path,)

    return (squeeze_field(lr, path=path), squeeze_field(hr, path=path),)


def load_ceddar_pmm(path: Path,) -> np.ndarray:
    arrays = load_npz(path)

    field = first_matching_key(arrays, ("pmm", "gen", "generated"), path=path,)

    return squeeze_field(field, path=path,)


def load_ceddar_ensemble(path: Path,) -> np.ndarray:
    arrays = load_npz(path)

    ens = first_matching_key(arrays, ("ens",), path=path,)

    # [M,1,H,W] -> [M,H,W]
    if ens.ndim == 4 and ens.shape[1] == 1:
        ens = ens[:, 0]

    if ens.ndim != 3:
        raise ValueError(f"{path}: expected ensemble [M,H,W], got {ens.shape}")

    return np.asarray(ens, dtype=float,)


def load_qm(path: Path,) -> np.ndarray:
    arrays = load_npz(path)

    field = first_matching_key(arrays, ("pmm", "qm", "gen", "generated"), path=path,)

    return squeeze_field(field, path=path,)


def dates_in(directory: Path,) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    return {path.stem for path in directory.glob("*.npz")}


# =============================================================================
# ISS implementation
# =============================================================================


@torch.no_grad()
def compute_iss_batch_at_scales(
    forecasts: np.ndarray,
    observation: np.ndarray,
    *,
    mask: np.ndarray,
    grid_km_per_px: float,
    scales_km: tuple[float, ...],
    threshold_mm: float,
    device: torch.device,
    eps: float = 1e-8,
    min_mse_rand: float = 1e-6,
) -> dict[float, np.ndarray]:
    """
    Vectorised form of the repository's
    compute_iss_at_scales().

    Parameters
    ----------
    forecasts
        [B,H,W] or [H,W].
    observation
        [H,W].
    mask
        [H,W] boolean evaluation mask.

    Returns
    -------
    dict[scale_km, array[B]]
        One ISS value per forecast field.

    Notes
    -----
    The equations intentionally match
    sbgm.evaluate.evaluate_prcp.eval_scale.metrics_scale.
    compute_iss_at_scales.

    Reduction is performed separately for each batch
    element so that an entire CEDDAR ensemble can be
    processed efficiently without changing the metric.
    """
    fc = np.asarray(forecasts, dtype=np.float32,)
    obs = np.asarray(observation, dtype=np.float32,)
    land = np.asarray(mask, dtype=bool,)

    if fc.ndim == 2:
        fc = fc[None, ...]
    if fc.ndim != 3:
        raise ValueError(f"Expected forecasts [B,H,W] or [H,W], got {fc.shape}")
    if obs.ndim != 2:
        raise ValueError(f"Expected observation [H,W], got {obs.shape}")
    if (land.shape != obs.shape or fc.shape[1:] != obs.shape):
        raise ValueError(f"Shape mismatch: forecasts={fc.shape}, observation={obs.shape}, mask={land.shape}")

    X = torch.as_tensor(fc, dtype=torch.float32, device=device,)[:, None]
    Y = torch.as_tensor(obs, dtype=torch.float32, device=device,)[None, None]
    M = torch.as_tensor(land, dtype=torch.float32, device=device,)[None, None]

    X = (X > float(threshold_mm)).float()
    Y = (Y > float(threshold_mm)).float()

    msum = M.sum().clamp_min(1.0)

    result: dict[float, np.ndarray] = {}

    for scale in scales_km:
        rad_px = int(max(1, round(float(scale) / float(grid_km_per_px))))
        kernel = 2 * rad_px + 1
        area = float(kernel * kernel)

        Xm = (F.avg_pool2d(X * M, kernel_size=kernel, stride=1, padding=rad_px,) * area)
        Ym = (F.avg_pool2d(Y * M, kernel_size=kernel, stride=1, padding=rad_px,) * area)

        Mm = (F.avg_pool2d(M, kernel_size=kernel, stride=1, padding=rad_px,) * area)

        Xs = Xm / (Mm + eps)
        Ys = Ym / (Mm + eps)

        # Separate reduction for every forecast field
        # in the batch.
        mse_mod = ((((Xs - Ys) ** 2) * M).sum(dim=(-1, -2, -3)) / msum)
        p_f = ((Xs * M).sum(dim=(-1, -2, -3)) / msum)
        p_o = ((Ys * M).sum(dim=(-1, -2, -3)) / msum)

        mse_rand = (p_f + p_o - 2.0 * p_f * p_o)

        iss = (1.0 - mse_mod / (mse_rand + eps))
        iss = torch.minimum(iss, torch.ones_like(iss),)
        
        bad = (~torch.isfinite(mse_rand) | (mse_rand < min_mse_rand))
        iss = torch.where(bad, torch.full_like(iss, torch.nan,), iss,)

        result[float(scale)] = (iss.detach().cpu().numpy().astype(float))
    return result


def validate_batch_implementation(
    member: np.ndarray,
    observation: np.ndarray,
    *,
    mask: np.ndarray,
    threshold_mm: float,
    scales_km: tuple[float, ...],
    grid_km_per_px: float,
) -> None:
    """
    Compare the vectorised implementation against
    the repository metric for one forecast field
    before processing the full dataset.
    """
    batched = compute_iss_batch_at_scales(
        member,
        observation,
        mask=mask,
        grid_km_per_px=grid_km_per_px,
        scales_km=scales_km,
        threshold_mm=threshold_mm,
        device=torch.device("cpu"),
    )

    core = compute_iss_at_scales(
        gen_bt=torch.as_tensor(member, dtype=torch.float32,)[None, None],
        hr_bt=torch.as_tensor(observation, dtype=torch.float32,)[None, None],
        mask=torch.as_tensor(mask, dtype=torch.bool,)[None, None],
        grid_km_per_px=grid_km_per_px,
        iss_scales_km=list(scales_km),
        thr_mm=threshold_mm,
    )

    for scale in scales_km:
        batch_value = float(batched[float(scale)][0])
        core_value = float(core[f"{int(scale)}km"])

        if not np.isclose(batch_value, core_value, atol=2e-6, rtol=1e-6, equal_nan=True):
            raise RuntimeError("Vectorised ISS implementation does not match repository metric: "
                f"threshold={threshold_mm}, scale={scale}, batch={batch_value}, core={core_value}")


# =============================================================================
# Device and summary helpers
# =============================================================================


def resolve_device(name: str,) -> torch.device:
    name = name.lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("mps")

        return torch.device("cpu")

    device = torch.device(name)

    if (device.type == "cuda" and not torch.cuda.is_available()):
        raise RuntimeError("CUDA requested but is not available")

    if device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but is not available")

    return device


def summarise_daily(daily: pd.DataFrame, *, expected_counts: dict[float, int],) -> pd.DataFrame:
    rows = []

    grouped = daily.groupby(["method", "threshold_mm", "scale_km",], sort=True,)

    for (method, threshold, scale,), group in grouped:
        values = group["iss"].to_numpy(float)
        values = values[np.isfinite(values)]

        expected = expected_counts[float(threshold)] # type: ignore

        if len(values) != expected:
            raise RuntimeError("Method-dependent sample size detected after observed-event selection: "
                f"method={method}, threshold={threshold}, scale={scale}, n={len(values)}, expected={expected}")

        mean = float(values.mean())

        if len(values) >= 2:
            std = float(values.std(ddof=1))
            sem = float(std / np.sqrt(len(values)))
        else:
            std = np.nan
            sem = np.nan

        rows.append(
            {
                "method": method,
                "threshold_mm": float(threshold), # type: ignore
                "scale_km": float(scale), # type: ignore
                "n_days": int(len(values)),
                "iss_mean": mean,
                "iss_std": std,
                "iss_sem": sem,
            }
        )

    return pd.DataFrame(rows)


@torch.no_grad()
def observed_iss_support(
    observation: np.ndarray,
    *,
    mask: np.ndarray,
    grid_km_per_px: float,
    scales_km: tuple[float, ...],
    threshold_mm: float,
    min_mse_rand: float = 1e-6,
) -> tuple[bool, dict[float, float]]:
    """
    Determine whether DANRA provides a non-degenerate ISS reference.
    Selection depends only on the observation and land mask.

    For a fixed observed exceedance frequency p_o,
        MSE_rand = p_f + p_o - 2 p_f p_o.
    Across all possible forecasts p_f in [0, 1], its minimum is
        min(p_o, 1 - p_o).
    Therefore requiring
        min(p_o, 1 - p_o) >= min_mse_rand
    guarantees a non-degenerate ISS denominator for every forecast
    method/member, independently of its predictions.
    """
    obs = np.asarray(observation, dtype=np.float32,)
    land = np.asarray(mask, dtype=bool,)

    Y = torch.as_tensor(obs, dtype=torch.float32,)[None, None]
    M = torch.as_tensor(land, dtype=torch.float32,)[None, None]

    Y = (Y > np.float32(threshold_mm)).float()

    msum = M.sum().clamp_min(1.0)

    p_obs = {}
    eligible = True

    for scale in scales_km:
        rad_px = int(max(1, round(float(scale) / float(grid_km_per_px))))

        kernel = 2 * rad_px + 1
        area = float(kernel * kernel)

        Ym = (F.avg_pool2d(Y * M, kernel_size=kernel, stride=1, padding=rad_px,) * area)
        Mm = (F.avg_pool2d(M, kernel_size=kernel, stride=1, padding=rad_px,) * area)

        Ys = Ym / (Mm + 1e-8)

        po = float(((Ys * M).sum() / msum).item())
        p_obs[float(scale)] = po

        support = min(po, 1.0 - po,)

        if (not np.isfinite(po) or support < min_mse_rand):
            eligible = False

    return eligible, p_obs

# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=("Regenerate ISS on DANRA-defined observed-event days so every method is "
            "evaluated on the same, method-independent sample."))

    parser.add_argument("--generation-root", type=Path, required=True, help=("Root containing the final CEDDAR model directory and baselines/qm/test."),)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL,)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(1.0, 5.0, 10.0, 20.0,),)
    parser.add_argument("--scales", type=float, nargs="+", default=(5.0, 10.0, 20.0, 40.0,),)
    parser.add_argument("--hr-dx-km", type=float, default=2.5,)
    parser.add_argument("--qm-subdir", type=Path, default=Path("baselines/qm/test"), help=("QM path relative to --generation-root."),)
    parser.add_argument("--device", default="cpu", help=("Torch device: cpu, mps, cuda, or auto."),)
    parser.add_argument("--max-dates", type=int, default=None, help=("Optional small subset for testing the pipeline."),)

    args = parser.parse_args()

    generation_root = (args.generation_root.expanduser().resolve())
    output_dir = (args.output_dir.expanduser().resolve())

    output_dir.mkdir(parents=True,exist_ok=True,)

    thresholds = tuple(sorted(float(x) for x in args.thresholds))
    scales = tuple(sorted(float(x) for x in args.scales))
    device = resolve_device(args.device)

    # -------------------------------------------------------------------------
    # Source paths
    # -------------------------------------------------------------------------

    ceddar_root = (generation_root / args.model)
    qm_root = (generation_root / args.qm_subdir)
    ens_dir = (ceddar_root / "ensembles_phys")
    pmm_dir = (ceddar_root / "pmm_phys")
    lrhr_dir = (ceddar_root / "lr_hr_phys")
    qm_dir = (qm_root / "pmm")

    source_dirs = {
        "ceddar_ensemble": ens_dir,
        "ceddar_pmm": pmm_dir,
        "ceddar_lr_hr": lrhr_dir,
        "qm": qm_dir,
    }

    for name, path in source_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{name}: {path}")

    # -------------------------------------------------------------------------
    # Common source dates
    # -------------------------------------------------------------------------

    date_sets = {name: dates_in(path) for name, path in source_dirs.items()}

    print("\nAvailable dates")
    print("---------------")

    for name, dates in date_sets.items():
        print(f"{name:20s}: {len(dates):4d}")

    common_dates = sorted(set.intersection(*date_sets.values()))

    if args.max_dates is not None:
        common_dates = common_dates[:args.max_dates]

    if not common_dates:
        raise RuntimeError("No common dates across CEDDAR and QM source files.")

    print(f"\nCommon source dates: {len(common_dates)}")
    print("First:", common_dates[:3])
    print("Last :", common_dates[-3:])
    print("Device:", device)

    # -------------------------------------------------------------------------
    # Static land mask
    # -------------------------------------------------------------------------

    land = np.asarray(legacy.load_land_mask(ceddar_root), dtype=bool,)

    print("\nLand pixels:", int(land.sum()), "/", land.size)

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    daily_rows: list[dict] = []
    event_rows: list[dict] = []

    eligible_counts = {float(threshold): 0 for threshold in thresholds}

    core_validated = False
    n_members_seen: set[int] = set()

    for i, date in enumerate(common_dates):
        lr, hr = load_ceddar_lr_hr(lrhr_dir / f"{date}.npz")
        pmm = load_ceddar_pmm(pmm_dir / f"{date}.npz")
        ensemble = load_ceddar_ensemble(ens_dir / f"{date}.npz")
        qm = load_qm(qm_dir / f"{date}.npz")

        hr_eval = np.asarray(hr, dtype=np.float32)
        pmm = np.asarray(pmm, dtype=np.float32)
        lr = np.asarray(lr, dtype=np.float32)
        qm = np.asarray(qm, dtype=np.float32)
        ensemble = np.asarray(ensemble, dtype=np.float32)

        n_members_seen.add(int(ensemble.shape[0]))

        expected_shape = land.shape

        fields = {
            "hr": hr,
            "bilinear_era5": lr,
            "ceddar_pmm": pmm,
            "qm": qm,
        }

        for name, field in fields.items():
            if field.shape != expected_shape:
                raise ValueError(f"{date}: {name} shape {field.shape} != mask shape {expected_shape}")

        if (ensemble.shape[1:] != expected_shape):
            raise ValueError(f"{date}: ensemble shape {ensemble.shape} incompatible with mask {expected_shape}")

        valid_land = (land & np.isfinite(hr))

        if not valid_land.any():
            raise RuntimeError(f"{date}: no finite DANRA land pixels")

        # A common evaluation mask must not depend
        # on method. Therefore method-side non-finite
        # values are treated as errors rather than
        # silently shrinking the mask.

        if not np.all(np.isfinite(pmm[valid_land])):
            raise RuntimeError(f"{date}: non-finite CEDDAR PMM values on valid land")
        if not np.all(np.isfinite(lr[valid_land])):
            raise RuntimeError(f"{date}: non-finite bilinear ERA5 values on valid land")
        if not np.all(np.isfinite(qm[valid_land])):
            raise RuntimeError(f"{date}: non-finite QM values on valid land")
        if not np.all(np.isfinite(ensemble[:, valid_land])):
            raise RuntimeError(f"{date}: non-finite CEDDAR ensemble values on valid land")

        n_land = int(valid_land.sum())

        for threshold in thresholds:
            event = (valid_land & (hr_eval > np.float32(threshold)))
            event_pixels = int(event.sum())

            event_fraction = (event_pixels / n_land)

            eligible, p_obs_by_scale = (
                observed_iss_support(
                    hr_eval,
                    mask=valid_land,
                    grid_km_per_px=float(args.hr_dx_km),
                    scales_km=scales,
                    threshold_mm=float(threshold),
                )
            )
            event_fraction = (event_pixels / n_land)
            event_row = {
                "date": date,
                "threshold_mm": float(threshold),
                "eligible": bool(eligible),
                "danra_event_pixels": (event_pixels),
                "danra_event_fraction": (event_fraction),
                "n_valid_land_pixels": (n_land),
            }
            for scale, po in p_obs_by_scale.items():
                event_row[f"danra_p_obs_{int(scale)}km"] = po
            event_rows.append(event_row)

            if not eligible:
                continue

            eligible_counts[float(threshold)] += 1

            # -------------------------------------------------------------
            # Validate vectorised metric once
            # against repository implementation.
            # -------------------------------------------------------------

            if not core_validated:
                validate_batch_implementation(
                    ensemble[0],
                    hr,
                    mask=valid_land,
                    threshold_mm=float(threshold),
                    scales_km=scales,
                    grid_km_per_px=float(args.hr_dx_km),
                )

                core_validated = True

                print("\nVectorised ISS validation against repository metric: PASSED")

            # -------------------------------------------------------------
            # Compute all methods on exactly
            # the same observed-event date.
            # -------------------------------------------------------------

            member_iss = (
                compute_iss_batch_at_scales(
                    ensemble,
                    hr,
                    mask=valid_land,
                    grid_km_per_px=float(args.hr_dx_km),
                    scales_km=scales,
                    threshold_mm=float(threshold),
                    device=device,
                )
            )

            pmm_iss = (
                compute_iss_batch_at_scales(
                    pmm,
                    hr,
                    mask=valid_land,
                    grid_km_per_px=float(args.hr_dx_km),
                    scales_km=scales,
                    threshold_mm=float(threshold),
                    device=device,
                )
            )

            lr_iss = (
                compute_iss_batch_at_scales(
                    lr,
                    hr,
                    mask=valid_land,
                    grid_km_per_px=float(args.hr_dx_km),
                    scales_km=scales,
                    threshold_mm=float(threshold),
                    device=device,
                )
            )

            qm_iss = (
                compute_iss_batch_at_scales(
                    qm,
                    hr,
                    mask=valid_land,
                    grid_km_per_px=float(args.hr_dx_km),
                    scales_km=scales,
                    threshold_mm=float(threshold),
                    device=device,
                )
            )

            for scale in scales:
                members_at_scale = np.asarray(member_iss[float(scale)], dtype=float,)

                # With an observed threshold exceedance,
                # undefined ISS should not occur.
                # Fail rather than silently use
                # method-specific N.
                if not np.all(np.isfinite(members_at_scale)):
                    bad = int(np.sum(~np.isfinite(members_at_scale)))

                    raise RuntimeError(
                        f"{date}, threshold={threshold:g}, scale={scale:g}: {bad}/{len(members_at_scale)} CEDDAR members have undefined ISS on an observed-event day.")

                values = {
                    "ceddar_members": float(members_at_scale.mean()),
                    "ceddar_pmm": float(pmm_iss[float(scale)][0]),
                    "bilinear_era5": float(lr_iss[float(scale)][0]),
                    "qm": float(qm_iss[float(scale)][0]),
                }

                for (method, value,) in values.items():
                    if not np.isfinite(value):
                        raise RuntimeError(
                            f"{date}, threshold={threshold:g}, scale={scale:g}, method={method}: undefined ISS on an observed-event day.")

                    daily_rows.append({
                            "date": date,
                            "threshold_mm": float(threshold),
                            "scale_km": float(scale),
                            "method": method,
                            "iss": value,
                            "danra_event_pixels": (event_pixels),
                            "danra_event_fraction": (event_fraction),
                            "n_valid_land_pixels": (n_land),
                            "n_members": (int(ensemble.shape[0]) if method == "ceddar_members" else 1),
                        })

        if ((i + 1) % 25 == 0 or i == 0 or i + 1 == len(common_dates)):
            counts_text = ", ".join(f">{thr:g}: " f"{eligible_counts[thr]}" for thr in thresholds)
            print(f"Processed {i + 1}/{len(common_dates)} | eligible so far: {counts_text}")

    if not daily_rows:
        raise RuntimeError("No eligible observed-event cases were produced.")

    # -------------------------------------------------------------------------
    # Save products
    # -------------------------------------------------------------------------

    daily = pd.DataFrame(daily_rows)
    events = pd.DataFrame(event_rows)
    summary = summarise_daily(daily, expected_counts=eligible_counts,)
    counts = pd.DataFrame([
        {
                "threshold_mm": threshold,
                "n_common_source_dates": (len(common_dates)),
                "n_eligible_observed_event_days": (eligible_counts[threshold]),}
            for threshold in thresholds
        ])

    daily_path = (output_dir / "iss_daily.csv")
    summary_path = (output_dir / "iss_summary.csv")
    events_path = (output_dir / "iss_event_days.csv")
    counts_path = (output_dir / "iss_threshold_counts.csv")

    daily.to_csv(daily_path, index=False,)
    summary.to_csv(summary_path, index=False,)
    events.to_csv(events_path, index=False,)
    counts.to_csv(counts_path, index=False,)

    metadata = {
        "description": ("ISS recomputed on a method-independent DANRA-defined observed-event sample."),
        "eligibility_rule": ("A date is included for a precipitation threshold only when the DANRA threshold-exceedance field yields a non-degenerate ISS random-reference denominator at every evaluated spatial scale. Selection depends only on DANRA and the land mask."),
        "iss_degeneracy_threshold": 1e-6,
        "threshold_operator": ">",
        "thresholds_mm_day": list(thresholds),
        "scales_km": list(scales),
        "hr_dx_km": float(args.hr_dx_km),
        "methods": {
            "ceddar_members": ("ISS computed separately for every CEDDAR ensemble member, then averaged across members within date."),
            "ceddar_pmm": ("ISS of CEDDAR probability-matched mean."),
            "bilinear_era5": ("ISS of ERA5 bilinearly regridded to the DANRA grid."),
            "qm": ("ISS of the per-pixel quantile-mapped ERA5 baseline."),
        },
        "generation_root": str(generation_root),
        "ceddar_root": str(ceddar_root),
        "qm_root": str(qm_root),
        "model": args.model,
        "device": str(device),
        "n_common_source_dates": len(common_dates),
        "eligible_days_by_threshold": {str(threshold): count for (threshold, count,) in eligible_counts.items()},
        "ensemble_member_counts_seen": sorted(n_members_seen),
        "common_source_date_first": (common_dates[0]),
        "common_source_date_last": (common_dates[-1]),
        "common_source_dates": (common_dates),
        "core_metric_validation": ("Vectorised implementation checked against compute_iss_at_scales on the first eligible case."),
        "sample_size_policy": ("No method-specific finite-value filtering is allowed after DANRA event selection. Any undefined ISS on an eligible date raises an error."),
    }

    metadata_path = (output_dir / "metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2,))

    print("\nSaved")
    print("-----")
    print(daily_path)
    print(summary_path)
    print(events_path)
    print(counts_path)
    print(metadata_path)

    print("\nEligible observed-event days")
    print("----------------------------")

    for threshold in thresholds:
        print(f">{threshold:g} mm/day: {eligible_counts[threshold]}")


if __name__ == "__main__":
    main()