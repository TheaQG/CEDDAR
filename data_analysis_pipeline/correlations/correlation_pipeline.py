"""
Module for running data correlation analysis.
For each HR-LR pair:
1. Instantiate DataLoader with correct variable, model, domain
2. Load aligned data (shared dates)
3. Apply transformations (e.g., detrending, normalization)
4. Compute correlations:
    - Temporal: over time for spatial means (and optionally for each grid point or single grid point)
    - Spatial: per-pixel correlations over time
5. Save results:
    - Correlation maps (.npy or .zarr)
    - Summary statistics (mean, median, std) (.npy or .json)
    - Visualizations (heatmaps, time series plots) (.png)
"""

import os
import logging
import os
import math
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from data_analysis_pipeline.stats_analysis.data_loading import DataLoader
from data_analysis_pipeline.correlations.correlation_methods import compute_temporal_correlation, compute_spatial_correlation, compute_temporal_corr_series_np
from data_analysis_pipeline.correlations.correlation_plotting import (plot_correlation_map, plot_temporal_series, plot_spatial_corr_grid, plot_temporal_grid, plot_temporal_pair, plot_temporal_correlations_grid)
from data_analysis_pipeline.stats_analysis.statistics import load_global_stats
from scor_dm.special_transforms import transform_from_stats
# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _remove_seasonality_field_dict(field_dict, dates, method="monthly"):
    """
    Given a dict {date: 2D array} and aligned list of dates, subtracts a seasonal
    cycle per pixel. method in {"monthly", "doy"}.
    Returns a NEW dict with anomalies.
    """
    if method not in ("monthly", "doy", None):
        raise ValueError(f"Unknown seasonality method: {method}")
    if method is None:
        return dict(field_dict)  # no-op copy

    # stack to (T,H,W)
    ordered = [field_dict[d] for d in dates]
    arr = np.stack([np.asarray(a, dtype=float) for a in ordered], axis=0)
    T, H, W = arr.shape

    if method == "monthly":
        keys = np.array([d.month for d in dates])
        key_space = range(1, 13)
    else:  # 'doy'
        # map Feb29 -> Feb28 (or 59th day in non-leap indexing)
        keys = np.array([ (d.timetuple().tm_yday if d.timetuple().tm_yday != 366 else 365) for d in dates ])
        key_space = range(1, 366)

    # subtract per-key climatology
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        if idx.size == 0:
            continue
        mu = np.nanmean(arr[idx, :, :], axis=0)
        arr[idx, :, :] = arr[idx, :, :] - mu

    # rebuild dict
    out = {}
    for i, d in enumerate(dates):
        out[d] = arr[i]
    return out



def run_data_correlations(cfg):
    """
        Main correlation analysis entry point.
        Loops over HR x LR variable paris and runs selected correlation methods.
    """

    # === Load config sections ===
    hr_cfg = cfg["highres"]
    lr_cfg = cfg["lowres"]
    data_cfg = cfg["data"]
    corr_cfg = cfg["correlation"]
    plot_cfg = cfg["plotting"]
        
    correlation_types = corr_cfg.get("analysis_types", ["temporal", "spatial"])
    split = data_cfg.get("split", "all")
    logger.info(f"Data split: {split}")
    method = corr_cfg.get("method", "pearson")

    # === Extract config entries ===
    figs_save_dir = cfg.get("paths", {}).get("output_dir") or plot_cfg.get("fig_save_dir", "./correlation_outputs")
    stats_save_dir = cfg.get("paths", {}).get("stats_save_dir", "./correlation_stats")
    stats_load_dir = cfg.get("paths", {}).get("stats_load_dir", stats_save_dir)
    if split is not None:
        # Append split to save path if not "all"
        figs_save_dir = os.path.join(figs_save_dir, split) if split != "all" else figs_save_dir
        stats_save_dir = os.path.join(stats_save_dir, split) if split != "all" else stats_save_dir
        stats_load_dir = os.path.join(stats_load_dir, split) if split != "all" else stats_load_dir
    os.makedirs(figs_save_dir, exist_ok=True)
    os.makedirs(stats_save_dir, exist_ok=True)

    # Normalization
    normalize = corr_cfg.get("normalize", False)
    if not os.path.exists(stats_load_dir):
        logger.warning(f"Stats load directory '{stats_load_dir}' does not exist. Normalization will be skipped.")
        normalize = False

    # Plotting/calc options
    temporal_remove_seasonality = corr_cfg.get("temporal_remove_seasonality", None)  # None | 'monthly' | 'doy'
    temporal_aggregate = corr_cfg.get("temporal_aggregate", None)  # None | 'weekly' | 'monthly'
    temporal_aggregate_how = corr_cfg.get("temporal_aggregate_how", "mean")  # 'mean'|'sum'
    temporal_grid = corr_cfg.get("temporal_grid_figure", False)
    temporal_grid_ncols = int(corr_cfg.get("temporal_ncols", 2))

    spatial_grid = corr_cfg.get("spatial_grid_figure", False)
    spatial_grid_ncols = int(corr_cfg.get("spatial_ncols", 2))
    spatial_remove_seasonality = corr_cfg.get("spatial_remove_seasonality", None)  # None|'monthly'|'doy'


    hr_vars = hr_cfg.get("variables", [])
    domain_size_hr = hr_cfg.get("domain_size", [])
    domain_size_hr_str = "x".join(map(str, domain_size_hr)) if domain_size_hr else "full"
    crop_region_hr = hr_cfg.get("crop_region", [])
    crop_region_hr_str = "_".join(map(str, crop_region_hr)) if crop_region_hr else "full"
    model_hr = hr_cfg.get("model", "")

    lr_vars = lr_cfg.get("condition_variables", [])
    domain_size_lr = lr_cfg.get("domain_size", [])
    domain_size_lr_str = "x".join(map(str, domain_size_lr)) if domain_size_lr else "full"
    crop_region_lr = lr_cfg.get("crop_region", [])
    crop_region_lr_str = "_".join(map(str, crop_region_lr)) if crop_region_lr else "full"
    model_lr = lr_cfg.get("model", "")

    # dictionary of possible transformations to apply to data before correlation
    transformations_dict = corr_cfg.get("transformations", {})

    data_dir = data_cfg.get("data_dir", ".")
    n_workers = int(data_cfg.get("n_workers", 4) or 4)
    verbose = data_cfg.get("verbose", False)

    for hr_var in hr_vars:
        logger.info(f"\n\n\n  ############## Running correlation analysis for HR variable '{hr_var}' ({model_hr}) ##############\n\n")
        logger.info(f"  N workers: {n_workers}")

        # === Load data ===
        hr_loader = DataLoader(
            base_dir=data_dir,
            n_workers=n_workers,
            variable=hr_var,
            model=model_hr,
            domain_size=domain_size_hr,
            split=split,
            crop_region=crop_region_hr,
            verbose=verbose,
        )

        hr_data = hr_loader.load()

        # Accumulators for grid figures
        spatial_maps_by_lr = {}
        temporal_series_by_lr = {}  # lr_var -> (hr_series, lr_series, dates)

        # Loop over LR variables
        for lr_var in lr_vars:
            logger.info(f"  -- LR variable '{lr_var}' ({model_lr}) --")

            # Load LR for this variable
            lr_loader = DataLoader(
                base_dir=data_dir,
                n_workers=n_workers,
                variable=lr_var,
                model=model_lr,
                domain_size=domain_size_lr,
                split=split,
                crop_region=crop_region_lr,
                verbose=verbose,
            )
            lr_data = lr_loader.load()

            # Align datasets on shared dates
            shared_dates = sorted(set(hr_data['timestamps']) & set(lr_data['timestamps']))
            logger.info(f"  Found {len(shared_dates)} shared dates between HR '{hr_var}' and LR '{lr_var}'")

            # Build date->cutout dicts
            hr_dict = {d: c if isinstance(c, np.ndarray) else np.asarray(c) for d, c in zip(hr_data['timestamps'], hr_data['cutouts']) if d in shared_dates}
            lr_dict = {d: c if isinstance(c, np.ndarray) else np.asarray(c) for d, c in zip(lr_data['timestamps'], lr_data['cutouts']) if d in shared_dates}

            # === Optional normalization (per variable transform) ===
            save_str_add = ""
            if normalize:
                try:
                    hr_transform_method = transformations_dict.get(hr_var, "zscore")
                    lr_transform_method = transformations_dict.get(lr_var, "zscore")
                    save_str_add = f"_transformed_{hr_transform_method}_and_{lr_transform_method}"
                    logger.info("          Applying normalization based on global stats")
                    hr_stats_dict = load_global_stats(
                        hr_var, model_hr, domain_size_hr_str, crop_region_hr_str, split=split, dir_load=stats_load_dir
                    )
                    lr_stats_dict = load_global_stats(
                        lr_var, model_lr, domain_size_lr_str, crop_region_lr_str, split=split, dir_load=stats_load_dir
                    )
                    logger.info(f"          Using transformation '{hr_transform_method}' for HR variable")
                    logger.info(f"          Using transformation '{lr_transform_method}' for LR variable")

                    for d in shared_dates:
                        hr_dict[d] = transform_from_stats(data=hr_dict[d], transform_type=hr_transform_method, cfg=cfg, stats=hr_stats_dict)  # type: ignore
                        if not isinstance(hr_dict[d], np.ndarray):
                            hr_dict[d] = np.array(hr_dict[d])
                        lr_dict[d] = transform_from_stats(data=lr_dict[d], transform_type=lr_transform_method, cfg=cfg, stats=lr_stats_dict)  # type: ignore
                        if not isinstance(lr_dict[d], np.ndarray):
                            lr_dict[d] = np.array(lr_dict[d])
                except Exception as e:
                    logger.warning(f"          Failed to load/apply global stats for normalization: {e}. Skipping normalization for this pair.")
                    save_str_add = ""

            # ========== TEMPORAL ==========
            if "temporal" in correlation_types:

                # Build series with/without deseasonalization and a monthly aggregate
                series_np = compute_temporal_corr_series_np(
                    hr_dict, lr_dict,
                    remove_seasonality=temporal_remove_seasonality,
                    monthly=True,
                    monthly_how=temporal_aggregate_how
                )
                # Collect for grid
                temporal_series_by_lr[lr_var] = series_np

            # ========== SPATIAL ==========
            if "spatial" in correlation_types:
                # Optional seasonality removal per-pixel before correlation
                if spatial_remove_seasonality:
                    hr_dict_anom = _remove_seasonality_field_dict(hr_dict, shared_dates, method=spatial_remove_seasonality)
                    lr_dict_anom = _remove_seasonality_field_dict(lr_dict, shared_dates, method=spatial_remove_seasonality)
                    corr_map = compute_spatial_correlation(
                        hr_dict, lr_dict,
                        method=method,
                        remove_seasonality=spatial_remove_seasonality,
                        timestamps=shared_dates
                    )
                    save_str_ssn = f"_deseason_{spatial_remove_seasonality}" if spatial_remove_seasonality else ""
                    spatial_maps_by_lr[lr_var] = corr_map
                    np.save(os.path.join(stats_save_dir,
                        f"spatial_corr_map_{hr_var}_{lr_var}_{model_hr}_vs_{model_lr}{save_str_ssn}.npy"),
                        corr_map)
                    save_str_ssn = f"_deseason_{spatial_remove_seasonality}"
                else:
                    corr_map = compute_spatial_correlation(hr_dict, lr_dict, method=method)
                    save_str_ssn = ""
                spatial_maps_by_lr[lr_var] = corr_map

                np.save(os.path.join(stats_save_dir, f"spatial_corr_map_{hr_var}_{lr_var}_{model_hr}_vs_{model_lr}{save_str_add}{save_str_ssn}.npy"), corr_map)

                # (No per-variable temporal figure here; grids only)


        # ======= GRID FIGURES after we finish all LR for this HR =======
        if spatial_grid and spatial_maps_by_lr:
            try:
                # Set grid ncols default to 4 if not provided
                spatial_grid_ncols = spatial_grid_ncols if spatial_grid_ncols else 4
                # 1) Shared colorbar, fixed vmin/vmax
                season_tag = "deseasonalized" if spatial_remove_seasonality else "with seasonality"

                fig = plot_spatial_corr_grid(
                    hr_var=hr_var,
                    lr_to_rmap=spatial_maps_by_lr,
                    ncols=spatial_grid_ncols if spatial_grid_ncols else 4,
                    per_subplot_cbar=False,
                    cbar_label="Correlation coefficient",
                    suptitle=f"Spatial correlations | HR={hr_var} ({season_tag})",
                    vmin=-1, vmax=1,
                    savepath=os.path.join(
                        figs_save_dir,
                        f"spatial_corr_grid_{hr_var}_{model_hr}_vs_{model_lr}"
                        f"{('_deseason_'+spatial_remove_seasonality) if spatial_remove_seasonality else ''}.png"
                    )
                )
                # 2) Individual colorbars
                fig = plot_spatial_corr_grid(
                    hr_var=hr_var,
                    lr_to_rmap=spatial_maps_by_lr,
                    ncols=spatial_grid_ncols if spatial_grid_ncols else 4,
                    per_subplot_cbar=True,
                    per_cbar_size="3%", per_cbar_pad=0.04,
                    wspace=0.25, hspace=0.35,
                    cbar_label="Correlation coefficient",
                    suptitle=f"Spatial correlations | HR={hr_var} ({season_tag})",
                    vmin=None, vmax=None,
                    savepath=os.path.join(
                        figs_save_dir,
                        f"spatial_corr_grid_indcbar_{hr_var}_{model_hr}_vs_{model_lr}"
                        f"{('_deseason_'+spatial_remove_seasonality) if spatial_remove_seasonality else ''}.png"
                    )
                )
                if plot_cfg.get("show", False):
                    plt.show()
                plt.close(fig)

            except Exception as e:
                logger.warning(f"plot_spatial_corr_grid failed: {e}. Skipping combined spatial grid.")

        if temporal_grid and temporal_series_by_lr:
            try:
                # Convert dict to ordered pair list for plotting
                pairs = [{"hr_name": hr_var, "lr_name": lr, "series": series} for lr, series in temporal_series_by_lr.items()]
                fig_out_dir = figs_save_dir
                plot_temporal_correlations_grid(
                    pairs,
                    hr_var_label=hr_var,
                    deseasonalize=bool(temporal_remove_seasonality),
                    out_dir=fig_out_dir,
                    fname_prefix=f"temporal_corr_grid_{hr_var}_{model_hr}_vs_{model_lr}",
                    ncols=temporal_grid_ncols if temporal_grid_ncols else 4,
                    nrows=2,
                    hr_color="#c44e52", lr_color="#2ca02c",
                )
            except Exception as e:
                logger.warning(f"plot_temporal_correlations_grid failed: {e}.")
