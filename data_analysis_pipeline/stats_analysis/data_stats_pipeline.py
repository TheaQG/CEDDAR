import os
import math
import logging
import numpy as np

from data_analysis_pipeline.stats_analysis.data_loading import DataLoader
from data_analysis_pipeline.stats_analysis.plotting import plot_cutout_example, visualize_statistics
from data_analysis_pipeline.stats_analysis.statistics import compute_statistics


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def run_data_statistics(cfg):
    all_results = {}

    split = cfg.get("data", {}).get("split", "all")
    plot_only = bool(cfg.get("global", {}).get("plot_only", False))
    if plot_only:
        logger.info("[stats] Plot-only: will try to load cached statistics from disk and skip heavy recomputation.")

    data_dir = cfg.get("data", {}).get("data_dir", ".")
    # Check if data_dir ends with _small, if so set data_batch_small to True
    if data_dir.endswith("_small"):
        small_data_batch = True
        logger.info(f"Detected small data batch from data_dir ending with '_small': {data_dir}")
    else:
        small_data_batch = False

    fig_save_dir = cfg.get("plotting", {}).get("fig_save_dir", ".")
    stats_save_dir = cfg.get("statistics", {}).get("stats_save_dir", ".")
    logger.info(f"  PATHS:")
    logger.info(f"          DATA: {data_dir}")
    logger.info(f"          FIGS SAVE: {fig_save_dir}")
    logger.info(f"          STATS SAVE: {stats_save_dir}")

    n_workers = int(cfg.get("data", {}).get("n_workers", 1))

    logger.info(f"Running data statistics with split='{split}', data_dir='{data_dir}', n_workers={n_workers}")
    
    def _estimate_streaming_need(loader, variable_name: str):
        """
        Decide whether we should avoid materializing the full dataset in RAM.
        Returns (use_streaming: bool, estimated_gib: float, n_entries: int | None).
        """
        stats_cfg = cfg.get("statistics", {})
        explicit_streaming = stats_cfg.get("streaming", None)
        max_ram_gib = float(stats_cfg.get("max_in_memory_gib", 4.0))
        bytes_per_value = int(stats_cfg.get("bytes_per_value", 4))  # float32 default

        file_list = loader._get_file_list()
        n_entries = len(file_list)
        expected_shape = loader.get_expected_shape()

        if expected_shape is None:
            logger.warning(f"[stats] Could not infer expected shape for {variable_name}; defaulting to streaming={bool(explicit_streaming)}")
            return bool(explicit_streaming), math.nan, n_entries

        n_pixels = int(expected_shape[0]) * int(expected_shape[1])
        estimated_bytes = n_entries * n_pixels * bytes_per_value
        estimated_gib = estimated_bytes / (1024 ** 3)

        if explicit_streaming is not None:
            use_streaming = bool(explicit_streaming)
        else:
            use_streaming = estimated_gib > max_ram_gib

        logger.info(
            f"[stats] Estimated in-memory stack for {variable_name}: "
            f"{n_entries} x {expected_shape[0]} x {expected_shape[1]} x {bytes_per_value} bytes "
            f"≈ {estimated_gib:.2f} GiB -> streaming={use_streaming}"
        )
        return use_streaming, estimated_gib, n_entries

    def process_variable(var_cfg, variable, agg_method, agg_time, level):
        logger.info(f"Processing {level.upper()} variable: {variable} ({var_cfg.get('model', '')}) | agg_method={agg_method} | agg_time={agg_time}")

        crop_region_str = "_".join(map(str, var_cfg.get("crop_region", []))) if var_cfg.get("crop_region", []) else "full"
        domain_size_str = "x".join(map(str, var_cfg.get("domain_size", []))) if var_cfg.get("domain_size", []) else "full"
        logger.info(f"Crop region: {crop_region_str} | Domain size: {domain_size_str}")

        if plot_only:
            from data_analysis_pipeline.stats_analysis.statistics import load_global_stats
            # Try load global stats
            global_stats = load_global_stats(variable, var_cfg.get("model",""), domain_size_str, crop_region_str, split, stats_save_dir)
            # Try load NPZ with time series and cutout stats
            stats_dir = os.path.join(
                stats_save_dir,
                var_cfg.get("model", ""),
                variable,
                split,
                domain_size_str,
                crop_region_str,
            )
            cand_small = os.path.join(stats_dir, f"stats_timeseries_cutout__{var_cfg.get('model','')}__{domain_size_str}__crop__{crop_region_str}__{variable}__{split}__small.npz")
            cand_full  = os.path.join(stats_dir, f"stats_timeseries_cutout__{var_cfg.get('model','')}__{domain_size_str}__crop__{crop_region_str}__{variable}__{split}.npz")
            npz_path = cand_small if os.path.exists(cand_small) else (cand_full if os.path.exists(cand_full) else None)
            time_series_stats = {}
            cutout_stats = {}
            if npz_path:
                try:
                    with np.load(npz_path) as npz:
                        # timestamps optional
                        if "timestamps_iso" in npz:
                            from datetime import datetime as _dt
                            time_series_stats["timestamps"] = [ _dt.fromisoformat(s) for s in npz["timestamps_iso"].tolist() ]
                        for k in ["mean","std","min","max","median","percentile_25","percentile_75"]:
                            if k in npz:
                                time_series_stats[k] = npz[k]
                        for k in ["mean","std","min","max","median","percentile_25","percentile_75"]:
                            nk = f"cutout_{k}"
                            if nk in npz:
                                cutout_stats[k] = npz[nk]
                    logger.info(f"[stats] Loaded cached full stats from {npz_path}")
                except Exception as e:
                    logger.warning(f"[stats] Failed to load cached NPZ stats at {npz_path}: {e}")
            else:
                logger.warning(f"[stats] No cached NPZ stats found at {stats_dir}. Skipping plots that require raw data.")
            # Collect results and (optionally) plot with data=None
            all_results[f"{level}__{variable}"] = {
                "global": global_stats if global_stats else {},
                "cutout": cutout_stats,
                "timeseries": time_series_stats,
            }
            if cfg.get("plotting", {}).get("visualize_data", False):
                # Also include the cutout area for path naming (<variable>__crop__{crop_region_str}) to distinguish from full-domain stats
                current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), f"{variable}__crop__{crop_region_str}", split)
                visualize_statistics(
                    variable,
                    None,  # data=None in plot-only
                    {
                        "global": global_stats if global_stats else {},
                        "cutout": cutout_stats,
                        "timeseries": time_series_stats,
                    },
                    cfg,
                    model=var_cfg.get("model", ""),
                    load_global=False,
                    domain_str=domain_size_str,
                    crop_region_str=crop_region_str,
                    split=split,
                    fig_save_path=current_fig_save_path,
                    dir_load_glob=stats_save_dir,
                    aggregated=False,
                    show_transformed=False,
                    transforms=[],
                    log_scale=False,
                )
            return

        loader = DataLoader(
            base_dir=data_dir,
            n_workers=n_workers,
            variable=variable,
            model=var_cfg.get("model", ""),
            domain_size=var_cfg.get("domain_size", []),
            split=split,
            crop_region=var_cfg.get("crop_region", []),
            verbose=cfg.get("data", {}).get("verbose", False),
        )
        use_streaming, estimated_gib, n_entries = _estimate_streaming_need(loader, variable)
        expected_shape = loader.get_expected_shape()

        if use_streaming:
            raw_data = {"loader": loader, "entries_iter": loader.iter_entries(), "expected_shape": expected_shape}
            logger.info(
                f"Initialized streaming stats for {variable} "
                f"(estimated full stack ≈ {estimated_gib:.2f} GiB; no full cutout list kept in memory)"
            )
        else:
            raw_data = loader.load()
            logger.info(f"Loaded {len(raw_data['cutouts'])} cutouts for {variable}")

        # In streaming mode, raw_data has no 'cutouts'. Build a small plotting subset when needed.
        def _get_plot_data():
            """Return a dict with 'cutouts' (+ optional 'timestamps') suitable for plotting."""
            plotting_cfg = cfg.get("plotting", {})
            ex_date = plotting_cfg.get("example_date", None)

            # 1) Prefer a deterministic example day if provided
            if ex_date:
                try:
                    d = loader.load_single_day(str(ex_date))  # YYYYMMDD

                    # normalize timestamps
                    ts = d.get("timestamps", None)
                    if ts is not None and not isinstance(ts, list):
                        d["timestamps"] = [ts]
                    # normalize shape: cutouts must be a list
                    if isinstance(d.get("cutouts", None), np.ndarray):
                        d["cutouts"] = [d["cutouts"]]
                    if not isinstance(d.get("cutouts", None), list):
                        d["cutouts"] = list(d["cutouts"])  # best-effort
                    return d
                except Exception as e:
                    logger.warning(
                        f"[stats] Failed to load example_date={ex_date} for plotting: {e}. Falling back to random samples."
                    )

            # 2) Otherwise, load a small random sample for plotting
            n = int(plotting_cfg.get("n_plot_samples", 8))
            d = loader.load_multi(n)
            if isinstance(d.get("cutouts", None), np.ndarray):
                d["cutouts"] = [d["cutouts"]]
            if not isinstance(d.get("cutouts", None), list):
                d["cutouts"] = list(d["cutouts"])  # best-effort
            return d

        # === Basic Statistics ===
        if cfg.get("statistics", {}).get("save_global_pixel_stats", True):
            save_glob_stats = True
            logger.info(f"Saving global statistics for {variable}")
        else:
            save_glob_stats = False
            logger.info(f"Not saving global statistics for {variable}")

        global_stats, cutout_stats, time_series_stats = compute_statistics(
                                                            raw_data,
                                                            print_stats=True,
                                                            return_all=True,
                                                            save_glob_stats=save_glob_stats,
                                                            variable=variable,
                                                            model=var_cfg.get("model", ""),
                                                            split=split,
                                                            domain_str=domain_size_str,
                                                            crop_region_str=crop_region_str,
                                                            cfg=cfg,
                                                            stats_save_path=stats_save_dir,
                                                            log_stats=variable in ['prcp', 'cape'],
                                                            pool_pixels=True,
                                                            small_data_batch=small_data_batch,
                                                            save_full_stats_npz=cfg.get("statistics", {}).get("save_full_stats_npz", True),
                                                            streaming=use_streaming,
                                                        )
        all_results[f"{level}__{variable}"] = {
            "global": global_stats,
            "cutout": cutout_stats,
            "timeseries": time_series_stats,
        }
        
        if cfg.get("plotting", {}).get("save_cutout_example", False):
            current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), f"{variable}__crop__{crop_region_str}", split)
            bounds = var_cfg.get("crop_region", [])
            logger.warning(f"Saving cutout example for variable {variable}, cutout region: {bounds}")
            plot_data = _get_plot_data() if use_streaming else raw_data
            plot_cutout_example(plot_data, variable, cfg, current_fig_save_path, bounds=bounds)

        if cfg.get("plotting", {}).get("visualize_data", False):
            current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), f"{variable}__crop__{crop_region_str}", split)
            logger.info(f"Generating raw data visualizations at {current_fig_save_path}")
            plot_data = _get_plot_data() if use_streaming else raw_data
            if variable in ['temp', 'ewvf', 'nwvf', 'msl', 'z_pl_1000', 'z_pl_250', 'z_pl_500', 'z_pl_850']:
                transforms = ['zscore']
                log_scale = False
            elif variable in ['prcp', 'cape']:
                transforms = ['log', 'log_zscore']
                log_scale = True
                # Make sure all data is positive for log plots
                plot_data['cutouts'] = [np.where(c <= 0, 1e-8, c) for c in plot_data['cutouts']]
            else:
                transforms = ['zscore']
                log_scale = False
                         
                         
            visualize_statistics(
                variable,
                plot_data,
                {
                    "global": global_stats,
                    "cutout": cutout_stats,
                    "timeseries": time_series_stats,
                },
                cfg,
                model=var_cfg.get("model", ""),
                load_global=True,
                domain_str=domain_size_str,
                crop_region_str=crop_region_str,
                split=split,
                fig_save_path=current_fig_save_path,
                dir_load_glob=stats_save_dir,
                aggregated=False,
                show_transformed=True,
                transforms=transforms,
                log_scale=log_scale,
            )

        if cfg.get("statistics", {}).get("aggregate", False):
            if use_streaming:
                logger.warning(
                    f"[stats] Skipping aggregated statistics for {level.upper()} variable '{variable}' "
                    f"because streaming mode is active. Aggregation currently requires materializing the full stack."
                )
            else:
                logger.info(f"Computing aggregated statistics for {level.upper()} variable '{variable}' using method: {agg_method}, time: {agg_time}")
                global_stats, cutout_stats, time_series_stats = compute_statistics(
                                                                    raw_data,
                                                                    aggregate=True,
                                                                    agg_method=agg_method,
                                                                    agg_time=cfg.get("data", {}).get("aggregation_time", "monthly"),
                                                                    return_all=True,
                                                                    print_stats=True,
                                                                    save_glob_stats=save_glob_stats,
                                                                    variable=variable,
                                                                    model=var_cfg.get("model", ""),
                                                                    split=split,
                                                                    domain_str=domain_size_str,
                                                                    crop_region_str=crop_region_str,
                                                                    cfg=cfg,
                                                                    stats_save_path=stats_save_dir,
                                                                    log_stats=variable in ['prcp', 'cape'],
                                                                    pool_pixels=True,
                                                                    small_data_batch=small_data_batch,
                                                                    save_full_stats_npz=cfg.get("statistics", {}).get("save_full_stats_npz", True),
                                                                )
                all_results[f"{level}__agg__{variable}"] = {
                    "global": global_stats,
                    "cutout": cutout_stats,
                    "timeseries": time_series_stats,
                }

                if cfg.get("plotting", {}).get("visualize_aggregated", False):
                    current_fig_save_path = os.path.join(fig_save_dir, var_cfg.get("model", ""), f"{variable}__crop__{crop_region_str}", split)
                    logger.info(f"Generating aggregated data visualizations at {current_fig_save_path}")
                    plot_data = _get_plot_data() if use_streaming else raw_data
                    if variable in ['temp', 'ewvf', 'nwvf', 'msl', 'z_pl_1000', 'z_pl_250', 'z_pl_500', 'z_pl_850']:
                        transforms = ['zscore']
                        log_scale = False
                    elif variable in ['prcp', 'cape']:
                        transforms = ['log', 'log_zscore']
                        log_scale = True
                        plot_data['cutouts'] = [np.where(c <= 0, 1e-8, c) for c in plot_data['cutouts']]
                    else:
                        transforms = ['zscore']
                        log_scale = False

                    visualize_statistics(
                        variable,
                        plot_data,
                        {
                            "global": global_stats,
                            "cutout": cutout_stats,
                            "timeseries": time_series_stats,
                        },
                        cfg,
                        model=var_cfg.get("model", ""),
                        fig_save_path=current_fig_save_path,
                        aggregated=True,
                        agg_method=agg_method,
                        agg_time=agg_time,
                        show_transformed=True,
                        transforms=transforms,
                        log_scale=log_scale,
                        load_global=True,
                        domain_str=domain_size_str,
                        crop_region_str=crop_region_str,
                        split=split,
                        dir_load_glob=stats_save_dir,
                    )

    # === Process HR variables ===
    for variable, agg_method in zip(cfg.get("highres", {}).get("variables", []), cfg.get("highres", {}).get("agg_methods", [])):
        agg_time = cfg.get("data", {}).get("aggregation_time", "monthly")
        logger.info(f"\nProcessing highres variable: {variable} ({cfg.get('highres', {}).get('model', '')})| agg_method={agg_method}\n")
        process_variable(cfg.get("highres", {}), variable, agg_method, agg_time, level="highres")

    # === Process LR variables ===
    for variable, agg_method in zip(cfg.get("lowres", {}).get("condition_variables", []), cfg.get("lowres", {}).get("agg_methods", [])):
        agg_time = cfg.get("data", {}).get("aggregation_time", "monthly")
        logger.info(f"\nProcessing lowres variable: {variable} ({cfg.get('lowres', {}).get('model', '')})| agg_method={agg_method}\n")
        process_variable(cfg.get("lowres", {}), variable, agg_method, agg_time, level="lowres")


    # logger.info("Finished running full data statistics pipeline. Final results:")
    # for key, value in all_results.items():
        
    #     for inner_key, inner_value in value.items():
    #         if inner_key in cfg.get("statistics", {}).get("print_results", []):
    #             logger.info(f" - {key} | {inner_key}: {inner_value}")
    #         # else:
    #         #     logger.info(f" - {key}: (not printed, not in print_results list)")

    return # all_results