import os
import random
import logging
import numpy as np
from data_analysis_pipeline.stats_analysis.data_loading import DataLoader
from data_analysis_pipeline.comparison.compare_fields import compare_single_day_fields
from data_analysis_pipeline.comparison.compare_timeseries import compare_over_time
from data_analysis_pipeline.comparison.compare_distributions import compare_power_spectra, batch_compare_power_spectra, compare_distributions, compare_seasonal_distributions
from data_analysis_pipeline.comparison.plot_utils import plot_samples_grid

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def run_comparison_pipeline(cfg):
    """
        Main entry point for running dataset comparisons. 
        (Distributional, Field-based, Time-series based)
    """

    # Global flags
    plot_only = cfg.get("global", {}).get("plot_only", False)

    # === Extract config sections ===
    highres_cfg = cfg["highres"]
    lowres_cfg = cfg["lowres"]
    comparison_cfg = cfg["comparison"]
    data_dir = cfg["paths"]["data_dir"]

    variable = comparison_cfg["variable"]
    split = comparison_cfg.get("split", "all")  # Optional data split (e.g., "train", "val", "test")
    transform = comparison_cfg.get("transformations", None)  # Optional transformations to apply
    model_hr = highres_cfg["model"]
    model_lr = lowres_cfg["model"]
    crop_hr = highres_cfg.get("crop_region", None)
    crop_lr = lowres_cfg.get("crop_region", None)
    domain_size_hr = highres_cfg.get("domain_size", None)
    domain_size_lr = lowres_cfg.get("domain_size", None)
    n_workers = comparison_cfg.get("n_workers", 4)
    mode = comparison_cfg["mode"]  # "field", "timeseries", "distribution", or "all"
    show = comparison_cfg.get("show", False)
    save_figures = comparison_cfg.get("save_figures", True)
    save_path = comparison_cfg.get("save_path", "./figures")
    verbose_data_loading = cfg.get("data", {}).get("verbose", False)
    if split is not None:
        # Append split to save path if not "all"
        save_path = os.path.join(save_path, split) if split != "all" else save_path
    print_results = comparison_cfg.get("print_results", True)
    max_days = comparison_cfg.get("max_days", None)  # Limit number of days to process (for testing)

    # Cache directory for precomputed comparison artifacts
    cache_dir = os.path.join(save_path if save_path else "./figures/comparison", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"[COMPARE] Mode: {'PLOT-ONLY' if plot_only else 'COMPUTE+PLOT'} | Cache: {cache_dir}")

    logger.info(f"\n[COMPARE] Saving figures to: {save_path if save_figures else 'Not saving figures'}")

    # === Instantiate loaders ===
    logger.info(f"HR: {model_hr} | LR: {model_lr} | Variable: {variable}")
    hr_loader = DataLoader(
        base_dir=data_dir,
        n_workers=n_workers,
        variable=variable,
        model=model_hr,
        domain_size=domain_size_hr,
        split=split,
        crop_region=crop_hr,
        verbose=verbose_data_loading,
        )

    lr_loader = DataLoader(
        base_dir=data_dir,
        n_workers=n_workers,
        variable=variable,
        model=model_lr,
        domain_size=domain_size_lr,
        split=split,
        crop_region=crop_lr,
        verbose=verbose_data_loading,
        )

    if mode == "field":

        ###############################
        #                             #
        # SINGLE/MULTI DAY COMPARISON #     Only load a single or a few specified days
        #                             #
        ###############################

        field_mode_cfg = next((m for m in comparison_cfg["modes"] if m["mode"] == "field"), {})
        comparison_types = list(field_mode_cfg.get("comparison_types", ["all"]))
        dates = list(field_mode_cfg.get("dates", []))  # Required for field mode]))
        mask = comparison_cfg.get("mask", None) # If null in config, becomes None here
        combine_into_grid = field_mode_cfg.get("combine_into_grid", True)
        n_max = field_mode_cfg.get("n_max", 5)

        if not dates:
            raise ValueError("For 'field' comparison mode, specific 'dates' must be provided in the config. At least one.")


        for comparison_type in (comparison_types if isinstance(comparison_types, list) else [comparison_types]):
            if comparison_type in ["field_statistics", "all"]:
                ###############################
                # SINGLE DAY FIELD STATISTICS #  Single day, numerical stats + diff map
                ###############################
                logger.info("\n\n           ### SINGLE DAY FIELD COMPARISON ###\n")
                
                # Only use one date for stats
                if isinstance(dates, list) and len(dates) > 1:
                    logger.warning("Multiple dates specified for single-day field comparison. Using the first date only.")
                    date = dates[0]
                else:
                    date = dates if isinstance(dates, str) else dates[0]
                
                logger.info(f"Loading single day data for {date} (field statistics)")
                hr_data = hr_loader.load_single_day(date)
                lr_data = lr_loader.load_single_day(date)

                result = compare_single_day_fields(
                            hr_data,
                            lr_data,
                            mask=mask,
                            variable=variable,
                            model1=model_hr,
                            model2=model_lr,
                            save_path=save_path if save_figures else "",
                            show=show,
                            print_results=print_results,
                            bounds=tuple(highres_cfg.get("crop_region", [200, 328, 380, 508])),
                            plot_only=plot_only,
                            cache_dir=cache_dir,                            
                            )
                logger.info("=========== Single day field comparison completed ===========\n")

            if comparison_type in ["qualitative_visual", "all"]:
                ###############################
                # QUALITATIVE VISUAL COMPARISON #  Single or multi-day, random samples with boxplots
                ###############################
                logger.info("\n\n           ### QUALITATIVE VISUAL COMPARISON ###\n")

                if isinstance(dates, list) and n_max:
                    if len(dates) > n_max:
                        logger.warning(f"More than {n_max} dates specified for qualitative visual comparison. Randomly selecting {n_max} dates.")
                        dates = random.sample(dates, n_max)
                elif isinstance(dates, str):
                    dates = [dates]

                hr_data = hr_loader.load_multi(dates)
                lr_data = lr_loader.load_multi(dates)

                # Find shared dates between loaded HR and LR data
                shared_dates = sorted(set(hr_data["timestamps"]) & set(lr_data["timestamps"]))
                dates = shared_dates

                hr_dict = {d: c for d, c in zip(hr_data["timestamps"], hr_data["cutouts"])}
                lr_dict = {d: c for d, c in zip(lr_data["timestamps"], lr_data["cutouts"])}

                # --- Read new options from config ---
                add_dk_outline = field_mode_cfg.get("add_dk_outline", True)
                include_difference = field_mode_cfg.get("include_difference", False)

                plot_samples_grid(
                    hr=hr_dict,
                    lr=lr_dict,
                    hr_model=model_hr,
                    lr_model=model_lr,
                    variable=variable,
                    dates=dates,
                    combine_into_grid=field_mode_cfg.get("combine_into_grid", True),
                    save_path=save_path if save_path is not None else f"./figures/comparison/{variable}",
                    show=show,
                    add_dk_outline=add_dk_outline,
                    include_difference=include_difference,
                    bounds=tuple(highres_cfg.get("crop_region", [200, 328, 380, 508])),                    
                    )
                logger.info("=========== Qualitative visual comparison completed ===========\n")
        

    else:

        ########################
        #                      #
        # FULL DATA COMPARISON #   Load all overlapping days from each dataset
        #                      #
        ########################

        hr_data = hr_loader.load()
        lr_data = lr_loader.load()
        hr_dates = set(hr_data["timestamps"])
        lr_dates = set(lr_data["timestamps"])
        
        shared_dates = sorted(hr_dates & lr_dates)
        
        logger.info(f"Found {len(shared_dates)} shared dates between HR and LR datasets.")
        
        if max_days:
            shared_dates = shared_dates[:max_days]
            logger.info(f"Limiting to first {max_days} days for testing.")

            hr_data = {date: cutout for date, cutout in zip(hr_data["timestamps"], hr_data["cutouts"]) if date in shared_dates}
            lr_data = {date: cutout for date, cutout in zip(lr_data["timestamps"], lr_data["cutouts"]) if date in shared_dates}
        else:
            shared_dates = shared_dates[:]
            hr_data = {date: cutout for date, cutout in zip(hr_data["timestamps"], hr_data["cutouts"]) if date in shared_dates}
            lr_data = {date: cutout for date, cutout in zip(lr_data["timestamps"], lr_data["cutouts"]) if date in shared_dates}

        if mode == "timeseries":

            ######################################
            #                                    #                             
            # DAILY STATS TIME SERIES COMPARISON #   Compare daily aggregated stats over all shared days
            #                                    #                             
            ######################################
            

            logger.info(f"\n\n           ### TIME SERIES COMPARISON OVER {len(shared_dates)} DAYS ###\n")

            result = compare_over_time(
                            hr_data,
                            lr_data,
                            model_hr,
                            model_lr,
                            variable,
                            save_path=save_path if save_figures else "",
                            show=show)
            # (no caching change here yet)
            if print_results:
                logger.info("=== Time series comparison summary ===")
                for metric, stats in result.items():
                    logger.info(f"{metric}: {stats:.4f}")

            logger.info("=========== Time series comparison completed ===========\n")

        elif mode == "distribution":
            
            #############################
            #                           #                             
            # DISTRIBUTIONAL COMPARISON #   Compare distributions over all shared days
            #                           #                             
            #############################


            distribution_mode_cfg = next((m for m in comparison_cfg["modes"] if m["mode"] == "distribution"), {})
            comparison_types = list(distribution_mode_cfg.get("comparison_types", ["all"]))

            if len(shared_dates) == 1:
                ##############
                # SINGLE DAY #
                ##############
                for comparison_type in (comparison_types if isinstance(comparison_types, list) else [comparison_types]):
                    logger.info(f"\n\n           ### DISTRIBUTIONAL COMPARISON ({comparison_type})###\n")

                    date = shared_dates[0]
                    logger.info(f"Only one date ({date}) found. Running single-day distribution comparison.")
                    if comparison_type in ["power_spectra", "all"]:

                        #############################
                        # POWER SPECTRUM COMPARISON #
                        #############################

                        logger.info("\n\n        ### SINGLE DAY POWER SPECTRUM COMPARISON ###\n")

                        ps_metrics = compare_power_spectra(
                                        np.array(hr_data[date]),
                                        np.array(lr_data[date]),
                                        model_hr,
                                        model_lr,
                                        variable,
                                        save_path=save_path if save_figures else "",
                                        show=show,
                                        cache_dir=cache_dir,
                                        plot_only=plot_only,
                                        )
                        if print_results:
                            logger.info(f"=== Power Spectra comparison for {date} ===")
                            if ps_metrics is not None:
                                for metric, value in ps_metrics.items():
                                    logger.info(f"{metric}: {value:.4f}")
                            else:
                                logger.warning("No power spectra metrics returned (ps_metrics is None).")

                        logger.info("=========== Single day Power spectrum comparison completed ===========\n")

                    if comparison_type in ["pixel_distributions", "all"]:
                        
                        #################################
                        # PIXEL DISTRIBUTION COMPARISON #
                        #################################

                        logger.info(f"\n\n           ### SINGLE DAY PIXEL DISTRIBUTION COMPARISON ###\n")

                        stats = compare_distributions(
                                        np.array(hr_data[date]),
                                        np.array(lr_data[date]),
                                        model_hr,
                                        model_lr,
                                        variable,
                                        save_path=save_path if save_figures else "",
                                        show=show,
                                        cache_dir=cache_dir,
                                        plot_only=plot_only,
                                        )
                        if print_results:
                            logger.info(f"=== Pixel Distribution comparison for {date} ===")
                            if stats is not None:
                                for metric, value in stats.items():
                                    logger.info(f"{metric}: {value:.4f}")
                            else:
                                logger.warning("No pixel distribution metrics returned (stats is None).")

                        logger.info("=========== Single day Pixel distribution comparison completed ===========\n")
            else:
                ##############
                # MULTI DAYS #
                ##############                
                logger.info(f"Multiple dates found ({len(shared_dates)}). Running batch distribution comparison.")
                for comparison_type in (comparison_types if isinstance(comparison_types, list) else [comparison_types]):
                    if comparison_type in ["power_spectra", "all"]:
                        
                        #############################
                        # POWER SPECTRUM COMPARISON #
                        #############################

                        logger.info(f"\n\n           ### BATCH POWER SPECTRUM COMPARISON ###\n")

                        ps_summary, ps_details = batch_compare_power_spectra(
                                        dataset1 = hr_data,
                                        dataset2 = lr_data,
                                        model1 = model_hr,
                                        model2 = model_lr,
                                        variable = variable,
                                        dx_model1 = highres_cfg.get("grid_spacing_km", 1.0),
                                        dx_model2 = lowres_cfg.get("grid_spacing_km", 1.0),
                                        save_path=save_path if save_figures else "",
                                        show=show,
                                        cache_dir=cache_dir,
                                        plot_only=plot_only,
                                        )
                        if print_results:
                            logger.info("=== Batch Power Spectra comparison summary ===")
                            for metric, stats in ps_summary.items():
                                if isinstance(stats, dict) and 'mean' in stats and 'std' in stats:
                                    logger.info(f"{metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
                                else:
                                    logger.info(f"{metric}: {stats:.4f}")

                        logger.info("=========== Batch Power spectrum comparison completed ===========\n")


                    if comparison_type in ["pixel_distributions", "all"]:
                        
                        #################################
                        # PIXEL DISTRIBUTION COMPARISON #
                        #################################

                        logger.info(f"\n\n           ### BATCH PIXEL DISTRIBUTION COMPARISON ###\n")

                        # Aggregate all days' data into single numpy arrays for distribution comparison
                        hr_array = np.concatenate([np.array(hr_data[date]).flatten() for date in shared_dates])
                        lr_array = np.concatenate([np.array(lr_data[date]).flatten() for date in shared_dates])
                        if variable in ['prcp']:
                            plot_log = True
                        else:
                            plot_log = False
                        pixel_stats = compare_distributions(
                                        hr_array,
                                        lr_array,
                                        model_hr,
                                        model_lr,
                                        variable,
                                        log_hist=plot_log,
                                        save_path=save_path if save_figures else "",
                                        show=show,
                                        cache_dir=cache_dir,
                                        plot_only=plot_only,
                                        )
                        if print_results:
                            logger.info("=== Batch Pixel Distribution comparison summary ===")
                            if pixel_stats is not None:
                                for metric, value in pixel_stats.items():
                                    logger.info(f"{metric}: {value:.4f}")
                            else:
                                logger.warning("No pixel distribution metrics returned (pixel_stats is None).")

                        logger.info("=========== Batch Pixel distribution comparison completed ===========\n")

                    if comparison_type in ["seasonal_histograms", "all"]:

                        ############################################
                        # PIXEL DISTRIBUTION COMPARISON - SEASONAL #
                        ############################################
                        logger.info(f"\n\n           ### SEASONAL PIXEL DISTRIBUTION COMPARISON ###\n")                        
                        if variable in ['prcp']:
                            plot_log = True
                        else:
                            plot_log = False
                        compare_seasonal_distributions(
                                        hr_data,
                                        lr_data,
                                        model_hr,
                                        model_lr,
                                        variable,
                                        log_hist=plot_log,
                                        save_path=save_path if save_figures else "",
                                        show=show)

                        logger.info("=========== Seasonal Pixel distribution comparison completed ===========\n")