import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from scor_dm.variable_utils import get_unit_for_variable, get_cmap_for_variable, get_color_for_model, get_color_for_variable
from scor_dm.special_transforms import transform_from_stats
from scor_dm.plotting_utils import plot_spatial_panel, apply_model_colors
from data_analysis_pipeline.stats_analysis.statistics import load_global_stats

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def plot_cutout_example(data,
                        variable,
                        cfg,
                        fig_save_path,
                        bounds: tuple[int, int, int, int] = (170, 350, 340, 520)
                        ):
    """
        Plot a single cutout example from the data (2D field).
        Can plot either a random cutout or the one corresponding to a specified date

        Args:
            data (dict): with keys 'cutouts' (list of 2D np.ndarrays) and optionally 'timestamps' (list of datetime objects)
            variable (str): name of the variable (e.g. "temp", "prcp")
            cfg (dict): Configuration dictionary 
    """

    cutouts = data["cutouts"]
    timestamps = data.get("timestamps", None)

    # === Select index ===
    specific_date = cfg.get("plotting", {}).get("example_date", None)
    if specific_date:
        specific_date = str(specific_date)
        match_idx = [i for i, ts in enumerate(timestamps) if ts.strftime("%Y%m%d") == specific_date] # Match YYYYMMDD format
        if not match_idx:
            logger.warning(f"Specified example_date {specific_date} not found in timestamps. Using random cutout instead.")
            idx = np.random.randint(len(cutouts))
        else:
            idx = match_idx[0]
            logger.info(f"Found matching date {specific_date} at index {idx}.")
    else:
        idx = np.random.randint(len(cutouts))
        logger.info(f"No specific date provided. Using random index {idx}.")
    
    cutout = cutouts[idx]

    cmap = get_cmap_for_variable(variable)
    unit = get_unit_for_variable(variable)
    # Optional gray background for near-zero precip
    under_color = None
    under_threshold = None
    if variable.lower() in ["prcp", "precip", "precipitation"]:
        under_threshold = float(cfg.get("plotting", {}).get("zero_floor_mm_day", 0.01))
        under_color = "#bdbdbd"

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_spatial_panel(
        ax,
        cutout,
        variable=variable,
        vmin=None,
        vmax=None,
        add_dk_outline=True,
        outline_color="darkgrey",
        outline_linewidth=0.8,
        title=f"{variable} on {timestamps[idx].strftime('%Y-%m-%d') if timestamps else 'N/A'}",
        under_color=under_color,
        under_threshold=under_threshold,
        bounds= bounds
    )

    # === Save or show ===
    plotting = cfg.get("plotting", {})
    show = plotting.get("show", False)
    save = plotting.get("save", True)

    if save:
        os.makedirs(fig_save_path, exist_ok=True)
        save_path = os.path.join(fig_save_path, f"example_cutout_{variable}_{idx}.png")
        logger.info(f"Saving example cutout plot to {save_path}")
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved cutout plot to {save_path}/example_cutout_{variable}_{idx}.png")
    
    if show:
        plt.show()
    plt.close(fig)


def visualize_statistics(variable,
                         data,
                         stats_dict,
                         cfg,
                         fig_save_path,
                         load_global=True,
                         model=None,
                         domain_str=None,
                         crop_region_str=None,
                         split=None,
                         dir_load_glob=None,
                         aggregated=False,
                         agg_method=None,
                         agg_time=None,
                         show_transformed=False,
                         transforms=['zscore'],
                         log_scale=False
                         ):
    """
        Visualize dataset statistics and distributions.
        Args:
            data: list of 2D np.ndarrays (cutouts) or 3D np.ndarrays (stacks; N, H, W)
            stats_dict: dict of per-timestep statistics (keys: mean, std, min, max etc.)
            cfg: configuration dictionary
            aggregated: boolean indicating if the statistics are aggregated
            agg_method: aggregation method used (if any)
            show_transformed: boolean indicating if the transformed data should be shown (i.e. the standardize/normalized)
    """
    
    plotting = cfg.get("plotting", {})
    show = plotting.get("show", False)
    save = plotting.get("save", True)
    fig_path = fig_save_path

    if save:
        os.makedirs(fig_path, exist_ok=True)

    model_color = get_color_for_model(model) if model is not None else None

    suffix = f"_agg_{agg_method}_{agg_time}" if aggregated and agg_method else "_daily"
    agg_str = f"{agg_time} {agg_method} aggregated " if aggregated and agg_method else "daily "

    # Prepare data stack
    if isinstance(data, dict) and "cutouts" in data:
        cutouts = data["cutouts"]
    elif isinstance(data, list):
        cutouts = data    
    else:
        raise ValueError("Unexpected data format. Provide a dict with 'cutouts' key or a list of cutouts.")

    stack = np.stack(cutouts)  # Shape: (T, H, W)
    flat = stack.flatten() # Shape: (T * H * W,)


    # === 1. Time series plots ===
    if "timeseries" in stats_dict:
        ts_stats = stats_dict["timeseries"]
        times = ts_stats.get("timestamps", np.arange(len(ts_stats.get("mean", []))))

        # === 1a. Plot mean with std as error bars ===
        if "mean" in ts_stats and "std" in ts_stats:
            fig, ax = plt.subplots(figsize=(10, 5))
            mean_series = np.array(ts_stats["mean"])
            std_series = np.array(ts_stats["std"])
            line_c = model_color or 'k'
            ax.plot(times, mean_series, color=line_c, lw=1, alpha=0.9, label='Mean')
            ax.errorbar(times, mean_series, yerr=std_series, fmt='.', ecolor=line_c, elinewidth=1, capsize=2, label='± Std Dev', color=line_c, alpha=0.6)
            apply_model_colors(ax)
            ax.set_title(f"{variable} Mean with Std Dev over Time, {agg_str}")
            ax.set_xlabel("Time")
            ax.set_ylabel(f"{variable} ({get_unit_for_variable(variable)})")
            ax.legend()
            fig.autofmt_xdate() # Auto-format date labels
            if save:
                logger.info(f"Saving mean ± std time series plot to {fig_path}/mean_std_time_series{suffix}.png")
                fig.savefig(os.path.join(fig_path, f"mean_std_time_series{suffix}.png"), dpi=300)
            if show:
                plt.show()
            plt.close(fig)

        # === 1b. Plot individual time series metrics in subplots ===
        keys = plotting.get("plot_stats", ['mean', 'std', 'min', 'max', 'median', 'percentile_25', 'percentile_75'])
        n_keys = len([k for k in keys if k in ts_stats])
        n_cols = 2
        n_rows = (n_keys + 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
        axs = axs.flatten()

        plotted_count = 0
        for k in keys:
            if k in ts_stats:
                axs[plotted_count].plot(times, ts_stats[k], label=k, alpha=0.8, color=(model_color or 'k'))
                axs[plotted_count].set_title(f"{variable} {k} over Time, {agg_str}")
                axs[plotted_count].set_xlabel("Time")
                axs[plotted_count].set_ylabel(f"{k} ({get_unit_for_variable(variable)})")
                axs[plotted_count].tick_params(axis='x', rotation=30)
                axs[plotted_count].legend()
                axs[plotted_count].grid(True)
                apply_model_colors(axs[plotted_count])
                plotted_count += 1
        for j in range(plotted_count, len(axs)):
            fig.delaxes(axs[j])

        if save:
            path = os.path.join(fig_path, f"time_series_subplots_{variable}{suffix}.png")
            logger.info(f"Saving time series subplots to {path}")
            fig.savefig(path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)




    # === 2. Histogram of all pixel values ===
    if data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        # Main histogram
        ax1.hist(flat, bins=100, color=model_color or "gray", alpha=0.7, label='Pixel Values')
        # Mean shown as vertical line
        ax1.axvline(np.mean(flat), color=model_color or "gray", linestyle='--', label='Mean')

        if show_transformed:
            if load_global:
                # Load global stats for transformations if available (if not, compute from data)
                logger.info(f"Loading global stats from {dir_load_glob} for variable {variable}, model {model}, domain {domain_str}, crop_region {crop_region_str}, split {split}")
                global_stats = load_global_stats(variable, model, domain_str, crop_region_str, split, dir_load_glob)
                logger.info(f"          Loaded global stats: {global_stats}")
            elif stats_dict and "global" in stats_dict:
                global_stats = stats_dict["global"]
            else:
                global_stats = None

            if global_stats is None:
                logger.info(f"No global stats provided or loaded. Computing from data for variable {variable}.")
                mean_val = np.mean(flat)
                std_val = np.std(flat)
                min_val = np.min(flat)
                max_val = np.max(flat)
                log_mean = np.mean(np.log(flat[flat > 0] + 1e-8))  # Avoid log(0)
                log_std = np.std(np.log(flat[flat > 0] + 1e-8))
                log_min = np.min(np.log(flat[flat > 0] + 1e-8))
                log_max = np.max(np.log(flat[flat > 0] + 1e-8))
                global_stats = {"mean": mean_val, "std": std_val, "min": min_val, "max": max_val, "log_mean": log_mean, "log_std": log_std, "log_min": log_min, "log_max": log_max}

            # Define a dict of colors for each transform
            colors = {
                "zscore": "orange",
                "minmax": "red",
                "log": "green",
                "log_zscore": "purple"
            }
            labels = {
                "zscore": "Z-Score",
                "minmax": "Min-Max",
                "log": "Log",
                "log_zscore": "Log Z-Score"
            }

            # Loop through requested transforms
            for transform in transforms:
                logger.info(f"\n          Applying transformation: {transform}\n")
                if global_stats is None:
                    logger.info(f"No global stats available for transformation {transform}. Skipping.")
                    continue
                
                transformed = transform_from_stats(flat, transform, cfg, global_stats)
                # Transformed is torch.Tensor, convert to numpy
                if transformed is not None:
                    transformed = transformed.numpy()
                
                if transformed is not None:
                    # Plot alongside original
                    ax1.hist(transformed, bins=100, color=colors[transform], alpha=0.5, label=labels[transform])

                    ax1.axvline(np.mean(transformed), color=colors[transform], linestyle='--', linewidth=1)#, label=f'{labels[transform]} Mean')
                    # Plot alongside zoomed inset
                    ax2.hist(transformed, bins=100, alpha=0.7, label=labels[transform], color=colors[transform])
                    ax2.axvline(np.mean(transformed), color=colors[transform], linestyle='--', linewidth=1)#, label=f'{labels[transform]} Mean')

        if log_scale:
            ax1.set_yscale('log')
            # ax2.set_yscale('log')

        ax1.set_title(f"Histogram of {variable} pixel values, {agg_str}")
        ax1.set_xlabel(f"{variable}, {get_unit_for_variable(variable)}")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        ax2.set_title(f"Zoomed Histogram of {variable} pixel values, {agg_str}")
        ax2.set_xlabel(f"{variable} Transformed Value")
        ax2.set_ylabel("Frequency")
        ax2.legend()

        if save:
            logger.info(f"Saving histogram plot to {fig_path}/histogram_pixels_{variable}_{suffix}.png")
            fig.savefig(os.path.join(fig_path, f"histogram_pixels_{variable}_{suffix}.png"), dpi=300)
        if show:
            plt.show()
        plt.close(fig)






    # === 3. Histogram of time-series statistics ===
    if "timeseries" in stats_dict:
        keys = plotting.get("plot_stats", ['mean', 'std', 'min', 'max', 'median', 'percentile_25', 'percentile_75'])

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for ax, k in zip(axs.flatten(), keys):
            if k in stats_dict["timeseries"]:
                ax.hist(stats_dict["timeseries"][k], bins=100, alpha=0.7, label=k, color=get_color_for_variable(variable, model if model is not None else ""))
                ax.set_title(f"Histogram of {variable} {k}, {agg_str}")
                if log_scale:
                    ax.set_yscale('log')
                ax.set_xlabel(f"{k} ({get_unit_for_variable(variable)})")
                ax.set_ylabel("Frequency")
        
        if save:
            logger.info(f"Saving histogram of time-series stats to {fig_path}/histogram_time_series_{variable}_{suffix}.png")
            fig.savefig(os.path.join(fig_path, f"histogram_time_series_{variable}_{suffix}.png"), dpi=300)
        if show:
            plt.show()
        plt.close(fig)


    # # === 4. Global summary bar plot ===
    # values = [np.mean(stats_dict[k]) for k in keys if k in stats_dict]
    # labels = [k for k in keys if k in stats_dict]
    
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.bar(labels, values, color='skyblue', alpha=0.7)
    # ax.set_title(f"Global Summary of {variable}, {agg_str}")

    # if save:
    #     fig.savefig(os.path.join(fig_path, f"global_summary_{variable}_{suffix}.png"), dpi=300)
    # if show:
    #     plt.show()
    # plt.close(fig)


def visualize_data(data, stats_dict, cfg, aggregated=False):
    """
        Visualize the data and statistics.
        Time-series, histograms, pixel-wise distributions, etc.
    """
    pass