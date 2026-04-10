import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from collections import defaultdict
from datetime import datetime
from sbgm.variable_utils import get_unit_for_variable, get_cmap_for_variable, get_color_for_model, get_color_for_variable
from sbgm.special_transforms import transform_from_stats
from sbgm.plotting_utils import plot_spatial_panel, apply_model_colors
from data_analysis_pipeline.stats_analysis.statistics import load_global_stats


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _plot_text_cfg(cfg):
    plotting = cfg.get("plotting", {})
    return {
        "title": int(plotting.get("title_fontsize", 18)),
        "subtitle": int(plotting.get("subtitle_fontsize", 14)),
        "label": int(plotting.get("label_fontsize", 13)),
        "tick": int(plotting.get("tick_fontsize", 11)),
        "legend": int(plotting.get("legend_fontsize", 11)),
    }


def _aggregate_timeseries_monthly(ts_stats):
    """
    Aggregate per-day time-series statistics into monthly summaries for plotting.
    This uses already-computed per-day scalar stats and does not require
    materializing the full (T,H,W) stack.
    """
    timestamps = ts_stats.get("timestamps", None)
    if not timestamps:
        return None

    month_groups = defaultdict(list)
    for i, ts in enumerate(timestamps):
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        month_groups[(ts.year, ts.month)].append(i)

    monthly_times = []
    monthly = {}
    scalar_keys = ["mean", "std", "min", "max", "median", "percentile_25", "percentile_75", "sum"]

    for key in scalar_keys:
        if key in ts_stats and ts_stats[key] is not None:
            arr = np.asarray(ts_stats[key], dtype=np.float64)
            monthly[key] = []
            for (year, month), idxs in sorted(month_groups.items()):
                monthly[key].append(float(np.nanmean(arr[idxs])))

    for (year, month), _idxs in sorted(month_groups.items()):
        monthly_times.append(datetime(year, month, 1))

    if not monthly:
        return None

    monthly["timestamps"] = monthly_times
    for key, values in list(monthly.items()):
        if key != "timestamps":
            monthly[key] = np.asarray(values, dtype=np.float64)
    return monthly


def _pretty_ts_label(stat_key: str):
    if stat_key == "mean":
        return "daily spatial mean"
    if stat_key == "std":
        return "daily spatial std"
    if stat_key == "min":
        return "daily spatial min"
    if stat_key == "max":
        return "daily spatial max"
    if stat_key == "sum":
        return "daily spatial sum"
    if stat_key == "median":
        return "daily spatial median"
    if stat_key == "percentile_25":
        return "daily spatial 25th percentile"
    if stat_key == "percentile_75":
        return "daily spatial 75th percentile"
    return stat_key


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

    if isinstance(cutouts, np.ndarray):
        cutouts = [cutouts]
    if timestamps is not None and not isinstance(timestamps, list):
        timestamps = [timestamps]

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

    text_cfg = _plot_text_cfg(cfg)
    fig, ax = plt.subplots(figsize=(8, 6))
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

    ax.set_title(f"{variable} on {timestamps[idx].strftime('%Y-%m-%d') if timestamps else 'N/A'}", fontsize=text_cfg["subtitle"])
    ax.tick_params(axis='both', labelsize=text_cfg["tick"])

    # === Save or show ===
    plotting = cfg.get("plotting", {})
    show = plotting.get("show", False)
    save = plotting.get("save", True)

    if save:
        os.makedirs(fig_save_path, exist_ok=True)
        save_path = os.path.join(fig_save_path, f"example_cutout_{variable}_{idx}.png")
        logger.info(f"Saving example cutout plot to {save_path}")
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved cutout plot to {save_path}")
    
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
    text_cfg = _plot_text_cfg(cfg)

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
    flat = stack.flatten()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        logger.warning(f"All pixel values are non-finite for {model}/{variable}/{split}. Skipping histogram plots.")
        return


    # === 1. Time series plots ===
    if "timeseries" in stats_dict:
        ts_stats = stats_dict["timeseries"]
        times = ts_stats.get("timestamps", np.arange(len(ts_stats.get("mean", []))))

        # === 1a. Plot mean with std as error bars ===
        if "mean" in ts_stats and "std" in ts_stats:
            fig, ax = plt.subplots(figsize=(12, 6))
            mean_series = np.array(ts_stats["mean"], dtype=np.float64)
            std_series = np.array(ts_stats["std"], dtype=np.float64)
            line_c = model_color or 'k'
            ax.plot(times, mean_series, color=line_c, lw=1.2, alpha=0.9, label='Mean')
            if variable.lower() in ["prcp", "precip", "precipitation"]:
                negative_mask = mean_series - std_series < 0
                if np.any(negative_mask):
                    logger.warning(
                        f"[plotting] Precipitation mean-std drops below zero for {negative_mask.sum()} timesteps; "
                        f"lower error bars are clamped at 0 in mean_std_time_series{suffix}.png"
                    )
                lower_err = np.minimum(std_series, np.maximum(mean_series, 0.0))
                yerr = np.vstack([lower_err, std_series])
                ax.errorbar(times, mean_series, yerr=yerr, fmt='.', ecolor=line_c, elinewidth=0.8, capsize=2, label='± Std Dev', color=line_c, alpha=0.6)
                ax.set_ylim(bottom=0)
            else:
                ax.errorbar(times, mean_series, yerr=std_series, fmt='.', ecolor=line_c, elinewidth=0.8, capsize=2, label='± Std Dev', color=line_c, alpha=0.6)
            apply_model_colors(ax)
            ax.set_title(f"{variable} daily spatial mean with std over time", fontsize=text_cfg["subtitle"])
            ax.set_xlabel("Time", fontsize=text_cfg["label"])
            ax.set_ylabel(f"{variable} ({get_unit_for_variable(variable)})", fontsize=text_cfg["label"])
            ax.tick_params(axis='both', labelsize=text_cfg["tick"])
            ax.legend(fontsize=text_cfg["legend"])
            fig.autofmt_xdate()
            if save:
                logger.info(f"Saving mean ± std time series plot to {fig_path}/mean_std_time_series{suffix}.png")
                fig.savefig(os.path.join(fig_path, f"mean_std_time_series{suffix}.png"), dpi=300)
            if show:
                plt.show()
            plt.close(fig)

        monthly_ts = _aggregate_timeseries_monthly(ts_stats)
        if monthly_ts is not None and plotting.get("plot_monthly_from_daily", True) and "mean" in monthly_ts and "std" in monthly_ts:
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_times = monthly_ts["timestamps"]
            monthly_mean = np.asarray(monthly_ts["mean"], dtype=np.float64)
            monthly_std = np.asarray(monthly_ts["std"], dtype=np.float64)
            line_c = model_color or 'k'
            ax.plot(monthly_times, monthly_mean, color=line_c, lw=1.5, alpha=0.95, label='Monthly mean of daily spatial mean')
            if variable.lower() in ["prcp", "precip", "precipitation"]:
                lower_err = np.minimum(monthly_std, np.maximum(monthly_mean, 0.0))
                yerr = np.vstack([lower_err, monthly_std])
                ax.errorbar(monthly_times, monthly_mean, yerr=yerr, fmt='.', ecolor=line_c, elinewidth=0.8, capsize=2, label='± Monthly mean of daily spatial std', color=line_c, alpha=0.6)
                ax.set_ylim(bottom=0)
            else:
                ax.errorbar(monthly_times, monthly_mean, yerr=monthly_std, fmt='.', ecolor=line_c, elinewidth=0.8, capsize=2, label='± Monthly mean of daily spatial std', color=line_c, alpha=0.6)
            apply_model_colors(ax)
            ax.set_title(f"{variable} monthly summary from daily time-series stats", fontsize=text_cfg["subtitle"])
            ax.set_xlabel("Time", fontsize=text_cfg["label"])
            ax.set_ylabel(f"{variable} ({get_unit_for_variable(variable)})", fontsize=text_cfg["label"])
            ax.tick_params(axis='both', labelsize=text_cfg["tick"])
            ax.legend(fontsize=text_cfg["legend"])
            fig.autofmt_xdate()
            if save:
                path = os.path.join(fig_path, f"mean_std_time_series_monthly_from_daily.png")
                logger.info(f"Saving monthly summary time series plot to {path}")
                fig.savefig(path, dpi=300)
            if show:
                plt.show()
            plt.close(fig)

        keys = plotting.get("plot_stats", ['mean', 'std', 'min', 'max', 'median', 'percentile_25', 'percentile_75'])
        available_keys = [k for k in keys if k in ts_stats and ts_stats[k] is not None]
        n_keys = len(available_keys)
        n_cols = 2
        n_rows = max(1, (n_keys + n_cols - 1) // n_cols)

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows), constrained_layout=True)
        axs = np.atleast_1d(axs).flatten()

        plotted_count = 0
        for k in available_keys:
            axs[plotted_count].plot(times, ts_stats[k], label=_pretty_ts_label(k), alpha=0.85, color=(model_color or 'k'))
            axs[plotted_count].set_title(f"{variable} {_pretty_ts_label(k)} over time", fontsize=text_cfg["subtitle"])
            axs[plotted_count].set_xlabel("Time", fontsize=text_cfg["label"])
            axs[plotted_count].set_ylabel(f"{_pretty_ts_label(k)} ({get_unit_for_variable(variable)})", fontsize=text_cfg["label"])
            axs[plotted_count].tick_params(axis='x', rotation=30, labelsize=text_cfg["tick"])
            axs[plotted_count].tick_params(axis='y', labelsize=text_cfg["tick"])
            axs[plotted_count].legend(fontsize=text_cfg["legend"])
            axs[plotted_count].grid(True)
            if variable.lower() in ["prcp", "precip", "precipitation"] and k in ["mean", "std", "min", "max", "sum"]:
                axs[plotted_count].set_ylim(bottom=0)
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
                    transformed = np.asarray(transformed)
                    transformed = transformed[np.isfinite(transformed)]
                    if transformed.size == 0:
                        logger.warning(f"Transformed data '{transform}' is all non-finite for {model}/{variable}/{split}. Skipping this transform in histograms.")
                        continue

                    # Plot alongside original
                    ax1.hist(transformed, bins=100, color=colors[transform], alpha=0.5, label=labels[transform])
                    ax1.axvline(np.mean(transformed), color=colors[transform], linestyle='--', linewidth=1)

                    # Plot alongside zoomed inset
                    ax2.hist(transformed, bins=100, alpha=0.7, label=labels[transform], color=colors[transform])
                    ax2.axvline(np.mean(transformed), color=colors[transform], linestyle='--', linewidth=1)

        if log_scale:
            ax1.set_yscale('log')
            ax2.set_yscale('log')

        ax1.set_title(f"Histogram of all {variable} pixel values ({agg_str.strip()})", fontsize=text_cfg["subtitle"])
        ax1.set_xlabel(f"{variable} ({get_unit_for_variable(variable)})", fontsize=text_cfg["label"])
        ax1.set_ylabel("Frequency", fontsize=text_cfg["label"])
        ax1.tick_params(axis='both', labelsize=text_cfg["tick"])
        ax1.legend(fontsize=text_cfg["legend"])

        ax2.set_title(f"Zoomed histogram of all {variable} pixel values ({agg_str.strip()})", fontsize=text_cfg["subtitle"])
        ax2.set_xlabel(f"{variable} transformed value", fontsize=text_cfg["label"])
        ax2.set_ylabel("Frequency", fontsize=text_cfg["label"])
        ax2.tick_params(axis='both', labelsize=text_cfg["tick"])
        ax2.legend(fontsize=text_cfg["legend"])

        if save:
            logger.info(f"Saving histogram plot to {fig_path}/histogram_pixels_{variable}_{suffix}.png")
            fig.savefig(os.path.join(fig_path, f"histogram_pixels_{variable}_{suffix}.png"), dpi=300)
        if show:
            plt.show()
        plt.close(fig)






    # === 3. Histogram of time-series statistics ===
    if "timeseries" in stats_dict:
        keys = plotting.get("plot_stats", ['mean', 'std', 'min', 'max', 'median', 'percentile_25', 'percentile_75'])
        available_keys = [k for k in keys if k in stats_dict["timeseries"] and stats_dict["timeseries"][k] is not None]
        if available_keys:
            n_cols = 2
            n_rows = max(1, (len(available_keys) + n_cols - 1) // n_cols)
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows), constrained_layout=True)
            axs = np.atleast_1d(axs).flatten()
            for ax, k in zip(axs, available_keys):
                values = np.asarray(stats_dict["timeseries"][k], dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    ax.set_visible(False)
                    continue
                ax.hist(values, bins=100, alpha=0.7, label=_pretty_ts_label(k), color=get_color_for_variable(variable, model if model is not None else ""))
                ax.set_title(f"Histogram of {_pretty_ts_label(k)}", fontsize=text_cfg["subtitle"])
                if log_scale:
                    ax.set_yscale('log')
                ax.set_xlabel(f"{_pretty_ts_label(k)} ({get_unit_for_variable(variable)})", fontsize=text_cfg["label"])
                ax.set_ylabel("Frequency", fontsize=text_cfg["label"])
                ax.tick_params(axis='both', labelsize=text_cfg["tick"])
                ax.legend(fontsize=text_cfg["legend"])
            for ax in axs[len(available_keys):]:
                fig.delaxes(ax)
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