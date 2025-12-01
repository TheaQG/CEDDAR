import logging
import datetime
import os
import json
import numpy as np
from typing import Dict, Any

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



def aggregate_data(data, agg_time, agg_method):
    """ 
        Aggregate data across multiple files or data chunks.
        This could involve averaging, summing, or otherwise combining the data.
    """
    aggregation_time = agg_time  # e.g., "weekly", "monthly", "yearly"
    
    cutouts = data["cutouts"]
    timestamps = data["timestamps"]

    # Convert timestamps to datetime objects if not already
    if not isinstance(timestamps[0], datetime.datetime):
        timestamps = [datetime.datetime.fromisoformat(ts) if isinstance(ts, str) else ts for ts in timestamps]

    # Group indices by aggregation_time
    groups = {}
    for idx, ts in enumerate(timestamps):
        if aggregation_time == "weekly":
            key = (ts.year, ts.isocalendar()[1])  # Year and week number
        elif aggregation_time == "monthly":
            key = (ts.year, ts.month)  # Year and month
        elif aggregation_time == "yearly":
            key = (ts.year,)  # Year only
        elif aggregation_time == "daily":
            # The data is already daily, so exit function
            logger.info("Aggregation time is daily, no aggregation performed.")
            cutouts_stack = np.stack(cutouts)

            return {
                "cutouts": cutouts_stack,
                "stack": cutouts_stack.flatten(),
                "timestamps": timestamps
            }
        else:
            raise ValueError(f"Unsupported aggregation_time: {aggregation_time}")

        # Store the index in the appropriate group
        groups.setdefault(key, []).append(idx)

    aggregated_cutouts = []
    aggregated_timestamps = []

    for key, indices in groups.items():
        # Group cutouts by the current key
        group_cutouts = [cutouts[i] for i in indices]
        # Stack the group arrays into a single array
        stack_group = np.stack(group_cutouts)  # Shape: (num_in_group, H, W)

        if agg_method == "mean":
            # Compute mean across time axis (axis = 0)
            agg_arrays = np.mean(stack_group, axis=0) #group_cutouts[0].copy(data=np.mean(stack_group, axis=0))
            
        elif agg_method == "sum":
            # Compute sum across time axis
            agg_arrays = np.sum(stack_group, axis=0) #group_cutouts[0].copy(data=np.sum(stack_group, axis=0))
            
        elif agg_method == "max":
            # Compute max across time axis
            agg_arrays = np.max(stack_group, axis=0) #group_cutouts[0].copy(data=np.max(stack_group, axis=0))

        elif agg_method == "min":
            # Compute min across time axis
            agg_arrays = np.min(stack_group, axis=0) #group_cutouts[0].copy(data=np.min(stack_group, axis=0))

        else:
            raise ValueError(f"Unsupported aggregation method: {agg_method}")

        aggregated_cutouts.append(agg_arrays)

        # Generate representative timestamp for the group (always start of _)
        if aggregation_time == "weekly":
            year, week = key
            dt = datetime.datetime(year, 1, 1) + datetime.timedelta(weeks=week-1) # Start of the week
        elif aggregation_time == "monthly":
            year, month = key
            dt = datetime.datetime(year, month, 1) # Start of the month
        elif aggregation_time == "yearly":
            (year, ) = key[0]
            dt = datetime.datetime(year, 1, 1) # Start of the year
        else:
            raise ValueError(f"Unsupported aggregation_time: {aggregation_time}")

        aggregated_timestamps.append(dt)

    cutouts_stack = np.stack(aggregated_cutouts)

    # Stack the aggregated cutouts and return
    return {
        "cutouts": cutouts_stack,
        "timestamps": aggregated_timestamps
    }






def compute_statistics(data,
                       aggregate=False,
                       agg_time="monthly",
                       agg_method="mean",
                       return_timeseries=True,
                       return_cutout_stats=True,
                       return_all=True,
                       print_stats=False,
                       save_glob_stats=True,
                       variable="variable",
                       model="model",
                       split="all",
                       domain_str="_589x789",
                       crop_region_str="_0_0_180_180",
                       small_data_batch=False,
                       cfg={},
                       stats_save_path=".",
                       log_stats=False,
                       pool_pixels=True,
                       save_full_stats_npz: bool = True,
                       ):
    """
        Compute statistics for the given data.
        Mean, std, min, max etc. per file or full stack
        If aggregate = True, aggregate temporally before computing statistcs
    """
    if return_all:
        return_timeseries = True
        return_cutout_stats = True

    cutouts = data["cutouts"]
    timestamps = data.get("timestamps", None)

    logger.info(f"Length of cutouts: {len(cutouts)}")
    logger.info(f"Shape of single cutout: {cutouts[0].shape}")
    
    if aggregate: 
        aggregation = aggregate_data(data, agg_time, agg_method)
        cutouts = aggregation["cutouts"]
        timestamps = aggregation["timestamps"]
        logger.info(f"After aggregation ({agg_method} over {agg_time}):")
        logger.info(f"  New length of cutouts: {len(cutouts)}")
        logger.info(f"  New shape of single cutout: {cutouts[0].shape}")

    stack = np.stack(cutouts)  # Shape: (T, H, W)
    global_flat = stack.flatten()  # Shape: (T * H * W,)

    # === 1. Global statistics across all time and pixels ===
    # Use global_stats to save these for training normalization
    global_stats_result = compute_global_stats(
        data_dict=data,
        variable=variable,
        model=model,
        domain_str=domain_str,
        split=split,
        crop_region_str=crop_region_str,
        cfg=cfg,
        stats_save_path=stats_save_path,
        save=save_glob_stats,
        log_stats=log_stats,
        pool_pixels=pool_pixels,
        small_data_batch=small_data_batch
    )

    # === 2. Per-timestep statistics (time-series) ===
    time_series_stats = {}
    if return_timeseries:
        time_series_stats = {
            "mean": np.mean(stack, axis=(1, 2)),  # Shape: (T,)
            "std": np.std(stack, axis=(1, 2)),    # Shape: (T,)
            "min": np.min(stack, axis=(1, 2)),    # Shape: (T,)
            "max": np.max(stack, axis=(1, 2)),    # Shape: (T,)
            "median": np.median(stack, axis=(1, 2)),  # Shape: (T,)
            "percentile_25": np.percentile(stack, 25, axis=(1, 2)),  # Shape: (T,)
            "percentile_75": np.percentile(stack, 75, axis=(1, 2))   # Shape: (T,)
        }
        if timestamps is not None:
            time_series_stats["timestamps"] = timestamps

    # === 3. Per-pixel statistics across all time ===
    cutout_stats = {}
    if return_cutout_stats:
        cutout_stats = {
            "mean": np.mean(stack, axis=0),  # Shape: (H, W)
            "std": np.std(stack, axis=0),    # Shape: (H, W)
            "min": np.min(stack, axis=0),    # Shape: (H, W)
            "max": np.max(stack, axis=0),    # Shape: (H, W)
            "median": np.median(stack, axis=0),  # Shape: (H, W)
            "percentile_25": np.percentile(stack, 25, axis=0),  # Shape: (H, W)
            "percentile_75": np.percentile(stack, 75, axis=0)   # Shape: (H, W)
        }


    if print_stats:
        logger.info("\n   COMPUTED BASIC STATS:")
        for key, value in global_stats_result.items():
            logger.info(f"          {key}: {value}")

    # === Optional save of full stats (time series + per-pixel) to NPZ for plot-only reloads ===
    if save_full_stats_npz:
        # Build save dir and filename consistent with global stats JSON
        save_dir = os.path.join(stats_save_path, model, variable, split)
        os.makedirs(save_dir, exist_ok=True)
        if small_data_batch:
            fname = f"stats_timeseries_cutout__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}__small.npz"
        else:
            fname = f"stats_timeseries_cutout__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}.npz"
        fpath = os.path.join(save_dir, fname)
        try:
            # Flatten dicts to arrays where possible; store timestamps as ISO strings if present
            ts = {}
            if return_timeseries and time_series_stats:
                ts = {k: np.array(v) for k, v in time_series_stats.items() if k != "timestamps"}
                if "timestamps" in time_series_stats and time_series_stats["timestamps"] is not None:
                    ts["timestamps_iso"] = np.array([
                        (t.isoformat() if hasattr(t, "isoformat") else str(t))
                        for t in time_series_stats["timestamps"]
                    ])
            co = {}
            if return_cutout_stats and cutout_stats:
                co = {f"cutout_{k}": np.array(v) for k, v in cutout_stats.items()}
            np.savez_compressed(fpath, **ts, **co)
            logger.info(f"[INFO] Saved time series and cutout stats to {fpath}")
        except Exception as e:
            logger.warning(f"[WARN] Failed to save NPZ full stats to {fpath}: {e}")

    return global_stats_result, cutout_stats, time_series_stats





def compute_global_stats(data_dict,
                      variable,
                      model,
                      split,
                      domain_str,
                      crop_region_str,
                      cfg,
                      stats_save_path,
                      small_data_batch=False,
                      save=False,
                      log_stats=False,
                      pool_pixels=True
                      ):
    """
        Compute global pixel-wise statistics over the stack of cutouts
        and save them for use in training normalization.
    """
    save_dir = os.path.join(stats_save_path, model, variable, split)
    os.makedirs(save_dir, exist_ok=True)

    cutouts = data_dict["cutouts"]
    # Gives the stats for each pixel position across time, returns shape (H, W)
    stacked = np.stack(cutouts) #np.stack([x.values for x in cutouts]) # Shape: (T, H, W)

    # Pool across pixels if specified, otherwise keep spatial dimensions
    if pool_pixels:
        stacked = stacked.flatten()  # Shape: (T * H * W,)

    global_mean = np.mean(stacked)
    global_std = np.std(stacked)
    global_min = np.min(stacked)
    global_max = np.max(stacked)

    # To avoid issues with log(0), we add a small constant
    # Instead of just global_min >= 0, only get log stats if asked for it
    if log_stats:
        stacked = np.where(stacked <= 0, 1e-8, stacked)  # Replace non-positive values with a small constant
        log_stack = np.log(stacked)
        log_mean = np.mean(log_stack)
        log_std = np.std(log_stack)
        log_min = np.min(log_stack)
        log_max = np.max(log_stack)
    else:
        log_mean = log_std = log_min = log_max = None


    stats = {
        "mean": global_mean,
        "std": global_std,
        "min": global_min,
        "max": global_max,
        "log_mean": log_mean,
        "log_std": log_std,
        "log_min": log_min,
        "log_max": log_max
    }

    split = cfg.get("data", {}).get("split", "unknown")

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if small_data_batch:
            filename = f"global_stats__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}__small.json"
        else:
            filename = f"global_stats__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}.json"
        filepath = os.path.join(save_dir, filename)

    
        with open(filepath, 'w') as f:
            for k, v in stats.items():
                if v is None:
                    stats[k] = None
                    logger.warning(f"{k} is None, saving as null in JSON.")
                else:
                    stats[k] = float(v)
            json.dump(stats, f)


        logger.info(f"[INFO] Global statistics saved to {filepath}")

    return stats



def load_global_stats(variable, model, domain_str, crop_region_str, split, dir_load):
    """
        Load previously saved global statistics for a given variable, model, domain, and crop region.
    """
    stats_load_dir = os.path.join(dir_load, model, variable, split)
    stats_load_path = os.path.join(stats_load_dir, f"global_stats__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}.json")
    
    if not os.path.exists(stats_load_path):
        logger.warning(f"Stats file not found: {stats_load_path}")
        return None
    logger.info(f"Loading stats from {stats_load_path}")

    with open(stats_load_path, "r") as f:
        stats = json.load(f)
    
    return stats











