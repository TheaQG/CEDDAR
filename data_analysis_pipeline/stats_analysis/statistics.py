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
                       streaming: bool = False, # Whether to compute stats in a streaming fashion (not fully implemented yet)
                       ):
    """
        Compute statistics for the given data.
        Mean, std, min, max etc. per file or full stack
        If aggregate = True, aggregate temporally before computing statistcs
    """
    if return_all:
        return_timeseries = True
        return_cutout_stats = True

    if streaming:
        entries_iter = data.get("entries_iter", None)
        loader = data.get("loader", None)
        expected_shape = data.get("expected_shape", None)

        if entries_iter is None and loader is not None and hasattr(loader, "iter_entries"):
            entries_iter = loader.iter_entries()
            if expected_shape is None and hasattr(loader, "get_expected_shape"):
                expected_shape = loader.get_expected_shape()

        if entries_iter is None:
            raise ValueError("streaming=True requires data['entries_iter'] or data['loader'] with iter_entries().")

        return compute_statistics_streaming(
            entries_iter=entries_iter,
            expected_shape=expected_shape,
            # pass through the same knobs:
            aggregate=aggregate,
            agg_time=agg_time,
            agg_method=agg_method,
            return_timeseries=return_timeseries,
            return_cutout_stats=return_cutout_stats,
            return_all=return_all,
            print_stats=print_stats,
            save_glob_stats=save_glob_stats,
            variable=variable,
            model=model,
            split=split,
            domain_str=domain_str,
            crop_region_str=crop_region_str,
            small_data_batch=small_data_batch,
            cfg=cfg,
            stats_save_path=stats_save_path,
            log_stats=log_stats,
            pool_pixels=pool_pixels,
            save_full_stats_npz=save_full_stats_npz,
        )

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
            "percentile_75": np.percentile(stack, 75, axis=(1, 2)),   # Shape: (T,)
            "sum": np.sum(stack, axis=(1, 2))       # Shape: (T,)
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
            "percentile_75": np.percentile(stack, 75, axis=0),   # Shape: (H, W)
            "sum": np.sum(stack, axis=0)       # Shape: (H, W)
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


def compute_statistics_streaming(
    entries_iter,
    expected_shape=None,
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
    crop_region_str="full",
    small_data_batch=False,
    cfg=None,
    stats_save_path=".",
    log_stats=False,
    pool_pixels=True,
    save_full_stats_npz: bool = True,
):
    """
    Streaming statistics computation.

    Avoids materializing the full (T,H,W) stack in memory.

    Computes:
      - Global (pooled) stats: mean/std/min/max/sum/count (+ optional log/asinh/boxcox)
      - Optional per-timestep stats (scalar over pixels): mean/std/min/max/sum
      - Optional per-pixel stats over time: mean/std/min/max/sum (maps) if pool_pixels=False

    Notes:
      - Exact median/percentiles are NOT computed in streaming mode.
      - Temporal aggregation (weekly/monthly/yearly) is NOT implemented in streaming mode;
        we warn and proceed un-aggregated.
    """
    import json
    import os
    import numpy as np

    if cfg is None:
        cfg = {}

    if return_all:
        return_timeseries = True
        return_cutout_stats = True

    if aggregate and agg_time not in ["daily", None, "none", ""]:
        logger.warning(
            "[streaming] aggregate=True with agg_time='%s' is not implemented in streaming mode. "
            "Proceeding without aggregation to avoid high memory use.",
            agg_time,
        )

    class _Welford:
        __slots__ = ("n", "mean", "M2", "min", "max", "sum", "n_nan")

        def __init__(self):
            self.n = 0
            self.mean = 0.0
            self.M2 = 0.0
            self.min = np.inf
            self.max = -np.inf
            self.sum = 0.0
            self.n_nan = 0  # count of dropped non-finite values

        def update_array(self, x: np.ndarray):
            # x is 1D float64
            if x.size == 0:
                return

            # Drop NaN/Inf so a single bad field doesn't poison the whole run
            finite = np.isfinite(x)
            if not np.all(finite):
                self.n_nan += int((~finite).sum())
                x = x[finite]
                if x.size == 0:
                    return

            x_min = float(np.min(x))
            x_max = float(np.max(x))
            x_sum = float(np.sum(x))
            x_n = int(x.size)
            x_mean = float(x_sum / x_n)
            x_M2 = float(np.sum((x - x_mean) ** 2))

            if self.n == 0:
                self.n = x_n
                self.mean = x_mean
                self.M2 = x_M2
                self.sum = x_sum
                self.min = x_min
                self.max = x_max
                return

            n_total = self.n + x_n
            delta = x_mean - self.mean
            self.mean = self.mean + delta * (x_n / n_total)
            self.M2 = self.M2 + x_M2 + (delta ** 2) * (self.n * x_n / n_total)
            self.n = n_total
            self.sum += x_sum
            if x_min < self.min:
                self.min = x_min
            if x_max > self.max:
                self.max = x_max

        @property
        def var(self):
            return self.M2 / self.n if self.n > 0 else np.nan

        @property
        def std(self):
            v = self.var
            return float(np.sqrt(v)) if np.isfinite(v) else np.nan

    # Per-pixel accumulators (only used if pool_pixels=False)
    sum_map = None
    sumsq_map = None
    min_map = None
    max_map = None

    # Per-timestep stats lists
    ts_mean, ts_std, ts_min, ts_max, ts_sum, ts_timestamps = [], [], [], [], [], []

    # Global pooled stats
    g = _Welford()

    do_log = bool(log_stats)
    g_log = _Welford() if do_log else None

    stats_cfg = cfg.get("statistics", {}) if isinstance(cfg, dict) else {}
    asinh_scale = float(stats_cfg.get("asinh_scale", 1.0))
    boxcox_lambda = float(stats_cfg.get("boxcox_lambda", 0.3))
    boxcox_eps = float(stats_cfg.get("boxcox_eps", 0.01))

    do_asinh_boxcox = bool(do_log and variable in ["prcp", "cape"])
    g_asinh = _Welford() if do_asinh_boxcox else None
    g_boxcox = _Welford() if do_asinh_boxcox else None

    first_shape = None
    count_t = 0
    first_nonfinite = None  # (i, ts, n_nonfinite_in_field)
    total_nonfinite_fields = 0

    for i, (arr, ts) in enumerate(entries_iter, 1):
        if arr is None:
            continue
        arr = np.asarray(arr)

        if first_shape is None:
            first_shape = arr.shape
            logger.info(f"[streaming] First cutout shape: {first_shape}")

            if expected_shape is not None and tuple(first_shape) != tuple(expected_shape):
                raise ValueError(
                    f"[streaming] Shape mismatch: got {first_shape} but expected {expected_shape}. "
                    f"(This usually indicates crop_region ordering problems.)"
                )

            if return_cutout_stats and not pool_pixels:
                sum_map = np.zeros(first_shape, dtype=np.float64)
                sumsq_map = np.zeros(first_shape, dtype=np.float64)
                min_map = np.full(first_shape, np.inf, dtype=np.float64)
                max_map = np.full(first_shape, -np.inf, dtype=np.float64)

        if first_shape is not None and arr.shape != first_shape:
            raise ValueError(f"[streaming] Inconsistent cutout shape at i={i}: {arr.shape} vs {first_shape}")

        a64 = arr.astype(np.float64, copy=False)

        # Detect non-finite values early
        if not np.isfinite(a64).all():
            n_bad = int((~np.isfinite(a64)).sum())
            total_nonfinite_fields += 1
            if first_nonfinite is None:
                first_nonfinite = (i, ts, n_bad)
            # We continue; pooled stats will ignore non-finite values
            # and per-timestep stats use nan-safe reductions below.

        if return_timeseries:
            ts_mean.append(float(np.nanmean(a64)))
            ts_std.append(float(np.nanstd(a64)))
            ts_min.append(float(np.nanmin(a64)))
            ts_max.append(float(np.nanmax(a64)))
            ts_sum.append(float(np.nansum(a64)))
            ts_timestamps.append(ts)

        if pool_pixels:
            x = a64.ravel()
            g.update_array(x)

            if do_log:
                # Work only on finite values; avoid log of <=0
                x_pos = np.where(x <= 0, 1e-8, x)
                g_log.update_array(np.log(x_pos))

            if do_asinh_boxcox:
                nonneg = np.where(x < 0, 0.0, x)

                a = max(asinh_scale, 1e-8)
                g_asinh.update_array(np.arcsinh(nonneg / a))

                x_pos2 = np.where(nonneg <= 0, boxcox_eps, nonneg + boxcox_eps)
                if abs(boxcox_lambda) < 1e-6:
                    bc = np.log(x_pos2)
                else:
                    bc = (np.power(x_pos2, boxcox_lambda) - 1.0) / boxcox_lambda
                g_boxcox.update_array(bc)

        else:
            # Per-pixel map stats
            sum_map += a64
            sumsq_map += a64 * a64
            np.minimum(min_map, a64, out=min_map)
            np.maximum(max_map, a64, out=max_map)

        count_t += 1
        if i % 500 == 0:
            logger.info(f"[streaming] Processed {i} entries...")

    if count_t == 0:
        raise ValueError("[streaming] No entries were processed (empty iterator).")

    if first_nonfinite is not None:
        i0, ts0, nbad0 = first_nonfinite
        logger.warning(
            "[streaming] Detected non-finite values (NaN/Inf) in %d fields. First occurrence at entry %d (timestamp=%s) with %d non-finite values. "
            "Global pooled stats will ignore non-finite values.",
            total_nonfinite_fields,
            i0,
            str(ts0),
            nbad0,
        )

    global_stats = {
        "mean": float(g.mean) if pool_pixels else None,
        "std": float(g.std) if pool_pixels else None,
        "min": float(g.min) if pool_pixels else None,
        "max": float(g.max) if pool_pixels else None,
        "sum": float(g.sum) if pool_pixels else None,
        "count": int(g.n) if pool_pixels else None,
        "yearly_sum": float(g.sum / (count_t / 365.0)) if pool_pixels else None,
        "log_mean": float(g_log.mean) if (do_log and pool_pixels) else None,
        "log_std": float(g_log.std) if (do_log and pool_pixels) else None,
        "log_min": float(g_log.min) if (do_log and pool_pixels) else None,
        "log_max": float(g_log.max) if (do_log and pool_pixels) else None,
        "asinh_mean": float(g_asinh.mean) if (do_asinh_boxcox and pool_pixels) else None,
        "asinh_std": float(g_asinh.std) if (do_asinh_boxcox and pool_pixels) else None,
        "asinh_min": float(g_asinh.min) if (do_asinh_boxcox and pool_pixels) else None,
        "asinh_max": float(g_asinh.max) if (do_asinh_boxcox and pool_pixels) else None,
        "asinh_scale": float(asinh_scale) if (do_asinh_boxcox and pool_pixels) else None,
        "boxcox_mean": float(g_boxcox.mean) if (do_asinh_boxcox and pool_pixels) else None,
        "boxcox_std": float(g_boxcox.std) if (do_asinh_boxcox and pool_pixels) else None,
        "boxcox_min": float(g_boxcox.min) if (do_asinh_boxcox and pool_pixels) else None,
        "boxcox_max": float(g_boxcox.max) if (do_asinh_boxcox and pool_pixels) else None,
        "boxcox_lambda": float(boxcox_lambda) if (do_asinh_boxcox and pool_pixels) else None,
        "dropped_nonfinite_global": int(g.n_nan) if pool_pixels else None,
        "dropped_nonfinite_log": int(g_log.n_nan) if (do_log and pool_pixels) else None,
        "dropped_nonfinite_asinh": int(g_asinh.n_nan) if (do_asinh_boxcox and pool_pixels) else None,
        "dropped_nonfinite_boxcox": int(g_boxcox.n_nan) if (do_asinh_boxcox and pool_pixels) else None,
    }

    if save_glob_stats:
        save_dir = os.path.join(stats_save_path, model, variable, split)
        os.makedirs(save_dir, exist_ok=True)
        if small_data_batch:
            filename = f"global_stats__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}__small.json"
        else:
            filename = f"global_stats__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}.json"
        filepath = os.path.join(save_dir, filename)

        out_stats = {k: (None if v is None else float(v)) for k, v in global_stats.items()}
        with open(filepath, "w") as f:
            json.dump(out_stats, f)
        logger.info(f"[INFO] Global statistics saved to {filepath} (streaming)")

    time_series_stats = {}
    if return_timeseries:
        time_series_stats = {
            "mean": np.array(ts_mean, dtype=np.float64),
            "std": np.array(ts_std, dtype=np.float64),
            "min": np.array(ts_min, dtype=np.float64),
            "max": np.array(ts_max, dtype=np.float64),
            "sum": np.array(ts_sum, dtype=np.float64),
            "median": None,
            "percentile_25": None,
            "percentile_75": None,
            "timestamps": ts_timestamps,
        }

    cutout_stats = {}
    if return_cutout_stats:
        if pool_pixels:
            cutout_stats = {}
        else:
            n = float(count_t)
            mean_map = sum_map / n
            var_map = (sumsq_map / n) - (mean_map * mean_map)
            var_map = np.maximum(var_map, 0.0)
            std_map = np.sqrt(var_map)
            cutout_stats = {
                "mean": mean_map.astype(np.float32),
                "std": std_map.astype(np.float32),
                "min": min_map.astype(np.float32),
                "max": max_map.astype(np.float32),
                "sum": sum_map.astype(np.float32),
                "median": None,
                "percentile_25": None,
                "percentile_75": None,
            }

    if print_stats:
        logger.info("\n   COMPUTED GLOBAL STATS (streaming):")
        for k, v in global_stats.items():
            logger.info(f"          {k}: {v}")

    if save_full_stats_npz:
        save_dir = os.path.join(stats_save_path, model, variable, split)
        os.makedirs(save_dir, exist_ok=True)
        if small_data_batch:
            fname = f"stats_timeseries_cutout__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}__small.npz"
        else:
            fname = f"stats_timeseries_cutout__{model}__{domain_str}__crop__{crop_region_str}__{variable}__{split}.npz"
        fpath = os.path.join(save_dir, fname)

        payload = {}
        if return_timeseries and time_series_stats:
            for k in ["mean", "std", "min", "max", "sum"]:
                if time_series_stats.get(k) is not None:
                    payload[k] = np.asarray(time_series_stats[k])
            if time_series_stats.get("timestamps") is not None:
                payload["timestamps_iso"] = np.array(
                    [(t.isoformat() if hasattr(t, "isoformat") else str(t)) for t in time_series_stats["timestamps"]]
                )

        if return_cutout_stats and cutout_stats:
            for k, v in cutout_stats.items():
                if v is None:
                    continue
                payload[f"cutout_{k}"] = np.asarray(v)

        try:
            np.savez_compressed(fpath, **payload)
            logger.info(f"[INFO] Saved time series and cutout stats to {fpath} (streaming)")
        except Exception as e:
            logger.warning(f"[WARN] Failed to save NPZ full stats to {fpath} (streaming): {e}")

    return global_stats, cutout_stats, time_series_stats


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

    # NaN/Inf safe reductions (if any bad values exist)
    global_mean = np.nanmean(stacked)
    global_std = np.nanstd(stacked)
    global_min = np.nanmin(stacked)
    global_max = np.nanmax(stacked)
    global_sum = np.nansum(stacked)
    global_count = int(np.isfinite(stacked).sum())
    global_yearly_sum = global_sum / (len(cutouts) / 365.0)  # Approximate yearly sum

    n_bad = int((~np.isfinite(stacked)).sum())
    if n_bad > 0:
        logger.warning(f"[global_stats] Detected {n_bad} non-finite (NaN/Inf) values for {model}/{variable}/{split}. Using nan-safe stats and ignoring them.")

    # Prepare containers for optional nonlinear-transform stats
    asinh_mean = asinh_std = asinh_min = asinh_max = None
    boxcox_mean = boxcox_std = boxcox_min = boxcox_max = None
    asinh_scale = None
    boxcox_lambda = None

    # Hyperparameters for nonlinear transforms (used both in stats and training)
    stats_cfg = cfg.get("statistics", {}) if isinstance(cfg, dict) else {}
    # asinh_scale is the mm/day scale where asinh transitions from linear to log-like
    asinh_scale_cfg = stats_cfg.get("asinh_scale", 1.0)
    boxcox_lambda_cfg = stats_cfg.get("boxcox_lambda", 0.3)
    boxcox_eps_cfg = stats_cfg.get("boxcox_eps", 0.01)

    # To avoid issues with log(0), we add a small constant
    # Instead of just global_min >= 0, only get log stats if asked for it
    if log_stats:
        stacked_pos = np.where(stacked <= 0, 1e-8, stacked)  # Replace non-positive values with a small constant
        log_stack = np.log(stacked_pos)
        log_mean = np.mean(log_stack)
        log_std = np.std(log_stack)
        log_min = np.min(log_stack)
        log_max = np.max(log_stack)
    else:
        log_mean = log_std = log_min = log_max = None

    # === Optional: asinh and Box–Cox stats for precip-like variables ===
    # These are used by PrcpAsinhZScoreTransform and PrcpBoxCoxZScoreTransform.
    # We compute them by default for prcp/cape if log_stats is True.
    if log_stats and variable in ["prcp", "cape"]:
        # Ensure non-negative input for these nonlinear transforms
        nonneg = np.where(stacked < 0, 0.0, stacked)

        # Asinh stats
        try:
            asinh_scale = float(asinh_scale_cfg)
        except Exception:
            asinh_scale = 1.0
        a = max(asinh_scale, 1e-8)
        asinh_stack = np.arcsinh(nonneg / a)
        asinh_mean = float(np.mean(asinh_stack))
        asinh_std = float(np.std(asinh_stack))
        asinh_min = float(np.min(asinh_stack))
        asinh_max = float(np.max(asinh_stack))

        # Box–Cox stats
        try:
            boxcox_lambda = float(boxcox_lambda_cfg)
        except Exception:
            boxcox_lambda = 0.3
        try:
            boxcox_eps = float(boxcox_eps_cfg)
        except Exception:
            boxcox_eps = 0.01

        # Shift to positive (>= eps) for Box–Cox
        x_pos = np.where(nonneg <= 0, boxcox_eps, nonneg + boxcox_eps)
        if abs(boxcox_lambda) < 1e-6:
            boxcox_stack = np.log(x_pos)
        else:
            boxcox_stack = (np.power(x_pos, boxcox_lambda) - 1.0) / boxcox_lambda

        boxcox_mean = float(np.mean(boxcox_stack))
        boxcox_std = float(np.std(boxcox_stack))
        boxcox_min = float(np.min(boxcox_stack))
        boxcox_max = float(np.max(boxcox_stack))


    stats = {
        "mean": global_mean,
        "std": global_std,
        "min": global_min,
        "max": global_max,
        "log_mean": log_mean,
        "log_std": log_std,
        "log_min": log_min,
        "log_max": log_max,
        "sum": global_sum,
        "count": global_count,
        "yearly_sum": global_yearly_sum,
        # Asinh-based transform stats
        "asinh_mean": asinh_mean,
        "asinh_std": asinh_std,
        "asinh_min": asinh_min,
        "asinh_max": asinh_max,
        "asinh_scale": asinh_scale,
        # Box–Cox transform stats
        "boxcox_mean": boxcox_mean,
        "boxcox_std": boxcox_std,
        "boxcox_min": boxcox_min,
        "boxcox_max": boxcox_max,
        "boxcox_lambda": boxcox_lambda,
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











