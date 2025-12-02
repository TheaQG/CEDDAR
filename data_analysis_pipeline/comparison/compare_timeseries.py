"""
Time series comparison module (pandas-free).
  - Aggregate per-day differences: bias over time, variance differences, time-varying correlations
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as _dt
from data_analysis_pipeline.comparison.compare_fields import compute_field_stats

from ceddar.variable_utils import get_color_for_model
from ceddar.plotting_utils import apply_model_colors

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Detect and normalize input structures ---

def _looks_like_date_key(k):
    if isinstance(k, (_dt.date, _dt.datetime, np.datetime64)):
        return True
    if isinstance(k, str):
        s = k.strip().replace("-", "")
        if len(s) == 8:
            try:
                _dt.datetime.strptime(s, "%Y%m%d")
                return True
            except Exception:
                pass
    return False

def _is_array_like(v):
    try:
        return hasattr(v, "__array__") or hasattr(v, "shape") or isinstance(v, (list, tuple))
    except Exception:
        return False

def _as_ts_and_cutouts(d: dict):
    """
    Normalize different input shapes to (timestamps_list, cutouts_list).
    Supports:
      A) {'timestamps': [...], 'cutouts': [...]}
      B) {date_key -> 2D array}
    Returns ([], []) if nothing recognized.
    """
    if not isinstance(d, dict):
        return [], []

    # Case A: parallel arrays
    ts = d.get("timestamps")
    cu = d.get("cutouts")
    if ts is None:
        for alias in ["dates", "date_list", "date_array", "times", "time"]:
            if alias in d:
                ts = d[alias]
                break
    if ts is not None and cu is not None and hasattr(ts, "__len__") and hasattr(cu, "__len__") and len(ts) == len(cu) and len(ts) > 0:
        return list(ts), list(cu)

    # Case B: dict keyed by dates
    meta = {"timestamps","dates","date_list","date_array","times","time","cutouts","X","Y","meta","attrs"}
    candidates = [(k, v) for k, v in d.items() if k not in meta]
    if candidates and all(_looks_like_date_key(k) and _is_array_like(v) for k, v in candidates[:min(10, len(candidates))]):
        def _sort_key(x):
            if isinstance(x, (_dt.date, _dt.datetime)):
                return x.toordinal()
            if isinstance(x, np.datetime64):
                return int(x.astype("datetime64[D]").astype(int))
            s = str(x).replace("-", "")
            try:
                return _dt.datetime.strptime(s, "%Y%m%d").toordinal()
            except Exception:
                return hash(s)
        candidates.sort(key=lambda kv: _sort_key(kv[0]))
        ks = [k for k, _ in candidates]
        vs = [v for _, v in candidates]
        return ks, vs

    return [], []

def _extract_timestamps(d: dict):
    """
    Robustly extract a list of timestamps from a data dict that may use
    different key names. Returns [] if none found.
    """
    if d is None:
        return []
    for k in ["timestamps", "dates", "date", "date_list", "date_array", "time", "times"]:
        ts = d.get(k, [])
        # Accept numpy arrays or lists; ignore scalars
        if ts is not None and hasattr(ts, "__len__") and len(ts) > 0:
            return list(ts)
    # Diagnostics
    logger.debug(f"_extract_timestamps: available keys={list(d.keys())}")
    return []

# Helper: normalize various date types to 'YYYYMMDD' string
def _normalize_date_key(d):
    """
    Return a canonical 'YYYYMMDD' string for a variety of input date types:
    - datetime.date/datetime.datetime
    - numpy.datetime64
    - 'YYYYMMDD' or 'YYYY-MM-DD' strings
    - Falls back to naive string slicing for other types
    """
    try:
        # Python datetime/date
        if isinstance(d, (_dt.datetime, _dt.date)):
            return d.strftime("%Y%m%d")
        # Numpy datetime64 -> 'YYYY-MM-DD'
        if isinstance(d, np.datetime64):
            s = np.datetime_as_string(d, unit='D')
            return s.replace("-", "")
        # String inputs
        if isinstance(d, str):
            s = d.strip()
            if "-" in s:
                s = s.replace("-", "")
            return s[:8]
    except Exception:
        pass
    # Fallback
    s = str(d)
    if "-" in s:
        s = s.replace("-", "")
    return s[:8]


#######################################################################
# === Daily domain-mean series + aggregation plotting ===
#######################################################################

def _compute_daily_domain_means(data_dict):
    """
    Accepts either {'timestamps','cutouts'} or {date -> 2D array}.
    Returns (dates_sorted: list[datetime.date], values_sorted: np.ndarray)
    """
    ts, cu = _as_ts_and_cutouts(data_dict)
    if not ts or not cu:
        return [], np.array([])
    # Convert to datetime.date and sort
    def _to_date(x):
        if isinstance(x, _dt.datetime):
            return x.date()
        if isinstance(x, _dt.date):
            return x
        if isinstance(x, np.datetime64):
            s = np.datetime_as_string(x, unit='D')
            return _dt.datetime.strptime(s, "%Y-%m-%d").date()
        s = str(x).replace("-", "")
        try:
            return _dt.datetime.strptime(s, "%Y%m%d").date()
        except Exception:
            # fallback: try ISO
            try:
                return _dt.datetime.strptime(str(x), "%Y-%m-%d").date()
            except Exception:
                return None
    pairs = [(d, np.nanmean(c)) for d, c in zip(ts, cu)]
    pairs = [( _to_date(d), v) for d, v in pairs if _to_date(d) is not None and np.isfinite(v)]

    pairs.sort(key=lambda kv: kv[0].toordinal()) # type: ignore
    if not pairs:
        return [], np.array([])
    dates = [d for d, _ in pairs]
    vals  = np.array([v for _, v in pairs], dtype=float)
    return dates, vals

def _aggregate_series(dates, values, freq='monthly', how='mean'):
    """
    Aggregate a daily series to monthly or weekly with mean +/- std.
    Returns (agg_dates, agg_mean, agg_std, counts)
    """
    if not dates or values.size == 0:
        return [], np.array([]), np.array([]), np.array([])
    from collections import defaultdict
    buckets = defaultdict(list)
    if freq == 'weekly':
        for d, v in zip(dates, values):
            iso = d.isocalendar()  # (year, week, weekday)
            key = (iso[0], iso[1])
            buckets[key].append((d, v))
        agg_dates = []
        means, stds, counts = [], [], []
        for (y, w), items in sorted(buckets.items()):
            # Find Monday
            monday = _dt.date.fromisocalendar(y, w, 1)
            agg_dates.append(monday)
            arr = np.array([v for _, v in items], dtype=float)
            mean = np.nanmean(arr) if how == 'mean' else np.nansum(arr)
            std  = np.nanstd(arr)
            means.append(mean)
            stds.append(std)
            counts.append(arr.size)
        return agg_dates, np.array(means), np.array(stds), np.array(counts)
    else:  # monthly
        for d, v in zip(dates, values):
            key = (d.year, d.month)
            buckets[key].append(v)
        agg_dates = [_dt.date(y, m, 15) for (y, m) in sorted(buckets.keys())]
        means, stds, counts = [], [], []
        for (y, m) in sorted(buckets.keys()):
            arr = np.array(buckets[(y, m)], dtype=float)
            means.append(np.nanmean(arr) if how == 'mean' else np.nansum(arr))
            stds.append(np.nanstd(arr))
            counts.append(arr.size)
        return agg_dates, np.array(means), np.array(stds), np.array(counts)

def plot_daily_series_dual(
    dict_data1,
    dict_data2,
    model1: str,
    model2: str,
    variable: str,
    save_path: str = "./figures",
    show: bool = False,
    fname_prefix: str = "",
    freqs = ("monthly", "weekly"),
    how: str = "mean",
    use_shaded: bool = True,
):
    """
    New figure: daily domain-mean scatter for each dataset + aggregated (monthly/weekly) mean with errorbars.
    Produces one figure per frequency in `freqs`.
    """
    os.makedirs(save_path, exist_ok=True)

    dates1, vals1 = _compute_daily_domain_means(dict_data1)
    dates2, vals2 = _compute_daily_domain_means(dict_data2)

    if not dates1 or not dates2:
        logger.warning("plot_daily_series_dual: empty daily series for one or both datasets; skipping.")
        return

    # Align the daily series to shared dates for fair overlays (optional)
    set1 = {d: v for d, v in zip(dates1, vals1)}
    set2 = {d: v for d, v in zip(dates2, vals2)}
    shared = sorted([d for d in set(set1.keys()) & set(set2.keys()) if d is not None])
    dates = shared
    vals1 = np.array([set1[d] for d in dates], dtype=float)
    vals2 = np.array([set2[d] for d in dates], dtype=float)

    c1 = get_color_for_model(model1)
    c2 = get_color_for_model(model2)
    for freq in freqs:
        fig, ax = plt.subplots(figsize=(13, 4.2), constrained_layout=True)
        # Scatter of daily means
        ax.scatter(np.asarray(dates), vals1, s=1.2, alpha=0.18, label=f"{model1} daily", color=c1)
        ax.scatter(np.asarray(dates), vals2, s=1.2, alpha=0.18, label=f"{model2} daily", color=c2)

        # Aggregated overlays with errorbars or shaded band
        d1, m1, s1, _ = _aggregate_series(dates, vals1, freq=freq, how=how)
        d2, m2, s2, _ = _aggregate_series(dates, vals2, freq=freq, how=how)

        if len(d1) > 0:
            if use_shaded:
                ax.plot(d1, m1, linewidth=1.4, label=f"{model1} {freq} {how}", color=c1)
                ax.fill_between(d1, m1 - s1, m1 + s1, color=c1, alpha=0.12, linewidth=0) # type: ignore
            else:
                ax.errorbar(d1, m1, yerr=s1, fmt='-', linewidth=0.9, markersize=0, # type: ignore
                            elinewidth=0.6, capsize=1.5, label=f"{model1} {freq} {how}", color=c1)
        if len(d2) > 0:
            if use_shaded:
                ax.plot(d2, m2, linewidth=1.4, label=f"{model2} {freq} {how}", color=c2)
                ax.fill_between(d2, m2 - s2, m2 + s2, color=c2, alpha=0.12, linewidth=0) # type: ignore
            else:
                ax.errorbar(d2, m2, yerr=s2, fmt='-o', linewidth=1.2, markersize=3, # type: ignore
                            label=f"{model2} {freq} {how}", color=c2)
        # Tighten x-limits to data span
        if len(dates) > 1:
            xmin = min(dates)
            xmax = max(dates)
            ax.set_xlim(xmin, xmax) # type: ignore

        ax.set_title(f"{variable} daily domain mean | {model1} vs {model2} ({freq} {how})")
        ax.set_xlabel("Date"); ax.set_ylabel(variable)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(frameon=False, ncol=2)
        apply_model_colors(ax)
        fname = f"{fname_prefix}{variable}_{model1}_vs_{model2}_daily_series_{freq}.png" if fname_prefix else f"{variable}_{model1}_vs_{model2}_daily_series_{freq}.png"
        out = os.path.join(save_path, fname)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        logger.info(f"      Saved daily series ({freq}) to {out}")
        if show:
            plt.show()
        plt.close(fig)

def compute_daily_metrics_over_time(dict_data1, dict_data2):
    """
    Accepts dictionaries with keys "cutouts" and "timestamps" for two datasets.
    Computes metrics for each matching day and returns a timeseries list of dicts
    """
    timeseries = []

    # Support both structures: {"timestamps","cutouts"} OR {date -> 2D array}
    timestamps1, cutouts1 = _as_ts_and_cutouts(dict_data1)
    timestamps2, cutouts2 = _as_ts_and_cutouts(dict_data2)

    if not timestamps1 or not timestamps2:
        logger.warning(
            "No timestamps found: len(ts1)=%d, len(ts2)=%d | keys1=%s | keys2=%s",
            len(timestamps1), len(timestamps2),
            list(dict_data1.keys()) if isinstance(dict_data1, dict) else type(dict_data1).__name__,
            list(dict_data2.keys()) if isinstance(dict_data2, dict) else type(dict_data2).__name__,
        )
        return timeseries
    
    # Normalize the timestamps to common key (YYYYMMDD)
    norm1 = [_normalize_date_key(t) for t in timestamps1]
    norm2 = [_normalize_date_key(t) for t in timestamps2]

    # Index maps
    idx_map1 = {d: i for i, d in enumerate(norm1)}
    idx_map2 = {d: i for i, d in enumerate(norm2)}

    shared_norm = sorted(set(norm1) & set(norm2))
    if not shared_norm:
        logger.warning("No overlapping dates between the two datasets.\n"
                       f"   Example ts1[0:3] = {norm1[:3]} (types: {[type(t).__name__ for t in timestamps1[:3]]})\n"
                       f"   Example ts2[0:3] = {norm2[:3]} (types: {[type(t).__name__ for t in timestamps2[:3]]})")
        return timeseries

    if not cutouts1 or not cutouts2:
        logger.warning(f"No cutouts found: len(cutouts1)={len(cutouts1)}, len(cutouts2)={len(cutouts2)}")
        return timeseries
    
    for key in shared_norm:
        i1 = idx_map1[key]
        i2 = idx_map2[key]
        data1 = cutouts1[i1]
        data2 = cutouts2[i2]

        # Parse a plotting-friendly date (datetime.date) if possible
        try:
            date = _dt.datetime.strptime(key, "%Y%m%d").date()
        except Exception:
            date = key  # fallback to raw string

        stats = compute_field_stats(data1, data2)
        stats['date'] = date
        timeseries.append(stats)

    return timeseries


def plot_daily_metrics_over_time(timeseries, save_path='./figures', title="Time Series", fname='daily_metrics', show=False, window_days=90):
    """
    Plots time series of each metric over time in subplots.
    """
    if not timeseries or 'date' not in timeseries[0]:
        logger.warning("Timeseries is empty or lacks 'date' - skipping time series plot.")
        return  # Stop plotting if no valid data

    # Extract metrics and dates
    metrics = [k for k in timeseries[0] if k != 'date']
    n_metrics = len(metrics)
    dates = [entry['date'] for entry in timeseries]
    values = {metric: [entry[metric] for entry in timeseries] for metric in metrics}

    # Create subplot grid
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    axs = axs.flatten()

    # Helper
    def moving_average(x, w):
        if w <= 1 or len(x) < w:
            return None
        kernel = np.ones(w, dtype=float) / w
        return np.convolve(np.asarray(x, dtype=float), kernel, mode='valid')
    
    # Sort by date to make the moving average meaningful
    def _to_sort_key(d):
        if isinstance(d, (_dt.date, _dt.datetime)):
            return d.toordinal()
        try:
            return _dt.datetime.strptime(str(d), "%Y-%m-%d").toordinal()
        except Exception:
            try:
                return _dt.datetime.strptime(str(d), "%Y%m%d").toordinal()
            except Exception:
                return hash(str(d))
    order_idx = sorted(range(len(dates)), key=lambda i: _to_sort_key(dates[i]))
    dates_sorted = [dates[i] for i in order_idx]
    values = {metric: [values[metric][i] for i in order_idx] for metric in metrics}

    for i, metric in enumerate(metrics):
        # Background: all daily points
        axs[i].scatter(dates_sorted, values[metric], s=4, alpha=0.2)
        # Rolling mean
        ma = moving_average(values[metric], window_days)
        if ma is not None:
            half = window_days // 2
            ma_dates = dates_sorted[half:half + len(ma)]
            axs[i].plot(ma_dates, ma, linewidth=1.5, label=f"{metric} {window_days}-day mean", color='k')
        axs[i].set_title(f"{metric}")
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel(metric)
        axs[i].tick_params(axis='x', rotation=45)
        axs[i].grid(True)
        if ma is not None:
            axs[i].legend(loc='best', frameon=False)

    # Hide unused subplots
    for j in range(n_metrics, len(axs)):
        fig.delaxes(axs[j])

    if save_path:
        plt.savefig(f"{save_path}/{fname}_timeseries.png")
        logger.info(f"      Saved time series plot to {save_path}/{fname}_timeseries.png")

    if show:
        plt.show()
    plt.close()


def compare_over_time(dict_data1, dict_data2, model1, model2, variable, save_path='./figures', show=False):
    """
    Computes and plots daily metrics over time.
    Returns summary statistics (mean and std) for each metric.
    """
    timeseries = compute_daily_metrics_over_time(dict_data1, dict_data2)

    if not timeseries:
        logger.warning("No timeseries data computed - skipping plotting and summary stats.")
        return {}

    if show or save_path:
        title = f"Daily Metrics Over Time: {variable} ({model1} vs {model2})"
        fname = f"{variable}_{model1}_vs_{model2}"
        plot_daily_metrics_over_time(timeseries, title=title, fname=fname, save_path=save_path, show=show, window_days=90)
        
        # New figure: daily domain-mean series for each dataset with aggregated overlays
        try:
            plot_daily_series_dual(
                dict_data1, dict_data2,
                model1, model2, variable,
                save_path=save_path, show=show,
                fname_prefix="", freqs=("monthly", "weekly"), how="mean",
                use_shaded=True,
            )
        except Exception as e:
            logger.warning(f"Failed to plot daily series dual figure: {e}")

    # Compute summary stats
    summary_stats = {}
    for entry in timeseries:
        for k, v in entry.items():
            if k == 'date':
                continue
            summary_stats.setdefault(f'{k}_values', []).append(v)

    metrics = [k for k in timeseries[0] if k != 'date']
    summary_stats = {}

    for metric in metrics:
        values = [entry[metric] for entry in timeseries]
        summary_stats[f"{metric}_mean"] = np.mean(values)
        summary_stats[f"{metric}_std"] = np.std(values)

    return summary_stats