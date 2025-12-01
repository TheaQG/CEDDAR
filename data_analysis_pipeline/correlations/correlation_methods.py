"""
    Module for computing correlations between high-resolution (HR) and low-resolution (LR) datasets.
    Implemented:
        - Temporal correlation (domain mean per day)
        - Spatial correlation (grid-point wise over time)

    TODO:
        - Lagged correlation analysis (temporal): to explore lead/lag relationships (i.e. if LR influences HR with a time delay) - use numpy.correlate or scipy.signal.correlate
        - Composite correlation maps: To understand spatial patterns associated with high/low values of one variable (similar to composites in climate science) - use numpy.where to select dates based on thresholds
        - Canonical correlation analysis (CCA): To identify pairs of linear combinations of LR and HR that are maximally correlated (sklearn.cross_decomposition.CCA)
        - Feature importance via Random Forest: Use ML to rank which LR variables are most predictive of HR variable (e.g. RandomForestRegressor feature_importances_)
        - Mutual information: To detect non-linear dependencies missed by correlation (use sklearn.feature_selection.mutual_info_regression)
        
"""
import math
from collections import defaultdict
from datetime import datetime

from scipy.stats import pearsonr, spearmanr
import numpy as np
import logging


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# === Seasonality helpers ===
def _month_id(dt): # (year,month)
    return (dt.year, dt.month)

def _week_id(dt): # ISO week (year, week)
    iso = dt.isocalendar()
    return (iso[0], iso[1])

def _year_id(dt): # year
    return dt.year



def build_climatology(ts, timestamps, method="monthly"):
    """
        ts: 1D array (time, )
        timestamps: list of datetime
        method: "monthly", "weekly", "yearly", or 'DOY' (day of year)
        returns: dict mapping season-key -> mean value
    """
    ts = np.asarray(ts)
    groups = defaultdict(list)

    method = (method or "monthly").lower()

    if method == "monthly":
        keys = [d.month for d in timestamps]
    elif method == "doy":
        # map Feb29 to Feb28 to avoid tiny groups
        keys = [(d.month, 28 if (d.month == 2 and d.day == 29) else d.day) for d in timestamps]
    else:
        raise ValueError(f"Unknown seasonality method: {method}")
    
    for k, v in zip(keys, ts):
        groups[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in groups.items()}
    
def remove_seasonality_ts(ts, timestamps, method="monthly"):
    """
        Subtracts seasonal cycle (monthly or day-of-year) from a 1D time series.
    """
    method = (method or "monthly").lower()
    ts = np.asarray(ts, dtype=float)
    clim = build_climatology(ts, timestamps, method=method)

    if method == "monthly":
        keys = [d.month for d in timestamps]
    elif method == "doy":
        keys = [(d.month, 28 if (d.month == 2 and d.day == 29) else d.day) for d in timestamps]
    else:
        raise ValueError(f"Unknown seasonality method: {method}")
    
    anomalies = np.array([t - clim[k] for t,k in zip(ts, keys)], dtype=float)
    return anomalies

def remove_seasonality_stack(stack, timestamps, method="monthly"):
    """
        For 3D stacks shaped (time, y, x). Subtract seasonal cycle per-pixel.
    """
    method = (method or "monthly").lower()
    T, H, W = stack.shape
    out = np.empty_like(stack, dtype=float)
    # build indices once
    if method == "monthly":
        keys = [d.month for d in timestamps]
        key_set = sorted(set(keys))
    # elif method == "weekly":
    #     keys = [_week_id(d.replace(year=2001)) for d in timestamps]
    #     key_set = sorted(set(keys))
    # elif method == "yearly":
    #     keys = [_year_id(d) for d in timestamps]
    #     key_set = sorted(set(keys))
    elif method == "doy":
        keys = [(d.month, 28 if (d.month == 2 and d.day == 29) else d.day) for d in timestamps]
        key_set = sorted(set(keys))
    else:
        raise ValueError(f"Unknown seasonality method: {method}")

    # pre-allocate masks per key for efficiency
    key_to_idx = {k: np.where(np.array(keys) == k)[0] for k in key_set}

    for y in range(H):
        row = stack[:, y, :]
        for x in range(W):
            ts = row[:, x].astype(float)
            # compute per-key means
            mu = {k: np.nanmean(ts[idx]) for k, idx in key_to_idx.items()}
            out[:, y, x] = np.array([ts[t] - mu[keys[t]] for t in range(T)], dtype=float)
    return out

# === Temporal aggregation ===
def aggregate_ts(ts, timestamps, freq="monthly", how="mean"):
    """
    freq: 'weekly' or 'monthly'
    how: 'mean' or 'sum'
    returns (agg_values, agg_dates) aligned for correlation/plotting
    """
    ts = np.asarray(ts, dtype=float)
    groups = defaultdict(list)
    for v, d in zip(ts, timestamps):
        key = _week_id(d) if freq == "weekly" else (d.year, d.month)
        groups[key].append(v)

    keys_sorted = sorted(groups.keys())
    agg_values, agg_dates = [], []
    for key in keys_sorted:
        vals = np.array(groups[key], dtype=float)
        agg_values.append(np.nanmean(vals) if how == "mean" else np.nansum(vals))
        # pick group mid-point for plotting
        if freq == "weekly":
            year, week = key
            # pick the Wednesday of that ISO week
            agg_dates.append(datetime.fromisocalendar(year, week, 3))
        else:
            y, m = key
            agg_dates.append(datetime(y, m, 15))
    return np.array(agg_values), agg_dates

# === Correlation helpers ===
def corrcoef_1d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0,1])

def spatial_corr_map(hr_stack, lr_stack, remove_seasonality=None):
    """
        hr_stack, lr_stack: (time, y, x)
        remove_seasonality: None | "monthly" | "doy" (later "weekly", "yearly")
        returns r_map (y, x)
    """
    if remove_seasonality:
        hr_stack = remove_seasonality_stack(hr_stack, timestamps=None, method=remove_seasonality) # timestamps must be captured in closure or passed as argument
        lr_stack = remove_seasonality_stack(lr_stack, timestamps=None, method=remove_seasonality)
    T, H, W = hr_stack.shape
    R = np.full((H, W), np.nan, dtype=float)
    # vecorized-ish loop over pixels
    for y in range(H):
        a = hr_stack[:, y, :]
        b = lr_stack[:, y, :]
        # compute correlation along time for each x in this row
        for x in range(W):
            R[y, x] = corrcoef_1d(a[:, x], b[:, x])
    return R
















def compute_temporal_correlation(
                                hr_data,
                                lr_data,
                                method='pearson',
                                timestamps=None, 
                                remove_seasonality=None, # None | 'monthly' | 'doy'
                                agg=None, # None | 'weekly' | 'monthly'
                                agg_how="mean"
                                ):
    """
        Compute correlation between HR and LR time series (domain mean per day),
        with optional deseasonalization and temporal aggregation applied *before* computing correlation.
        Parameters:
            hr_data, lr_data: dict mapping date -> 2D array (H, W)
            method: 'pearson' or 'spearman'
            timestamps: list of datetime objects corresponding to the keys in hr_data/lr_data
            remove_seasonality: None | 'monthly' | 'doy'
            agg: None | 'weekly' | 'monthly'
            agg_how: 'mean' or 'sum' (only relevant if agg is not None)
        Returns:
            dict with keys:
                'correlation': correlation coefficient
                'series_hr_raw': processed HR time series (1D array)
                'series_lr_raw': processed LR time series (1D array)
                'series_hr_proc': processed HR time series after deseasonalization and aggregation (1D array)
                'series_lr_proc': processed LR time series after deseasonalization and aggregation (1D array)
                'timestamps_raw': original timestamps (list of datetime)
                'timestamps_proc': timestamps after aggregation (list of datetime)
    """

    # Align data by dates
    dates  = sorted(set(hr_data.keys()) & set(lr_data.keys()))
    if timestamps is None:
        timestamps = dates
    else:
        # keep only shared
        timestamps = [d for d in timestamps if d in hr_data and d in lr_data]
    if not timestamps:
        return dict(correlation=np.nan, series_hr_raw=np.array([]), series_lr_raw=np.array([]),
                    series_hr_proc=np.array([]), series_lr_proc=np.array([]),
                    timestamps_raw=[], timestamps_proc=[])
    # Domain mean per day (raw)
    hr_series = np.array([np.nanmean(hr_data[d]) for d in timestamps], dtype=float)
    lr_series = np.array([np.nanmean(lr_data[d]) for d in timestamps], dtype=float)

    hr_proc, lr_proc, t_proc = hr_series, lr_series, list(timestamps)

    # Deseasonalize first (if requested)
    if remove_seasonality is not None:
        hr_proc = remove_seasonality_ts(hr_proc, t_proc, method=remove_seasonality)
        lr_proc = remove_seasonality_ts(lr_proc, t_proc, method=remove_seasonality)

    # Then aggregate (if requested)
    if agg is not None:
        hr_proc, t_proc = aggregate_ts(hr_proc, t_proc, freq=agg, how=agg_how)
        lr_proc, _      = aggregate_ts(lr_proc, t_proc, freq=agg, how=agg_how)

    # Correlation on processed series
    if method == 'pearson':
        r = corrcoef_1d(hr_proc, lr_proc)
    elif method == 'spearman':
        m = np.isfinite(hr_proc) & np.isfinite(lr_proc)
        r, _ = spearmanr(hr_proc[m], lr_proc[m]) if np.sum(m) >= 2 else (np.nan, None)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return {
        'correlation': float(r) if r is not None else np.nan, # type: ignore
        'series_hr_raw': hr_series,
        'series_lr_raw': lr_series,
        'series_hr_proc': np.asarray(hr_proc),
        'series_lr_proc': np.asarray(lr_proc),
        'timestamps_raw': list(timestamps),
        'timestamps_proc': list(t_proc),
    }


# === Numpy-only temporal corr helper ===
def compute_temporal_corr_series_np(
    hr_data,
    lr_data,
    *,
    remove_seasonality: str | None = None,  # None | 'monthly' | 'doy'
    monthly: bool = True,
    monthly_how: str = "mean",
):
    """
    Numpy-only routine that builds daily domain-mean series for HR and LR,
    optionally removes seasonality, then (optionally) aggregates to monthly.
    Returns a dict:
      {
        "raw":     {"hr": np.ndarray, "lr": np.ndarray, "dates": list[datetime], "r": float},
        "monthly": {"hr": np.ndarray, "lr": np.ndarray, "dates": list[datetime], "r": float}
      }
    """
    # Align dates
    shared = sorted(set(hr_data.keys()) & set(lr_data.keys()))
    if not shared:
        return {
            "raw": {"hr": np.array([]), "lr": np.array([]), "dates": [], "r": np.nan},
            "monthly": {"hr": np.array([]), "lr": np.array([]), "dates": [], "r": np.nan},
        }

    # Daily domain means (aligned)
    hr_daily = np.array([np.nanmean(hr_data[d]) for d in shared], dtype=float)
    lr_daily = np.array([np.nanmean(lr_data[d]) for d in shared], dtype=float)

    # Optional deseasonalization (same timestamps for both)
    if remove_seasonality is not None:
        hr_anom = remove_seasonality_ts(hr_daily, shared, method=remove_seasonality)
        lr_anom = remove_seasonality_ts(lr_daily, shared, method=remove_seasonality)
    else:
        hr_anom, lr_anom = hr_daily, lr_daily

    # Daily correlation
    r_raw = corrcoef_1d(hr_anom, lr_anom)
    out = {
        "raw": {
            "hr": hr_anom,
            "lr": lr_anom,
            "dates": list(shared),
            "r": float(r_raw) if r_raw == r_raw else np.nan,
        }
    }

    # Monthly aggregation (optional)
    if monthly:
        hr_m, dates_m = aggregate_ts(hr_anom, shared, freq="monthly", how=monthly_how)
        lr_m, _       = aggregate_ts(lr_anom, shared, freq="monthly", how=monthly_how)
        r_m = corrcoef_1d(hr_m, lr_m)
        out["monthly"] = {
            "hr": np.asarray(hr_m),
            "lr": np.asarray(lr_m),
            "dates": dates_m,
            "r": float(r_m) if r_m == r_m else np.nan,
        }w
    else:
        out["monthly"] = {"hr": np.array([]), "lr": np.array([]), "dates": [], "r": np.nan}

    return out



def compute_spatial_correlation(
    hr_data, lr_data, method="pearson",
    remove_seasonality: str | None = None,
    timestamps: list | None = None
):
    """
    Compute per-pixel correlation over time, optionally removing a seasonal
    cycle ('monthly' or 'doy') for each pixel prior to correlation.
    """
    # align dates & build stacks
    dates_shared = sorted(set(hr_data.keys()) & set(lr_data.keys()))
    hr_stack = np.array([hr_data[d] for d in dates_shared], dtype=float)
    lr_stack = np.array([lr_data[d] for d in dates_shared], dtype=float)

    if remove_seasonality is not None:
        ts_use = timestamps if timestamps is not None else dates_shared
        hr_stack = remove_seasonality_stack(hr_stack, ts_use, method=remove_seasonality)
        lr_stack = remove_seasonality_stack(lr_stack, ts_use, method=remove_seasonality)

    T, H, W = hr_stack.shape
    corr_map = np.full((H, W), np.nan)
    for i in range(H):
        for j in range(W):
            x = hr_stack[:, i, j]; y = lr_stack[:, i, j]
            if method == "pearson":
                corr = corrcoef_1d(x, y)
            else:
                m = np.isfinite(x) & np.isfinite(y)
                corr, _ = spearmanr(x[m], y[m]) if np.sum(m) >= 2 else (np.nan, None)
            corr_map[i, j] = corr
    return corr_map