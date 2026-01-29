import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import logging
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data_analysis_pipeline.correlations.correlation_methods import (
    remove_seasonality_ts,
    aggregate_ts,
    corrcoef_1d
    )

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# === SPATIAL: grid of subplots (one figure per HR variable) ===
def plot_spatial_corr_grid(hr_var, lr_to_rmap, *,
                           vmin=None,
                           vmax=None,
                           cmap='RdBu_r',
                           ncols=4, figsize_per_subplot=(4.2, 4.2),
                           suptitle=None,
                           wspace=0.25, hspace=0.35, title_pad=6,
                           per_subplot_cbar=False,
                           cbar_label="Correlation coefficient",
                           # For global colorbar (added as extra column)
                           cbar_width_ratio=0.06,
                           # For per-subplot colorbars (axes_grid1)
                           per_cbar_size="3%", per_cbar_pad=0.04,
                           savepath=None, show=False):                           
    """
        lr_to_rmap: dict like {'prcp': r2d, 'temp': r2d, ...}
        Draws a grid o correlation maps (one figure per HR variable).
        - By default uses a SINGLE global colorbar on the right.
        - If per_subplot_cbar=True, adds a small colorbar to the right of EACH subplot instead
    """
    lr_vars = list(lr_to_rmap.keys())
    N = len(lr_vars)
    ncols = min(ncols, N)
    nrows = math.ceil(N / ncols)
    fig_w = figsize_per_subplot[0] * (ncols + (0 if per_subplot_cbar else cbar_width_ratio))
    fig_h = figsize_per_subplot[1] * nrows
    
    if per_subplot_cbar:
        # Standard grid; we'll append a cbar to each axes with axes_grid1
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False,
                                 gridspec_kw=dict(wspace=wspace, hspace=hspace), constrained_layout=True)
        cax_global = None
    else:
        # Build a gridspec with an extra narrow column for the shared colorbar
        fig = plt.figure(figsize=(fig_w, fig_h))
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(nrows, ncols + 1,
                               width_ratios=[1] * ncols + [cbar_width_ratio * ncols],
                               wspace=wspace, hspace=hspace)
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                axes[i, j] = fig.add_subplot(gs[i, j])
        cax_global = fig.add_subplot(gs[:, -1])

    im_last = None

    for i, lr in enumerate(lr_vars):
        # If vmin or vmax are None, compute from data
        if vmin is None or vmax is None:
            rmap = lr_to_rmap[lr]
            vmin = np.nanmin(rmap) if vmin is None else vmin
            vmax = np.nanmax(rmap) if vmax is None else vmax
        rmap = lr_to_rmap[lr]
        ax = axes[i // ncols, i % ncols]
        im_last = ax.imshow(rmap, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        ax.set_title(f"{hr_var} vs {lr}", pad=title_pad)
        ax.set_xticks([]); ax.set_yticks([])

        if per_subplot_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=per_cbar_size, pad=per_cbar_pad)
            cb = fig.colorbar(im_last, cax=cax)
            cb.set_label(cbar_label)

    # Turn off unused axes (when N not divisible by ncols)
    for j in range(N, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if not per_subplot_cbar and im_last is not None and cax_global is not None:
        cb = fig.colorbar(im_last, cax=cax_global)
        cb.set_label(cbar_label)

    if suptitle:
        fig.suptitle(suptitle, y=0.98)
        plt.subplots_adjust(top=0.92) # leave space for suptitle

    if savepath:
        fig.savefig(savepath, dpi=300)
        logger.info(f"Saved spatial correlation grid plot to {savepath}")

    if show:
        plt.show()
    plt.close(fig)
    
    return fig

# === TEMPORAL: raw + aggregated with optional seasonality removal ===
# correlation_plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_temporal_correlations_grid(
    pairs,
    *, 
    hr_var_label: str,
    deseasonalize: bool,
    out_dir: str,
    fname_prefix: str,
    ncols: int = 4, nrows: int = 2,
    hr_color="tab:red", lr_color="tab:green",
):
    """
        pairs[i]["series"] is expected from compute_temporal_corr_series_np:
                {"raw": {"hr","lr","dates","r"}, "monthly": {"hr","lr","dates","r"}}
    """
    N = len(pairs)
    ncols = max(1, min(ncols, N))
    nrows = max(1, nrows if nrows * ncols >= N else int(np.ceil(N / ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 2.6*nrows),
                             sharex=True, constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    season_tag = "deseasonalized" if deseasonalize else "with seasonality"
    fig.suptitle(f"Temporal correlations | HR={hr_var_label} ({season_tag})", y=0.98)
    plt.subplots_adjust(top=0.92) # leave space for suptitle

    for ax, p in zip(axes, pairs):
        ser = p["series"]

        # Raw daily: faint, thin
        d_raw = ser["raw"]["dates"]
        hr_raw = ser["raw"]["hr"]; lr_raw = ser["raw"]["lr"]
        if len(d_raw) and hr_raw.size and lr_raw.size:
            ax.plot(d_raw, hr_raw, lw=0.6, alpha=0.25, color=hr_color, label=f'{p["hr_name"]} (raw)')
            ax.plot(d_raw, lr_raw, lw=0.6, alpha=0.25, color=lr_color, label=f'{p["lr_name"]} (raw)')

        # Monthly: thicker
        d_m = ser["monthly"]["dates"]
        hr_m = ser["monthly"]["hr"]; lr_m = ser["monthly"]["lr"]
        if len(d_m) and hr_m.size and lr_m.size:
            ax.plot(d_m, hr_m, lw=1.1, alpha=0.9, color=hr_color, label=f'{p["hr_name"]} (monthly)')
            ax.plot(d_m, lr_m, lw=1.1, alpha=0.9, color=lr_color, label=f'{p["lr_name"]} (monthly)')

        r_raw = ser["raw"]["r"]
        r_month = ser["monthly"]["r"]
        ax.set_title(f'{p["hr_name"]} vs {p["lr_name"]} | r(raw)={r_raw:.02f}, r(monthly)={r_month:.02f}', fontsize=9)

        # Ticks: yearly major, label every 5th year to reduce clutter 
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.25)
        # Small, non-intrusive legend
        ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)

    # Only bottom row shows x tick labels & “Date”
    for a in axes[:-ncols]:
        a.label_outer()
    for a in axes[-ncols:]:
        a.set_xlabel("Date")

    # Save with seasonality indicator
    suffix = "deseas-yes" if deseasonalize else "deseas-no"
    fname = f"{fname_prefix}__{suffix}.png"
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, fname), dpi=300)
    plt.close(fig)

def plot_temporal_pair(hr_var, lr_var, hr_ts, lr_ts, timestamps,
                       remove_seasonality=None, agg=None, agg_how="mean",
                       ax=None, colors=None, title=None, save_path=None, show=False):
    """
    hr_ts, lr_ts: 1D arrays (time,) – typically spatial means
    remove_seasonality: None | 'monthly' | 'doy'
    agg: None | 'weekly' | 'monthly'
    returns: (r_raw, r_agg, ax)
    """
    import matplotlib.dates as mdates

    hr = np.asarray(hr_ts, dtype=float)
    lr = np.asarray(lr_ts, dtype=float)

    if remove_seasonality:
        hr = remove_seasonality_ts(hr, timestamps, method=remove_seasonality)
        lr = remove_seasonality_ts(lr, timestamps, method=remove_seasonality)

    r_raw = corrcoef_1d(hr, lr)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3.2))
        created_fig = True
    else:
        fig = ax.figure

    # background raw series (thinner, cleaner)
    ax.plot(timestamps, hr, label=f"{hr_var} (raw)", alpha=0.25, linewidth=0.6)
    ax.plot(timestamps, lr, label=f"{lr_var} (raw)", alpha=0.25, linewidth=0.6)

    r_agg = np.nan
    if agg is not None:
        hr_a, t_a = aggregate_ts(hr, timestamps, freq=agg, how=agg_how)
        lr_a, _   = aggregate_ts(lr, timestamps, freq=agg, how=agg_how)
        r_agg = corrcoef_1d(hr_a, lr_a)
        ax.plot(t_a, hr_a, label=f"{hr_var} ({agg})", linewidth=1.0)
        ax.plot(t_a, lr_a, label=f"{lr_var} ({agg})", linewidth=1.0)

    # cosmetics
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{hr_var} / {lr_var}")
    rtxt = f"r(raw)={r_raw:.2f}" + (f", r({agg})={r_agg:.2f}" if agg else "")
    ax.set_title(title or f"Temporal correlation ({rtxt})")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Saved temporal correlation plot to {save_path}")

    if show:
        plt.show()
    if created_fig:
        plt.close(fig)
    return r_raw, r_agg, ax



def plot_temporal_grid(hr_var, lr_to_series, timestamps,
                       remove_seasonality=None, agg=None, agg_how="mean",
                       ncols=4, figsize_per_subplot=(9, 2.8), suptitle=None, savepath=None):
    """
    lr_to_series: dict {lr_var: (hr_ts, lr_ts)} – both 1D aligned arrays
    Makes one figure with subplots for each LR var.
    """
    lr_vars = list(lr_to_series.keys())
    N = len(lr_vars)
    ncols = min(ncols, N)
    nrows = math.ceil(N / ncols)
    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.5)

    for i, lr in enumerate(lr_vars):
        hr_ts, lr_ts = lr_to_series[lr]
        ax = axes[i // ncols, i % ncols]
        r_raw, r_agg, _ = plot_temporal_pair(
            hr_var=hr_var, lr_var=lr,
            hr_ts=hr_ts, lr_ts=lr_ts, timestamps=timestamps,
            remove_seasonality=remove_seasonality, agg=agg, agg_how=agg_how, ax=ax, title=None
        )
        ax.set_title(f"{hr_var} vs {lr} | r(raw)={r_raw:.2f}" + (f", r({agg})={r_agg:.2f}" if agg else ""))

    for j in range(N, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    if suptitle:
        fig.suptitle(suptitle, y=0.99)
        plt.subplots_adjust(top=0.92) # leave space for suptitle
    
    if savepath:
        fig.savefig(savepath, dpi=300)
        logger.info(f"Saved temporal correlation grid plot to {savepath}")
    return fig



















def plot_temporal_series(hr_series, lr_series, dates, variable1, variable2, model1, model2, save_path=None, show=False):
    """
        Plot temporal series of HR and LR data.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, hr_series, label=f'{variable1} ({model1}) (mean)', marker='o', markersize=3)
    ax.plot(dates, lr_series, label=f'{variable2} ({model2}) (mean)', marker='x', markersize=3)
    ax.set_xlabel('Date')
    ax.set_ylabel(f"{variable1} / {variable2}")
    ax.set_title(f"Temporal correlation of spatial mean {variable1} / {variable2} | {model1} vs {model2}")
    # Add a text box with correlation coefficient
    if len(hr_series) == len(lr_series) and len(hr_series) > 1:
        corr_coef = np.corrcoef(hr_series, lr_series)[0, 1]
        textstr = f'Correlation: {corr_coef:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    ax.legend()
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved temporal series plot to {save_path}")
    if show:
        plt.show()
    plt.close()

def plot_correlation_map(corr_map, variable1, variable2, model1, model2, save_path=None, show=False):
    """
        Plot spatial correlation map.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr_map, cmap='RdBu_r') #, vmin=-1, vmax=1)
    ax.set_title(f"Spatial correlation map of {variable1} / {variable2} | {model1} vs {model2}")
    ax.invert_yaxis()
    fig.colorbar(cax, ax=ax, label='Correlation coefficient')
    plt.axis('off')
    if save_path:
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved correlation map plot to {save_path}")
    if show:
        plt.show()
    plt.close()