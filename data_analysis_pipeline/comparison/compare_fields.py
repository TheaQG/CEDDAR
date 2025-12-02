"""
    Single-day field comparison module.
        - Field-level: Bias, RMSE, correlation, etc.
        - Spatial: Error maps, differen fields, ratio maps
"""
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from ceddar.variable_utils import get_cmap_for_variable, get_unit_for_variable
from ceddar.plotting_utils import plot_spatial_panel, get_dk_lsm_outline, overlay_outline

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def compute_field_stats(data_model1, data_model2, mask=None):
    """
    Compute basic statistics between two 2D data fields.
    
    Parameters:
        data_model1 (np.ndarray): First data field.
        data_model2 (np.ndarray): Second data field.
        mask (np.ndarray, optional): Mask to apply to the data fields.
        
    Returns:
        dict: Dictionary containing bias, RMSE, and correlation.
    """

    if mask is not None:
        data_model1 = data_model1[mask]
        data_model2 = data_model2[mask]

    diff = data_model1 - data_model2
    return {
        'bias': np.mean(diff),
        'rmse': np.sqrt(np.mean(diff**2)),
        'corr': np.corrcoef(data_model1.flatten(), data_model2.flatten())[0, 1],
        'std_diff': np.std(diff)
    }

def plot_difference_map(
            data_model1,
            data_model2,
            model1="HR model",
            model2="LR model",
            variable="variable",
            title='Difference Map',
            save_path=None,
            show=False,
            bounds=None):
    """
    Plot the difference map between two data fields and the two fields themselves.
    
    Parameters:
        data_model1 (np.ndarray): First data field.
        data_model2 (np.ndarray): Second data field.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. If None, the plot is shown instead.
    """
    diff_map = data_model1 - data_model2
    unit = get_unit_for_variable(variable)

    # Shared vmin/vmax for HR/LR panels
    finite1 = np.asarray(data_model1)[np.isfinite(data_model1)]
    finite2 = np.asarray(data_model2)[np.isfinite(data_model2)]
    if finite1.size and finite2.size:
        vmin = min(finite1.min(), finite2.min())
        vmax = max(finite1.max(), finite2.max())
    else:
        vmin, vmax = None, None

    # Symmetric range for the difference
    finite_diff = np.asarray(diff_map)[np.isfinite(diff_map)]
    dmax = float(np.max(np.abs(finite_diff))) if finite_diff.size else None

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.ravel()  # Flatten

    bnds = tuple(bounds) if bounds is not None else (200, 328, 380, 508)
    is_precip = variable.lower() in ["prcp", "precip", "precipitation"]

    # Panel 1: model1 with correct cmap + DK outline
    im1 = plot_spatial_panel(
        axs[0],
        np.asarray(data_model1),
        variable=variable,
        vmin=vmin, vmax=vmax,
        add_dk_outline=True,
        outline_color="darkgrey",
        outline_linewidth=0.8,
        title=f"{variable} - {model1}",
        under_color="#bdbdbd" if is_precip else None,
        under_threshold=0.01 if is_precip else None,
        bounds=bnds,
    )
    axs[0].figure.axes[-1].set_ylabel(f"[{unit}]" if unit else "")

    # Panel 2: model2 with correct cmap + DK outline
    im2 = plot_spatial_panel(
        axs[1],
        np.asarray(data_model2),
        variable=variable,
        vmin=vmin, vmax=vmax,
        add_dk_outline=True,
        outline_color="darkgrey",
        outline_linewidth=0.8,
        title=f"{variable} - {model2}",
        under_color="#bdbdbd" if is_precip else None,
        under_threshold=0.01 if is_precip else None,
        bounds=bnds,
    )
    axs[1].figure.axes[-1].set_ylabel(f"[{unit}]" if unit else "")

    # Panel 3: difference with diverging cmap
    vmin_d, vmax_d = ((-dmax, dmax) if dmax is not None else (None, None))
    try:
        diff_cmap = get_cmap_for_variable(f"{variable}_bias")
    except Exception:
        diff_cmap = "bwr"
    im3 = axs[2].imshow(diff_map, cmap=diff_cmap, vmin=vmin_d, vmax=vmax_d, origin="lower")
    axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].set_title(f"{variable} - Difference ({model1} - {model2})")
    # DK outline on difference panel
    try:
        mask = get_dk_lsm_outline(bnds)
        if mask is not None:
            mask = np.flipud(np.asarray(mask))  # align with imshow origin='lower'
        overlay_outline(axs[2], mask, color="darkgrey", linewidth=0.8)
    except Exception:
        pass    
    divider = make_axes_locatable(axs[2])
    cax3 = divider.append_axes("right", size="3.5%", pad=0.05)
    cb3 = plt.colorbar(im3, cax=cax3, orientation="vertical")
    cb3.set_label(f"[{unit}]" if unit else "")

    fig.suptitle(title, fontsize=16)
    
    if save_path:
        out = f"{save_path}/{variable}_{model1}_vs_{model2}_difference_map.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        logger.info(f"      Saved difference map to {out}")
    if show:
        plt.show()
    plt.close()


def compare_single_day_fields(
                    data_model1,
                    data_model2,
                    mask=None,
                    variable="variable",
                    model1="Model 1",
                    model2="Model 2",
                    save_path=None,
                    show=False,
                    print_results=True,
                    bounds=None,
                    plot_only: bool = False,
                    cache_dir: str | None = None): # type: ignore
    """
    Compare two 2D fields from the same date.
    Supports inputs as either raw arrays or {'cutout': ..., 'timestamp': ...} dicts.
    """

    """
    Compare two 2D fields from the same date.
    Supports inputs as either raw arrays or {'cutout': ..., 'timestamp': ...} dicts.
    """

    def _cache_fname(date_str):
        if cache_dir is None or date_str is None:
            return None
        safe = f"{variable}_{model1}_vs_{model2}_{date_str}_fields.npz".replace(' ', '_')
        return os.path.join(cache_dir, safe)

    def _save_cache(path, arr1, arr2, stats):
        if path is None:
            return
        try:
            import numpy as np
            np.savez_compressed(path, data_model1=np.asarray(arr1), data_model2=np.asarray(arr2), **stats)
            logger.info(f"      [cache] Saved field cache to {path}")
        except Exception as e:
            logger.warning(f"      [cache] Failed to save field cache to {path}: {e}")

    def _load_cache(path):
        try:
            import numpy as np
            with np.load(path) as z:
                a1 = z["data_model1"]
                a2 = z["data_model2"]
                stats = {k: float(z[k]) for k in ["bias","rmse","corr","std_diff"] if k in z}
            logger.info(f"      [cache] Loaded field cache from {path}")
            return a1, a2, stats
        except Exception as e:
            logger.warning(f"      [cache] Failed to load field cache {path}: {e}")
            return None, None, {}

    # === DICT-WRAPPED INPUTS ===
    timestamp1 = None
    timestamp2 = None
    if isinstance(data_model1, dict):
        timestamp1 = data_model1.get("timestamp", None)
        data_model1 = data_model1.get("cutouts", data_model1)
    if isinstance(data_model2, dict):
        timestamp2 = data_model2.get("timestamp", None)
        data_model2 = data_model2.get("cutouts", data_model2)

    if timestamp1 and timestamp2 and timestamp1 != timestamp2:
        logger.warning(f"Comparing fields with different timestamps: {timestamp1} vs {timestamp2}")

    date_str = timestamp1.strftime("%Y%m%d") if timestamp1 else None

    cache_path = _cache_fname(date_str)
    if plot_only and cache_path and os.path.exists(cache_path):
        # Load arrays and stats from cache, skip recompute
        dm1, dm2, stats_loaded = _load_cache(cache_path)
        if dm1 is not None and dm2 is not None:
            if print_results and stats_loaded:
                logger.info(f"[cache] Stats (loaded): bias={stats_loaded.get('bias', np.nan):.3f}, rmse={stats_loaded.get('rmse', np.nan):.3f}, corr={stats_loaded.get('corr', np.nan):.3f}, std_diff={stats_loaded.get('std_diff', np.nan):.3f}")
            if show or save_path:
                title = f'{variable} | {model1} vs {model2} | {date_str if date_str else "N/A"}'
                plot_difference_map(dm1, dm2, model1=model1, model2=model2, variable=variable, title=title, save_path=save_path if save_path else None, show=show, bounds=bounds)
            return stats_loaded if stats_loaded else {"bias": np.nan, "rmse": np.nan, "corr": np.nan, "std_diff": np.nan}

    # === COMPUTE STATS ===
    stats = compute_field_stats(data_model1, data_model2, mask)

    # Save compute results for future plot-only runs
    if cache_path:
        _save_cache(cache_path, data_model1, data_model2, stats)

    if print_results:
        logger.info(f"\nComparison stats for {variable} on {date_str if date_str else 'N/A'}:")
        logger.info(f"  Models: {model1} vs {model2}")
        logger.info(f"  Bias: {stats['bias']:.3f}, RMSE: {stats['rmse']:.3f}, Corr: {stats['corr']:.3f}, Std Diff: {stats['std_diff']:.3f}\n")

    # === PLOT DIFF MAP ===
    if show or save_path:
        title = f'{variable} | {model1} vs {model2} | {date_str if date_str else "N/A"}'
        plot_difference_map(
            data_model1=data_model1,
            data_model2=data_model2,
            model1=model1,
            model2=model2,
            variable=variable,
            title=title,
            save_path=save_path if save_path else None,
            show=show,
            bounds=bounds
            )

    return stats
    