"""
    Distributional comparison module.
        - Compare full data distributions: PDFs, CDFs, histograms, energy Power spectra
        - Include statistical tests: KS-test, Wasserstein distance, etc.
"""
import os
import logging
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from numpy.fft import fft2, fftshift
from collections import defaultdict
from typing import Optional
from ceddar.variable_utils import get_unit_for_variable, get_color_for_model

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def compute_2d_power_spectrum(
                data: np.ndarray,
                detrend: bool = False) -> np.ndarray:
    """
        Compute the 2D power spectrum of a 2D field.
        Optionally removes the mean or linear trend first
    """

    if detrend:
        data = data - np.mean(data)
    
    fft_data = fft2(data)
    power_spectrum = np.abs(fft_data)**2
    power_spectrum = fftshift(power_spectrum)  # Shift zero frequency to center
    return power_spectrum

def radial_average(ps_2d: np.ndarray) -> np.ndarray:
    """
        Compute the isotropic (radially averaged) 1D power spectrum from a 2D power spectrum.

    """
    y, x = np.indices(ps_2d.shape)
    center = np.array(ps_2d.shape) // 2
    r = np.hypot(x - center[1], y - center[0])
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), ps_2d.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)  # type: ignore Avoid division by zero
    
    return radialprofile

def compare_power_spectra(
    data_model1: np.ndarray,
    data_model2: np.ndarray,
    model1: str = 'Model 1',
    model2: str = 'Model 2',
    dx_model1: float = 2.5,
    dx_model2: float = 2.5,
    variable: str = 'Variable',
    save_path: Optional[str]= None,
    loglog: bool = True,
    return_metrics: bool = True,
    show: bool = False,
    cache_dir: Optional[str] = None,
    plot_only: bool = False,
    ):
    """
        Compare radially averaged power spectra of two 2D datasets.
        - Low spatial frequencies corresponds to large-scale structures (waves, fronts, smooth gradients)
        - High spatial frequencies corresponds to small-scale structures (local turbulence, variability, topographic noise/effects)
    """
    cache_path = None
    if cache_dir:
        cache_name = f"{variable}_{model1}_vs_{model2}_ps_single.npz".replace(" ", "_")
        cache_path = os.path.join(cache_dir, cache_name)

    if plot_only and cache_path and os.path.exists(cache_path):
        try:
            with np.load(cache_path) as z:
                wavelengths = z["wavelengths"]
                ps1 = z["ps1"]
                ps2 = z["ps2"]
            # plotting branch (same as below)
            _, ax = plt.subplots(figsize=(10, 6))
            if loglog:
                ax.loglog(wavelengths, ps1, label=model1, color=get_color_for_model(model1))
                ax.loglog(wavelengths, ps2, label=model2, color=get_color_for_model(model2))
            else:
                ax.plot(wavelengths, ps1, label=model1, color=get_color_for_model(model1))
                ax.plot(wavelengths, ps2, label=model2, color=get_color_for_model(model2))
            ax.set_title(f'{variable} | {model1} vs {model2} | Radially Averaged Power Spectrum Comparison')
            ax.set_xlabel('Wavelength (km)'); ax.set_ylabel('Power Spectrum Density')
            ax.legend(); ax.grid(True, which='both', ls='--', alpha=0.5)
            fname = f"{variable}_{model1}_vs_{model2}_power_spectrum".replace(" ", "_")
            if save_path:
                if not os.path.exists(save_path): os.makedirs(save_path)
                plt.savefig(os.path.join(save_path, f'{fname}.png'), dpi=300)
                logger.info(f"      [cache] Saved power spectra figure from cache to {save_path}/{fname}.png")
            if show: plt.show()
            plt.close()
        except Exception as e:
            logger.warning(f"[cache] Failed to plot from cached power spectra: {e}")
        return None

    # Compute radial power spectra (1D)
    ps1 = radial_average(compute_2d_power_spectrum(data_model1))
    ps2 = radial_average(compute_2d_power_spectrum(data_model2))

    title = f"{variable} | {model1} vs {model2} | "
    fname = f"{variable}_{model1}_vs_{model2}_power_spectrum".replace(" ", "_")

    # Frequency bins converted to wavelengths
    nx = data_model1.shape[1]
    logger.info(f"Data shape: {data_model1.shape}, nx: {nx}")
    dx = dx_model1  # grid spacing in km
    wavelengths = (nx * dx) / np.arange(1, len(ps1)+1, dtype=np.float64)  # Avoid division by zero

    # Nyquist limit cutoff
    nyquist_limit = 2 * dx  # = 5 km for dx=2.5
    mask = wavelengths >= nyquist_limit
    wavelengths = wavelengths[mask]
    ps1 = ps1[mask]
    ps2 = ps2[mask]
    
    # Save cache
    if cache_path:
        try:
            np.savez_compressed(cache_path, wavelengths=wavelengths, ps1=ps1, ps2=ps2)
            logger.info(f"      [cache] Saved single-day spectra cache to {cache_path}")
        except Exception as e:
            logger.warning(f"      [cache] Failed to save spectra cache: {e}")

    # Plotting
    _, ax = plt.subplots(figsize=(10, 6))
    if loglog:
        ax.loglog(wavelengths, ps1, label=model1, color=get_color_for_model(model1))
        ax.loglog(wavelengths, ps2, label=model2, color=get_color_for_model(model2))
    else:
        ax.plot(wavelengths, ps1, label=model1, color=get_color_for_model(model1))
        ax.plot(wavelengths, ps2, label=model2, color=get_color_for_model(model2))

    ax.set_title(f'{title}Radially Averaged Power Spectrum Comparison')
    ax.set_xlabel('Wavelength (km)')
    ax.set_ylabel('Power Spectrum Density')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'{fname}.png'), dpi=300)
        logger.info(f"      Saved power spectra comparison to {save_path}/{fname}.png")
    if show:
        plt.show()
    plt.close()

    # Compute evaluation metrics
    if return_metrics:
        eps = 1e-8  # Small constant to avoid division by zero
        mse_spec = np.mean((ps1 - ps2)**2) 
        log_mse_spec = np.mean((np.log(ps1 + eps) - np.log(ps2 + eps))**2)
        ratio = ps2 / (ps1 + eps)
        metrics = {
            'mse_spectrum': float(mse_spec),
            'log_mse_spectrum': float(log_mse_spec),
            'mean_ratio': float(np.mean(ratio)),
            'std_ratio': float(np.std(ratio)),
            'min_ratio': float(np.min(ratio)),
            'max_ratio': float(np.max(ratio))
        }
        return metrics

def batch_compare_power_spectra(
    dataset1: dict,
    dataset2: dict,
    model1: str = 'Model 1',
    model2: str = 'Model 2',
    dx_model1: float = 2.5,
    dx_model2: float = 2.5,
    variable: str = 'Variable',
    show_plot: bool = False,
    loglog: bool = True,
    save_path: str = './figures/power_spectra_comparison',
    return_all_metrics: bool = True,
    show: bool = False,
    cache_dir: Optional[str] = None,
    plot_only: bool = False
    ):
    """
        Compare power spectra for multiple field pairs over time
        Assumes dataset1 and dataset2 are dictionaries with time keys and 2D numpy array values (i.e. mapping date -> 2D field)

        Returns average metrics and optionally all daily metrics
    """
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"{variable}_{model1}_vs_{model2}_ps_batch.npz".replace(" ", "_"))

    if plot_only and cache_path and os.path.exists(cache_path):
        try:
            z = np.load(cache_path)
            wavelengths = z["wavelengths"]; mean_ps1 = z["mean_ps1"]; mean_ps2 = z["mean_ps2"]
            # plotting block (reuse current plot branch using these arrays)
            title = f"{variable} | {model1} vs {model2} | "
            fname = f"{variable}_{model1}_vs_{model2}_power_spectrum".replace(" ", "_")
            fig, ax = plt.subplots(figsize=(10, 6))
            if loglog:
                ax.loglog(wavelengths, mean_ps1, label=f'{model1} Mean', color=get_color_for_model(model1))
                ax.loglog(wavelengths, mean_ps2, label=f'{model2} Mean', color=get_color_for_model(model2))
            else:
                ax.plot(wavelengths, mean_ps1, label=f'{model1} Mean', color=get_color_for_model(model1), linewidth=2)
                ax.plot(wavelengths, mean_ps2, label=f'{model2} Mean', color=get_color_for_model(model2), linewidth=2)
            ax.set_title(f'{title}\nMean (over time) Radially Averaged Power Spectrum Comparison')
            ax.set_xlabel('Wavelength (km)')
            tick_vals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
            tick_vals = [tv for tv in tick_vals if wavelengths.min() <= tv <= wavelengths.max()]
            ax.set_xticks(tick_vals)
            ax.set_xscale('log')
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.tick_params(axis='x', which='major', labelsize=10)
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[1], xlim[0])
            ax.set_ylabel('Power Spectrum Density')
            ax.legend()
            ax.grid(True, which='both', ls='--', alpha=0.5)
            important_scales = {
                'Large-scale front': 256,
                'Mesoscale': 64,
                'Convective': 8
            }
            for label, wl in important_scales.items():
                if wavelengths.min() <= wl <= wavelengths.max():
                    ax.axvline(wl, linestyle='--', color='gray', alpha=0.5)
                    ax.text(wl, ax.get_ylim()[1], label, rotation=90, va='top', ha='right', fontsize=8)
            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(os.path.join(save_path, f'{fname}.png'), dpi=300)
                logger.info(f"      [cache] Saved batch power spectra figure from cache to {save_path}/{fname}.png")
            if show:
                plt.show()
            plt.close()
            # return average metrics if present
            if "avg_metrics" in z and "std_metrics" in z:
                return dict(z["avg_metrics"].item()), dict(z["std_metrics"].item())
            return {}, {}
        except Exception as e:
            logger.warning(f"[cache] Failed to load batch spectra cache: {e}")
            return {}, {}

    title = f"{variable} | {model1} vs {model2} | "
    fname = f"{variable}_{model1}_vs_{model2}_power_spectrum".replace(" ", "_")

    all_metrics = defaultdict(list) # Store lists of metrics for each time point
    spectra_1 = []
    spectra_2 = []

    shared_dates = sorted(set(k for k in dataset1 if isinstance(dataset1[k], np.ndarray)) & set(k for k in dataset2 if isinstance(dataset2[k], np.ndarray)))
    logger.info(f"Found {len(shared_dates)} shared dates for batch power spectra comparison.")
    if not shared_dates:
        raise ValueError("No overlapping dates between the two datasets.")

    for date in shared_dates:
        field1 = dataset1[date]
        field2 = dataset2[date]
        ps1 = radial_average(compute_2d_power_spectrum(field1))
        ps2 = radial_average(compute_2d_power_spectrum(field2))
        spectra_1.append(ps1)
        spectra_2.append(ps2)
        eps = 1e-8
        mse_spec = np.mean((ps1 - ps2)**2)
        log_mse_spec = np.mean((np.log(ps1 + eps) - np.log(ps2 + eps))**2)
        ratio = ps2 / (ps1 + eps)
        all_metrics['date'].append(date)
        all_metrics['mse_spectrum'].append(float(mse_spec))
        all_metrics['log_mse_spectrum'].append(float(log_mse_spec))
        all_metrics['mean_ratio'].append(float(np.mean(ratio)))
        all_metrics['std_ratio'].append(float(np.std(ratio)))
        all_metrics['min_ratio'].append(float(np.min(ratio)))
        all_metrics['max_ratio'].append(float(np.max(ratio)))

    # Convert to arrays 
    spectra_1 = np.stack(spectra_1)
    spectra_2 = np.stack(spectra_2)
    mean_ps1 = np.mean(spectra_1, axis=0)
    std_ps1 = np.std(spectra_1, axis=0)
    mean_ps2 = np.mean(spectra_2, axis=0)
    std_ps2 = np.std(spectra_2, axis=0)

    first_sample = next(v for v in dataset1.values() if isinstance(v, np.ndarray) and v.ndim == 2)
    nx = first_sample.shape[1]
    dx = dx_model1  # grid spacing in km
    logger.info(f"Data shape: {first_sample.shape}, nx: {nx}")
    logger.info(f"Grid spacing dx: {dx} km")
    wavelengths = (nx * dx) / np.arange(1, len(mean_ps1)+1, dtype=np.float64)  # Avoid division by zero
    nyquist_limit = 2 * dx  # = 5 km for dx=2.5
    mask = wavelengths >= nyquist_limit
    wavelengths = wavelengths[mask]
    mean_ps1 = mean_ps1[mask]
    mean_ps2 = mean_ps2[mask]
    std_ps1 = std_ps1[mask]
    std_ps2 = std_ps2[mask]

    # Save batch spectra cache
    if cache_dir:
        try:
            if cache_path is not None:
                np.savez_compressed(cache_path, wavelengths=wavelengths, mean_ps1=mean_ps1, mean_ps2=mean_ps2)
                logger.info(f"      [cache] Saved batch spectra cache to {cache_path}")
            else:
                logger.warning("      [cache] cache_dir set but cache_path is None; skipping save.")
        except Exception as e:
            logger.warning(f"      [cache] Failed to save batch spectra cache: {e}")

    # Plot mean spectra
    if show_plot or save_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        if loglog:
            ax.loglog(wavelengths, mean_ps1, label=f'{model1} Mean', color=get_color_for_model(model1))
            ax.loglog(wavelengths, mean_ps2, label=f'{model2} Mean', color=get_color_for_model(model2))
        else:
            ax.plot(wavelengths, mean_ps1, label=f'{model1} Mean', color=get_color_for_model(model1), linewidth=2)
            ax.plot(wavelengths, mean_ps2, label=f'{model2} Mean', color=get_color_for_model(model2), linewidth=2)
        ax.set_title(f'{title}\nMean (over time) Radially Averaged Power Spectrum Comparison')
        ax.set_xlabel('Wavelength (km)')

        # Log-scale wavelength ticks (rounded powers of 2)
        tick_vals = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        tick_vals = [tv for tv in tick_vals if wavelengths.min() <= tv <= wavelengths.max()]
        ax.set_xticks(tick_vals)
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='x', which='major', labelsize=10)


        # # Shade region below Nyquist wavelength (2 * dx = 5 km)
        # nyquist_limit = 2 * 2.5  # = 5 km
        # ax.axvspan(wavelengths.min(), nyquist_limit, color='gray', alpha=0.2, label='Below Nyquist')

        # Reverse x-axis to show large scales on left
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[1], xlim[0])
        ax.set_ylabel('Power Spectrum Density')
        ax.legend()
        ax.grid(True, which='both', ls='--', alpha=0.5)

        important_scales = {
            'Large-scale front': 256,
            'Mesoscale': 64,
            'Convective': 8
        }

        for label, wl in important_scales.items():
            if wavelengths.min() <= wl <= wavelengths.max():
                ax.axvline(wl, linestyle='--', color='gray', alpha=0.5)
                ax.text(wl, ax.get_ylim()[1], label, rotation=90, va='top', ha='right', fontsize=8)

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'{fname}.png'), dpi=300)
            logger.info(f"      Saved power spectra comparison to {save_path}/{fname}.png")
        if show:
            plt.show()
        plt.close()

    # Average metrics
    avg_metrics = {
        k: float(np.mean(v)) for k, v in all_metrics.items() if k != 'date'
    }
    std_metrics = {
        k: float(np.std(v)) for k, v in all_metrics.items() if k != 'date'
    }

    if return_all_metrics:
        return avg_metrics | std_metrics, all_metrics # '|' merges two dicts in Python 3.9+
    else:
        return avg_metrics | std_metrics







def compute_distribution_stats(data_model1, data_model2):
    """
        Return comparison statistics between flattened distributions
    """
    ks_stat, ks_pvalue = ks_2samp(data_model1.flatten(), data_model2.flatten())
    w_distance = wasserstein_distance(data_model1.flatten(), data_model2.flatten())
    
    return {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'wasserstein_distance': w_distance
    }

def plot_histograms(data_model1,
                    data_model2,
                    bins=100,
                    model1 ='Model 1',
                    model2 ='Model 2',
                    variable='Variable',
                    log=False,
                    save=True,
                    show=False,
                    save_path='./figures',
                    color_model1=None,
                    color_model2=None,
                    edge=True,
                    hist1=None,
                    hist2=None,
                    edges=None):
    """
        Plot overlaid histograms of two datasets for visual comparison.
    """

    title = f"{variable} | {model1} vs {model2} | Histogram Comparison"
    fname = f"{variable}_{model1}_vs_{model2}_histogram".replace(" ", "_")

    if color_model1 is None:
        color_model1 = get_color_for_model(model1)
    if color_model2 is None:
        color_model2 = get_color_for_model(model2)
    edgecolor = 'black' if edge else None

    fig, ax = plt.subplots(figsize=(10, 6))

    if hist1 is not None and hist2 is not None and edges is not None:
        # plot from precomputed bins
        ax.step(edges[:-1], hist1, where='post', color=color_model1, label=f"{model1}, {variable}")
        ax.step(edges[:-1], hist2, where='post', color=color_model2, label=f"{model2}, {variable}")
        if log:
            ax.set_yscale('log')
        ax.set_xlabel(f'{variable} ({get_unit_for_variable(variable)})')
        ax.set_ylabel('Density')
        ax.set_title(title)
        plt.legend()
        if save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(f"{save_path}/{fname}", dpi=300)
            logger.info(f"      Saved histogram to {save_path}/{fname}")
        if show:
            plt.show()
        return

    ax.hist(data_model1.flatten(), bins=bins, alpha=0.55, label=f"{model1}, {variable}",
            density=True, color=color_model1, edgecolor=edgecolor, linewidth=0.3)
    ax.hist(data_model2.flatten(), bins=bins, alpha=0.55, label=f"{model2}, {variable}",
            density=True, color=color_model2, edgecolor=edgecolor, linewidth=0.3)
    
    if log:
        ax.set_yscale('log')
    
    ax.set_xlabel(f'{variable} ({get_unit_for_variable(variable)})')
    ax.set_ylabel('Density')
    ax.set_title(title)
    
    plt.legend()
    
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        plt.savefig(f"{save_path}/{fname}", dpi=300)
        logger.info(f"      Saved histogram to {save_path}/{fname}")
    if show:
        plt.show()

def compare_distributions(
    data_model1: np.ndarray,
    data_model2: np.ndarray,
    model1: str = 'Model 1',
    model2: str = 'Model 2',
    variable: str = 'Variable',
    bins: int = 100,
    log_hist: bool = False,
    save_figures: bool = True,
    show: bool = False,
    save_path: str = './figures/distribution_comparison',
    return_metrics: bool = True,
    color_model1: str = '#1f77b4',
    color_model2: str = '#ff7f0e',
    cache_dir: Optional[str] = None,
    plot_only: bool = False,    
    ):
    """
        Wrapper function to compute and plot distribution comparison between two datasets.
        
        Parameters:
            data_model1 (np.ndarray): First data field.
            data_model2 (np.ndarray): Second data field.
            model1 (str): Name of the first model.
            model2 (str): Name of the second model.
            variable (str): Variable name for titles/labels.
            bins (int): Number of bins for histograms.
            log_hist (bool): Whether to use logarithmic scale for histogram y-axis.
            save_figures (bool): Whether to save the histogram figure.
            show (bool): Whether to display the histogram figure.
            save_path (str): Directory to save figures.
            return_metrics (bool): Whether to return computed statistics.

        Returns:
            dict: Dictionary containing KS statistic, p-value, and Wasserstein distance if return_metrics is True.
    """
    cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, f"{variable}_{model1}_vs_{model2}_hist.npz".replace(" ", "_"))

    if plot_only and cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as z:
            hist1 = z["hist1"]; hist2 = z["hist2"]; edges = z["edges"]
        plot_histograms(None, None, bins=bins, model1=model1, model2=model2, variable=variable, log=log_hist, save=save_figures, show=show, save_path=save_path, color_model1=color_model1, color_model2=color_model2, edge=True, hist1=hist1, hist2=hist2, edges=edges)
        return None

    if show or save_figures:
        # Compute histograms and save cache if enabled
        hist1, edges = np.histogram(data_model1.flatten(), bins=bins, density=True)
        hist2, _ = np.histogram(data_model2.flatten(), bins=edges, density=True)
        if cache_path:
            try:
                np.savez_compressed(cache_path, hist1=hist1, hist2=hist2, edges=edges)
                logger.info(f"      [cache] Saved histogram cache to {cache_path}")
            except Exception as e:
                logger.warning(f"      [cache] Failed to save histogram cache: {e}")        
        plot_histograms(
            data_model1,
            data_model2,
            bins=bins,
            model1=model1,
            model2=model2,
            variable=variable,
            log=log_hist,
            save=save_figures,
            show=show,
            save_path=save_path,
            color_model1=color_model1,
            color_model2=color_model2,
            hist1=hist1,
            hist2=hist2,
            edges=edges
        )
    
    if return_metrics:
        stats = compute_distribution_stats(data_model1, data_model2)
        return stats

def get_season(dt):
    """
        Return meteorological season for a given datetime object.
    """
    month = dt.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def compare_seasonal_distributions(
    dataset1: dict,
    dataset2: dict,
    model1: str = 'Model 1',
    model2: str = 'Model 2',
    variable: str = 'Variable',
    bins: int = 100,
    log_hist: bool = False,
    save_figures: bool = True,
    show: bool = False,
    save_path: str = './figures/seasonal_distribution_comparison',
    ):
    """
        Plot seasonal histograms comparing HR and LR models.
        Produces:
        (1) Two-panel figure: each model, all four seasons overlaid
        (2) Four-panel figure: each season, both models overlaid
    """

    # === Organize data by season ===
    season_bins_model1 = {'Winter': [], 'Spring': [], 'Summer': [], 'Autumn': []}
    season_bins_model2 = {'Winter': [], 'Spring': [], 'Summer': [], 'Autumn': []}

    for date, arr1 in dataset1.items():
        if date in dataset2:
            arr2 = dataset2[date]
            season = get_season(date)
            season_bins_model1[season].append(np.array(arr1).flatten())
            season_bins_model2[season].append(np.array(arr2).flatten())

    # Concatenate all seasonal arrays
    for season in season_bins_model1:
        season_bins_model1[season] = np.concatenate(season_bins_model1[season]) if season_bins_model1[season] else np.array([]) # type: ignore
        season_bins_model2[season] = np.concatenate(season_bins_model2[season]) if season_bins_model2[season] else np.array([]) # type: ignore

    # === Plot 1: 1x2 panels, each model, seasonal histograms (STEP LINES) ===
    # Uses outlines only to avoid fill overlap - log-scale via axis for consistent behaviour
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, sharey=True)
    colors = {'Winter': '#3366cc', 'Spring': '#2ca02c', 'Summer': '#ffbf00', 'Autumn': '#c44e52'}

    for season, color in colors.items():
        if len(season_bins_model1[season]) > 0:
            axs1[0].hist(season_bins_model1[season], bins=bins, density=True, histtype='step', linewidth=1.6, color=color, label=season)
        if len(season_bins_model2[season]) > 0:
            axs1[1].hist(season_bins_model2[season], bins=bins, density=True, histtype='step', linewidth=1.6, color=color, label=season)

    axs1[0].set_title(f'{model1}')
    axs1[1].set_title(f'{model2}')
    for ax in axs1:
        if log_hist:
            ax.set_yscale('log')
        ax.legend(frameon=False)
        ax.set_xlabel(f'{variable} ({get_unit_for_variable(variable)})')
        ax.set_ylabel('Log count' if log_hist else 'Density')
        ax.grid(True, which='both', ls='--', alpha=0.3)
    
    fig1.suptitle(f"{variable} | Seasonal Histogram Comparison (by model)", fontsize=16)

    # === Plot 2: 2x2 panels, each season, both models (STYLE BY MODEL) ===
    # Same hue per season, distinguish models by linestyle/fill/hatch
    fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, sharey=True)
    axs2 = axs2.flatten()
    season_palette_fill = {'Winter': '#6fa3ff', 'Spring': '#7cd67a', 'Summer': '#ffd24d', 'Autumn': '#f28e8e'}
    season_palette_line = {'Winter': '#1f4ba5', 'Spring': '#1d7f1d', 'Summer': '#e6ac00', 'Autumn': '#a93a3a'}
    for i, season in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
        # Model 1: light fill + solid outline
        if len(season_bins_model1[season]) > 0:
            axs2[i].hist(season_bins_model1[season], bins=bins, density=True, histtype='stepfilled', alpha=0.25, color=season_palette_fill[season], edgecolor=season_palette_line[season], linewidth=1.2, label=model1, zorder=1)
        # Model 3: no fill, dashed outline (on top)
        if len(season_bins_model2[season]) > 0:
            axs2[i].hist(season_bins_model2[season], bins=bins, density=True, histtype='step', linewidth=1.8, linestyle='--', color=season_palette_line[season], label=model2, zorder=2)
            
        axs2[i].set_title(f'{season}')
        if log_hist:
            axs2[i].set_yscale('log')
        axs2[i].legend(frameon=False)
        axs2[i].set_xlabel(f'{variable} ({get_unit_for_variable(variable)})')
        axs2[i].set_ylabel('Log count' if log_hist else 'Density')
        axs2[i].grid(True, which='both', ls='--', alpha=0.3)

    fig2.suptitle(f"{variable} | Seasonal Histogram Comparison (by season)", fontsize=16)

    if save_figures:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fname1 = f"{variable}_{model1}_vs_{model2}_seasonal_histogram_by_model".replace(" ", "_")
        fname2 = f"{variable}_{model1}_vs_{model2}_seasonal_histogram_by_season".replace(" ", "_")
        fig1.savefig(os.path.join(save_path, f'{fname1}.png'), dpi=300)
        fig2.savefig(os.path.join(save_path, f'{fname2}.png'), dpi=300)
        logger.info(f"      Saved seasonal histograms to {save_path}/{fname1}.png and {fname2}.png")
    if show:
        plt.show()
    plt.close('all')


