from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import logging
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Colormap

from sbgm.variable_utils import get_cmap_for_variable
from sbgm.evaluate.evaluate_prcp.plot_utils import _nice, _savefig, _ensure_dir
from sbgm.plotting_utils import get_dk_lsm_outline, overlay_outline
logger = logging.getLogger(__name__)

SET_DPI = 300

# Styling helper
def _get_var_style(var: str):
    """
    Returns (cmap, vmin, vmax, cbar_label) for the value row.
    """
    if var in {"mean", "p95", "p99"}:
        return (get_cmap_for_variable("prcp"), 0.0, None, "mm/day")
    if var in {"sum", "rx1", "rx5"}:
        return (get_cmap_for_variable("prcp"), 0.0, None, "mm")
    if var == "wetfreq":
        return ("Blues", 0.0, 1.0, "fraction of days")
    # generic fallback
    return ("viridis", 0.0, None, "")

def _load_npz(tables_dir: Path, tag: str):
    p = tables_dir / f"{tag}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)

def _draw_single(
    ax,
    data,
    title: str,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label="",
    *,
    dk_mask=None,
    add_stats: bool = False,
    add_colorbar: bool = True,
):
    if data is None:
        ax.axis("off")
        return None

    arr = np.asarray(data)

    # Display with the same north-up orientation as the DK outline.
    # Do not force-mask the ocean here: when eval_land_only=True the ocean is
    # already NaN in the saved arrays; when eval_land_only=False the ocean should
    # remain visible.
    arr_plot = np.flipud(arr.astype(float, copy=False))

    # Ensure NaNs are transparent so the ocean background shows through
    try:
        if isinstance(cmap, Colormap):
            cmap_use = cmap
        else:
            cmap_use = plt.get_cmap(cmap) # type: ignore
        cmap_use = cmap_use.copy()
        cmap_use.set_bad(alpha=0.0)
    except Exception:
        cmap_use = cmap

    im = ax.imshow(arr_plot, cmap=cmap_use, vmin=vmin, vmax=vmax)

    # Keep this so all spatial plots have the same orientation as elsewhere
    # ax.invert_yaxis()

    if dk_mask is not None and dk_mask.shape == arr.shape:
        overlay_outline(ax, np.flipud(dk_mask))

    if add_stats:
        flat = arr_plot.ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size > 0:
            mu = float(np.nanmean(flat))
            sd = float(np.nanstd(flat))
            title = f"{title}  |  {mu:.2f} ± {sd:.2f}"

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    if add_colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=12)

    return im

def plot_spatial_maps(eval_root: str | Path) -> None:
    """
    Compose multi-panel figures per group and variable:

      | Row 0: value maps for HR, Ensemble mean, Ensemble spread (std), and LR (where available).
      | Row 1: ratio maps vs HR for Ensemble mean and LR.
      | Row 2: bias maps vs HR for Ensemble mean and LR.

    Notes:
      - PMM/Generated columns are intentionally omitted.
      - Value colorbars start at 0 to avoid misleading grays on precipitation.
    """
    eval_root = Path(eval_root)
    tables = eval_root / "tables"
    figs   = _ensure_dir(eval_root / "figures")

    # Discover groups by NPZ names
    tags = [p.stem for p in tables.glob("spatial_*.npz")]
    # Build mapping group -> dict(source->npz)
    buckets: Dict[str, Dict[str, Path]] = {}
    for stem in tags:
        parts = stem.split("_", 2)  # ["spatial", src, group]
        if len(parts) != 3:
            continue
        _, src, group = parts
        buckets.setdefault(group, {})[src] = tables / f"{stem}.npz"

    if not buckets:
        logger.warning("[plot_spatial] No spatial_* NPZ files found under %s", str(tables))
        return

    variables = ["mean","sum","rx1","rx5","p95","p99","wetfreq"]
    dk_mask = get_dk_lsm_outline()

    for group, src_map in sorted(buckets.items()):
        # Load available sources (HR, EnsMean, EnsStd, LR)
        npz_hr      = _load_npz(tables, f"spatial_hr_{group}")
        npz_ensmean = _load_npz(tables, f"spatial_ensmean_{group}")
        npz_ensstd  = _load_npz(tables, f"spatial_ensstd_{group}")
        npz_lr      = _load_npz(tables, f"spatial_lr_{group}")

        for var in variables:
            arrs: List[np.ndarray] = []
            titles: List[str] = []
            idx_map = {}  # keys: "hr", "ens", "ensstd", "lr"
            cmap, vmin, vmax, clabel = _get_var_style(var)

            if npz_hr is not None and var in npz_hr:
                idx_map["hr"] = len(arrs); arrs.append(npz_hr[var]); titles.append(f"HR | {var}")
            if npz_ensmean is not None and var in npz_ensmean:
                idx_map["ens"] = len(arrs); arrs.append(npz_ensmean[var]); titles.append(f"Ensemble mean | {var}")
            # Ensemble spread (std) column removed from spatial multi-panel plots                
            # if npz_ensstd is not None and var in npz_ensstd:
            #     idx_map["ensstd"] = len(arrs); arrs.append(npz_ensstd[var]); titles.append(f"Ensemble spread (std) | {var}")
            if npz_lr is not None and var in npz_lr:
                idx_map["lr"] = len(arrs); arrs.append(npz_lr[var]); titles.append(f"LR | {var}")

            if not arrs:
                continue

            # robust vmin/vmax across value arrays (1–99th) computed over land-only pixels (avoid ocean zeros)
            stack_chunks: List[np.ndarray] = []
            for a in arrs:
                if a is None:
                    continue
                aa = np.asarray(a)
                vals = aa.reshape(-1)
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    stack_chunks.append(vals)

            stack_vals = np.concatenate(stack_chunks) if stack_chunks else np.array([], dtype=float)

            if stack_vals.size > 0:
                if vmin is None:
                    vmin = float(np.nanpercentile(stack_vals, 1.0))
                if vmax is None:
                    vmax = float(np.nanpercentile(stack_vals, 99.0))
                # Precipitation should not go negative; only clamp if tiny numerical noise
                if np.isfinite(vmin) and vmin < 0.0:
                    vmin = 0.0
                if np.isfinite(vmin) and np.isfinite(vmax) and vmin >= vmax:
                    vmin, vmax = float(np.nanmin(stack_vals)), float(np.nanmax(stack_vals))
                    if vmin < 0.0:
                        vmin = 0.0

            _nice()

            # helpers for ratio/bias vs HR
            def _safe_ratio(num, den):
                if num is None or den is None:
                    return None
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = num / den
                r[~np.isfinite(r)] = np.nan
                return r

            def _safe_bias(num, den):
                if num is None or den is None:
                    return None
                with np.errstate(invalid="ignore"):
                    b = num - den
                b[~np.isfinite(b)] = np.nan
                return b

            hr_arr   = npz_hr[var]       if (npz_hr is not None and var in npz_hr) else None
            ens_arr  = npz_ensmean[var]  if (npz_ensmean is not None and var in npz_ensmean) else None
            lr_arr   = npz_lr[var]       if (npz_lr is not None and var in npz_lr) else None

            rat_ens  = _safe_ratio(ens_arr, hr_arr)
            rat_lr   = _safe_ratio(lr_arr,  hr_arr)
            bias_ens = _safe_bias(ens_arr, hr_arr)
            bias_lr  = _safe_bias(lr_arr,  hr_arr)

            # Layout: row 0 = values, row 1 = ratios, row 2 = biases
            ncols = len(arrs)
            fig, axs = plt.subplots(3, ncols, figsize=(4.0*ncols, 10.5), squeeze=False)
            # Reserve space on the right for the shared row-0 colorbar
            fig.subplots_adjust(right=0.89, wspace=0.05, hspace=0.18)

            # Row 0: values (μ ± σ in titles)
            first_im = None
            for j, a in enumerate(arrs):
                im = _draw_single(
                    axs[0, j], a, titles[j],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar_label=clabel, dk_mask=dk_mask,
                    add_stats=True,
                    add_colorbar=False,   # shared cbar below
                )
                if first_im is None and im is not None:
                    first_im = im

            # Shared colorbar for row-0 value maps (bigger + easier to read)
            # Place it in an explicit cax so it cannot overlap the last panel.
            if first_im is not None:
                # [left, bottom, width, height] in figure coordinates; top row occupies ~top third
                cax = fig.add_axes([0.92, 0.69, 0.015, 0.22])
                cb = fig.colorbar(first_im, cax=cax)
                if clabel:
                    cb.set_label(clabel, fontsize=12)

            # Initialize rows 1–2 as empty
            for j in range(ncols):
                axs[1, j].axis("off")
                axs[2, j].axis("off")

            # Row 1: ratios
            if "ens" in idx_map and rat_ens is not None:
                _draw_single(
                    axs[1, idx_map["ens"]], rat_ens, f"EnsMean/HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=0.5, vmax=1.5,
                    cbar_label="ratio", dk_mask=dk_mask, add_stats=True
                )
                axs[1, idx_map["ens"]].axis("on")
            if "lr" in idx_map and rat_lr is not None:
                _draw_single(
                    axs[1, idx_map["lr"]], rat_lr, f"LR/HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=0.5, vmax=1.5,
                    cbar_label="ratio", dk_mask=dk_mask, add_stats=True
                )
                axs[1, idx_map["lr"]].axis("on")

            # Row 2: biases (symmetric limits)
            bias_vals = []
            if bias_ens is not None: bias_vals.append(np.abs(bias_ens).ravel())
            if bias_lr  is not None: bias_vals.append(np.abs(bias_lr).ravel())
            bmax = float(np.nanpercentile(np.concatenate(bias_vals), 99.0)) if bias_vals else (float(vmax) if vmax is not None else 1.0)

            if "ens" in idx_map and bias_ens is not None:
                _draw_single(
                    axs[2, idx_map["ens"]], bias_ens, f"EnsMean-HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=-bmax, vmax=bmax,
                    cbar_label="bias", dk_mask=dk_mask, add_stats=True
                )
                axs[2, idx_map["ens"]].axis("on")
            if "lr" in idx_map and bias_lr is not None:
                _draw_single(
                    axs[2, idx_map["lr"]], bias_lr, f"LR-HR | {var}",
                    cmap=get_cmap_for_variable("prcp_bias"), vmin=-bmax, vmax=bmax,
                    cbar_label="bias", dk_mask=dk_mask, add_stats=True
                )
                axs[2, idx_map["lr"]].axis("on")

            fig.suptitle(f"{group}: {var}", fontsize=16)
            _savefig(fig, figs / f"spatial_{group}_{var}.png", dpi=SET_DPI)