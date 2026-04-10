
import torch
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, Union, List, Dict, Tuple

mpl.rcParams["hatch.linewidth"] = 0.3  # thinner hatches by default

from sbgm.utils import _squeeze_geo_value
from sbgm.variable_utils import (
    get_units,
    get_cmaps,
    get_cmap_for_variable,
    get_color_for_model,
    get_color_for_model_cycle,
)

# Set up logging
logger = logging.getLogger(__name__)


# ------------------------------
# Full-domain LSM cache (for LR context overlays)
# ------------------------------
_FULL_LSM_CACHE: dict[tuple[int, int, str], np.ndarray] = {}

def _load_full_lsm(cfg: dict) -> np.ndarray | None:
    """
    Load the raw full-domain LSM (H,W) from cfg['paths']['lsm_path'] or DATA_DIR fallback.
    Returns array in the same orientation as your plotted fields (already flipud in data loading).
    """
    try:
        p = None
        if isinstance(cfg, dict):
            p = cfg.get("paths", {}).get("lsm_path", None)
        if p is None:
            base = os.environ.get("DATA_DIR", None)
            if base is None:
                return None
            p = os.path.join(base, "data_lsm/truth_fullDomain/lsm_full.npz")

        if not os.path.exists(p):
            return None

        d = np.load(p, allow_pickle=True)
        key = "data" if "data" in getattr(d, "files", []) else d.files[0]
        arr = np.asarray(d[key])
        arr = np.squeeze(arr)
        # Match your data loading convention (you do np.flipud when reading lsm_full_domain)
        arr = np.flipud(arr).copy()
        return arr
    except Exception:
        return None

# ==============================
# Visualization helpers
# ==============================

class _ZeroFirstNormalize(Normalize):
    """
    Normalize where *exact zeros* get mapped to a dedicated first colormap color.

    Mapping:
      - x == 0 -> [0, zero_frac)
      - x > 0  -> [zero_frac, 1]

    Intended for precipitation-like variables where 0 is qualitatively special.
    """
    def __init__(self, vmin=None, vmax=None, *, zero_frac: float = 0.06, min_positive: float | None = None, clip: bool = False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.zero_frac = float(zero_frac)
        self.min_positive = None if min_positive is None else float(min_positive)

    def __call__(self, value, clip=None):
        v = np.asarray(value)
        out = np.zeros_like(v, dtype=float)

        nan_mask = ~np.isfinite(v)
        zero_mask = (v == 0) & (~nan_mask)
        pos_mask = (v > 0) & (~nan_mask)

        vmin = self.vmin
        vmax = self.vmax

        if vmin is None or vmax is None:
            if np.any(pos_mask):
                vv = v[pos_mask]
                if vmin is None:
                    vmin = float(np.nanmin(vv))
                if vmax is None:
                    vmax = float(np.nanmax(vv))
            else:
                vmin = 0.0 if vmin is None else float(vmin)
                vmax = 1.0 if vmax is None else float(vmax)

        vmin_pos = float(vmin)
        if self.min_positive is not None:
            vmin_pos = max(vmin_pos, self.min_positive)

        denom = (float(vmax) - vmin_pos)
        denom = denom if denom != 0 else 1.0

        out[zero_mask] = 0.0

        if np.any(pos_mask):
            scaled = (v[pos_mask] - vmin_pos) / denom
            scaled = np.clip(scaled, 0.0, 1.0)
            out[pos_mask] = self.zero_frac + (1.0 - self.zero_frac) * scaled

        out[nan_mask] = np.nan
        return out

def _to_numpy_2d(x):
    if x is None:
        return None
    import numpy as np
    import torch
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 2:
        x = x.reshape((-1, x.shape[-2], x.shape[-1]))[0]
    return x

def _is_precip_var(var: str) -> bool:
    s = str(var).lower()
    return s in ("prcp", "tp", "precip", "precipitation")


def _get_cfg_vis(cfg: dict | None) -> dict:
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("visualization", {}) or {}


def _overlay_landsea_outline(ax, lsm_mask, *, color="lightgrey", linewidth=1.1):
    try:
        m = _to_numpy_2d(lsm_mask)
        if m is None:
            return
        ax.contour(m.astype(float, copy=False), levels=[0.5], colors=color, linewidths=linewidth)
    except Exception:
        return


def _overlay_ocean_background(
    ax,
    ocean_mask: np.ndarray,
    *,
    style: str = "hatch",
    facecolor: str = "#f3f3f3",
    hatch: str = "//",
    alpha: float = 0.15,
):
    """
    Draw a distinct background on ocean pixels.

    style:
      - 'hatch': hatch pattern (recommended)
      - 'solid': solid tinted ocean
      - 'none' : nothing
    """
    style = str(style).lower()
    if style in ("none", "off", "false", "0"):
        return

    m = _to_numpy_2d(ocean_mask)
    if m is None:
        return

    try:
        if style == "solid":
            ax.contourf(m.astype(int), levels=[0.5, 1.5], colors=[facecolor], alpha=alpha)
        else:
            ax.contourf(m.astype(int), levels=[0.5, 1.5], colors=[facecolor], alpha=alpha)
            cf = ax.contourf(m.astype(int), levels=[0.5, 1.5], colors="none", hatches=[hatch], alpha=0.0)
            for c in cf.collections:
                c.set_edgecolor("none")
    except Exception:
        return


def _build_precip_zero_cmap(base_cmap, *, zero_color: str = "#ffffff", n: int = 256):
    """Create a ListedColormap whose first color is a dedicated 'exact zero' color."""
    try:
        cm = mcm.get_cmap(base_cmap) if isinstance(base_cmap, str) else base_cmap
    except Exception:
        cm = base_cmap

    colors = cm(np.linspace(0, 1, n)) # type: ignore
    colors[0] = mcolors.to_rgba(zero_color) # type: ignore
    return mcolors.ListedColormap(colors) # type: ignore

# --- Robust conversion for imshow ---
import numpy as _np
import torch as _torch
from pathlib import Path
from datetime import datetime


def _savefig(fig, out_path: Path, dpi: int = 300):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _nice():
    # lightweight, you can override with your global style later
    plt.rcParams.update({
        "figure.figsize": (5.5, 4.0),
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

def _to_date_safe(s: str) -> Optional[datetime]:
    s = s.strip()
    # accept "YYYY-MM-DD" and "YYYYMMDD"
    try:
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"

def _extract_land_mask(sample: dict) -> np.ndarray | None:
    """Return boolean land mask [H,W] if present in sample."""
    for k in ("lsm_hr", "lsm", "land_sea_mask", "mask"):
        if k in sample and sample[k] is not None:
            m = _to_numpy_2d(sample[k])
            if m is None:
                continue
            return (m >= 0.5)
    return None

def _masked_finite_values(arr2d: np.ndarray, land_mask: np.ndarray | None) -> np.ndarray:
    """Flatten finite values, optionally land-masked."""
    a = np.asarray(arr2d)
    a = np.squeeze(a)
    if a.ndim != 2:
        a = a.reshape((-1, a.shape[-2], a.shape[-1]))[0]

    if land_mask is not None and np.asarray(land_mask).shape == a.shape:
        a = a[land_mask]
    else:
        a = a.ravel()

    a = a[np.isfinite(a)]
    return a

def _shared_minmax(arrs: list[np.ndarray], land_mask: np.ndarray | None = None) -> tuple[float | None, float | None]:
    vals = []
    for a in arrs:
        if a is None:
            continue
        v = _masked_finite_values(a, land_mask)
        if v.size:
            vals.append(v)
    if not vals:
        return None, None
    vcat = np.concatenate(vals, axis=0)
    return float(vcat.min()), float(vcat.max())


# ------------------------------
# DK outline via LSM (cached)
# ------------------------------
_DK_LSM_CACHE: dict[tuple[int, int, int, int], np.ndarray] | np.ndarray | None = None

def _load_dk_lsm_outline(
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    base: str | None = None,
    rel_path: str = "data_lsm/truth_fullDomain/lsm_full.npz",
    key_candidates: tuple[str, ...] = ("lsm_hr", "lsm", "mask", "roi", "lsm_full", "data", "arr_0"),
) -> np.ndarray | None:
    """Load and crop a land-sea mask and return a boolean [H,W] mask for Denmark.
    bounds is interpreted as (y0, y1, x0, x1) with y1/x1 exclusive; e.g., (200,328,380,508) → 128x128.
    """
    try:
        # Resolve base path for data directory
        if base is None:
            base = os.environ.get("DATA_DIR", None)
        if base is None:
            logger.warning("[DK_LSM] DATA_DIR not set and no base path provided; cannot load LSM.")
            return None

        logger.info("[DEBUG] Loading DK LSM outline from %s/%s", base, rel_path)
        p = Path(base) / rel_path
        if not p.exists():
            logger.warning("[DEBUG] DK LSM outline file not found: %s", str(p))
            return None
        d = np.load(p, allow_pickle=True)
        # Print the keys available in the npz file for debugging
        arr = None
        if hasattr(d, "files"):
            for k in key_candidates:
                if k in d.files:
                    arr = d[k]
                    break
        if arr is None:
            logger.warning("[DEBUG] DK LSM outline: no suitable key found in %s", str(p))
            return None
        a = np.asarray(arr)
        # normalize to [H,W]
        if a.ndim == 4 and a.shape[:2] == (1, 1):
            a = a.squeeze(0).squeeze(0)
        elif a.ndim == 3 and a.shape[0] == 1:
            a = a.squeeze(0)
        y0, y1, x0, x1 = bounds
        a = np.flipud(a)  # flip vertically if needed
        a = a[y0:y1, x0:x1]
        m = (a >= 0.5)
        # m = np.flipud(m)  # flip back to original orientation

        logger.info("[DEBUG] DK LSM outline loaded with shape %s", str(m.shape))
        return m.astype(bool, copy=False)
    except Exception as e:
        logger.exception("[DEBUG] Exception while loading DK LSM outline: %s", str(e))
        return None

def get_dk_lsm_outline(
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    base: str | None = None,
) -> np.ndarray | None:
    """
    Return cached DK outline mask (boolean [H,W]) for the requested `bounds`.
    Caches per-bounds so different crops return correctly sized masks.
    """
    global _DK_LSM_CACHE
    # For backward compat, treat _DK_LSM_CACHE as a dict if needed
    if isinstance(_DK_LSM_CACHE, dict):
        cache = _DK_LSM_CACHE
    else:
        # Single cache for default bounds (legacy)
        cache = {}
        if _DK_LSM_CACHE is not None:
            cache[(200, 328, 380, 508)] = _DK_LSM_CACHE
        _DK_LSM_CACHE = cache
    try:
        key = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    except Exception:
        key = (200, 328, 380, 508)
    if key in _DK_LSM_CACHE:
        return _DK_LSM_CACHE[key]
    m = _load_dk_lsm_outline(bounds=key, base=base)
    if m is not None:
        _DK_LSM_CACHE[key] = m
    return m

def overlay_outline(ax, mask: np.ndarray | None, *, color: str = "black", linewidth: float = 0.8):
    """Overlay a contour outline (level 0.5) on the given axes if mask is provided."""
    if mask is None:
        return
    try:
        ax.contour(mask.astype(float, copy=False), levels=[0.5], colors=color, linewidths=linewidth)
    except Exception:
        pass


# === Centralized imshow for variables with DK outline ===
def imshow_variable(
    ax,
    img2d: np.ndarray,
    *,
    variable: str,
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    add_outline: bool = True,
    outline_color: str = "lightgrey",
    outline_linewidth: float = 0.7,
    # Ocean handling
    show_ocean: bool = True,
    lsm_mask: np.ndarray | None = None,
    ocean_background: str = "hatch",  # 'hatch'|'solid'|'none'
    ocean_facecolor: str = "#f3f3f3",
    ocean_hatch: str = "////",
    ocean_alpha: float = 0.25,
    ocean_overlay: bool = False, # draw ocean styling on top, but keep data visible
    ocean_data_alpha: float | None = None, # if set, ocean pixels are dimmed (e.g. 0.4)
    # Precip exact-zero appearance
    precip_zero_color: str | None = "#ffffff",
    precip_zero_frac: float = 0.06,
    precip_min_positive: float | None = None,
    # Back-compat: under-color for <vmin
    under_color: str | None = None,
    under_threshold: float | None = None,
    # Optional cfg for defaults
    cfg: dict | None = None,
):
    """
    Centralized imshow used across plotting.

    Adds:
      1) land/sea outline overlay on all maps (when LSM available)
      2) distinct ocean background when show_ocean=False
      3) precipitation-only exact-zero color

    Returns: image handle from imshow.
    """
    if img2d is None:
        raise ValueError("imshow_variable: img2d is None")

    vis = _get_cfg_vis(cfg)

    outline_color = str(vis.get("outline_color", outline_color))
    outline_linewidth = float(vis.get("outline_linewidth", outline_linewidth))

    ocean_background = str(vis.get("ocean_background", ocean_background))
    ocean_facecolor = str(vis.get("ocean_facecolor", ocean_facecolor))
    ocean_hatch = str(vis.get("ocean_hatch", ocean_hatch))
    ocean_alpha = float(vis.get("ocean_alpha", ocean_alpha))
    # Optional hatch linewidth (Matplotlib controls this globally via rcParams)
    hatch_lw = vis.get("ocean_hatch_linewidth", None) if isinstance(vis, dict) else None
    if hatch_lw is not None:
        try:
            mpl.rcParams["hatch.linewidth"] = float(hatch_lw)
        except Exception:
            pass
    precip_zero_color = vis.get("prcp_zero_color", precip_zero_color)
    precip_zero_frac = float(vis.get("prcp_zero_frac", precip_zero_frac))
    precip_min_positive = vis.get("prcp_min_positive", precip_min_positive)

    arr = np.asarray(img2d)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))[0]

    cm_in = cmap or get_cmap_for_variable(variable)
    try:
        cm_obj = mcm.get_cmap(cm_in) if isinstance(cm_in, str) else cm_in
    except Exception:
        cm_obj = cm_in

    # Optional legacy set-under behavior
    if under_color is not None:
        w_ext = getattr(cm_obj, "with_extremes", None)
        if callable(w_ext):
            try:
                cm_obj = w_ext(under=under_color)
            except Exception:
                pass
        else:
            s_under = getattr(cm_obj, "set_under", None)
            if callable(s_under):
                try:
                    s_under(under_color)
                except Exception:
                    pass

    # If we are hiding the ocean, prepare a mask and background
    ocean_mask = None
    if (not show_ocean) and (lsm_mask is not None):
        m = np.asarray(lsm_mask)
        if m.ndim != 2:
            m = np.squeeze(m)
        # Only use the provided mask if it matches the plotted field.
        # In Paper2 large-domain context, LR panels can be 589x789 while lsm_hr is 128x128.
        # In that case, silently skip ocean masking/background instead of crashing.
        if m.shape == arr.shape:
            ocean_mask = (m < 0.5)
            _overlay_ocean_background(
                ax,
                ocean_mask,
                style=ocean_background,
                facecolor=ocean_facecolor,
                hatch=ocean_hatch,
                alpha=ocean_alpha,
            )
        else:
            ocean_mask = None


    # Precipitation exact-zero handling
    is_prcp = _is_precip_var(variable)
    norm = None
    if is_prcp and precip_zero_color is not None:
        cm_obj = _build_precip_zero_cmap(cm_obj, zero_color=str(precip_zero_color))
        norm = _ZeroFirstNormalize(vmin=vmin, vmax=vmax, zero_frac=precip_zero_frac, min_positive=precip_min_positive)

    # Legacy under-threshold -> vmin
    if under_threshold is not None and vmin is None and norm is None:
        vmin = float(under_threshold)

    # Matplotlib does not allow passing both `norm` and `vmin/vmax` to imshow.
    # If we built a Normalize (precip exact-zero), push vmin/vmax into the norm so
    # caller-provided limits still control the scale (e.g. consistent HR vs LR).
    vmin_arg = vmin
    vmax_arg = vmax
    if norm is not None:
        if vmin is not None:
            try:
                norm.vmin = float(vmin)
            except Exception:
                pass
        if vmax is not None:
            try:
                norm.vmax = float(vmax)
            except Exception:
                pass
        vmin_arg = None
        vmax_arg = None

    # Plot: if ocean is hidden, make ocean NaNs transparent so background shows through
    if ocean_mask is not None:
        plot_arr = arr.copy()
        plot_arr[ocean_mask] = np.nan
        try:
            cm_obj = cm_obj.copy() # type: ignore
            cm_obj.set_bad(alpha=0.0)
        except Exception:
            pass
        im = ax.imshow(plot_arr, cmap=cm_obj, vmin=vmin_arg, vmax=vmax_arg, norm=norm, interpolation="nearest", origin="lower")
    else:
        im = ax.imshow(arr, cmap=cm_obj, vmin=vmin_arg, vmax=vmax_arg, norm=norm, interpolation="nearest", origin="lower")


    # ------------------------------------------------------------
    # Outline / land-sea context handling
    # ------------------------------------------------------------

    # 1) If provided lsm_mask doesn't match the plotted field shape, discard it
    #    (prevents tiny Denmark-in-corner artifacts when LR is 589x789 but lsm_hr is 128x128)
    if lsm_mask is not None:
        mm = np.asarray(lsm_mask)
        mm = np.squeeze(mm)
        if mm.ndim == 2 and mm.shape != arr.shape:
            lsm_mask = None

    # 2) Detect Paper2 large_domain LR-context panel shape
    lr_ctx_shape = None
    if isinstance(cfg, dict):
        paper2 = cfg.get("paper2", {}) or {}
        scfg = paper2.get("spatial_context", {}) or {}
        if str(scfg.get("mode", "")).lower() == "large_domain":
            lr_ctx = scfg.get("lr_context_size", None)
            if lr_ctx is not None:
                lr_ctx_shape = tuple(lr_ctx)

    # Check if this panel matches the LR-context shape (e.g. 589x789) where we want to use the full-domain LSM for
    is_lr_context_panel = (lr_ctx_shape is not None and tuple(arr.shape) == tuple(lr_ctx_shape))

    # 3) If this is an LR-context panel and we have no matching lsm_mask, load full-domain LSM for outlines/masking
    if lsm_mask is None and is_lr_context_panel and isinstance(cfg, dict):
        print("[DEBUG] imshow_variable: LR context panel (variable=%s, shape=%s) with no LSM provided; attempting to load full-domain LSM for outlines/masking.", variable, arr.shape)
        try:
            cache_key = (arr.shape[0], arr.shape[1], str(cfg.get("paths", {}).get("lsm_path", "")))
            if cache_key in _FULL_LSM_CACHE:
                lsm_mask = _FULL_LSM_CACHE[cache_key]
            else:
                full_lsm = _load_full_lsm(cfg)  # must return (H,W) matching full domain, already flipped consistently
                if full_lsm is not None and tuple(full_lsm.shape) == tuple(arr.shape):
                    _FULL_LSM_CACHE[cache_key] = full_lsm
                    lsm_mask = full_lsm
        except Exception:
            pass

    # 4) Outline selection rules:
    #    - LR context panels: outline comes from full-domain LSM (lsm_mask)
    #    - Non-LR-context panels: outline comes from DK outline for `bounds` (or matching lsm_mask if present)
    # If variable is topo or lsm, skip outline since it's redundant with the data
    if variable in ("topo", "lsm", "land_sea_mask", "mask"):
        add_outline = False

    # Do not draw a local DK outline on large-context LR panels when no matching
    # context/full-domain LSM is available. This avoids the tiny Denmark outline
    # appearing inside the 589x789 context panel.
    if add_outline and is_lr_context_panel and lsm_mask is None:
        add_outline = False

    if add_outline:
        mask_for_outline = None

        # First choice for co-located / local panels: use the in-sample mask if it matches.
        if lsm_mask is not None:
            mm = np.asarray(lsm_mask)
            mm = np.squeeze(mm)
            if mm.ndim == 2 and mm.shape == arr.shape:
                mask_for_outline = mm

        # For large LR-context panels, only use the full-domain mask.
        if mask_for_outline is None and is_lr_context_panel:
            mask_for_outline = None

        # Last fallback for local/co-located panels only: use the cached DK outline.
        if mask_for_outline is None and (not is_lr_context_panel):
            dk = get_dk_lsm_outline(bounds=bounds)
            if dk is not None:
                dk = np.asarray(dk)
                dk = np.squeeze(dk)
                if dk.ndim == 2 and dk.shape == arr.shape:
                    mask_for_outline = dk

        if mask_for_outline is not None:
            _overlay_landsea_outline(ax, mask_for_outline, color=outline_color, linewidth=outline_linewidth)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# === Apply model color scheme to lines/markers ===
def apply_model_colors(
    ax,
    *,
    exclude_kind: str | None = None,
    assume_kind: str | None = None,
):
    """
    Recolors existing line artists in `ax` to follow model colors defined in variable_utils.get_color_for_model.
    - We infer the intended model from each legend label or Line2D label.
    - If exclude_kind == 'seasonal', we skip recoloring (keeps user's preferred seasonal palette).
    - If assume_kind provided, it's only metadata; current logic relies on labels.

    Typical usage right after plotting lines and before/after ax.legend().
    """
    if exclude_kind and exclude_kind.lower() == "seasonal":
        return

    # Collect handles and labels robustly
    handles, labels = ax.get_legend_handles_labels()
    if not labels:
        # Try to infer from lines if no legend exists yet
        lines = [l for l in ax.get_lines() if hasattr(l, "get_label")]
        handles = lines
        labels = [l.get_label() for l in lines]

    for h, lab in zip(handles, labels):
        key = None
        if lab is not None:
            s = lab.strip().lower()
            if any(k in s for k in ["hr", "danra", "truth", "high-res"]):
                key = "hr"
            elif "pmm" in s:
                key = "pmm"
            elif any(k in s for k in ["gen", "generated", "model", "ens"]):
                key = "generated"
            elif any(k in s for k in ["lr", "era5", "low-res"]):
                key = "lr"
        if key is None:
            continue
        try:
            color = get_color_for_model(key)
            # Set both face/edge colors as appropriate
            if hasattr(h, "set_color"):
                h.set_color(color)
            if hasattr(h, "set_markerfacecolor"):
                h.set_markerfacecolor(color)
            if hasattr(h, "set_markeredgecolor"):
                h.set_markeredgecolor(color)
            if hasattr(h, "set_facecolor"):
                h.set_facecolor(color)
            if hasattr(h, "set_edgecolor"):
                h.set_edgecolor(color)
        except Exception:
            # best-effort; keep going
            pass


# === Convenience wrapper for spatial panel plotting ===
def plot_spatial_panel(
    ax,
    img2d: np.ndarray,
    *,
    bounds: tuple[int, int, int, int] = (200, 328, 380, 508),
    variable: str,
    vmin: float | None = None,
    vmax: float | None = None,
    add_dk_outline: bool = True,
    outline_color: str = "darkgrey",
    outline_linewidth: float = 0.8,
    title: str | None = None,
    under_color: str | None = None,
    under_threshold: float | None = None,
):
    """
    Convenience wrapper used by spatial map routines to ensure:
      - correct variable colormap
      - DK outline overlay
      - tight axis cosmetics
    """
    im = imshow_variable(
        ax,
        img2d,
        variable=variable,
        vmin=vmin,
        vmax=vmax,
        # add_dk_outline=add_dk_outline,
        outline_color=outline_color,
        outline_linewidth=outline_linewidth,
        under_color=under_color,
        under_threshold=under_threshold,
        bounds=bounds,
    )
    if title:
        ax.set_title(title, fontsize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    ax.figure.colorbar(im, cax=cax, orientation="vertical")
    return im
# === IO-agnostic maybe_compute helper ===
def maybe_compute(cache_exists: bool, plot_only: bool, compute_fn, *args, **kwargs):
    """
    Calling-side helper for heavy steps:
      - If plot_only is True and cache_exists, SKIP compute_fn and return None.
      - Else, run compute_fn(*args, **kwargs) and return its result.

    This keeps plotting_utils IO-agnostic; the caller is responsible for checking
    the concrete cache path(s) and for loading cached results when plot_only=True.
    """
    if plot_only and cache_exists:
        logger.info("[plot_only] Skipping heavy computation because cache exists.")
        return None
    return compute_fn(*args, **kwargs)

def _to_imshow_image(arr, prefer_channel: int = 0):
    """
    Return (img, was_rgb) where `img` is suitable for plt.imshow.
      - 2D → as is
      - 3D (H,W,3|4) → RGB(A)
      - 3D (C,H,W) → pick channel `prefer_channel` (or squeeze if C==1)
      - torch.Tensor → to cpu().numpy()
    """
    if isinstance(arr, _torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = _np.asarray(arr)

    if arr.ndim == 2:
        return arr, False

    if arr.ndim == 3:
        H, W = arr.shape[-2], arr.shape[-1]
        # RGB(A) as (H,W,3|4)
        if arr.shape[0] == H and arr.shape[1] == W and arr.shape[-1] in (3, 4):
            return arr, True
        # Channel-first (C,H,W)
        if arr.shape[0] in (1, 2, 3, 4):
            C = arr.shape[0]
            if C == 1:
                return arr[0], False
            ch = max(0, min(prefer_channel, C - 1))
            return arr[ch], False
        # Generic fallback for (H,W,C)
        if arr.shape[-1] in (3, 4) and arr.shape[0] == H and arr.shape[1] == W:
            return arr, True

    # Last resort: squeeze singletons or take first slice
    squeezed = _np.squeeze(arr)
    if squeezed.ndim == 2:
        return squeezed, False
    view = squeezed.reshape((-1, squeezed.shape[-2], squeezed.shape[-1]))[0]
    return view, False

def plot_sample(sample, cfg, figsize=None):
    """
    Plot a single dataset sample with a layout that works for both co-located and
    large-domain Paper2 setups.

    Main behaviour:
        - Prefer physical/original fields when available.
        - Show both local and large-domain LR panels when present.
        - Use panel-specific LSM masks so the outline follows the actual sample geometry
        - Automatically wrap to multiple rows when many panels are present.
    """
    hr_model = cfg['highres']['model']
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    hr_units, lr_units = get_units(cfg)
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    default_lr_cmap = 'inferno'
    extra_cmap_dict = {"topo": "terrain", "sdf": "coolwarm", "lsm": "binary"}

    vis = cfg.get('visualization', {}) or {}
    show_ocean = bool(vis.get('show_ocean', True))
    show_scaled_if_no_original = bool(vis.get('show_both_orig_scaled', False))
    add_boxplot_per_panel = bool(vis.get('add_boxplot_per_panel', True))

    paper2 = cfg.get('paper2', {}) or {}
    spatial_cfg = paper2.get('spatial_context', {}) or {}
    spatial_mode = str(spatial_cfg.get('mode', '')).lower()
    is_large_domain = spatial_mode == 'large_domain'

    def _to_np(x):
        if x is None:
            return None
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _to_2d(x):
        if x is None:
            return None
        arr = _to_np(x)
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def _unit_for_lr_base(base_name: str) -> str:
        conds = list(cfg['lowres'].get('condition_variables', []))
        if base_name in conds:
            idx = conds.index(base_name)
            if idx < len(lr_units):
                return lr_units[idx]
        return '—'

    def _panel_title(panel_kind, base_name=None, is_physical=True):
        if panel_kind == 'hr':
            return f"HR {hr_model} ({var})\nphysical [{hr_units}]" if is_physical else f"HR {hr_model} ({var})\nscaled"
        if panel_kind == 'lr_local':
            unit = _unit_for_lr_base(base_name)
            return f"LR {lr_model} ({base_name}) local\nphysical [{unit}]" if is_physical else f"LR {lr_model} ({base_name}) local\nscaled"
        if panel_kind == 'lr_context':
            unit = _unit_for_lr_base(base_name)
            return f"LR {lr_model} ({base_name}) context\nphysical [{unit}]" if is_physical else f"LR {lr_model} ({base_name}) context\nscaled"
        if panel_kind == 'extra':
            if base_name == 'topo':
                return 'Topography'
            if base_name == 'lsm':
                return 'Land/Sea Mask\n(LSM)'
            if base_name == 'sdf':
                return 'SDF'
            return str(base_name)
        return str(base_name)

    def _pick_display_array(primary_key, original_key=None):
        if original_key is not None:
            orig = sample.get(original_key, None)
            if orig is not None:
                arr = _to_2d(orig)
                if arr is not None:
                    return arr, True, original_key
        arr = _to_2d(sample.get(primary_key, None))
        if arr is not None:
            return arr, False, primary_key
        return None, False, primary_key

    panel_specs = []

    # HR target panel
    hr_primary = f"{var}_hr"
    hr_original = f"{var}_hr_original"
    hr_img, hr_is_physical, hr_source_key = _pick_display_array(hr_primary, hr_original)
    if hr_img is not None:
        panel_specs.append({
            'kind': 'hr',
            'base_name': var,
            'source_key': hr_source_key,
            'img': hr_img,
            'is_physical': hr_is_physical,
            'cmap': hr_cmap,
            'lsm_key': 'lsm_hr' if 'lsm_hr' in sample else ('lsm' if 'lsm' in sample else None),
            'show_ocean': show_ocean,
        })

    # LR panels: prefer local + context variants when they exist
    cond_vars = list(cfg['lowres'].get('condition_variables', []))
    for base in cond_vars:
        local_key = f"{base}_lr_local"
        local_original_key = f"{base}_lr_local_original"
        context_key = f"{base}_lr"
        context_original_key = f"{base}_lr_original"

        if local_key in sample:
            img, is_physical, source_key = _pick_display_array(local_key, local_original_key)
            if img is not None:
                panel_specs.append({
                    'kind': 'lr_local',
                    'base_name': base,
                    'source_key': source_key,
                    'img': img,
                    'is_physical': is_physical,
                    'cmap': lr_cmap_dict.get(base, default_lr_cmap) if lr_cmap_dict is not None else default_lr_cmap,
                    'lsm_key': 'lsm_hr' if 'lsm_hr' in sample else ('lsm' if 'lsm' in sample else None),
                    'show_ocean': True,
                })

        if context_key in sample:
            img, is_physical, source_key = _pick_display_array(context_key, context_original_key)
            if img is not None:
                same_as_local = False
                if local_key in sample:
                    try:
                        same_as_local = tuple(np.asarray(img).shape) == tuple(np.asarray(_to_2d(sample[local_key])).shape)
                    except Exception:
                        same_as_local = False
                if is_large_domain or (not same_as_local) or (local_key not in sample):
                    panel_specs.append({
                        'kind': 'lr_context',
                        'base_name': base,
                        'source_key': source_key,
                        'img': img,
                        'is_physical': is_physical,
                        'cmap': lr_cmap_dict.get(base, default_lr_cmap) if lr_cmap_dict is not None else default_lr_cmap,
                        'lsm_key': None,
                        'show_ocean': True,
                    })

    # Extras
    extra_keys = cfg.get('stationary_conditions', {}).get('geographic_conditions', {}).get('geo_variables', None)
    if extra_keys is not None:
        for extra_key in extra_keys:
            source_key = extra_key

            # Prefer local HR versions for quicklook, so these panels stay square
            # and match the HR / co-located domain.
            if extra_key == 'lsm' and 'lsm_hr' in sample and sample['lsm_hr'] is not None:
                source_key = 'lsm_hr'
            elif extra_key == 'topo' and 'topo_hr' in sample and sample['topo_hr'] is not None:
                source_key = 'topo_hr'

            if source_key in sample and sample[source_key] is not None:
                panel_specs.append({
                    'kind': 'extra',
                    'base_name': extra_key,
                    'source_key': source_key,
                    'img': _to_2d(sample[source_key]),
                    'is_physical': True,
                    'cmap': extra_cmap_dict.get(extra_key, 'viridis'),
                    'lsm_key': None,
                    'show_ocean': True,
                })

    if not panel_specs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.axis('off')
        ax.set_title('No sample panels available')
        return fig, np.array([ax])

    # Shared limits within comparable groups
    def _shared_limits(specs):
        arrays = []
        for spec in specs:
            arr = np.asarray(spec['img'], dtype=float)
            if not spec.get('show_ocean', True) and spec.get('lsm_key') == 'lsm_hr' and ('lsm_hr' in sample):
                m = _to_2d(sample['lsm_hr'])
                if m is not None and m.shape == arr.shape:
                    arr = np.where(m >= 0.5, arr, np.nan)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                arrays.append(finite)
        if not arrays:
            return None, None
        vals = np.concatenate(arrays)
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    physical_target_specs = [
        s for s in panel_specs
        if s['base_name'] == var and s['kind'] in {'hr', 'lr_local', 'lr_context'} and s['is_physical']
    ]
    scaled_target_specs = [
        s for s in panel_specs
        if s['base_name'] == var and s['kind'] in {'hr', 'lr_local', 'lr_context'} and not s['is_physical']
    ]
    physical_limits = _shared_limits(physical_target_specs)
    scaled_limits = _shared_limits(scaled_target_specs)

    n_panels = len(panel_specs)
    if n_panels <= 4:
        ncols = n_panels
    elif n_panels <= 6:
        ncols = 3
    else:
        ncols = 4
    nrows = int(np.ceil(n_panels / ncols))

    if figsize is None:
        figsize = (5.2 * ncols, 4.8 * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.subplots_adjust(wspace=0.5, hspace=0.45)
    fig.suptitle(f"Sample from train dataset, {var} (HR: {hr_model}, LR: {lr_model})", fontsize=16)

    flat_axs = axs.flatten()

    for ax, spec in zip(flat_axs, panel_specs):
        img = np.asarray(spec['img'], dtype=float)
        lsm_for_plot = None
        if spec['lsm_key'] is not None and spec['lsm_key'] in sample:
            lsm_for_plot = _to_2d(sample[spec['lsm_key']])
            if lsm_for_plot is not None and tuple(np.asarray(lsm_for_plot).shape) != tuple(np.asarray(img).shape):
                lsm_for_plot = None

        if not spec['show_ocean'] and lsm_for_plot is not None and lsm_for_plot.shape == img.shape:
            img = np.where(lsm_for_plot >= 0.5, img, np.nan)

        if spec['base_name'] == var and spec['kind'] in {'hr', 'lr_local', 'lr_context'}:
            if spec['is_physical'] and all(v is not None for v in physical_limits):
                vmin, vmax = physical_limits
            elif (not spec['is_physical']) and all(v is not None for v in scaled_limits):
                vmin, vmax = scaled_limits
            else:
                vmin, vmax = np.nanmin(img), np.nanmax(img)
        else:
            vmin, vmax = np.nanmin(img), np.nanmax(img)

        # For large-domain LR context panels, only draw outline if a matching full-domain
        # LSM actually exists. Otherwise do not fall back to a DK local outline.
        add_outline = (spec['base_name'] not in {'lsm', 'topo', 'sdf'})
        if spec['kind'] == 'lr_context':
            # Only draw a context-domain outline when we truly have a matching
            # full-domain/context LSM. Never fall back to a local DK outline here.
            if lsm_for_plot is None or tuple(np.asarray(lsm_for_plot).shape) != tuple(np.asarray(img).shape):
                add_outline = False

        im = imshow_variable(
            ax,
            img,
            variable=spec['base_name'],
            vmin=vmin,
            vmax=vmax,
            cmap=spec['cmap'],
            add_outline=add_outline,
            outline_linewidth=1.25,
            show_ocean=spec['show_ocean'],
            lsm_mask=lsm_for_plot,
            cfg=cfg,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(_panel_title(spec['kind'], spec['base_name'], spec['is_physical']), fontsize=10)

        _box_mask = None
        if spec['lsm_key'] is not None and spec['lsm_key'] in sample:
            _box_mask = sample[spec['lsm_key']]

        _add_colorbar_and_boxplot(
            fig,
            ax,
            im,
            img,
            boxplot=add_boxplot_per_panel and spec['kind'] in {'hr', 'lr_local', 'lr_context'},
            ylim=(vmin, vmax),
            boxplot_mask=_box_mask,
        )

    for ax in flat_axs[n_panels:]:
        ax.axis('off')

    return fig, axs



def _finite_flat(arr):
    """Return finite values flattened (NaNs masked out)"""
    if arr is None:
        return np.empty((0,), dtype=float)
    # Ensure NumPy array (avoid torch boolean indexing deprecation)
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    else:
        arr = np.asarray(arr)
    mask = np.isfinite(arr)
    return arr[mask].ravel()

def _add_colorbar_and_boxplot(fig, ax, im, img_data, *, boxplot=True, ylim=None, boxplot_mask=None):
    """
    Attach a boxplot (left) and a colorbar (right) to an image axis using axes_divider.
    The boxplot is vertical, minimal styling and hides ticks/frames.
    """
    divider = make_axes_locatable(ax)

    # order: [ax | boxplot | colorbar]
    bax = divider.append_axes("right", size="9%", pad=0.12) if boxplot else None
    cax = divider.append_axes("right", size="4.5%", pad=0.12)

    cb = fig.colorbar(im, cax=cax, orientation='vertical')

    # Put ticks/labels on the right only and keep them compact.
    cb.ax.yaxis.set_ticks_position("right")
    cb.ax.yaxis.set_label_position("right")
    cb.ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=True,
        labelleft=False,
        labelright=True,
        labelsize=8,
        pad=2,
    )

    # Choose a formatter based on data spread so we do not get visually duplicated
    # labels such as repeated "0.0" or cramped rounded values.
    try:
        finite = _finite_flat(img_data)
        if finite.size:
            data_min = float(np.nanmin(finite))
            data_max = float(np.nanmax(finite))
        else:
            data_min, data_max = 0.0, 1.0

        if ylim is not None:
            try:
                y0, y1 = float(ylim[0]), float(ylim[1])
                if np.isfinite([y0, y1]).all() and y1 > y0:
                    data_min, data_max = y0, y1
            except Exception:
                pass

        if np.isfinite([data_min, data_max]).all() and data_max > data_min:
            ticks = np.linspace(data_min, data_max, 4)
            cb.set_ticks(ticks)
            spread = data_max - data_min
            if spread < 0.1:
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            elif spread < 1.0:
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            else:
                cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            cb.set_ticks([data_min])
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        cb.ax.minorticks_off()
        cb.ax.yaxis.get_offset_text().set_visible(False)
    except Exception:
        pass

    if boxplot and bax is not None:
        # If a mask is provided, compute the boxplot only over masked (typically land-only) pixels.
        # Convention: mask >= 0.5 is land, < 0.5 is ocean.
        _bp_src = img_data
        if boxplot_mask is not None:
            try:
                m = boxplot_mask
                if torch.is_tensor(m):
                    m = m.detach().cpu().numpy()
                m = np.asarray(m).squeeze()

                src = np.asarray(_bp_src)
                if m.shape == src.shape:
                    _bp_src = np.where(m >= 0.5, _bp_src, np.nan)
            except Exception:
                pass

        vals = _finite_flat(_bp_src)
        if vals.size:
            vals = np.asarray(vals, dtype=float)
            bax.boxplot(
                vals,
                vert=True,
                widths=0.85,
                showmeans=True,
                meanprops=dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick'),
                flierprops=dict(marker='o', markerfacecolor='none', markersize=2, linestyle='None', markeredgecolor='darkgreen', alpha=0.4),
                medianprops=dict(linestyle='-', linewidth=2, color='black'),
            )
            if ylim is not None:
                try:
                    y0, y1 = float(ylim[0]), float(ylim[1])
                    if np.isfinite([y0, y1]).all() and y1 > y0:
                        bax.set_ylim(y0, y1)
                except Exception as e:
                    logger.warning(f"Could not set boxplot ylim {ylim}: {e}")

            # Cosmetic cleanup: boxplot should never contribute labels/ticks that can
            # visually collide with the colorbar tick labels.
            bax.set_xticks([])
            bax.set_yticks([])
            bax.tick_params(
                axis='both',
                which='both',
                left=False,
                right=False,
                labelleft=False,
                labelright=False,
                bottom=False,
                top=False,
                labelbottom=False,
            )
            bax.set_frame_on(False)
        else:
            bax.axis('off')


def plot_samples_and_generated(
        samples,
        generated,
        cfg,
        *,
        dates: Optional[List[str]] = None,
        transform_back_bf_plot=False,
        back_transforms=None,
        n_samples_threshold=5,
        figsize=(15, 15),
):
    """
    Like ``plot_samples`` but adds an extra left-most column with “Generated”
    images (one per sample).

    Parameters
    ----------
    samples : dict | list[dict]
        The usual batch/list accepted by ``plot_samples``.
    generated : torch.Tensor | np.ndarray | list
        Shape (B,1,H,W) or list/tuple of length B with 2-D arrays.
    transform_back_bf_plot : bool, default False
        Apply inverse scaling before display.
    back_transforms : dict[str, Callable], optional
        Mapping *plot-key* → inverse-transform function.  Only used when
        *transform_back_bf_plot* is ``True``.
    """
    # Extract configuration for plotting
    hr_model = cfg['highres']['model']
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    hr_units, lr_units = get_units(cfg)
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    default_lr_cmap = 'viridis'
    extra_cmap_dict = {"topo": "terrain", "lsm": "binary", "sdf": "coolwarm"}
    
    cfg_vis = cfg.get('visualization', {})
    show_ocean = cfg_vis.get('show_ocean', False)
    force_matching_scale = cfg_vis.get('force_matching_scale', True)
    global_min = cfg_vis.get('global_min', None)
    global_max = cfg_vis.get('global_max', None)
    extra_keys = cfg_vis.get('extra_keys', None)
    scaling = cfg_vis.get('scaling', True)
    add_boxplot_per_panel = bool(cfg_vis.get('add_boxplot_per_panel', True))
    add_boxplot_summary = bool(cfg_vis.get('add_boxplot_summary', False))
    summary_boxplot_keys = cfg_vis.get('summary_boxplot_keys', None)  # list of keys for summary boxplot column

    plot_dual_lr_channel = 0
    try: 
        plot_dual_lr_channel = int(cfg_vis.get('plot_dual_lr_channel', 0))
    except Exception:
        pass

    # ------------------------------------------------------------------ utils
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def maybe_inverse(k, arr, verbose=False):
        if transform_back_bf_plot and back_transforms and k in back_transforms:
            if verbose:
                logger.info(f"Applying inverse transformation for key: {k}")
                logger.info(f"Found inverse transformation for key: {k}")
            return back_transforms[k](arr)
        if verbose:
            if not transform_back_bf_plot:
                logger.info("transform_back_bf_plot is False, skipping inverse transform.")
            elif back_transforms is None:
                logger.info("No back_transforms provided, skipping inverse transform.")
            elif k not in back_transforms:
                logger.info(f"No inverse transformation found for key: {k}")
        return arr
    
    def _lr_base_name_from_key(k: str) -> str:
        # "prcp_lr" -> "prcp"; "msl_lr_original" -> "msl"
        if k.endswith("_lr_original"):
            return k[:-12]
        if k.endswith("_lr"):
            return k[:-3]
        return k

    def maybe_inverse_dual_lr(key: str, arr, prefer_channel: int = 0, verbose: bool = False):
        """
        Dual-LR safe inverse transform:
          - If arr is (2,H,W) we pick the plotted channel first and apply the correct inverse transform
            key for that channel if available.
          - For non-dual tensors, fall back to maybe_inverse(key, ...).
        """
        if not (transform_back_bf_plot and back_transforms):
            return arr

        # Only special-case LR keys (others use the regular path)
        if not (key.endswith("_lr") or key.endswith("_lr_original")):
            return maybe_inverse(key, arr, verbose=verbose)

        a = to_numpy(arr)

        # If this is a dual-LR stacked tensor, pick the channel first
        if a.ndim == 3 and a.shape[0] == 2:
            ch = int(prefer_channel)
            ch = 0 if ch not in (0, 1) else ch
            a2 = a[ch, :, :]
            base = _lr_base_name_from_key(key)

            # Determine which stats-space ch0 represents (config-dependent)
            dual_lr = bool(cfg.get("lowres", {}).get("dual_lr", False))
            lr_main_scale = str(cfg.get("lowres", {}).get("lr_main_var_scale", "LR")).upper()

            if dual_lr:
                # Convention: ch0 = "main", ch1 = "lr_only"
                # If main is HR-scaled -> prefer *_lr_hrspace when available
                if ch == 0 and lr_main_scale == "HR":
                    inv_key_candidates = [f"{base}_lr_hrspace", f"{base}_lrspace", f"{base}_lr", key]
                else:
                    inv_key_candidates = [f"{base}_lr_lrspace", f"{base}_lrspace", f"{base}_lr", key]
            else:
                inv_key_candidates = [f"{base}_lr", key]

            for kk in inv_key_candidates:
                if kk in back_transforms and callable(back_transforms[kk]):
                    if verbose:
                        logger.info(f"[plot] Applying inverse transform for '{key}' using '{kk}' (channel={ch}).")
                    return back_transforms[kk](a2)

            # Fallback: no inverse available for the chosen channel
            if verbose:
                logger.info(f"[plot] No inverse transform found for '{key}' (channel={ch}); returning scaled values.")
            return a2

        # Non-dual case: apply inverse directly if available
        return maybe_inverse(key, a, verbose=verbose)

    def _prep_for_limits(sample_dict, key):
        """Prep image like in plotting (inverse, mask, squeeze) for consistent vlim calc"""
        if key is None or key not in sample_dict or sample_dict[key] is None:
            return None
        arr = to_numpy(sample_dict[key]).squeeze()
        arr = _squeeze_geo_value(arr, key)
        arr = maybe_inverse_dual_lr(key, arr, prefer_channel=plot_dual_lr_channel)
        if not show_ocean and key in {gen_key, hr_key, f"{hr_key}_original"}:
            if "lsm_hr" in sample_dict and sample_dict["lsm_hr"] is not None:
                mask = to_numpy(sample_dict["lsm_hr"]).squeeze()
                arr = np.where(mask < 1, np.nan, arr)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(axis=0)
        return arr
    
    def _finite_minmax(arrs):
        """Compute global min/max over a list of arrays, ignoring NaNs."""
        vals = []
        for a in arrs:
            if a is None:
                continue
            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()
            else:
                a = np.asarray(a)
            af = a[np.isfinite(a)]
            if af.size:
                vals.append(af)
        if not vals:
            return None, None
        all_vals = np.concatenate(vals)
        return float(np.nanmin(all_vals)), float(np.nanmax(all_vals))


    # -------------------------------------------------------- unpack samples
    if isinstance(samples, dict):              # turn single batch-dict → list
        B = None
        for v in samples.values():
            if torch.is_tensor(v):
                B = v.shape[0]
                break
            if isinstance(v, list) and v and torch.is_tensor(v[0]):
                B = len(v)
                break
        if B is None:
            raise ValueError("Could not determine batch size (B) from samples dictionary.")
        sample_list = []
        for i in range(B):
            d = {}
            for k, v in samples.items():
                if torch.is_tensor(v):
                    d[k] = v[i]
                elif isinstance(v, (list, tuple)) and len(v) == B:
                    d[k] = v[i]
                else:
                    d[k] = v
            sample_list.append(d)
    else:
        sample_list = list(samples)

    sample_list = sample_list[:n_samples_threshold]

    # ------------------------------------------------------- generated batch
    # logger.info(f"Generated shape: {generated.shape}")
    gen_np = to_numpy(generated)
    if gen_np.ndim == 4:               # (B, 1, H, W), multiple samples with 1 channel
        gen_np = gen_np[:, 0, :, :]
    elif gen_np.ndim == 3:             # (B, H, W), multiple samples
        pass
    elif gen_np.ndim == 2:             # (H, W), single samples
        gen_np = np.expand_dims(gen_np, axis=0)
    else:
        raise ValueError(f"Unexpected shape for generated samples: {gen_np.shape}")

    gen_np = gen_np[:len(sample_list)]
    # logger.info(f"Generated shape after slicing: {gen_np.shape}")

    # inject into dicts
    gen_key = "generated"
    for d, im in zip(sample_list, gen_np):
        d[gen_key] = im

    # --------------------------------------------------- assemble key order
    hr_key = f"{var}_hr"
    lr_keys = sorted(k for k in sample_list[0] if k.endswith("_lr"))
    # Decide which LR key to use for matching (if any)
    matching_lr_key = f"{var}_lr" if f"{var}_lr" in lr_keys else None
    original_keys = [k + "_original"
                     for k in (hr_key, *lr_keys)
                     if k + "_original" in sample_list[0]]

    plot_keys = [gen_key, hr_key, *lr_keys, *original_keys]
    if extra_keys:
        plot_keys.extend(extra_keys)


    # -------------------------------------------------- Pooled colourlims (per sample)
    per_row_vlims = None
    if not (force_matching_scale and global_min is not None and global_max is not None):
        per_row_vlims = []
        for sd in sample_list:
            arrs = []
            arrs.append(_prep_for_limits(sd, gen_key))
            arrs.append(_prep_for_limits(sd, hr_key))
            if matching_lr_key is not None:
                arrs.append(_prep_for_limits(sd, matching_lr_key))
            vmin_row, vmax_row = _finite_minmax(arrs)
            # Fallback: if empty (all-NaN), compute from HR only
            if vmin_row is None or vmax_row is None:
                hr_only = _prep_for_limits(sd, hr_key)
                vmin_row, vmax_row = _finite_minmax([hr_only])
            per_row_vlims.append( (vmin_row, vmax_row) )

            
    # -------------------------------------------------------------- figure
    n_rows, n_cols = len(sample_list), len(plot_keys)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)

    # If requested, add a summary boxplot column, rebuild figure with +1 column to the right
    if add_boxplot_summary:
        plt.close(fig)
        n_rows, n_cols = len(sample_list), len(plot_keys) + 1
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        summary_col_idx = n_cols - 1
    else:
        summary_col_idx = None

    # Ensure axs is always 2D
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = axs[np.newaxis, :]
    if n_cols == 1:
        axs = axs[:, np.newaxis]

    fig.suptitle(f"Generated vs. conditions – {var} (HR {hr_model} / LR {lr_model}) ")

    for r, sample in enumerate(sample_list):
        # For the summary column, collect distributions here:
        summary_vals = [] # list of (label, values)
        # If user provided explicit keys for the summary boxplot, use those; else default gen + HR + (matching LR)

        if summary_boxplot_keys is not None:
            row_summary_keys = [k for k in summary_boxplot_keys if k in sample]
        else:
            row_summary_keys = [gen_key, hr_key]
            if matching_lr_key is not None:
                row_summary_keys.append(matching_lr_key)
            row_summary_keys = [k for k in row_summary_keys if k in sample]

        for c, key in enumerate(plot_keys):
            ax = axs[r, c]
            if key not in sample or sample[key] is None:
                ax.axis('off')
                continue
            # Add date if provided as y-axis label on first column
            if c == 0 and dates is not None and r < len(dates):
                date_str = str(dates[r])
                if date_str:
                    ax.set_ylabel(date_str, fontsize=10)


            # ========= Retrieve image data =========
            img_data = to_numpy(sample[key]).squeeze()
            img_data = _squeeze_geo_value(img_data, key)
            img_data = maybe_inverse_dual_lr(key, img_data, prefer_channel=plot_dual_lr_channel)

            # Decide per-panel ocean display:
            #   - HR + generated: respect cfg `show_ocean`
            #   - LR conditions: ALWAYS show full field (conditioning uses the whole field)
            is_lr_panel = bool(key.endswith("_lr") or key.endswith("_lr_original"))
            show_ocean_panel = True if is_lr_panel else bool(show_ocean)

            # For HR/gen images, optionally mask out ocean using lsm_hr
            if (not show_ocean_panel) and key in {gen_key, hr_key, f"{hr_key}_original"}:
                if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                    mask = to_numpy(sample["lsm_hr"]).squeeze()
                    img_data = np.where(mask < 1, np.nan, img_data)

            # cmap selection
            if key in {gen_key, hr_key, f"{hr_key}_original"}:
                cmap = hr_cmap
            elif key.endswith('_lr') or key.endswith('_lr_original'):
                base = key.replace('_lr', '').replace('_lr_original', '')
                cmap = (lr_cmap_dict or {}).get(base, default_lr_cmap)
            else:
                cmap = (extra_cmap_dict or {}).get(key, 'viridis')

            # vmin/vmax 
            if force_matching_scale and global_min is not None and global_max is not None:
                vmin = global_min.get(key, np.nanmin(img_data)) if isinstance(global_min, dict) else global_min
                vmax = global_max.get(key, np.nanmax(img_data)) if isinstance(global_max, dict) else global_max
            else:
                use_row_pool = (key == gen_key) or (key == hr_key) or (matching_lr_key is not None and key == matching_lr_key)
                if use_row_pool and per_row_vlims is not None:
                    vmin, vmax = per_row_vlims[r]
                    # If degenerate or non-finite, fallback to per-image
                    if (vmin is None) or (vmax is None) or (not np.isfinite([vmin, vmax]).all()):
                        vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                else:
                    vmin, vmax = np.nanmin(img_data), np.nanmax(img_data)
                

            # Ensure 2D
            if img_data.ndim == 3 and img_data.shape[0] == 1:
                img_data = img_data.squeeze(0)

            img2d, _ = _to_imshow_image(img_data, prefer_channel=plot_dual_lr_channel)
            lsm_for_plot = None
            if "lsm_hr" in sample and sample["lsm_hr"] is not None:
                lsm_for_plot = sample["lsm_hr"]
            elif "lsm" in sample and sample["lsm"] is not None:
                lsm_for_plot = sample["lsm"]

            if lsm_for_plot is not None and torch.is_tensor(lsm_for_plot):
                lsm_for_plot = lsm_for_plot.detach().cpu().numpy().squeeze()
            elif lsm_for_plot is not None:
                lsm_for_plot = np.asarray(lsm_for_plot).squeeze()

            im = imshow_variable(
                ax,
                img2d,
                variable=(var if key in {gen_key, hr_key, f"{hr_key}_original"} else key),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                add_outline=True,
                show_ocean=show_ocean_panel,
                lsm_mask=lsm_for_plot,
                cfg=cfg,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            # ========= If LR conditions, add LSM contour =========
            # Specifically NOT the HR lsm, if we change LR geographical domain
            if key.endswith("_lr") and "lsm" in sample and sample["lsm"] is not None and bool(cfg_vis.get("overlay_lsm_contour", True)):
                lsm_data = to_numpy(sample["lsm"]).squeeze()
                try:
                    ax.contour(np.array(lsm_data, copy=False), levels=[0.5], colors="darkgrey", linewidths=0.5)
                except Exception as e:
                    logger.warning(f"LSM contour failed on {key}: {e}")

            # ========= column headers (title logic) =========
            if r == 0:
                if scaling:
                    if transform_back_bf_plot and back_transforms and key in back_transforms:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nback-transformed [{hr_units}]",
                            **{k: f"LR {lr_model} ({k[:-3]})\nback-transformed [{lr_units[cfg['lowres']['condition_variables'].index(k[:-3])] if (k[:-3] in cfg.get('lowres', {}).get('condition_variables', [])) else 'unknown'}]" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\nscaled" for k in original_keys},
                        }
                    else:
                        titles = {
                            gen_key: "Generated",
                            hr_key: f"HR {hr_model}, {var}\nscaled",
                            **{k: f"LR {lr_model} ({k[:-3]})\nscaled" for k in lr_keys},
                            **{k: f"LR {lr_model} ({k[:-12]})\noriginal [{lr_units[cfg['lowres']['condition_variables'].index(k[:-12])] if (k[:-12] in cfg.get('lowres', {}).get('condition_variables', [])) else 'unknown'}]" for k in original_keys},
                        }
                else:
                    titles = {
                        gen_key: "Generated",
                        hr_key: f"HR {hr_model}, {var}\nno scaling [{hr_units}]",
                        **{k: f"LR {lr_model} ({k[:-3]})\nno scaling [{lr_units[lr_keys.index(k[:-3])] if k[:-3] in lr_keys else 'unknown'}]" for k in lr_keys},
                        **{k: f"LR {lr_model}" for k in lr_keys},
                    }


                ax.set_title(titles.get(key, key), fontsize=9)
            
            # ========= Add per-panel boxplot if requested next to colorbar =========
            if key.endswith(("generated", "_hr", "_lr", "_hr_original", "_lr_original")) and add_boxplot_per_panel:
                # LR panels: show full field, but use LAND-ONLY pixels in the boxplot when cfg show_ocean=False
                _bp_mask = None
                if (not bool(show_ocean)) and (key.endswith("_lr") or key.endswith("_lr_original")) and ("lsm_hr" in sample) and (sample["lsm_hr"] is not None):
                    _bp_mask = sample["lsm_hr"]

                _add_colorbar_and_boxplot(
                    fig,
                    ax,
                    im,
                    img2d,
                    boxplot=True,
                    ylim=(vmin, vmax),
                    boxplot_mask=_bp_mask,
                )
            else:
                # Still add a colorbar but no boxplot for non-variable maps / extras
                divide = make_axes_locatable(ax)
                cax = divide.append_axes("right", size="5%", pad=0.1)
                fig.colorbar(im, cax=cax, orientation='vertical')
            
            # ========= Collect for the summary boxplot if requested =========
            if add_boxplot_summary and key in row_summary_keys:
                vals = _finite_flat(img2d)
                if vals.size:
                    if key == gen_key:
                        label = hr_key.replace('_hr', ' gen')
                    elif key == hr_key:
                        label = hr_key.replace('_hr', ' hr')
                    elif key.endswith('_lr'):
                        label = key.replace('_lr', ' lr')
                    else:
                        label = key
                    summary_vals.append((label, vals))
            # End of column loop 

        # ========= Draw the summary column for this row, if requested ========
        if add_boxplot_summary and summary_col_idx is not None:
            axd = axs[r, summary_col_idx]
            axd.clear()
            if summary_vals:
                labels, data = zip(*summary_vals)
                axd.boxplot(data, vert=True, widths=0.7, showmeans=True,
                            meanprops=dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick'),
                            flierprops=dict(marker='o', markerfacecolor='none', markersize=2, linestyle='None', markeredgecolor='darkgreen', alpha=0.35),
                            medianprops=dict(linestyle='-', linewidth=1.2, color='black'),
                )
                
                axd.tick_params(axis='y', labelsize=8)
                # Only add x-ticks  on last row
                if r == n_rows - 1:
                    axd.set_xticks(range(1, len(labels) + 1))
                    axd.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
                else:
                    axd.set_xticks([])
                # Only add title on first row
                if r == 0:
                    axd.set_title("Pixel distribution summary", fontsize=9)
                axd.set_frame_on(False)
                    
            else:
                axd.axis('off')
    fig.text(0.5, 0.01, f"Dual-LR plotting: channel {plot_dual_lr_channel} (if applicable) (ch0~HR z-space, ch1~LR z-space)", ha='center', fontsize=8, va='bottom', color='gray')
    # Tighten layout
    fig.tight_layout()

    return fig, axs

def plot_training_monitor_generated(
        sample,
        generated_members,
        cfg,
        *,
        date: Optional[str] = None,
        transform_back_bf_plot: bool = False,
        back_transforms=None,
        figsize=(16, 4.8),
):
    """
    Lightweight training-monitor plot for one case:
      [local LR | HR truth | ensemble member 1 | ... | ensemble member N | summary]

    The summary panel shows min / mean / max over the ensemble and a small text block
    with extrema for LR, HR and ensemble aggregate.
    """
    hr_model = cfg['highres']['model']
    lr_model = cfg['lowres']['model']
    var = cfg['highres']['variable']
    hr_units, lr_units = get_units(cfg)
    hr_cmap, lr_cmap_dict = get_cmaps(cfg)
    cfg_vis = cfg.get('visualization', {})
    show_ocean = bool(cfg_vis.get('show_ocean', False))
    plot_dual_lr_channel = int(cfg_vis.get('plot_dual_lr_channel', 0) or 0)

    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _lr_base_name_from_key(k: str) -> str:
        if k.endswith('_lr_original'):
            return k[:-12]
        if k.endswith('_lr'):
            return k[:-3]
        return k

    def maybe_inverse(k, arr):
        if transform_back_bf_plot and back_transforms and k in back_transforms:
            return back_transforms[k](arr)
        return arr

    def maybe_inverse_dual_lr(key: str, arr, prefer_channel: int = 0):
        if not (transform_back_bf_plot and back_transforms):
            return arr
        if not (key.endswith('_lr') or key.endswith('_lr_original')):
            return maybe_inverse(key, arr)

        a = to_numpy(arr)
        if a.ndim == 3 and a.shape[0] == 2:
            ch = 0 if prefer_channel not in (0, 1) else int(prefer_channel)
            a2 = a[ch, :, :]
            base = _lr_base_name_from_key(key)
            dual_lr = bool(cfg.get('lowres', {}).get('dual_lr', False))
            lr_main_scale = str(cfg.get('lowres', {}).get('lr_main_var_scale', 'LR')).upper()

            if dual_lr:
                if ch == 0 and lr_main_scale == 'HR':
                    inv_key_candidates = [f"{base}_lr_hrspace", f"{base}_lrspace", f"{base}_lr", key]
                else:
                    inv_key_candidates = [f"{base}_lr_lrspace", f"{base}_lrspace", f"{base}_lr", key]
            else:
                inv_key_candidates = [f"{base}_lr", key]

            for kk in inv_key_candidates:
                if kk in back_transforms and callable(back_transforms[kk]):
                    return back_transforms[kk](a2)
            return a2

        return maybe_inverse(key, a)

    def _apply_land_mask(arr, mask):
        if mask is None:
            return arr
        m = to_numpy(mask).squeeze()
        a = np.asarray(arr)
        if a.shape == m.shape:
            return np.where(m >= 0.5, a, np.nan)
        return a

    def _finite_minmax(arrs):
        vals = []
        for a in arrs:
            if a is None:
                continue
            a = np.asarray(a)
            af = a[np.isfinite(a)]
            if af.size:
                vals.append(af)
        if not vals:
            return None, None
        all_vals = np.concatenate(vals)
        return float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    def _stats_text(name, arr):
        a = np.asarray(arr)
        af = a[np.isfinite(a)]
        if af.size == 0:
            return f"{name}: no finite values"
        return (
            f"{name}:\n"
            f"  min={np.nanmin(af):.3f}\n"
            f"  mean={np.nanmean(af):.3f}\n"
            f"  max={np.nanmax(af):.3f}"
        )

    hr_key = f"{var}_hr"
    lr_key = f"{var}_lr_local" if f"{var}_lr_local" in sample else f"{var}_lr"

    lr_arr = to_numpy(sample[lr_key]).squeeze()
    hr_arr = to_numpy(sample[hr_key]).squeeze()
    lr_arr = _squeeze_geo_value(lr_arr, lr_key)
    hr_arr = _squeeze_geo_value(hr_arr, hr_key)
    lr_arr = maybe_inverse_dual_lr(lr_key, lr_arr, prefer_channel=plot_dual_lr_channel)
    hr_arr = maybe_inverse(hr_key, hr_arr)

    gen_np = to_numpy(generated_members)
    if gen_np.ndim == 4 and gen_np.shape[1] == 1:
        gen_np = gen_np[:, 0, :, :]
    elif gen_np.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected generated_members shape: {gen_np.shape}")

    gen_np = np.asarray([maybe_inverse('generated', g) for g in gen_np])

    lsm_mask = None
    if 'lsm_hr' in sample and sample['lsm_hr'] is not None:
        lsm_mask = sample['lsm_hr']
    elif 'lsm' in sample and sample['lsm'] is not None:
        lsm_mask = sample['lsm']

    if not show_ocean:
        hr_arr = _apply_land_mask(hr_arr, lsm_mask)
        gen_np = np.asarray([_apply_land_mask(g, lsm_mask) for g in gen_np])

    ens_mean = np.nanmean(gen_np, axis=0)
    ens_min = np.nanmin(gen_np, axis=0)
    ens_max = np.nanmax(gen_np, axis=0)

    vmin, vmax = _finite_minmax([lr_arr, hr_arr, ens_mean, ens_min, ens_max, *list(gen_np)])
    if vmin is None or vmax is None:
        vmin, vmax = np.nanmin(hr_arr), np.nanmax(hr_arr)

    n_members = int(gen_np.shape[0])
    n_cols = 2 + n_members + 1
    fig, axs = plt.subplots(1, n_cols, figsize=figsize)
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([axs])

    if date is not None:
        fig.suptitle(f"Training monitor – {var} – {date}")
    else:
        fig.suptitle(f"Training monitor – {var}")

    panels = [
        ('Local LR', lr_arr, (lr_cmap_dict or {}).get(var, 'viridis')),
        (f'HR {hr_model}', hr_arr, hr_cmap),
    ]
    for i in range(n_members):
        panels.append((f'Gen m{i+1}', gen_np[i], hr_cmap))

    for ax, (title, arr, cmap) in zip(axs[:-1], panels):
        img2d, _ = _to_imshow_image(arr, prefer_channel=plot_dual_lr_channel)
        lsm_for_plot = None
        if lsm_mask is not None:
            lsm_for_plot = to_numpy(lsm_mask).squeeze()
        is_lr_panel = title.startswith('Local LR')
        im = imshow_variable(
            ax,
            img2d,
            variable=(lr_key if is_lr_panel else var),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            add_outline=not is_lr_panel,
            show_ocean=(True if is_lr_panel else show_ocean),
            lsm_mask=lsm_for_plot,
            cfg=cfg,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.08)
        fig.colorbar(im, cax=cax, orientation='vertical')

    ax_sum = axs[-1]
    summary_img = ens_mean
    img2d, _ = _to_imshow_image(summary_img, prefer_channel=plot_dual_lr_channel)
    lsm_for_plot = to_numpy(lsm_mask).squeeze() if lsm_mask is not None else None
    im = imshow_variable(
        ax_sum,
        img2d,
        variable=var,
        vmin=vmin,
        vmax=vmax,
        cmap=hr_cmap,
        add_outline=True,
        show_ocean=show_ocean,
        lsm_mask=lsm_for_plot,
        cfg=cfg,
    )
    ax_sum.set_xticks([])
    ax_sum.set_yticks([])
    ax_sum.set_title('Ensemble mean', fontsize=9)
    divider = make_axes_locatable(ax_sum)
    cax = divider.append_axes('right', size='5%', pad=0.08)
    fig.colorbar(im, cax=cax, orientation='vertical')

    stats_lines = [
        _stats_text('LR', lr_arr),
        _stats_text('HR', hr_arr),
        _stats_text('Ens mean', ens_mean),
        _stats_text('Ens min', ens_min),
        _stats_text('Ens max', ens_max),
    ]
    ax_sum.text(
        1.22, 0.5,
        "\n\n".join(stats_lines),
        transform=ax_sum.transAxes,
        va='center',
        ha='left',
        fontsize=8,
        family='monospace',
    )

    fig.tight_layout()
    return fig, axs

# ===============================
# Metrics plotting helpers
# ===============================

def _ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def _safe_savefig(fig, save_dir: str, filename: str, dpi=300):
    _ensure_dir(save_dir)
    full = os.path.join(save_dir, filename)
    fig.savefig(full, dpi=dpi, bbox_inches='tight')
    logger.info(f"[plot] Saved figure to {full}")

def plot_live_training_metrics(
        steps: List[int],
        edm_cosine: List[float],
        hr_lr_corr: List[float],
        *,
        save_dir: str,
        n_samples: Optional[int] = None,
        filename: str = "live_metrics.png",
        show: bool = False,
        title: str | None = None,
        land_only: bool = False
):
    """
        Line plots for lightweight in-loop metrics collected over steps.
    """
    title = title or "Live Training Metrics"
    if land_only:
        title += " (land only)"

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    fig, ax = plt.subplots(figsize=(8, 5))
    steps_np = np.asarray(steps, dtype=float)
    if len(steps_np) == 0:
        logger.warning("No steps provided for live training metrics plot.")
        return
    
    def _maybe_plot(y, label): 
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        if ok.any():
            ax.plot(steps_np[ok], y[ok], label=label, lw=2)

    _maybe_plot(edm_cosine, "EDM Cosine Similarity")
    _maybe_plot(hr_lr_corr, "HR-LR Correlation")

    ax.set_xlabel("Global step")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)


# ------------------------------
# FSS at multiple spatial scales
# ------------------------------
def plot_fss_epoch(
    fss: Dict[str, float],
    *,
    save_dir: str,
    filename: str = "fss_epoch.png",
    title: str = "FSS at scales",
    show: bool = False,
):
    """
    Bar plot for a single-epoch FSS dictionary, e.g. {'5km': 0.7, '10km': 0.8, ...}
    """
    if not fss:
        logger.warning("[plot] plot_fss_epoch: empty dict; skipping.")
        return
    # Sort by numeric km if possible
    def _km_key(k):
        try:
            return float(k.replace("km", ""))
        except Exception:
            return float("inf")
    keys = sorted(fss.keys(), key=_km_key)
    vals = [fss[k] for k in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(keys, vals)
    ax.set_ylim(0, 1)
    ax.set_ylabel("FSS")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

def plot_fss_history(
    fss_hist: List[Dict[str, float]],
    epoch_list: Optional[List[int]] = None,
    *,
    save_dir: str,
    n_samples: Optional[int] = None,
    filename: str = "fss_history.png",
    title: str = "FSS over epochs",
    show: bool = False,
):
    """
    Line plot over epochs. Each scale gets its own line.
    fss_hist: list of dicts per epoch, e.g. [{'5km':..,'10km':..}, {...}, ...]
    """
    if not fss_hist:
        logger.warning("[plot] plot_fss_history: empty history; skipping.")
        return
    # Collect all scales
    scales = sorted({k for d in fss_hist for k in d.keys()},
                    key=lambda k: float(k.replace("km", "")) if "km" in k else float("inf"))

    if epoch_list is not None and len(epoch_list) == len(fss_hist):
        epochs = np.asarray(epoch_list, dtype=float)
    else:
        epochs = np.arange(1, len(fss_hist) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for s in scales:
        y = [d.get(s, np.nan) for d in fss_hist]
        y = np.asarray(y, dtype=float)
        ok = np.isfinite(y)
        if ok.any():
            ax.plot(epochs[ok], y[ok], label=s, lw=2)

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    ax.set_xlabel("Epoch")
    ax.set_ylabel("FSS")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Scale")
    ax.set_title(title)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

# ------------------------------
# PSD slope (β) comparisons
# ------------------------------
def plot_psd_slope_epoch(
    psd: Dict[str, float],
    *,
    save_dir: str,
    filename: str = "psd_slope_epoch.png",
    title: str = "PSD slope (log–log)",
    show: bool = False,
):
    """
    Bar plot comparing gen vs HR slopes (if available) with delta text.
    Expected keys: 'psd_slope_gen', optionally 'psd_slope_hr' and 'psd_slope_delta'
    """
    gen = psd.get("psd_slope_gen", np.nan)
    hr = psd.get("psd_slope_hr", np.nan)
    has_hr = np.isfinite(hr)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    labels = ["Gen"] + (["HR"] if has_hr else [])
    vals = [gen] + ([hr] if has_hr else [])
    ax.bar(labels, vals, color=["#4c72b0", "#55a868"][:len(labels)])
    ax.set_ylabel("Slope β")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)

    # annotate delta if both present
    if has_hr and np.isfinite(gen):
        delta = psd.get("psd_slope_delta", float(hr - gen))
        ax.text(0.5, max(vals) + 0.02, f"Δ (HR–Gen) ≈ {delta:.3f}",
                ha="center", va="bottom", transform=ax.get_xaxis_transform())

    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

def plot_psd_slope_history(
    psd_hist: List[Dict[str, float]],
    epoch_list: Optional[List[int]] = None,
    *,
    save_dir: str,
    n_samples: Optional[int] = None,
    filename: str = "psd_slope_history.png",
    title: str = "PSD slope over epochs",
    show: bool = False,
):
    """
    Line plots of β_gen and β_hr over epochs (if HR available), plus Δ on a secondary axis.
    """
    if not psd_hist:
        logger.warning("[plot] plot_psd_slope_history: empty history; skipping.")
        return

    if epoch_list is not None and len(epoch_list) == len(psd_hist):
        epochs = np.asarray(epoch_list, dtype=float)
    else:
        epochs = np.arange(1, len(psd_hist) + 1, dtype=float)
    gen = np.array([d.get("psd_slope_gen", np.nan) for d in psd_hist], dtype=float)
    hr  = np.array([d.get("psd_slope_hr", np.nan) for d in psd_hist], dtype=float)
    delta = np.array([d.get("psd_slope_delta", np.nan) for d in psd_hist], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ln1 = ax1.plot(epochs, gen, label="β_gen", lw=2)
    ln2 = []
    if np.isfinite(hr).any():
        ln2 = ax1.plot(epochs, hr, label="β_hr", lw=2)

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Slope β")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)

    ax2 = None
    if np.isfinite(delta).any():
        ax2 = ax1.twinx()
        ln3 = ax2.plot(epochs, delta, "--", label="Δ(HR–Gen)", lw=2)
        ax2.set_ylabel("Δ β")
        lines = ln1 + ln2 + ln3
    else:
        lines = ln1 + ln2

    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="best")

    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

# -----------------------------------------
# Quantiles & wet-day frequency comparisons
# -----------------------------------------
def plot_quantiles_wetday_epoch(
    q: Dict[str, float],
    *,
    save_dir: str,
    filename: str = "quantiles_wetday_epoch.png",
    title: str = "Quantiles and wet-day frequency",
    show: bool = False,
):
    """
    Grouped bar chart for P95, P99, wet-day freq (gen vs HR if available).
    Expected keys: 'gen_p95','gen_p99','gen_wet_freq' and optionally 'hr_*'
    """
    keys = [("p95", "gen_p95", "hr_p95"),
            ("p99", "gen_p99", "hr_p99"),
            ("wetfreq", "gen_wet_freq", "hr_wet_freq")]
    labels = []
    gen_vals, hr_vals = [], []
    for lab, gk, hk in keys:
        labels.append(lab.upper())
        gen_vals.append(q.get(gk, np.nan))
        hr_vals.append(q.get(hk, np.nan))

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, gen_vals, width, label="Gen")
    if np.isfinite(hr_vals).any():
        ax.bar(x + width/2, hr_vals, width, label="HR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)

def plot_quantiles_wetday_history(
    q_hist: List[Dict[str, float]],
    epoch_list: Optional[List[int]] = None,
    *,
    save_dir: str,
    n_samples: Optional[int] = None,
    filename: str = "quantiles_wetday_history.png",
    title: str = "P95/P99/Wet-day over epochs",
    show: bool = False,
):
    """
    Line plots for P95/P99/wet-day across epochs (gen and HR where available).
    """
    if not q_hist:
        logger.warning("[plot] plot_quantiles_wetday_history: empty history; skipping.")
        return

    if epoch_list is not None and len(epoch_list) == len(q_hist):
        epochs = np.asarray(epoch_list, dtype=float)
    else:
        epochs = np.arange(1, len(q_hist) + 1, dtype=float)

    def _series(gk, hk):
        g = np.array([d.get(gk, np.nan) for d in q_hist], dtype=float)
        h = np.array([d.get(hk, np.nan) for d in q_hist], dtype=float)
        return g, h

    series = [
        ("P95", "gen_p95", "hr_p95"),
        ("P99", "gen_p99", "hr_p99"),
        ("Wet-day freq", "gen_wet_freq", "hr_wet_freq"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, (name, gk, hk) in zip(axes, series):
        g, h = _series(gk, hk)
        okg = np.isfinite(g)
        if okg.any():
            ax.plot(epochs[okg], g[okg], label="Gen", lw=2)
        okh = np.isfinite(h)
        if okh.any():
            ax.plot(epochs[okh], h[okh], label="HR", lw=2)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        if name == "Wet-day freq":
            ax.set_ylim(0, 1)

    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Epoch")
    axes[2].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="best")

    if n_samples is not None:
        title = f"{title} (n={n_samples} samples)"

    fig.suptitle(title)
    fig.tight_layout()
    _safe_savefig(fig, save_dir, filename)
    if show:
        plt.show()
    plt.close(fig)


