"""
Shared visual identity for the revised CEDDAR GMD manuscript.

Usage
-----
At the start of each manuscript figure script:
    from gmd_revision_manuscript_plots.style import (
        apply_style,
        method_style,
        panel_label,
        PRECIP_CMAP,
    )

    apply_style()

Then:
    ax.plot(x, y, **method_style("qm"))
    panel_label(ax, "(a)")

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# ================================================================================
# Core manuscript palette
# ================================================================================

DANRA = "#4b4b4b"
ERA5 = "#997938"
QM = "#ab513b"

CEDDAR = "#35B19F"
CEDDAR_DARK = "#288C7D"

UNET = "#0a4714"
NEUTRAL = "#7570b3"

BLACK = "#202020"
AXIS_GREY = "#3f3f3f"
GRID_GREY = "#b8b8b8"
LIGHT_GREY = "#d9d9d9"
VERY_LIGHT_GREY = "#f3f3f3"
WHITE = "#ffffff"


# ================================================================================
# Precipitation and bias colormaps
# ================================================================================

PRECIP_COLORS = [
    "#ffffff",
    "#c9f3df",
    "#66c29a",
    "#2b8c67",
    "#26534A",
]

PRECIP_CMAP = LinearSegmentedColormap.from_list(
    "precip_white_tealgray",
    PRECIP_COLORS,
    N=256,
)

# Dry/under-threshold pixels
PRECIP_CMAP.set_under("#c2c2c2") 


BIAS_COLORS = [
    "#8c510a", # negative: muted brown
    "#ffffff", # zero: white
    "#2b8c85", # positive: teal-gray
]

BIAS_CMAP = LinearSegmentedColormap.from_list(
    "bias_brown_white_tealgray",
    BIAS_COLORS,
    N=256
)


# ================================================================================
# Seasonal identity
# ================================================================================

SEASON_FACE = {
    "DJF": (0.35, 0.55, 0.85, 0.20),
    "MAM": (0.60, 0.80, 0.60, 0.20),
    "JJA": (0.95, 0.85, 0.40, 0.22),
    "SON": (0.95, 0.70, 0.60, 0.20),
}


# ================================================================================
# Method identity
# ================================================================================
# CEDDAR mean, median, and ensemble members belong to the same model family - 
# therefore same teal identity, but distinguished by marker/linestyle

METHOD_STYLES = {
    "danra": {
        "label": "DANRA",
        "color": DANRA,
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.5,
        "markersize": 4.5,
        "zorder": 20,
    },
    "era5_bilinear": {
        "label": "Bilinear ERA5",
                "color": ERA5,
                "marker": "o",
                "linestyle": "--",
                "linewidth": 1.25,
                "markersize": 4.2,
                "zorder": 10,
    },
    "qm": {
        "label": "QM",
                "color": QM,
                "marker": "s",
                "linestyle": ":",
                "linewidth": 1.35,
                "markersize": 4.3,
                "zorder": 12,
    },
    "ceddar_members": {
        "label": "CEDDAR members",
                "color": CEDDAR,
                "marker": "o",
                "linestyle": "-",
                "linewidth": 1.5,
                "markersize": 4.5,
                "zorder": 25,
    },
    "ceddar_mean": {
        "label": "CEDDAR mean",
                "color": CEDDAR,
                "marker": "D",
                "linestyle": "-",
                "linewidth": 1.45,
                "markersize": 4.5,
                "zorder": 24,
    },
    "ceddar_median": {
        "label": "CEDDAR median",
                "color": CEDDAR,
                "marker": "^",
                "linestyle": "--",
                "linewidth": 1.35,
                "markersize": 4.7,
                "zorder": 23,
    },
    "ceddar_pmm": {
        "label": "CEDDAR PMM",
        "color": CEDDAR_DARK,
        "marker": "v",
        "linestyle": "-.",
        "linewidth": 1.4,
        "markersize": 4.7,
        "zorder": 22,
    },
    "unet": {
        "label": "U-Net",
        "color": UNET,
        "marker": "P",
        "linestyle": "-",
        "linewidth": 1.25,
        "markersize": 4.2,
        "zorder": 8,
    },
}

# Aliases:
METHOD_ALIASES = {
    # DANRA
    "danra": "danra",
    "hr": "danra",
    "hr_danra": "danra",

    # ERA5
    "era5": "era5_bilinear",
    "lr": "era5_bilinear",
    "bilinear": "era5_bilinear",
    "bilinear_era5": "era5_bilinear",
    "era5_bilinear": "era5_bilinear",

    # QM
    "qm": "qm",
    "quantile_mapping": "qm",

    #CEDDAR ensemble
    "ensemble": "ceddar_members",
    "members": "ceddar_members",
    "ceddar": "ceddar_members",
    "gen": "ceddar_members",
    "generated": "ceddar_members",

    # Deterministic summaries
    "ceddar_mean": "ceddar_mean",
    "ensemble_mean": "ceddar_mean",

    "ceddar_median": "ceddar_median",
    "ensemble_median": "ceddar_median",

    "ceddar_pmm": "ceddar_pmm",
    "ensemble_pmm": "ceddar_pmm",
    "pmm": "ceddar_pmm",

    # Legacy U-net baseline
    "unet": "unet",
    "unet_sr": "unet",
}


def canonical_method(name:str) -> str:
    """Map legacy/revision method labels to manuscript canonical names."""
    key = str(name).strip().lower()
    key = key.replace(" ", "_")
    return METHOD_ALIASES.get(key, key)


def method_style(
        method: str,
        *,
        label: bool = True,
        **overrides,
) -> dict: 
    """Return plot-ready style kwargs for a method. To suppress automatic legend label set label=False."""
    method = canonical_method(method)

    if method not in METHOD_STYLES:
        raise KeyError(
            f"Unknown manuscript method: '{method}'."
            f"Available methods are: {sorted(METHOD_STYLES)}"
        )

    style = METHOD_STYLES[method].copy()
    if not label:
        style.pop("label", None)
    style.update(overrides)

    return style


def method_color(method: str) -> str:
    """Return only canonical manuscript color for a method."""
    return method_style(method)["color"]


def method_label(method: str) -> str:
    """Return only canonical manuscript legend label."""
    return method_style(method)["label"]


# ================================================================================
# CEDDAR ensemble-member uncertainty/variability styling
# ================================================================================

MEMBER_BAND = {
    "color": CEDDAR,
    "alpha": 0.18,
    "linewidth": 0,
    "zorder": 5,
}

MEMBER_POINTS = {
    "color": CEDDAR,
    "alpha": 0.32,
    "s": 12,
    "edgecolors": "none",
    "zorder": 5,
}

MEMBER_MEDIAN = {
    "color": CEDDAR,
    "marker": "D",
    "markersize": 5.0,
    "linestyle": "-",
    "linewidth": 1.5,
    "zorder": 25,
}


# ================================================================================
# Reference-line styles
# ================================================================================

REFERENCE_LINE = {
    "color": "#666666",
    "linestyle": "--",
    "linewidth": 0.9,
    "alpha": 0.8,
    "zorder": 1,
}

PERCENTILE_LINE = {
    "color": "#4d4d4d",
    "linestyle": "--",
    "linewidth": 0.65,
    "alpha": 0.55,
    "zorder": 1,
}


# ================================================================================
# Figure sizes
# ================================================================================

FIGSIZE_SINGLE = (3.6, 3.0)
FIGSIZE_WIDE = (7.2, 3.8)
FIGSIZE_2X2 = (7.2, 6.0)
FIGSIZE_3PANEL = (7.2, 2.8)
FIGSIZE_FULL = (7.2, 7.0)


# ================================================================================
# Global Matplotlib style
# ================================================================================

RC_PARAMS = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": [
        "DejaVu Sans",
        "Arial",
        "Liberation Sans",
    ],
    "font.size": 9.0,
    
    # Axes
    "axes.titlesize": 9.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.0,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": AXIS_GREY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    
    # Grid: dotted/light style
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_GREY,
    "grid.linestyle": ":",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.50,
    
    # Tick labels
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "xtick.color": AXIS_GREY,
    "ytick.color": AXIS_GREY,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    
    # Lines
    "lines.linewidth": 1.3,
    "lines.markersize": 4.5,

    # Legends
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "legend.handlelength": 2.1,
    "legend.handletextpad": 0.6,
    "legend.columnspacing": 1.2,

    # Figure
    "figure.facecolor": "white",
    "figure.dpi": 100,

    # Saving
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    
    # PDFs: editable text, not Type-3 fonts
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Math
    "mathtext.default": "regular",

    # Thin hatching
    "hatch.linewidth": 0.3,
}


def apply_style() -> None:
    """Apply CEDDAR manuscript style globally for current script"""
    mpl.rcParams.update(RC_PARAMS)


# ================================================================================
# Axis helpers
# ================================================================================

def style_axes(
        ax,
        *,
        grid: str | bool = "both",
) -> None:
    """Apply manuscript axis styling. "both", "x", "y", or False for grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_axisbelow(True)

    if grid is False:
        ax.grid(False)
    elif grid in ("x", "y"):
        ax.grid(
            True,
            axis=grid,
            linestyle=":",
            alpha=RC_PARAMS["grid.alpha"],
            linewidth=RC_PARAMS["grid.linewidth"],
        )
    else:
        ax.grid(
            True,
            linestyle=":",
            alpha=RC_PARAMS["grid.alpha"],
            linewidth=RC_PARAMS["grid.linewidth"],
        )


def style_map_axes(ax) -> None:
    """Minimal styling for spatial/map panels"""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def set_season_background(
        ax,
        season: str,
) -> None:
    """Apply low-opacity seasonal panel background"""
    season = season.upper()

    if season in SEASON_FACE:
        ax.set_facecolor(SEASON_FACE[season])


def panel_label(
        ax,
        label: str,
        *,
        x: float = -0.10,
        y: float = 1.04,
        fontsize: float = 10.5,
) -> None:
    """ Place consistent panel label outside upper-left corner. Example: panel_label(ax, "(a)")"""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )


def zero_line(
        ax,
        *,
        axis: str = "y",
        value: float = 0.0,
        **kwargs,
) -> None:
    """Add zero-reference line"""
    style_kwargs = REFERENCE_LINE.copy()
    style_kwargs.update(kwargs)

    if axis == "x":
        ax.axvline(value, **style_kwargs)
    elif axis == "y":
        ax.axhline(value, **style_kwargs)


def one_to_one(
        ax,
        *,
        low: float = 0.0,
        high: float = 1.0,
        **kwargs,
) -> None:
    """Add a 1:1 calibration/reference line"""
    style = REFERENCE_LINE.copy()
    style.update(kwargs)

    ax.plot([low, high], [low, high], **style)


# ================================================================================
# Boxplot styling
# ================================================================================

def boxplot_style(
        color: str,
        *,
        fill_alpha: float = 0.28,
) -> dict:
    """ Consistent boxplot styling. Usage: ax.boxplot(data, patch_artist=True, **boxplot_style(CEDDAR))"""
    rgba = mpl.colors.to_rgba(color, alpha=fill_alpha) # type: ignore

    return {
        "boxprops": {
            "facecolor": rgba,
            "edgecolor": color,
            "linewidth": 0.9,
        },
        "medianprops": {
            "color": BLACK,
            "linewidth": 1.2,
        },
        "whiskerprops": {
            "color": color,
            "linewidth": 0.9, 
        },
        "capprops": {
            "color": color,
            "linewidth": 0.9
        },
        "flierprops": {
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": color,
            "markeredgewidth": 0.65,
            "markersize": 2.5,
            "alpha": 0.50,
        },
    }


# ================================================================================
# Legend helpers
# ================================================================================

def unique_legend(
        axes: Iterable,
        *,
        order: list[str] | None = None,
):
    """Collect unique legend entires across multiple axes"""

    by_label = {}

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label and not label.startswith("_"):
                by_label.setdefault(label, handle)

    if order is None:
        labels = list(by_label)
    else:
        labels = [label for label in order if label in by_label]
        labels.extend(label for label in by_label if label not in labels)

    handles = [by_label[label] for label in labels]
    return handles, labels


def method_legend_handle(
    method: str,
) -> Line2D:
    """Create clean legend handle from canonical method style"""

    style = method_style(method)

    return Line2D(
        [0],
        [0],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        markersize=style["markersize"],
        label=style["label"],
    )


# ================================================================================
# Sigma* paletter
# ================================================================================

def sigma_colors(
        values,
        *,
        dark_for_low_sigma: bool=True,
) -> dict[float, tuple]:
    """Return green shades for sigma* sweeps (Matplotlib Greens colormap)"""

    values = sorted(float(value) for value in values)

    n = len(values)

    if not n:
        return {}

    cmap = mpl.colormaps["Greens"] # type: ignore

    colors = {}

    for i, value in enumerate(values):
        fraction = (i / max(1, n-1))

        if dark_for_low_sigma:
            position = 0.90 - 0.65 * fraction
        else:
            position = 0.25 + 0.65 * fraction

        colors[value] = cmap(position)

    return colors


# ================================================================================
# Output helper
# ================================================================================

def save_figure(
        fig,
        path: str | Path,
        *,
        dpi: int = 300,
        also_png: bool = True,
) -> None:
    """Save publication-quality PDF and optional HR PNG."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stem = (path.with_suffix("") if path.suffix else path)

    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",)

    if also_png:
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white",)