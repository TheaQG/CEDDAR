"""Small axes-only helpers. Figure scripts own layout, legends, colourbars and saving."""
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import style
from sbgm.plotting_utils import (imshow_variable, _add_colorbar_and_boxplot)

SEASONS = ('DJF', 'MAM', 'JJA', 'SON')
METHODS = ('era5_bilinear', 'qm', 'ceddar_mean', 'ceddar_median', 'ceddar_pmm')


def number(row, key):
    value = row[key]
    return float(value) if value is not None and value != '' else np.nan


def column(rows, key):
    return np.array([number(r, key) for r in rows])


def unique(rows, keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate rows for {keys}; select one subset/season first')
    return result


def finish(ax, title=None, label=None):
    style.style_axes(ax)
    if title:
        ax.set_title(title, loc='left')
    if label:
        style.panel_label(ax, label)


def boxes(ax, groups, methods, horizontal=False):
    """All finite values and outliers retained; boxes describe variation, not a CI."""
    for i, (values, method) in enumerate(zip(groups, methods), 1):
        values = np.asarray(values, dtype=float)
        ax.boxplot(values[np.isfinite(values)], positions=[i], widths=.55,
                   vert=not horizontal, patch_artist=True, showfliers=True,
                   **style.boxplot_style(style.method_color(method)))


def map_field(
    ax,
    values,
    *,
    land=None,
    norm=None,
    vmin=None,
    vmax=None,
    cmap=None,
    variable="prcp",
    show_ocean=True,
    add_outline=True,
    add_colorbar=False,
    add_boxplot=False,
    outline_color="darkgrey",
    outline_linewidth=0.65,
    title=None,
    label=None,
):
    """Plot one manuscript spatial field using the legacy CEDDAR map identity.

    The displayed field is not land-masked by default. The land-sea mask is
    instead used for the Danish coastline outline and, where requested, for a
    land-only summary boxplot.

    Parameters
    ----------
    add_colorbar
        Attach a vertical colorbar using the legacy axes-divider layout.
    add_boxplot
        Attach the legacy land-only boxplot between the map and colorbar.
        Requires add_colorbar=True.
    """

    values = np.asarray(values, dtype=float).squeeze()

    if values.ndim != 2:
        raise ValueError(f"A map requires one 2D field, got {values.shape}")

    if land is not None:
        land = np.asarray(land).squeeze()

        if land.shape != values.shape:
            raise ValueError(f"Land mask and field shapes differ: {land.shape} versus {values.shape}")

    # Current manuscript code sometimes passes a plain Normalize.
    # The legacy precipitation helper constructs its own zero-aware
    # normalization, so only retain its explicit limits.
    if norm is not None:
        if vmin is None:
            vmin = norm.vmin
        if vmax is None:
            vmax = norm.vmax

    image = imshow_variable(
        ax,
        values,
        variable=variable,
        vmin=vmin,
        vmax=vmax,
        cmap=style.PRECIP_CMAP if cmap is None else cmap,  # type: ignore[arg-type]
        add_outline=add_outline,
        outline_color=outline_color,
        outline_linewidth=outline_linewidth,
        show_ocean=show_ocean,
        lsm_mask=land,
        precip_zero_color="#ffffff",
    )

    if add_colorbar:
        _add_colorbar_and_boxplot(
            ax.figure,
            ax,
            image,
            values,
            boxplot=add_boxplot,
            ylim=(
                (vmin, vmax)
                if vmin is not None and vmax is not None
                else None
            ),
            boxplot_mask=land,
        )
    elif add_boxplot:
        _add_boxplot_only(
            ax,
            values,
            land=land,
            ylim=(
                (vmin, vmax)
                if vmin is not None and vmax is not None
                else None
            ),
        )

    if title:
        ax.set_title(title)

    if label:
        style.panel_label(ax, label)

    return image


def _add_boxplot_only(ax, values, *, land=None, ylim=None,):
    """Attach the legacy land-only boxplot without adding a colorbar"""

    values = np.asarray(values, dtype=float)

    if land is not None:
        land = np.asarray(land).squeeze()

        if land.shape == values.shape:
            values = np.where(land >= 0.5, values, np.nan)

    values = values[np.isfinite(values)]

    divider = make_axes_locatable(ax)

    bax = divider.append_axes("right", size="8%", pad=0.025,)

    if values.size:
        bax.boxplot(
            values,
            vert=True,
            widths=0.9,
            showmeans=True,
            meanprops=dict(marker="x", markerfacecolor="firebrick", markeredgecolor="firebrick", markersize=4,),
            flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="darkgreen", markersize=1.8, linestyle="none", alpha=0.35,),
            medianprops=dict(linestyle="-", linewidth=1.4, color="black",)
        )

        if ylim is not None:
            bax.set_ylim(*ylim)

        bax.set_xticks([])
        bax.set_yticks([])
        bax.set_frame_on(False)

    else:
        bax.axis("off")