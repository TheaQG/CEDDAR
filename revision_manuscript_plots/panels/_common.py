"""Small axes-only helpers. Figure scripts own layout, legends, colourbars and saving."""
import numpy as np
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
        cmap=style.PRECIP_CMAP if cmap is None else cmap,
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

    if title:
        ax.set_title(title)

    if label:
        style.panel_label(ax, label)

    return image
