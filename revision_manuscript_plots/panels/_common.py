"""Small axes-only helpers. Figure scripts own layout, legends, colourbars and saving."""
import numpy as np
from .. import style

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


def map_field(ax, values, *, land=None, norm=None, vmin=None, vmax=None,
              cmap=None, origin='lower', extent=None, title=None, label=None):
    """Return the image for a caller-owned shared colourbar; no invented coordinates.

    Use the same norm across comparable panels. Without extent the coordinates are
    grid indices. Caller must select the saved field's correct orientation.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError('A map requires one 2D field')
    if land is not None:
        land = np.asarray(land, dtype=bool)
        if land.shape != values.shape:
            raise ValueError('Land mask and field shapes differ')
        values = np.where(land, values, np.nan)
    image = ax.imshow(np.ma.masked_invalid(values), origin=origin, extent=extent,
                      cmap=style.PRECIP_CMAP if cmap is None else cmap,
                      norm=norm, vmin=vmin, vmax=vmax)
    style.style_map_axes(ax)
    if title:
        ax.set_title(title)
    if label:
        style.panel_label(ax, label)
    return image
