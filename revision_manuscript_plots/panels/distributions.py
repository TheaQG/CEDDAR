"""Legacy distributions, spectra, and saved physical examples. No reevaluation"""

from ._common import np, style, number, unique, finish, map_field


def example(ax, data, date, method='danra', *, member=None, reference=None, **kwargs):
    """Draw one field or difference. Member is zero-based. Pass shared norm/extent"""

    day = data['dates'][str(date)]
    field = day['fields'][method] if member is None else day['ensemble'][member]

    if reference is not None:
        field = field - day['fields'][reference]
    kwargs.setdefault('land', day['land'])

    return map_field(ax, field, **kwargs)


def seasonal(ax, data, season, *, baselines=None, label=None):
    """Saved daily histograms. GEN is PMM, not member ensemble. 
    Baselines maps method names to separately loaded legacy distribution bundles.
    Densities integrate to one over the saved histogram range, excluding its tails.
    """

    series = [('danra', data, 'counts_hr'), ('era5_bilinear', data, 'counts_lr'), ('ceddar_pmm', data, 'counts_gen')]
    series += [(method, bundle, 'counts_gen') for method, bundle in (baselines or {}).items()]

    result = {}
    dates = data['arrays']['dist_daily']['dates']

    for method, bundle, key in series:
        daily = bundle['arrays']['dist_daily']

        if not np.array_equal(dates, daily['dates']):
            raise ValueError(f"{method}: histogram dates differ; align before comparison")

        indices = bundle['season_indices'][season]

        if not len(indices):
            raise ValueError(f"No dates for {season} in {method}")

        bins = np.asarray(daily['bins'])
        counts = np.asarray(daily[key])[indices].sum(axis=0)
        density = counts/counts.sum()/np.diff(bins) if counts.sum() else np.full(counts.shape, np.nan)

        ax.plot((bins[1:] + bins[:-1])/2, np.where(density > 0, density, np.nan), **style.method_style(method, marker=''))

        result[method] = dict(bins=bins, density=density, dates=dates[indices])

    ax.set(yscale='log', xlabel=r'Precipitation mm day$^{-1}$', ylabel='Probability density')

    style.set_season_background(ax, season)
    finish(ax, season, label)

    return result


def psd(ax, data, *, baselines=None, label=None):
    """PMM and mean member PSDs keep distinct labels. No ensemble-mean substitution"""

    arrays = data['arrays']['scale_psd_curves']
    series = [('danra', arrays, 'psd_hr'), ('era5_bilinear', arrays, 'psd_lr'), ('ceddar_pmm', arrays, 'psd_gen')]

    if 'psd_gen_ens_mean' in arrays:
        series.append(('ceddar_members', arrays, 'psd_gen_ens_mean'))

    series += [(m, d['arrays']['scale_psd_curves'], 'psd_gen') for m, d in (baselines or {}).items()]

    result = {}
    for method, saved, key in series:
        k, power = np.asarray(saved['k']), np.asarray(saved[key])
        power = np.nanmean(power, axis=0) if power.ndim == 2 else power
        valid = (k > 0) & np.isfinite(power) & (power > 0)

        opts = style.method_style(method, marker='')

        if method == 'ceddar_members':
            opts['label'] = 'CEDDAR members (mean PSD)'

        ax.plot(1/k[valid], power[valid], **opts)

        result[method] = dict(k=k, power=power)

    ax.set(xscale='log', yscale='log', xlabel='Wavelength (km)', ylabel='Spectral power')
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    finish(ax, 'Isotropic power spectral density', label)

    return result
    

def tails(ax, data, metrics=('P95', 'P99', 'P99.9', 'P99.99'), *, baselines=None, label=None):
    """Grouped saved tail statistics."""

    lookup = unique(data['tables']['ext_tails'], ('which',))

    series = [('danra', lookup['HR',]), ('era5_bilinear', lookup['LR',]), ('ceddar_pmm', lookup['GEN',])]

    if ('GEN_ENS',) in lookup:
        series.append(('ceddar_members', lookup['GEN_ENS',]))

    for method, bundle in (baselines or {}).items():
        series.append((method, unique(bundle['tables']['ext_tails'], ('which',))['GEN',]))

    width, x, result = .8 / len(series), np.arange(len(metrics)), {}

    for i, (method, row) in enumerate(series):
        values = [number(row, key) for key in metrics]
        ax.bar(x+(i-(len(series)-1)/2)*width, values, width, color=style.method_color(method), label=style.method_label(method))
        result[method] = values

    ax.set_xticks(x, metrics)
    ax.set_ylabel(r'Precipitation (mm day$^{-1}$)' if all(m.startswith('P') for m in metrics) else 'Frequency / rate')
    finish(ax, 'Tail statistics', label)

    return result
