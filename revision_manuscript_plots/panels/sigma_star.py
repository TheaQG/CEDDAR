"""One explicitly selected sigma* run per panel; metadata stays with its bundle."""
from ._common import np, style, number, column, unique, finish


def psd(ax, data, values=None, *, label=None):
    saved = data['arrays']['sigma_psd_curves']
    k, sigmas = np.asarray(saved['k']), np.asarray(saved['sigma_vals'])
    values = sigmas if values is None else np.asarray(values, dtype=float)
    colors, result = style.sigma_colors(values), {}
    for method, key in [('danra', 'psd_hr_mean'), ('era5_bilinear', 'psd_lr_mean')]:
        power = np.asarray(saved[key])
        valid = (k > 0) & np.isfinite(power) & (power > 0)
        ax.plot(1/k[valid], power[valid], **style.method_style(method, marker=''))
    for value in values:
        idx = np.flatnonzero(np.isclose(sigmas, value, rtol=0, atol=1e-8))
        if len(idx) != 1:
            raise ValueError(f'Expected one saved sigma*={value:g}')
        power = np.asarray(saved['psd_gen_mean'])[idx[0]]
        valid = (k > 0) & np.isfinite(power) & (power > 0)
        ax.plot(1/k[valid], power[valid], color=colors[float(value)], label=rf'CEDDAR ($\sigma^*={value:.2f}$)')
        result[float(value)] = dict(k=k, power=power)
    if 'lr_nyquist' in saved and float(saved['lr_nyquist']) > 0:
        ax.axvline(1/float(saved['lr_nyquist']), **style.REFERENCE_LINE)
    ax.set(xscale='log', yscale='log', xlabel='Wavelength (km)', ylabel='Spectral power')
    if not ax.xaxis_inverted():
        ax.invert_xaxis()
    finish(ax, 'PSD of ensemble mean', label)
    return result


def metric(ax, data, key='crps', *, errorbars='std', label=None):
    """Equal-weight daily means; std or nominal std/sqrt(n), never ensemble spread.

    SEM assumes independent dates. It is descriptive, not an autocorrelation-aware
    confidence interval. No bars where fewer than two finite dates are available.
    """
    titles = dict(r_lp='Low-pass PMM-LR correlation', slope_gen='PSD slope of ensemble mean',
                  crps='Ensemble CRPS (mm/day)', hk_gain='High-k power ratio')
    if key not in titles or errorbars not in ('std', 'sem', None):
        raise ValueError('Unknown metric or errorbars; use std, sem or None')
    rows = data['tables']['metrics_by_sigma']
    if not rows:
        raise ValueError('No daily sigma* metrics')
    unique(rows, ('date', 'sigma_star'))
    sigmas = sorted({number(r, 'sigma_star') for r in rows})
    result = {}
    for value in sigmas:
        selected = [r for r in rows if number(r, 'sigma_star') == value]
        y = column(selected, key)
        finite = np.isfinite(y)
        y = y[finite]
        std = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
        result[value] = dict(mean=float(np.mean(y)) if len(y) else np.nan, std=std,
                             sem=std/np.sqrt(len(y)) if len(y) else np.nan,
                             n=len(y), dates=[r['date'] for r, ok in zip(selected, finite) if ok])
    means = [result[s]['mean'] for s in sigmas]
    ax.plot(sigmas, means, **style.method_style('ceddar_members'))
    if errorbars:
        errors = np.array([result[s][errorbars] for s in sigmas])
        valid = np.isfinite(errors)
        ax.errorbar(np.asarray(sigmas)[valid], np.asarray(means)[valid], yerr=errors[valid],
                    fmt='none', color=style.CEDDAR, capsize=2)
    if key == 'hk_gain':
        ax.axhline(1, **style.REFERENCE_LINE)
    if key == 'slope_gen':
        refs = [number(r, 'slope_hr') for r in rows if number(r, 'sigma_star') == sigmas[0]]
        if np.any(np.isfinite(refs)):
            ax.axhline(np.nanmean(refs), label='DANRA', **style.REFERENCE_LINE)
    ax.set(xlabel=r'$\sigma^*$', ylabel=titles[key])
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    finish(ax, titles[key], label)
    return result
