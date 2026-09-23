"""Calibration panels; legacy PIT/daily scatter and revision summaries are explicit."""
from ._common import np, style, SEASONS, number, column, unique, boxes, finish


def crps(ax, data, subset='all_land', *, label=None):
    if subset not in ('all_land', 'observed_wet'):
        raise ValueError('Choose all_land or observed_wet')
    rows = [r for r in data['tables']['crps'] if r['subset'] == subset]
    unique(rows, ('date',))
    seasons = ('ALL', *SEASONS)
    groups = [column([r for r in rows if s == 'ALL' or r['season'] == s], 'crps') for s in seasons]
    groups = [g[np.isfinite(g)] for g in groups]
    boxes(ax, groups, ['ceddar_members']*len(groups))
    ax.set_xticks(range(1, 6), [f'{s}\nn={len(g)}' for s, g in zip(seasons, groups)])
    ax.set(ylabel='Daily ensemble CRPS (mm/day)', ylim=(0, None))
    finish(ax, 'All land' if subset == 'all_land' else 'Observed wet land', label)
    return dict(zip(seasons, groups))


def reliability(ax, data, threshold=1., *, source='revision', count_ax=None, label=None):
    if source == 'revision':
        rows = [r for r in data['tables']['reliability'] if r['subset'] == 'all_land' and number(r, 'threshold') == threshold]
        xkey, ykey, nkey = 'mean_forecast_probability', 'observed_frequency', 'n_cases'
    elif source == 'legacy':
        rows = data['tables'][f'prob_reliability_{threshold:.1f}mm']
        xkey, ykey, nkey = 'prob_pred', 'freq_obs', 'count'
    else:
        raise ValueError('source must be revision or legacy')
    if not rows:
        raise ValueError(f'No reliability rows for threshold {threshold}')
    x, y, n = column(rows, xkey), column(rows, ykey), column(rows, nkey)
    valid = (n > 0) & np.isfinite(x) & np.isfinite(y)
    ax.plot(x[valid], y[valid], **style.method_style('ceddar_members'))
    ax.plot([0, 1], [0, 1], **style.REFERENCE_LINE)
    if count_ax is not None:
        centers = column(rows, 'bin_center') if source == 'legacy' else (column(rows, 'bin_left')+column(rows, 'bin_right'))/2
        count_ax.bar(centers, n, width=.7/len(rows), color=style.CEDDAR, alpha=.15)
        count_ax.set_ylabel('Bin count')
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Forecast probability', ylabel='Observed frequency')
    finish(ax, f'Reliability: ≥{threshold:g} mm/day', label)
    return rows


def pit(ax, data, bins=20, *, label=None):
    """Legacy continuous PIT; no independent-pixel confidence band is assumed."""
    values = np.asarray(data['arrays']['prob_pit_values']['pit']).ravel()
    values = values[np.isfinite(values)]
    if not len(values) or np.any((values < 0) | (values > 1)):
        raise ValueError('PIT needs finite values within [0, 1]')
    ax.hist(values, bins=np.linspace(0, 1, bins+1), density=True, color=style.CEDDAR, rwidth=.95)
    ax.axhline(1, **style.REFERENCE_LINE)
    ax.set(xlim=(0, 1), xlabel='PIT', ylabel='Density')
    finish(ax, f'PIT; n={len(values):,}', label)
    return values


def ranks(ax, data, *, label=None):
    rows = sorted([r for r in data['tables']['rank_histogram'] if r['subset'] == 'all_land'], key=lambda r: number(r, 'rank'))
    unique(rows, ('rank',))
    if not rows:
        raise ValueError('No all-land rank histogram')
    ax.bar(column(rows, 'rank'), column(rows, 'frequency'), color=style.CEDDAR)
    ax.axhline(1/len(rows), **style.REFERENCE_LINE)
    ax.set(xlabel='Observation rank (0 … M)', ylabel='Relative frequency')
    finish(ax, 'Randomized rank histogram', label)
    return rows


def spread_skill(ax, data, *, source='revision', label=None):
    if source == 'revision':
        rows = sorted([r for r in data['tables']['spread_skill'] if r['subset'] == 'all_land'], key=lambda r: number(r, 'bin'))
        x, y = column(rows, 'mean_spread'), column(rows, 'rmse')
        ax.plot(x, y, **style.method_style('ceddar_members'))
        title, ylabel = 'Binned spread-skill', 'Ensemble-mean RMSE (mm/day)'
    elif source == 'legacy':
        rows = data['tables']['prob_spread_skill']
        x, y = column(rows, 'spread_mean'), column(rows, 'skill_mean')
        ax.scatter(x, y, color=style.CEDDAR, s=10, alpha=.5)
        title, ylabel = 'Daily spread-skill', 'Saved daily skill (mm/day)'
    else:
        raise ValueError('source must be revision or legacy')
    finite = np.r_[x[np.isfinite(x)], y[np.isfinite(y)]]
    if finite.size:
        ax.plot([0, finite.max()], [0, finite.max()], **style.REFERENCE_LINE)
    ax.set(xlabel='Ensemble spread (mm/day)', ylabel=ylabel, xlim=(0, None), ylim=(0, None))
    finish(ax, title, label)
    return rows


def coverage(ax, data, *, label=None):
    rows = sorted([r for r in data['tables']['coverage_summary']
                   if r['subset'] == 'all_land' and r['season'] == 'ALL' and r['metric'] == 'empirical_coverage'],
                  key=lambda r: number(r, 'nominal_coverage'))
    if not rows:
        raise ValueError('No all-land coverage summary')
    x = column(rows, 'nominal_coverage')
    ax.plot(x, column(rows, 'median'), **style.method_style('ceddar_members'))
    ax.fill_between(x, column(rows, 'p25'), column(rows, 'p75'), color=style.CEDDAR, alpha=.18)
    ax.plot([0, 1], [0, 1], **style.REFERENCE_LINE)
    ax.set(xlabel='Nominal central coverage', ylabel='Daily coverage: median and IQR', xlim=(0, 1), ylim=(0, 1))
    finish(ax, 'Ensemble interval coverage', label)
    return rows
