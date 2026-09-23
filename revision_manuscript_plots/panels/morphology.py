"""Object and SAL panels reuse the established revision aggregation helpers."""

from ._common import np, style, number, finish, boxes

METHODS = ('era5_bilinear', 'qm', 'ceddar_members')

def objects(ax, data, analysis='absolute', metric='object_count_absolute_error', *, label=None):
    from revision_evaluation.plot_morphology import date_summaries, common_dates

    if analysis not in ('absolute', 'equal_area') or metric not in ('object_count_absolute_error', 'largest_object_fraction'):
        raise ValueError(f"Unsupported analysis or metric: analysis={analysis}, metric={metric}. Choose analysis in ('absolute', 'equal_area') and metric in ('object_count_absolute_error', 'largest_object_fraction').")

    rows = date_summaries(data['tables'][f'objects_{analysis}'], [metric], analysis, paired=True)

    thresholds = sorted({r['reference_threshold'] for r in rows})
    cohorts = {q: common_dates([r for r in rows if r['reference_threshold'] == q]) for q in thresholds}
    count = metric == 'object_count_absolute_error'

    summary = {}

    for method in METHODS:
        mid, low, high = [], [], []
        for q in thresholds:
            group = [r for r in rows if r['method'] == method and r['reference_threshold'] == q and r['date'] in cohorts[q]]

            mid.append(float((np.mean if count else np.median)([r['value'] for r in group])) if group else np.nan)
            low.append(float(np.median([r['member_q25'] for r in group])) if group else np.nan)
            high.append(float(np.median([r['member_q75'] for r in group])) if group else np.nan)

        ax.plot(thresholds, mid, **style.method_style(method))

        if method == 'ceddar_members' and not count:
            ax.fill_between(thresholds, low, high, color=style.CEDDAR, alpha=.18)

        summary[method] = dict(values=mid, lower=low, upper=high)

    ax.axhline(0, **style.REFERENCE_LINE)
    ax.set_xticks(thresholds, [f'{q:g}\nn={len(cohorts[q])}' for q in thresholds])
    ax.set(xlabel=r'DANRA reference threshold (mm day$^{-1}$)', ylabel='Mean avsolute object-count error' if count else 'Largest-object fraction - DANRA')

    if count:
        ax.set_ylim(bottom=0)

    finish(ax, 'Absolute thresholds' if analysis == 'absolute' else 'Equal-area control', label)
    return dict(summary=summary, dates=cohorts, daily=rows)


def sal(ax, data, component='S', threshold=1., *, label=None):
    from revision_evaluation.plot_morphology import date_summaries, common_dates

    titles = dict(S='Structure', L='Location', A='Amplitude')
    if component not in titles:
        raise ValueError(f"Unsupported component: {component}. Choose from 'S', 'L', 'A'.")

    rows = [r for r in data['tables']['sal_absolute'] if number(r, 'reference_threshold') == threshold]
    if not rows:
        raise ValueError(f"No SAL rows at threshold {threshold}.")

    daily = date_summaries(rows, [component], 'sal')
    dates = common_dates(daily)
    groups = [[r['value'] for r in daily if r['method'] == m and r['date'] in dates] for m in METHODS]

    boxes(ax, groups, METHODS)
    ax.set_xticks([1, 2, 3], ['Bilinear\nERA5', 'QM', 'CEDDAR\nmembers'])
    ax.axhline(0, **style.REFERENCE_LINE)
    ax.set_ylim((-.03, 2.05) if component == 'L' else(-2.05, 2.05))
    finish(ax, f'{titles[component]} ({component}); n={len(dates)}', label)

    return dict(dates=dates, values=dict(zip(METHODS, groups)), daily=daily, threshold=threshold)