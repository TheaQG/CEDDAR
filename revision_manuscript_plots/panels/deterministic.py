"""Daily paired errors and pooled event detection from revision.load_deterministic"""
from ._common import np, style, METHODS, number, unique, boxes, finish

def daily_errors(ax, data, metric='mae', *, methods=METHODS[1:], label=None):

    if metric not in ['mae', 'rmse']:
        raise ValueError(f'Unsupported metric: {metric}. Choose from "mae" or "rmse"')

    rows = [r for r in data['tables']['daily_continuous_metrics'] if r['subset'] == 'all_land']
    lookup = unique(rows, ('date', 'method'))

    dates = sorted(r['date'] for r in rows if r['method'] == 'era5_bilinear')
    if not dates:
        raise ValueError('No dates found for era5_bilinear reference')

    result, groups, labels = {}, [], []

    for method in methods:
        if {r['date'] for r in rows if r['method'] == method} != set(dates):
            raise ValueError(f'{method}: dates differ from bilinear reference')

        delta = []
        for date in dates:
            pred, ref = lookup[date, method], lookup[date, 'era5_bilinear']

            if number(pred, 'n_pixels') != number(ref, 'n_pixels'):
                raise ValueError(f'{method}/{date}: pixel support differs')

            # Compute the difference between the predicted and reference metric
            delta.append(number(pred, metric) - number(ref, metric))

        delta = np.asarray(delta)
        delta = delta[np.isfinite(delta)]

        fraction = float(np.mean(delta < 0) if delta.size else np.nan)

        result[method] = {"values": delta, "fraction_lower": fraction, "n": len(delta)}

        groups.append(delta)
        labels.append(f'{style.method_label(method)}\n{100*fraction:.f}% lower; n={len(delta)}')

    boxes(ax, groups, methods, horizontal=True,)
    ax.set_yticks(range(1, len(methods)+1), labels)
    ax.invert_yaxis()
    ax.axvline(0, **style.REFERENCE_LINE)
    ax.set_xlabel(r'Method - bilinear ERA5 mm day$^{-1}$')

    finish(ax, f'Daily {metric.upper()} difference', label)

    return result


def event_detection(ax, data, metric='pod', *, methods=METHODS, label=None):
    titles = {
        'pod': 'Probability of detection',
        'far': 'False-alarm ratio',
        'csi': 'Critical success index',
    }

    if metric not in titles:
        raise ValueError(f'Choose from {tuple(titles)}')

    rows = [
        r for r in data['tables']['event_detection_metrics']
        if r['season'] == 'ALL' and r['subset'] == 'all_land'
    ]

    unique(rows, ('method', 'threshold'))

    result = {}

    for method in methods:
        selected = sorted([r for r in rows if r['method'] == method], key=lambda r: number(r, 'threshold'))

        if not selected:
            raise ValueError(f'No data (detection rows) found for method: {method}')

        result[method] = selected

        ax.plot([number(r, 'threshold') for r in selected], [number(r, metric) for r in selected], **style.method_style(method))
    ax.set(ylim=(0,1), xlabel=r'Threshold (mm day$^{-1}$)', ylabel=titles[metric])

    finish(ax, titles[metric], label)

    return result