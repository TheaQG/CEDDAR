"""Own-wet-mask diagnostics. Member summaries preserve one statistic per member."""
from ._common import np, style, METHODS, SEASONS, number, column, unique, finish, map_field


def occurrence(ax, data, *, label=None):
    rows = [r for r in data['tables']['seasonal_decomposition'] if r['season'] == 'ALL']
    lookup = unique(rows, ('method',))
    members = [r for r in data['tables']['ensemble_member_decomposition'] if r['season'] == 'ALL']
    methods = ('danra', 'era5_bilinear', 'qm', 'ceddar_members', *METHODS[2:])
    result = {}

    for i, method in enumerate(methods):
        values = column(members, 'wet_frequency') if method == 'ceddar_members' else np.array([number(lookup[method,], 'wet_frequency')])
        values = values[np.isfinite(values)]
        result[method] = values

        if values.size:
            lo, mid, hi = np.quantile(values, [0.25, 0.5, 0.75])
            ax.plot([i], [mid], **style.method_style(method, linestyle='none'))

            if method == 'ceddar_members':
                ax.scatter(i+np.linspace(-.12, .12, len(values)), values, color=style.CEDDAR, s=8, alpha=0.3)
                ax.vlines(i, lo, hi, color=style.CEDDAR, linewidth=3)

    ax.set_xticks(range(len(methods)), [style.method_label(m) for m in methods], rotation=35, ha='right')
    ax.set_ylabel('Wet pixel-day frequency')
    finish(ax, 'Precipitation occurrence', label)

    return result


def curves(ax, data, *, metric=None, quantiles=False, label=None):
    if quantiles:
        rows = data['tables']['conditional_intensity']
        members = data['tables']['ensemble_member_conditional_intensity']

        ticks, keys = ('P50', 'P90', 'P99'), ('p50', 'p90', 'p99')
        lookup = unique(rows, ('method',))
    else:
        rows = data['tables']['seasonal_decomposition']
        members = data['tables']['ensemble_member_decomposition']

        ticks = SEASONS
        keys = SEASONS
        lookup = unique(rows, ('method','season'))

    x, result = np.arange(len(ticks)), {}

    for method in ('danra', *METHODS):
        y = [number(lookup[method, ], k) for k in keys] if quantiles else [number(lookup[method, s], metric) for s in SEASONS]
        ax.plot(x, y, **style.method_style(method))
        result[method] = y

    if quantiles:
        unique(members, ('member',))
        values = np.array([[number(r, k) for k in keys] for r in members])
    else:
        lookup = unique(members, ('member', 'season'))
        ids = sorted({r['member'] for r in members})
        values = np.array([[number(lookup[m, s], metric) for s in SEASONS] for m in ids])

    if not values.size:
        raise ValueError("No saved statistics found for ensemble members.")

    lo, mid, hi = np.quantile(values, [0.25, 0.5, 0.75], axis=0)
    ax.fill_between(x, lo, hi, color=style.CEDDAR, alpha=0.18)
    ax.plot(x, mid, **style.method_style('ceddar_members'))
    ax.set_xticks(x, ticks)
    ax.set_ylabel('Wet pixel-day frequency' if metric == 'wet_frequency' else r'Precipitation conditional on wet (mm day$^{-1}$)')

    finish(ax, "Wet-pixel quantiles" if quantiles else 'seasonal '+(metric or '').replace('_', ' '), label)

    result['ceddar_members'] = dict(values=values, median=mid, q25=lo, q75=hi)
    return result


def conditional_intensity(ax, data, *, label=None):
    return curves(ax, data, quantiles=True, label=label)


def seasonal(ax, data, metric='wet_frequency', *, label=None):
    if metric not in ('wet_frequency', 'conditional_mean_wet'):
        raise ValueError(f"Unsupported metric: {metric}. Choose 'wet_frequency' or 'conditional_mean_wet'.")
    return curves(ax, data, metric=metric, label=label)


def accumulation(ax, data, source, group='2019', *, reference=None, **kwargs):
    """Saved sum map, or ratio to a supplied 2D reference. Zero denominators masked
    Source is explicit NPZ tag, e.g. hr, lr, ensmean. Missing maps raise."""

    field = np.asarray(data['arrays'][f'spatial_{source}_{group}']['sum'], dtype=float).squeeze()

    if reference is not None:
        reference = np.asarray(reference)

        if reference.shape != field.shape:
            raise ValueError("Reference and field must have the same shape.")


        field = np.divide(field, reference, out=np.full_like(field, np.nan), where=reference > 0)

    return map_field(ax, field, **kwargs)