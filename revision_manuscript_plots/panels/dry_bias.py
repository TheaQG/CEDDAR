"""Own-wet-mask diagnostics. Member summaries preserve one statistic per member."""
from ._common import np, style, METHODS, SEASONS, number, column, unique, finish, map_field


def occurrence(ax, data, *, label=None):
    """Plot overall wet-pixel frequency.

    Deterministic methods are shown as bars. For the CEDDAR ensemble,
    the bar height is the median across the 32 members and the error bar
    shows the inter-member IQR (25th--75th percentile).
    """

    rows = [r for r in data["tables"]["seasonal_decomposition"] if r["season"] == "ALL" ]
    lookup = unique(rows, ("method",))

    members = [r for r in data["tables"]["ensemble_member_decomposition"] if r["season"] == "ALL"]

    methods = (
        "danra",
        "era5_bilinear",
        "qm",
        "ceddar_members",
        *METHODS[2:],
    )

    labels = {
        "danra": "DANRA",
        "era5_bilinear": "Bilinear\nERA5",
        "qm": "QM",
        "ceddar_members": "CEDDAR\nmembers",
        "ceddar_mean": "CEDDAR\nmean",
        "ceddar_median": "CEDDAR\nmedian",
        "ceddar_pmm": "CEDDAR\nPMM",
    }

    result = {}

    x = np.arange(len(methods))

    danra_frequency = number(lookup["danra",], "wet_frequency",)

    # ------------------------------------------------------------------
    # Deterministic methods
    # ------------------------------------------------------------------

    for i, method in enumerate(methods):
        if method == "ceddar_members":
            continue

        value = number(lookup[method,],"wet_frequency",)
        values = np.array([value], dtype=float,)
        values = values[np.isfinite(values)]

        result[method] = values

        if not values.size:
            continue

        ax.bar(
            i,
            values[0],
            width=0.68,
            color=style.method_color(method),
            alpha=0.90,
            zorder=5,
        )

    # ------------------------------------------------------------------
    # CEDDAR ensemble members: median bar + IQR error bar
    # ------------------------------------------------------------------

    member_values = column(members, "wet_frequency",)
    member_values = member_values[np.isfinite(member_values)]

    result["ceddar_members"] = member_values

    member_index = methods.index("ceddar_members")

    if member_values.size:
        q25, median, q75 = np.quantile(member_values, [0.25, 0.50, 0.75],)

        # Median member value as the bar height.
        ax.bar(
            member_index,
            median,
            width=0.68,
            color=style.method_color("ceddar_members"),
            alpha=0.90,
            zorder=5,
        )

        # Inter-member IQR.
        ax.errorbar(
            member_index,
            median,
            yerr=np.array([[median - q25], [q75 - median],]),
            fmt="none",
            ecolor=style.AXIS_GREY,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=15,
        )

    if np.isfinite(danra_frequency):
        ax.axhline(danra_frequency, color=style.method_color("danra"), linestyle="--", linewidth=1.0, alpha=0.65, zorder=2)

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    ax.set_xticks(x, [labels.get(method, style.method_label(method),) for method in methods],)
    ax.tick_params(axis="x", rotation=38,)
    ax.set_ylabel("Wet pixel-day\nfrequency")
    ax.set_ylim(bottom=0,)

    finish(ax, "Precipitation occurrence", label,)

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

    if not quantiles:
        for i, season in enumerate(SEASONS):
            face = style.SEASON_FACE.get(season)
            if face is not None:
                ax.axvspan(i - 0.5, i + 0.5, color=face, zorder=0,)


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
    ax.set_ylabel('Wet pixel-day frequency' if metric == 'wet_frequency' else 'Precipitation conditional\n'+r'on wet (mm day$^{-1}$)')

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


def spatial_map(ax, values, *, land, mode="sum", relative=False, vmin=None, vmax=None, label=None,):
    """Plot accumulated/mean precipitation or difference from DANRA."""

    cmap = (style.BIAS_CMAP if relative else style.PRECIP_CMAP)

    # Important:
    # Difference fields can be negative, so they must NOT use the
    # precipitation-specific zero normalization in imshow_variable().
    variable = ("bias" if relative else "prcp")

    image = map_field(
        ax,
        values,
        land=land,
        variable=variable,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_ocean=False,
        add_outline=True,
        add_boxplot=False,
        add_colorbar=False,
    )

    if label:
        style.panel_label(ax, label,)

    return image
