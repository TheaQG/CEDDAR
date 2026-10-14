"""Daily paired errors and pooled event detection from revision.load_deterministic"""
from ._common import np, style, METHODS, number, unique, boxes, finish


def daily_errors(ax, data, metric="mae", *, methods=METHODS, label=None, show_method_labels=True, show_improvement=True,):
    if metric not in ["mae", "rmse"]:
        raise ValueError(f'Unsupported metric: {metric}. Choose from "mae" or "rmse"')

    rows = [r for r in data["tables"]["daily_continuous_metrics"] if r["subset"] == "all_land"]
    lookup = unique(rows, ("date", "method"),)
    dates = sorted(r["date"] for r in rows if r["method"] == "era5_bilinear")

    if not dates:
        raise ValueError("No dates found for era5_bilinear reference")

    result = {}
    groups = []
    fractions = []

    for method in methods:
        if {r["date"] for r in rows if r["method"] == method} != set(dates):
            raise ValueError(f"{method}: dates differ from bilinear reference")

        delta = []

        for date in dates:
            pred = lookup[date, method]
            ref = lookup[date, "era5_bilinear"]

            if (number(pred, "n_pixels") != number(ref, "n_pixels")):
                raise ValueError(f"{method}/{date}: pixel support differs")

            delta.append(number(pred, metric) - number(ref, metric))

        delta = np.asarray(delta, dtype=float,)
        delta = delta[np.isfinite(delta)]

        if method == "era5_bilinear":
            fraction = np.nan
        else:
            fraction = float(np.mean(delta < 0))

        result[method] = {
            "values": delta,
            "fraction_lower": fraction,
            "n": len(delta),
        }

        groups.append(delta)
        fractions.append(fraction)

    boxes(
        ax,
        groups,
        methods,
        horizontal=True,
        flier_alpha=0.22,
        flier_markersize=2.2,
    )

    positions = np.arange(1, len(methods) + 1,)

    ax.set_yticks(positions)

    if show_method_labels:
        ax.set_yticklabels([style.method_label(method) for method in methods])
    else:
        ax.tick_params(axis="y", labelleft=False,)

    # Explicit limits are safer than invert_yaxis()
    # when axes share their y-axis.
    ax.set_ylim(len(methods) + 0.5, 0.5,)
    ax.axvline(0, **style.REFERENCE_LINE,)
    ax.set_xlabel(r"Method $-$ bilinear ERA5 (mm day$^{-1}$)")

    if show_improvement:
        # Small column heading.
        ax.text(
            0.98,
            1.01,
            "Days improved",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="0.4",
        )

        # x in axes coordinates, y in data coordinates.
        transform = (ax.get_yaxis_transform())

        for y, method, fraction in zip(positions, methods, fractions,):
            text = ("ref." if method == "era5_bilinear" else f"{100 * fraction:.0f}%")

            ax.text(
                0.98,
                y,
                text,
                transform=transform,
                ha="right",
                va="center",
                fontsize=8,
                color=style.method_color(method),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.4),
            )

    finish(ax, f"Daily {metric.upper()} difference", label, label_position=(-0.14, 1.08),)

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


def spatial_correlation(ax, data,*, methods=METHODS, label=None,):
    """
    Daily spatial Pearson correlation with DANRA.

    All displayed methods use the common set of dates for which
    correlation is finite for every method.
    """
    rows = [r for r in data["tables"]["daily_spatial_correlation"] if r["subset"] == "all_land"]
    unique(rows, ("date", "method"),)

    # ---------------------------------------------------------
    # Common finite dates across all methods
    # ---------------------------------------------------------

    valid_sets = []

    for method in methods:
        valid_dates = {r["date"] for r in rows if (r["method"] == method and np.isfinite(number(r, "correlation")))}

        if not valid_dates:
            raise ValueError(f"No finite correlations for {method}")

        valid_sets.append(valid_dates)

    common_dates = set.intersection(*valid_sets)

    if not common_dates:
        raise ValueError("No common finite correlation dates")

    # ---------------------------------------------------------
    # Extract paired distributions
    # ---------------------------------------------------------

    groups = []
    # labels = []
    result = {}

    for method in methods:
        selected = sorted([r for r in rows if (r["method"] == method and r["date"] in common_dates)], key=lambda r: r["date"],)

        values = np.asarray([number(r, "correlation",) for r in selected], dtype=float,)

        if (len(values) != len(common_dates)):
            raise ValueError(f"{method}: expected {len(common_dates)} common dates, found {len(values)}")

        if not np.all(np.isfinite(values)):
            raise ValueError(f"{method}: non-finite value remained after common-date filtering")

        result[method] = {
            "values": values,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "n": len(values),
        }

        groups.append(values)
        # labels.append(style.method_label(method))

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    boxes(ax, groups, methods, horizontal=True,)

    positions = np.arange(1, len(methods) + 1)
    ax.set_yticks(positions)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(len(methods) + 0.5, 0.5)
    # r = 0 means no linear spatial association.
    ax.axvline(0, **style.REFERENCE_LINE,)
    ax.set_xlabel("Pearson $r$ with DANRA")

    finish(
        ax,
        (f"Daily spatial correlation ($N={len(common_dates)}$)"),
        label,
        label_position=(-0.14, 1.08,),
    )

    return result