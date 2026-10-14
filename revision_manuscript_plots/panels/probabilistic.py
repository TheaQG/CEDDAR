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


def reliability(ax, data, threshold=1.0, *, source="revision", count_ax=None, label=None,):
    if source == "revision":
        rows = [r for r in data["tables"]["reliability"] if (r["subset"] == "all_land" and number(r, "threshold",) == threshold)]
        rows = sorted(rows, key=lambda r: number(r, "bin",),)

        xkey = "mean_forecast_probability"
        ykey = "observed_frequency"
        nkey = "n_cases"
    elif source == "legacy":
        rows = data["tables"][f"prob_reliability_{threshold:.1f}mm"]

        xkey = "prob_pred"
        ykey = "freq_obs"
        nkey = "count"
    else:
        raise ValueError("source must be revision or legacy")

    if not rows:
        raise ValueError(f"No reliability rows for threshold {threshold}")

    x = column(rows, xkey,)
    y = column(rows, ykey,)
    n = column(rows, nkey,)

    valid = ((n > 0) & np.isfinite(x) & np.isfinite(y))

    # ----------------------------------------------------------
    # Bin populations
    # ----------------------------------------------------------

    if count_ax is not None:
        if source == "revision":
            left = column(rows, "bin_left",)
            right = column(rows, "bin_right",)
            width = (right - left)
        else:
            centres = column(rows, "bin_center",)
            width = np.full(len(rows), 0.9 / len(rows),)
            left = (centres - width / 2)

        count_ax.bar(
            left,
            n,
            width=width * 0.88,
            align="edge",
            color=style.CEDDAR,
            alpha=0.12,
            edgecolor="none",
            zorder=0,
        )

        count_ax.set(xlim=(0, 1), ylim=(0, None),)
        count_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),)
        count_ax.grid(False)

        # Put reliability curves visually above count bars.
        count_ax.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_alpha(0)

    # ----------------------------------------------------------
    # Reliability
    # ----------------------------------------------------------

    ax.plot(x[valid], y[valid], **style.method_style( "ceddar_members"),)
    ax.plot([0, 1], [0, 1], **style.REFERENCE_LINE,)

    ax.set(xlim=(0, 1.05), ylim=(0, 1.05), xlabel="Forecast probability", ylabel="Observed frequency",)

    finish(ax, rf"$\geq${threshold:g} mm day$^{{-1}}$", label, label_position=(-0.16, 1.08))

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
    finish(ax, 'Randomized rank histogram', label, label_position=(-0.12, 1.08))
    return rows


def spread_skill(ax, data, *, label=None,):
    """Daily spread-skill scatter with pooled bins and daily linear fit."""

    # Pooled pixel-level spread bins
    rows = sorted(
        [
            r
            for r in data["tables"]["spread_skill"]
            if r["subset"] == "all_land"
        ],
        key=lambda r: number(r, "bin",),
    )

    x_bin = column(rows, "mean_spread",)
    y_bin = column(rows, "rmse",)


    # Daily domain-mean spread-skill
    daily_rows = data["tables"].get("spread_skill_daily", [],)
    daily_rows = [r for r in daily_rows if r["subset"] == "all_land"]

    x_daily = np.array([])
    y_daily = np.array([])
    x_valid = np.array([])
    y_valid = np.array([])

    fit_result = None

    if daily_rows:
        x_daily = column(daily_rows, "mean_spread",)
        y_daily = column(daily_rows, "rmse",)
        valid_daily = (np.isfinite(x_daily) & np.isfinite(y_daily))

        x_valid = x_daily[valid_daily]
        y_valid = y_daily[valid_daily]

        ax.scatter(
            x_valid,
            y_valid,
            color=style.CEDDAR,
            s=10,
            alpha=0.20,
            edgecolors="none",
            zorder=2,
            label="Daily",
        )

    # Axis ranges
    finite_x = np.r_[x_bin[np.isfinite(x_bin)], x_valid,]
    finite_y = np.r_[y_bin[np.isfinite(y_bin)], y_valid,]

    if finite_x.size:
        x_limit = (1.05 * float(np.nanmax(finite_x)))
    else: 
        x_limit = 1.0

    if finite_y.size:
        y_limit = (1.05 * float(np.nanmax(finite_y)))
    else:
        y_limit = 1.0


    # Descriptive OLS fit to daily points
    if len(x_valid) >= 2:
        slope, intercept = np.polyfit(x_valid, y_valid, 1,)
        r = np.corrcoef(x_valid, y_valid,)[0, 1]

        x_fit = np.linspace(0, x_limit, 200,)
        y_fit = (slope * x_fit + intercept)

        ax.plot(
            x_fit,
            y_fit,
            color=style.NEUTRAL,
            linestyle="-.",
            linewidth=0.98,
            zorder=3,
            label="Daily linear fit",
        )

        ax.text(
            0.04,
            0.96,
            (rf"$y={slope:.2f}x{intercept:+.2f}$" + "\n" + rf"$r={r:.2f}$"),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color=style.AXIS_GREY
        )

        fit_result = {"slope": slope,"intercept": intercept, "r": r,}


    # Pooled spread bins

    ax.plot(
        x_bin,
        y_bin,
        color=style.CEDDAR_DARK,
        marker="o",
        linestyle="-",
        linewidth=1.25,
        markersize=5.5,
        zorder=5,
        label="Pooled bins",
    )


    # 1:1 spread = skill reference

    reference_limit = min(x_limit, y_limit)

    ax.plot([0, reference_limit], [0, reference_limit], **style.REFERENCE_LINE,)
    ax.set(xlabel=r"Ensemble spread (mm day$^{-1}$)", ylabel=r"Ensemble-mean RMSE (mm day$^{-1}$)", xlim=(0, x_limit), ylim=(0, y_limit))
    ax.legend(frameon=False, fontsize=6.5, loc="lower right",)

    finish(ax, "Spread-skill", label, label_position=(-0.14, 1.08))

    return {
        "bins": rows,
        "daily": daily_rows,
        "fit": fit_result,
    }


def coverage(ax, data, *, label=None):
    """Central interval coverage; error bars show inter-date IQR."""

    specs = (
        ("all_land", "All land", "o", "-", style.CEDDAR, -0.008),
        ("observed_wet", "Observed wet land", "^", "--", style.CEDDAR_DARK, +0.008),
    )

    result = {}

    for subset, legend_label, marker, linestyle, color, offset in specs:
        rows = sorted(
            [
                r
                for r in data["tables"]["coverage_summary"]
                if (
                    r["subset"] == subset
                    and r["season"] == "ALL"
                    and r["metric"] == "empirical_coverage"
                )
            ],
            key=lambda r: number(r, "nominal_coverage", ),
        )

        if not rows:
            continue

        x = column(rows, "nominal_coverage",)
        median = column(rows, "median",)
        p25 = column(rows, "p25",)
        p75 = column(rows, "p75",)

        yerr = np.vstack((median - p25, p75 - median,))
        x_plot = x + offset

        ax.errorbar(
            x_plot,
            median,
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.1,
            markersize=4,
            capsize=2.5,
            alpha=0.95,
            label=legend_label,
            zorder=4,
        )

        result[subset] = rows

    ax.plot([0, 1], [0, 1], **style.REFERENCE_LINE,)
    ax.set(xlabel="Nominal central coverage", ylabel="Empirical coverage", xlim=(0.48, 0.92), ylim=(0.05, 1.0),)
    ax.set_xticks((0.5, 0.8, 0.9))
    ax.legend(frameon=False, fontsize=7.5, loc="upper left",)

    finish(ax, "Central interval coverage", label, label_position=(-0.12, 1.08),)

    return result
