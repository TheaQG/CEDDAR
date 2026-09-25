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


def _hist_quantile(bins, counts, probability,):
    """Legacy-compatible histogram quantile using bin midpoints"""

    bins = np.asarray(bins, dtype=float)
    counts = np.asarray(counts, dtype=float)

    total = counts.sum()

    if total <= 0:
        return np.nan

    cumulative = np.cumsum(counts) / total

    index = int(np.clip(np.searchsorted(cumulative, probability,), 0, len(bins) - 2))

    return float(0.5 * (bins[index] + bins[index + 1]))



def _ensemble_member_densities(ensemble, season,):
    """Seasonal histogram density for each ensemble member."""
    indices = ensemble["season_indices"][season]
    bins = np.asarray(ensemble["bins"], dtype=float,)

    # [date, member, bin]
    counts = np.asarray(ensemble["counts_by_member"])[indices]

    # Sum over dates, retain member dimension: [member, bin]
    counts = counts.sum(axis=0)
    widths = np.diff(bins)
    totals = counts.sum(axis=1, keepdims=True)

    densities = np.divide(counts, totals * widths[None, :], out=np.full(counts.shape, np.nan, dtype=float,), where=totals > 0)
    centres = (bins[:-1] + bins[1:]) / 2

    return centres, densities


def seasonal(ax, data, season, *, baselines=None, ensemble=None, label=None, xlim=(-5, 120), show_percentiles=True, percentiles = ((0.95, "P95"), (0.99, "P99"), (0.999, "P99.9"), (0.9999, "P99.99"))):
    """Saved seasonal precipitation histograms. counts_gen is the saved PMM distribution, not pooled ensemble members."""

    series = [('danra', data, 'counts_hr'), ('era5_bilinear', data, 'counts_lr'), ('ceddar_pmm', data, 'counts_gen')]
    series += [(method, bundle, 'counts_gen') for method, bundle in (baselines or {}).items()]

    result = {}
    reference_daily = (data["arrays"]["dist_daily"])
    dates = reference_daily["dates"]

    danra_counts = None
    danra_bins = None

    eps = 1e-12

    for method, bundle, key in series:
        daily = bundle['arrays']['dist_daily']

        if not np.array_equal(dates, daily['dates']):
            raise ValueError(f"{method}: histogram dates differ; align before comparison")

        indices = bundle['season_indices'][season]

        if not len(indices):
            raise ValueError(f"No dates for {season} in {method}")

        bins = np.asarray(daily['bins'], dtype=float)
        counts = np.asarray(daily[key])[indices].sum(axis=0)

        if counts.sum():
            density = (counts / counts.sum() / np.diff(bins))
        else:
            density = np.full(counts.shape, np.nan)

        centres = (bins[1:] + bins[:-1]) / 2

        ax.plot(centres, np.maximum(density, eps), **style.method_style(method, marker=''))

        result[method] = dict(bins=bins, counts=counts, density=density, dates=dates[indices])

        if method == "danra":
            danra_counts = counts
            danra_bins = bins

    if ensemble is not None:
        ensemble_dates = np.asarray(ensemble["dates"]).astype(str)
        reference_dates = np.asarray(dates).astype(str)

        if not np.array_equal(reference_dates, ensemble_dates,):
            raise ValueError("CEDDAR ensemble histogram dates differ from the legacy distribution evaluation")

        indices = ensemble["season_indices"][season]

        if not len(indices):
            raise ValueError(f"No dates for {season} in ensemble")

        bins = np.asarray(ensemble["bins"], dtype=float)
        counts = np.asarray(ensemble["counts_pooled"])[indices].sum(axis=0)

        if counts.sum():
            density = (counts / counts.sum() / np.diff(bins))
        else:
            density = np.full(counts.shape, np.nan,)

        centres, member_density = (_ensemble_member_densities(ensemble, season))
        q25 = np.nanpercentile(member_density, 25, axis=0)
        median = np.nanmedian(member_density, axis=0)
        q75 = np.nanpercentile(member_density, 75, axis=0)
        valid_band = (np.isfinite(q25) & np.isfinite(q75) & (q25 > 0) & (q75 > 0))

        ax.fill_between(centres, q25, q75, where=valid_band, color=style.method_color("ceddar_members"), alpha=0.16, linewidth=0, zorder=4)
        ax.plot(centres, np.maximum(median, eps), **style.method_style("ceddar_members", marker="",),)
        # ax.plot(centres, np.maximum(density, eps), **style.method_style("ceddar_members", marker="",),)

        result["ceddar_members"] = dict(bins=bins, counts=counts, density=density, dates=ensemble_dates[indices],)
     
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 1)
    ax.set_xlim(*xlim)
    ax.set_xlabel("")
    ax.set_ylabel("Probability density")

    style.set_season_background(ax, season)

    # ========================================================================================
    # DANRA seasonal percentiles
    # ========================================================================================
    reference_percentiles = {}

    if (show_percentiles and danra_counts is not None):
        for probability, name in percentiles:
            value = _hist_quantile(danra_bins, danra_counts, probability)
            reference_percentiles[name] = value

            if not np.isfinite(value):
                continue

            ax.axvline(value, **style.PERCENTILE_LINE)
            ax.text(value, 0.94, name, transform=ax.get_xaxis_transform(), rotation=90, va='top', ha="right", fontsize=6.8, color=style.AXIS_GREY)

    n_dates = len(result["danra"]["dates"])

    ax.set_title(f"{season}   ($n={n_dates}$)", loc="center", pad=4)
    style.style_axes(ax)

    if label:
        style.panel_label(ax, label)

    result["reference_percentiles"] = reference_percentiles

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


def _ensemble_member_quantiles(ensemble, season, metrics,):
    """Seasonal precipitation quantiles for each ensemble member."""

    indices = ensemble["season_indices"][season]
    bins = np.asarray(ensemble["bins"],dtype=float,)

    # [date, member, bin]
    counts = np.asarray(ensemble["counts_by_member"])[indices]

    # Sum over dates, preserve member dimension:
    # [member, bin]
    counts = counts.sum(axis=0)
    n_members = counts.shape[0]

    values = np.full((n_members, len(metrics),), np.nan, dtype=float,)

    for member in range(n_members):
        for index, (probability, _,) in enumerate(metrics):
            values[member, index,] = _hist_quantile(bins, counts[member], probability,)
            
    return values


def seasonal_tails(
    ax,
    data,
    season,
    *,
    baselines=None,
    ensemble=None,
    metrics=((0.95, "P95"), (0.99, "P99"), (0.999, "P99.9"), (0.9999, "P99.99"),),
    mode="lines",
    label=None,
):
    """Seasonal upper-tail quantiles derived from saved daily histograms."""

    if mode not in {"lines", "bars",}:
        raise ValueError("mode must be 'lines' or 'bars'")

    series = [("danra", data, "counts_hr",), ("era5_bilinear", data, "counts_lr",), ("ceddar_pmm", data, "counts_gen",),]
    series += [(method, bundle, "counts_gen",) for method, bundle in (baselines or {}).items()]

    reference_dates = (data["arrays"]["dist_daily"]["dates"])

    result = {}

    for method, bundle, key in series:

        daily = bundle["arrays"]["dist_daily"]

        if not np.array_equal(reference_dates, daily["dates"],):
            raise ValueError(f"{method}: histogram dates differ; align before comparison")

        indices = bundle["season_indices"][season]
        bins = np.asarray(daily["bins"], dtype=float,)
        counts = np.asarray(daily[key])[indices].sum(axis=0)

        result[method] = np.array([_hist_quantile(bins, counts, probability) for probability, _ in metrics])

    member_quantiles = None

    # ------------------------------------------------------------------
    # Full CEDDAR ensemble members
    # ------------------------------------------------------------------

    if ensemble is not None:
        ensemble_dates = np.asarray(ensemble["dates"]).astype(str)

        if not np.array_equal(np.asarray(reference_dates).astype(str), ensemble_dates,):
            raise ValueError("CEDDAR ensemble histogram dates differ from the legacy distribution evaluation")

        member_quantiles = _ensemble_member_quantiles(ensemble, season, metrics,)

        # Useful to retain in returned diagnostics.
        result["ceddar_members_pooled"] = np.array([_hist_quantile(ensemble["bins"], ensemble["counts_pooled"][ensemble["season_indices"][season]].sum(axis=0), probability,) for probability, _ in metrics])


    y = np.arange(len(metrics))


    # ------------------------------------------------------------------
    # Lines
    # ------------------------------------------------------------------

    if mode == "lines":
        # Plot deterministic / summary methods normally.
        for method, values in result.items():
            # This is retained only for diagnostics, not manuscript plotting.
            if method == "ceddar_members_pooled":
                continue

            ax.plot(values, y, **style.method_style(method),)

        # Plot member median + inter-member IQR.
        if member_quantiles is not None:
            member_median = np.nanmedian(member_quantiles, axis=0,)
            member_q25 = np.nanpercentile(member_quantiles, 25, axis=0,)
            member_q75 = np.nanpercentile(member_quantiles, 75, axis=0,)

            ax.errorbar(
                member_median,
                y,
                xerr=np.vstack((member_median - member_q25, member_q75 - member_median,)),
                fmt="o-",
                color=style.method_color("ceddar_members"),
                linewidth=1.4,
                markersize=4.5,
                capsize=2.5,
                elinewidth=0.9,
                label=style.method_label("ceddar_members"),
                zorder=20,
            )


    # ------------------------------------------------------------------
    # Bars
    # ------------------------------------------------------------------

    else:

        methods = [method for method in result if method != "ceddar_members_pooled"]

        if member_quantiles is not None:
            methods.append("ceddar_members")

        height = 0.78 / len(methods)

        for index, method in enumerate(methods):
            offset = (index - (len(methods) - 1) / 2) * height

            if method == "ceddar_members":
                member_median = np.nanmedian(member_quantiles, axis=0,)
                member_q25 = np.nanpercentile(member_quantiles, 25, axis=0,)
                member_q75 = np.nanpercentile(member_quantiles, 75, axis=0,)

                ax.barh(
                    y + offset,
                    member_median,
                    height=height,
                    color=style.method_color("ceddar_members"),
                    label=style.method_label("ceddar_members"),
                    zorder=10,
                )

                ax.errorbar(
                    member_median,
                    y + offset,
                    xerr=np.vstack((member_median - member_q25, member_q75 - member_median,)),
                    fmt="none",
                    ecolor=style.AXIS_GREY,
                    elinewidth=0.8,
                    capsize=2,
                    zorder=20,
                )

            else:
                values = result[method]

                ax.barh(
                    y + offset,
                    values,
                    height=height,
                    color=style.method_color(method),
                    label=style.method_label(method),
                )


    # ============================================================
    # Axis formatting
    # ============================================================

    ax.set_yticks(y, [name for _, name in metrics],)

    ax.set_xlabel("")
    ax.set_ylabel("")

    style.set_season_background(ax, season,)
    style.style_axes(ax)

    if label:
        style.panel_label(ax, label,)

    return result

