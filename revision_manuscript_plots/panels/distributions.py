"""Legacy distributions, spectra, and saved physical examples. No reevaluation"""

from ._common import np, style, number, unique, finish, map_field


def _land_mask_2d(land, shape):
    if land is None:
        return None

    mask = np.asarray(land)
    mask = np.squeeze(mask)

    if mask.shape != shape:
        raise ValueError(f"Land-mask shape {mask.shape} does not match field {shape}")

    return mask >= 0.5


def _draw_hatched_ocean(
    ax,
    land,
    *,
    facecolor="#f5f5f5",
    hatch="////",
    hatch_color="#b5b5b5",
):
    """
    Draw a light hatched background over ocean pixels.
    """
    ocean = (~land).astype(int)

    # Light neutral ocean background.
    ax.contourf(
        ocean,
        levels=[0.5, 1.5],
        colors=[facecolor],
        zorder=-2,
    )

    # Hatch overlay.
    cf = ax.contourf(
        ocean,
        levels=[0.5, 1.5],
        colors="none",
        hatches=[hatch],
        zorder=5,
    )

    for collection in cf.collections:
        collection.set_edgecolor(hatch_color)
        collection.set_linewidth(0.0)


def _draw_ocean_tint(
    ax,
    land,
    *,
    alpha=0.38,
    facecolor="white",
):
    """
    Visually de-emphasise ocean while retaining the full LR field.
    """
    ocean = (~land).astype(int)

    ax.contourf(
        ocean,
        levels=[0.5, 1.5],
        colors=[facecolor],
        alpha=alpha,
        zorder=4,
    )

    # Redraw coastline above the tint.
    ax.contour(
        land.astype(float),
        levels=[0.5],
        colors="#aaaaaa",
        linewidths=0.5,
        zorder=6,
    )


def example(
    ax,
    data,
    date,
    method="danra",
    *,
    member=None,
    reference=None,
    ocean_style=None,
    ocean_alpha=0.38,
    **kwargs,
):
    """
    Draw one physical example field.

    ocean_style:
        "hatch" : mask ocean data and show a hatched background
        "tint"  : retain ocean data but visually de-emphasise it
        None    : no special ocean treatment
    """
    day = data["dates"][str(date)]

    if member is None:
        field = day["fields"][method]
    else:
        field = day["ensemble"][member]

    field = np.asarray(field, dtype=float,)

    if reference is not None:
        field = (field - np.asarray(day["fields"][reference], dtype=float,))

    land = _land_mask_2d(day.get("land"), field.shape,)

    kwargs.setdefault("land", day.get("land"),)

    # We handle the ocean explicitly here.
    kwargs["show_ocean"] = True

    if (ocean_style == "hatch" and land is not None):
        # Draw hatch underneath the precipitation image.
        _draw_hatched_ocean(ax, land,)

        # Make ocean transparent in the actual precipitation layer.
        field = field.copy()
        field[~land] = np.nan

    image = map_field(ax, field, **kwargs,)

    if (ocean_style == "tint" and land is not None):
        # LR remains a full-field quantity; only de-emphasise ocean.
        _draw_ocean_tint(ax, land, alpha=ocean_alpha,)

    return image


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


def _external_seasonal_histogram(product, reference_daily, season, *, name,):
    """Return seasonal counts from an externally derived histogram product."""

    reference_dates = np.asarray(reference_daily["dates"]).astype(str)

    dates = np.asarray(product["dates"]).astype(str)

    if not np.array_equal(reference_dates, dates,):
        raise ValueError(f"{name}: histogram dates differ from the reference distribution evaluation")

    reference_bins = np.asarray(reference_daily["bins"], dtype=float,)
    bins = np.asarray(product["bins"], dtype=float,)

    if (bins.shape != reference_bins.shape or not np.allclose(bins, reference_bins, rtol=0, atol=1e-12,)):
        raise ValueError(f"{name}: histogram bins differ from the reference distribution evaluation")

    indices = product["season_indices"][season]

    if not len(indices):
        raise ValueError(f"No dates for {season} in {name}")

    daily_counts = np.asarray(product["counts"])
    counts = daily_counts[indices].sum(axis=0)

    return (bins, counts, dates[indices],)


def seasonal(ax, data, season, *, baselines=None, ensemble=None, qm_histograms=None, label=None, xlim=(-5, 120), show_percentiles=True, percentiles = ((0.95, "P95"), (0.99, "P99"), (0.999, "P99.9"), (0.9999, "P99.99"))):
    """Saved seasonal precipitation histograms. counts_gen is the saved PMM distribution, not pooled ensemble members."""

    series = [('danra', data, 'counts_hr'), ('era5_bilinear', data, 'counts_lr'), ('ceddar_pmm', data, 'counts_gen')]
    series += [(method, bundle, 'counts_gen') for method, bundle in (baselines or {}).items()]

    result = {}
    reference_daily = (data["arrays"]["dist_daily"])
    dates = reference_daily["dates"]

    if (qm_histograms is not None and "qm" in (baselines or {})):
        raise ValueError("QM was supplied both through legacy baselines and corrected qm_histograms.")

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

    # Corrected land-only QM distribution
    if qm_histograms is not None:
        bins, counts, qm_dates = (_external_seasonal_histogram(qm_histograms, reference_daily, season, name="QM"))

        if counts.sum():
            density = (counts / counts.sum() / np.diff(bins))
        else:
            density = np.full(counts.shape, np.nan)

        centres = (bins[1:] + bins[:-1]) / 2
        ax.plot(centres, np.maximum(density, eps), **style.method_style("qm", marker=''))

        result["qm"] = {
            "bins": bins,
            "counts": counts,
            "density": density,
            "dates": qm_dates
        }

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


def psd(ax, data, *, baselines=None, label=None,):
    """Isotropic PSD with legacy-compatible ERA5 scale handling.

    ERA5 is the bilinearly interpolated LR field evaluated on the HR
    spatial grid (`psd_lr_hr`). Its native Nyquist wavelength is retained
    as the boundary between physically supported and unresolved scales.

    Saved uncertainty bands are 95% intervals around the temporal mean,
    not min/max day-to-day ranges.
    """

    arrays = data["arrays"]["scale_psd_curves"]
    eps = 1e-12

    k = np.asarray(arrays["k"], dtype=float,)
    valid_k = (np.isfinite(k) & (k > 0))
    wavelength = (1.0 / k[valid_k])

    # Keep wavelength decreasing from large -> small.
    order = np.argsort(wavelength)[::-1]
    wavelength = wavelength[order]

    # -------
    # Helpers
    # -------
    def mean_curve(saved, key):
        values = np.asarray(saved[key], dtype=float,)

        if values.ndim == 2:
            values = np.nanmean(values, axis=0,)
        values = values[valid_k][order]

        return np.maximum(values, eps,)

    def saved_band(saved, lo_key, hi_key, ):
        if (lo_key not in saved or hi_key not in saved):
            return None

        lo = np.asarray(saved[lo_key], dtype=float,)[valid_k][order]
        hi = np.asarray(saved[hi_key], dtype=float,)[valid_k][order]

        return (np.maximum(lo, eps), np.maximum(hi, eps),)

    def draw_band(saved, lo_key, hi_key, method, *, alpha=0.10,):
        band = saved_band(saved, lo_key, hi_key,)

        if band is None:
            return
        lo, hi = band

        ax.fill_between(
            wavelength,
            lo,
            hi,
            color=style.method_color(method),
            alpha=alpha,
            linewidth=0,
            zorder=1,
        )

    result = {}

    # =====
    # DANRA
    # =====

    danra = mean_curve(arrays, "psd_hr")

    ax.plot(wavelength, danra, **style.method_style("danra", marker="",))#, zorder=6,)
    draw_band(arrays, "psd_hr_ci_lo", "psd_hr_ci_hi", "danra", alpha=0.12,)

    result["danra"] = {"k": k, "power": danra,}

    # ====================================================
    # Bilinear ERA5. IMPORTANT: use psd_lr_hr, not psd_lr.
    # ====================================================

    era5_key = ("psd_lr_hr" if "psd_lr_hr" in arrays else "psd_lr")
    era5 = mean_curve(arrays, era5_key,)

    lr_nyquist = float(np.asarray(arrays.get("lr_nyquist", 0.0)).squeeze())

    if lr_nyquist > 0:

        lambda_nyquist = (1.0 / lr_nyquist)
        resolved = (wavelength >= lambda_nyquist)
        unresolved = (wavelength < lambda_nyquist)

        # Interpolate ERA5 PSD exactly at the native Nyquist wavelength
        log_lambda = np.log(wavelength[::-1])
        log_power = np.log(era5[::-1])
        power_nyquist = np.exp(np.interp(np.log(lambda_nyquist), log_lambda, log_power,))

        # Solid: resolved scales + exact Nyquist point.
        wavelength_resolved = np.r_[wavelength[resolved], lambda_nyquist,]
        era5_resolved = np.r_[era5[resolved], power_nyquist,]

        # Sort back to descending wavelength.
        order_resolved = np.argsort(wavelength_resolved)[::-1]
        wavelength_resolved = wavelength_resolved[order_resolved]
        era5_resolved = era5_resolved[order_resolved]

        ax.plot(
            wavelength_resolved,
            era5_resolved,
            color=style.method_color("era5_bilinear"),
            linestyle="-",
            linewidth=1.2,
            label="Bilinear ERA5",
            zorder=5,
        )

        # Dashed: exact Nyquist point + unresolved scales.
        wavelength_unresolved = np.r_[lambda_nyquist, wavelength[unresolved],]
        era5_unresolved = np.r_[power_nyquist, era5[unresolved],]
        order_unresolved = np.argsort(wavelength_unresolved)[::-1]

        wavelength_unresolved = wavelength_unresolved[order_unresolved]
        era5_unresolved = era5_unresolved[order_unresolved]

        ax.plot(
            wavelength_unresolved,
            era5_unresolved,
            color=style.method_color("era5_bilinear"),
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            label="_nolegend_",
            zorder=4,
        )

    else:
        ax.plot(wavelength, era5, **style.method_style("era5_bilinear",marker="",), zorder=5,)
        lambda_nyquist = None

    result["era5_bilinear"] = {
        "k": k,
        "power": era5,
        "lambda_nyquist": lambda_nyquist,
    }

    # ==========
    # CEDDAR PMM
    # ==========

    pmm = mean_curve(arrays, "psd_gen",)

    ax.plot(wavelength, pmm, **style.method_style("ceddar_pmm", marker="",))#, zorder=6,)
    draw_band(arrays, "psd_gen_ci_lo", "psd_gen_ci_hi", "ceddar_pmm", alpha=0.08,)

    result["ceddar_pmm"] = {
        "k": k,
        "power": pmm,
    }

    # ================================
    # CEDDAR members (mean member PSD)
    # ================================

    if "psd_gen_ens_mean" in arrays:
        members = mean_curve(arrays, "psd_gen_ens_mean",)

        opts = style.method_style("ceddar_members", marker="",)
        opts["label"] = ("CEDDAR members\n(mean PSD)")

        ax.plot(wavelength, members, **opts)#, zorder=7,)
        draw_band(arrays, "psd_gen_ens_ci_lo", "psd_gen_ens_ci_hi", "ceddar_members", alpha=0.12,)

        result["ceddar_members"] = {
            "k": k,
            "power": members,
        }

    # ========================
    # Additional baselines: QM
    # ========================

    for method, bundle in (baselines or {}).items():
        saved = bundle["arrays"]["scale_psd_curves"]

        baseline_k = np.asarray(saved["k"], dtype=float,)

        # Do not silently interpolate a baseline onto the CEDDAR grid.
        if (baseline_k.shape != k.shape or not np.allclose(
                baseline_k,
                k,
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError(f"{method}: PSD k-grid differs from the reference evaluation")

        baseline = mean_curve(saved, "psd_gen",)

        ax.plot(wavelength, baseline, **style.method_style(method, marker="",))#, zorder=5,)

        # QM has the same saved CI structure if generated by scale eval.
        draw_band(saved, "psd_gen_ci_lo", "psd_gen_ci_hi", method, alpha=0.06,)

        result[method] = {
            "k": baseline_k,
            "power": baseline,
        }

    # ========================
    # Scale-regime annotations
    # ========================
    # ERA5 native Nyquist when available.
    if lr_nyquist > 0:
        low_k_max = lr_nyquist
    else:
        low_k_max = 1.0 / 200.0

    high_k_min = (1.0 / 20.0)
    lambda_low = (1.0 / low_k_max)
    lambda_high = (1.0 / high_k_min)

    for value in (lambda_low, lambda_high,):
        ax.axvline(
            value,
            color=style.AXIS_GREY,
            linestyle="--",
            linewidth=0.8,
            alpha=0.75,
            zorder=2,
        )

    transform = (ax.get_xaxis_transform())

    ax.text(
        lambda_low * 1.025,
        0.74,
        rf"low-k  $\lambda={lambda_low:.0f}$ km",
        transform=transform,
        rotation=90,
        color=style.AXIS_GREY,
        fontsize=6.5,
        ha="center",
        va="bottom",
        zorder=3,
    )

    ax.text(
        lambda_high * 1.025,
        0.74,
        rf"high-k  $\lambda={lambda_high:.0f}$ km",
        transform=transform,
        rotation=90,
        color=style.AXIS_GREY,
        fontsize=6.5,
        ha="center",
        va="bottom",
        zorder=3,
    )

    # ====
    # Axes
    # ====

    ax.set(xscale="log", yscale="log", xlabel=r"Wavelength $\lambda$(km)", ylabel="Spectral power",)
    ax.set_ylim(bottom=5e-3)
    if not ax.xaxis_inverted():
        ax.invert_xaxis()

    finish(ax, "Isotropic power spectral density", label, label_position=(-0.10, 1.06),)

    # Both major and minor log-grid lines.
    ax.grid(True, which="major", linestyle=":", linewidth=0.65, alpha=0.50,)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.35,)

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
    qm_histograms=None,
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

    reference_daily = (data["arrays"]["dist_daily"])
    if qm_histograms is not None and "qm" in (baselines or {}):
        raise ValueError("QM was supplied both through legacy baselines and corrected QM histograms")
        
    for method, bundle, key in series:

        daily = bundle["arrays"]["dist_daily"]

        if not np.array_equal(reference_dates, daily["dates"],):
            raise ValueError(f"{method}: histogram dates differ; align before comparison")

        indices = bundle["season_indices"][season]
        bins = np.asarray(daily["bins"], dtype=float,)
        counts = np.asarray(daily[key])[indices].sum(axis=0)

        result[method] = np.array([_hist_quantile(bins, counts, probability) for probability, _ in metrics])

    # Corrected land-only QM tails
    if qm_histograms is not None:
        bins, counts, _ = (_external_seasonal_histogram(qm_histograms, reference_daily, season, name="QM"))

        result["qm"] = np.array([_hist_quantile(bins, counts, probability) for probability, _ in metrics])

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
                member_median = np.nanmedian(member_quantiles, axis=0,) # type: ignore
                member_q25 = np.nanpercentile(member_quantiles, 25, axis=0,) # type: ignore
                member_q75 = np.nanpercentile(member_quantiles, 75, axis=0,) # type: ignore

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

