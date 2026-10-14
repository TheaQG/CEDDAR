from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

from sbgm.variable_utils import get_color_for_model, get_cmap_for_variable
from sbgm.plotting_utils import get_dk_lsm_outline, overlay_outline


@dataclass
class SigmaStarData:
    summary: pd.DataFrame
    daily: pd.DataFrame
    psd: dict[str, np.ndarray]
    meta: dict


def load_sigma_star_data(eval_dir: str | Path) -> SigmaStarData:
    """
    Load saved sigma* evaluation products.

    Expected structure
    ------------------
    eval_dir/
        sigma_control_meta.json
        tables/
            agg_summary.csv
            metrics_by_sigma.csv
            sigma_psd_curves.npz
    """
    eval_dir = Path(eval_dir)
    table_dir = eval_dir / "tables"

    summary = pd.read_csv(table_dir / "agg_summary.csv")
    daily = pd.read_csv(table_dir / "metrics_by_sigma.csv")

    with np.load(table_dir / "sigma_psd_curves.npz") as f:
        psd = {key: f[key].copy() for key in f.files}

    meta = {}
    for name in ("sigma_control.json", "sigma_control_meta.json"):
        meta_path = eval_dir / name
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            break

    summary = summary.sort_values("sigma_star").reset_index(drop=True)

    # Basic consistency checks: fail loudly rather than silently plotting
    # mismatched versions of the saved products.
    sigma_summary = summary["sigma_star"].to_numpy(float)
    sigma_psd = np.asarray(psd["sigma_vals"], dtype=float)

    if not np.allclose(sigma_summary, sigma_psd):
        raise ValueError(
            "sigma grid differs between agg_summary.csv and sigma_psd_curves.npz"
        )

    sigma_meta = np.asarray(meta.get("sigma_star_grid", sigma_summary), dtype=float)
    if not np.allclose(sigma_summary, sigma_meta):
        raise ValueError(
            "sigma grid differs between saved tables and sigma_control_meta.json"
        )

    return SigmaStarData(
        summary=summary,
        daily=daily,
        psd=psd,
        meta=meta,
    )


def _metric_error(
    data: SigmaStarData,
    key: str,
    *,
    mode: str = "sem",
) -> np.ndarray:
    """
    Across-date variation for a metric.

    sem = sample standard deviation / sqrt(N finite dates)
    std = population standard deviation across dates
    """
    if mode not in {"sem", "std", "none"}:
        raise ValueError("mode must be one of: sem, std, none")

    sigma = data.summary["sigma_star"].to_numpy(float)

    if mode == "none":
        return np.full(sigma.shape, np.nan)

    errors = np.full(sigma.shape, np.nan)

    for i, s in enumerate(sigma):
        values = data.daily.loc[
            np.isclose(data.daily["sigma_star"], s, atol=1e-8, rtol=0),
            key,
        ].to_numpy(float)

        values = values[np.isfinite(values)]

        if len(values) < 2:
            continue

        if mode == "sem":
            errors[i] = values.std(ddof=1) / np.sqrt(len(values))
        else:
            errors[i] = values.std(ddof=0)

    return errors


def plot_psd(
    ax,
    data: SigmaStarData,
    *,
    add_colorbar: bool = True,
):
    """
    Panel (a): PSD of the daily ensemble-mean field, averaged across dates.
    """
    d = data.psd

    k = np.asarray(d["k"], float)
    sigma = np.asarray(d["sigma_vals"], float)
    hr = np.asarray(d["psd_hr_mean"], float)
    lr = np.asarray(d["psd_lr_mean"], float)
    gen = np.asarray(d["psd_gen_mean"], float)

    lr_nyquist = float(d["lr_nyquist"])
    band = np.asarray(d["psd_band_km"], float)

    valid = k > 0
    wavelength = 1.0 / k[valid]

    order = np.argsort(wavelength)[::-1]
    wavelength = wavelength[order]

    hr = hr[valid][order]
    lr = lr[valid][order]
    gen = gen[:, valid][:, order]

    ax.plot(
        wavelength,
        hr,
        color=get_color_for_model("hr"),
        lw=2.2,
        label="DANRA",
        zorder=10,
    )
    ax.plot(
        wavelength,
        lr,
        color=get_color_for_model("lr"),
        lw=1.8,
        ls="--",
        label="ERA5",
        zorder=9,
    )

    # Darker green = lower sigma*, lighter = higher sigma*.
    base_cmap = plt.get_cmap("Greens_r")  # type: ignore
    # Avoid nearly-white end of Greens_r so every sigma* remains visible
    cmap = LinearSegmentedColormap.from_list("Greens_r_truncated", base_cmap(np.linspace(0.05, 0.78, 256)))
    norm = Normalize(vmin=float(sigma.min()), vmax=float(sigma.max()))

    for i, s in enumerate(sigma):
        is_reference = np.isclose(s, 1.0)

        ax.plot(
            wavelength,
            gen[i],
            color=cmap(norm(s)),
            lw=2.3 if is_reference else 1.6,
            alpha=1.0 if is_reference else 0.95,
            zorder=6 if is_reference else 4,
        )

    # Slope-fit range.
    lo, hi = sorted(band)
    ax.axvspan(
        lo,
        hi,
        color="0.85",
        alpha=0.4,
        lw=0,
        label=f"PSD-slope band ({lo:g}–{hi:g} km)",
        zorder=0,
    )

    # Native LR Nyquist expressed as wavelength.
    if np.isfinite(lr_nyquist) and lr_nyquist > 0:
        lambda_nyquist = 1.0 / lr_nyquist
        ax.axvline(
            lambda_nyquist,
            color="0.25",
            ls=":",
            lw=1.2,
            label=rf"ERA5 Nyquist ($\lambda \approx {lambda_nyquist:.0f}$ km)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()

    ax.set_xlabel(r"Wavelength $\lambda$ (km)", fontsize=12)
    ax.set_ylabel("Spectral power", fontsize=12)
    ax.set_title("Power spectral density", fontsize=13)

    ax.tick_params(axis="both", which="major", labelsize=10.5)
    ax.tick_params(axis="both", which="minor", labelsize=9.5)

    ax.grid(True, which="both", ls=":", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(frameon=False, fontsize=10, loc="upper right", handlelength=2.4)

    if add_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = ax.figure.colorbar(
            sm,
            ax=ax,
            pad=0.015,
            fraction=0.035,
        )
        cb.set_label(r"$\sigma^*$", fontsize=11)
        cb.ax.tick_params(labelsize=10)
        cb.set_ticks(sigma[::2])
        cb.ax.invert_yaxis()

    return ax


def plot_metric(
    ax,
    data: SigmaStarData,
    key: str,
    *,
    error_mode: str = "sem",
):
    """
    Panels (b)--(d).

    Supported keys:
      r_lp       : low-pass PMM--ERA5 correlation
      slope_gen  : per-date generated PSD slope
      crps       : ensemble CRPS
    """
    specs = {
        "r_lp": {
            "summary_key": "r_lp_mean",
            "title": "Large-scale coherence",
            "ylabel": "Low-pass correlation\n(PMM, ERA5)",
        },
        "slope_gen": {
            "summary_key": "slope_gen_mean",
            "title": "Mesoscale PSD slope",
            "ylabel": r"PSD slope (5--20 km)",
        },
        "crps": {
            "summary_key": "crps_mean",
            "title": "Probabilistic skill",
            "ylabel": r"CRPS (mm day$^{-1}$)",
        },
    }

    if key not in specs:
        raise KeyError(f"Unknown sigma* metric: {key}")

    spec = specs[key]

    sigma = data.summary["sigma_star"].to_numpy(float)
    values = data.summary[spec["summary_key"]].to_numpy(float)
    errors = _metric_error(data, key, mode=error_mode)

    color = get_color_for_model("pmm")

    ax.axvline(
        1.0,
        color="0.55",
        lw=1.0,
        ls=":",
        zorder=0,
    )

    ax.plot(
        sigma,
        values,
        marker="o",
        ms=4,
        lw=1.5,
        color=color,
    )

    finite = np.isfinite(errors)
    if finite.any():
        ax.errorbar(
            sigma[finite],
            values[finite],
            yerr=errors[finite],
            fmt="none",
            color=color,
            lw=1,
            capsize=2,
        )

    if key == "slope_gen":
        hr = data.summary["slope_hr_mean"].to_numpy(float)
        hr_ref = float(np.nanmean(hr))

        ax.axhline(
            hr_ref,
            color=get_color_for_model("hr"),
            ls="--",
            lw=1.2,
            label="DANRA",
        )
        ax.legend(frameon=False, fontsize=8)

    if key == "crps":
        threshold = data.meta.get("metrics", {}).get("crps_rain_thresh")
        land_only = data.meta.get("metrics", {}).get("eval_land_only", False)

        if threshold is not None:
            subtitle = rf"DANRA $>{float(threshold):g}$ mm day$^{{-1}}$"
            if land_only:
                subtitle += ", land"
            ax.text(
                0.02,
                0.96,
                subtitle,
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

    ax.set_title(spec["title"], fontsize=12)
    ax.set_ylabel(spec["ylabel"], fontsize=10.5)
    ax.set_xlabel(r"$\sigma^*$", fontsize=11)
    ax.tick_params(labelsize=9.5)

    ax.set_xticks(sigma[::2])
    ax.set_xlim(sigma.min() - 0.01, sigma.max() + 0.01)

    ax.grid(True, ls=":", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    return ax


@dataclass
class SigmaStarExample:
    path: Path
    category: str
    date: str
    sigma_star: np.ndarray
    arrays: dict[str, np.ndarray]


def _scalar_string(x) -> str:
    a = np.asarray(x)

    if a.ndim == 0:
        value = a.item()
    else:
        value = a.reshape(-1)[0]

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    return str(value)


def _selected_samples_dir(eval_dir: str | Path) -> Path:
    eval_dir = Path(eval_dir)

    for name in ("selected_samples", "selected_examples"):
        p = eval_dir / name
        if p.is_dir():
            return p

    raise FileNotFoundError(f"No selected_samples/ directory found below {eval_dir}")


def list_selected_samples(eval_dir: str | Path) -> pd.DataFrame:
    """
    Return the selected sample cases available for plotting.
    """
    sample_dir = _selected_samples_dir(eval_dir)

    # Prefer the explicit manifest if the evaluator wrote one.
    for name in ("selected_samples.csv", "selected_examples.csv", "manifest.csv",):
        p = sample_dir / name
        if p.is_file():
            return pd.read_csv(p)

    # Fall back to reading lightweight metadata from NPZ files.
    rows = []

    for p in sorted(sample_dir.glob("*.npz")):
        with np.load(p, allow_pickle=False) as f:
            category = (_scalar_string(f["category"]) if "category" in f else p.stem.split("__")[0])
            date = (_scalar_string(f["date"]) if "date" in f else p.stem.split("__")[-1])

        rows.append({
                "category": category,
                "date": date,
                "file": p.name,
            })

    return pd.DataFrame(rows)


def load_selected_sample(
    eval_dir: str | Path,
    *,
    category: str | None = None,
    date: str | None = None,
) -> SigmaStarExample:
    """
    Load one compact example bundle written by sigma* evaluation.
    """
    sample_dir = _selected_samples_dir(eval_dir)

    candidates = sorted(sample_dir.glob("*.npz"))

    if not candidates:
        raise FileNotFoundError(f"No NPZ example bundles found in {sample_dir}")

    matches = []

    for p in candidates:
        with np.load(p, allow_pickle=False) as f:
            this_category = (_scalar_string(f["category"]) if "category" in f else p.stem.split("__")[0])

            this_date = (_scalar_string(f["date"]) if "date" in f else p.stem.split("__")[-1])

        if category is not None and this_category != category:
            continue
        if date is not None and this_date != str(date):
            continue

        matches.append((p, this_category, this_date))

    if len(matches) == 0:
        available = list_selected_samples(eval_dir)

        raise ValueError(f"No selected example matches category={category!r}, date={date!r}.\n\nAvailable cases:\n {available.to_string(index=False)}")

    if len(matches) > 1:
        raise ValueError("Example selection is ambiguous. Specify both --example-category and --example-date.")

    path, this_category, this_date = matches[0]

    with np.load(path, allow_pickle=False) as f:
        arrays = {key: f[key].copy() for key in f.files}

    sigma_key = next((key for key in ("sigma_star", "sigma_values", "sigma") if key in arrays), None,)

    if sigma_key is None:
        raise KeyError(f"{path} contains no sigma-star coordinate.")

    return SigmaStarExample(
        path=path,
        category=this_category,
        date=this_date,
        sigma_star=np.asarray(arrays[sigma_key], dtype=float,),
        arrays=arrays,
    )


def get_example_fields(
    example: SigmaStarExample,
    *,
    sigma_values: list[float],
    source: str = "member",
    member_index: int | None = None,
    mask_ocean: bool = False,
) -> dict[float, np.ndarray]:
    """
    Extract paired example fields from a compact sample bundle.

    source:
        "member"   -> same saved ensemble member across sigma*
        "pmm"      -> PMM across sigma*
        "ens_mean" -> ensemble mean across sigma*
    """
    if source not in {"member", "pmm", "ens_mean"}:
        raise ValueError("source must be 'member', 'pmm', or 'ens_mean'")

    if source == "member":
        if "members" not in example.arrays:
            raise KeyError(f"{example.path} does not contain 'members'")

        values = np.asarray(example.arrays["members"], dtype=float,)

        # [sigma, saved_member, y, x]
        if values.ndim != 4:
            raise ValueError(f"Expected members shape [S,N,H,W], got {values.shape}")

        saved_indices = np.asarray(example.arrays.get( "member_indices", np.arange(values.shape[1]),), dtype=int,)

        if member_index is None:
            position = 0
            member_index = int(saved_indices[position])
        else:
            where = np.where(saved_indices == int(member_index))[0]

            if len(where) == 0:
                raise ValueError(f"Member {member_index} was not saved. Available original member indices:  {saved_indices.tolist()}")

            position = int(where[0])

        values = values[:, position]

    else:
        if source not in example.arrays:
            raise KeyError(f"{example.path} does not contain {source!r}")

        values = np.asarray(example.arrays[source], dtype=float,)

        if values.ndim != 3:
            raise ValueError(f"Expected {source} shape [S,H,W], got {values.shape}")

    mask = example.arrays.get("mask")

    fields = {}

    for requested in sigma_values:
        matches = np.where(np.isclose(example.sigma_star, float(requested), atol=5e-4, rtol=0,))[0]

        if len(matches) != 1:
            raise ValueError(f"sigma*={requested:.2f} not uniquely found. Available values: {example.sigma_star.tolist()}")

        field = np.asarray(values[int(matches[0])], dtype=float,)

        if mask_ocean and mask is not None:
            field = np.where(np.asarray(mask).squeeze() > 0.5, field, np.nan,)

        fields[float(requested)] = field

    return fields


def plot_example_strip(
    axes,
    fields: dict[float, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    mask_ocean: bool = False,
    show_outline: bool = True,
    outline_alpha: float = 0.45,
    outline_lw: float = 0.7,
    percentile: float = 99.5,
    vmax: float | None = None,
):
    """
    Panel (e): paired field across selected sigma* values.

    Parameters
    ----------
    mask :
        Land-sea mask on the same grid as the generated fields.
        Land is assumed to be > 0.5.
    mask_ocean :
        If True, ocean pixels are masked from the precipitation field.
    show_outline :
        Draw the land-sea boundary derived directly from ``mask``.
    """
    sigma_values = list(fields.keys())

    mask_bool = None
    if mask is not None:
        mask_bool = np.asarray(mask).squeeze() > 0.5

    # Determine a common colour scale from exactly what will be displayed.
    display_fields = {}

    for sigma, field in fields.items():
        field = np.asarray(field, dtype=float)

        if mask_ocean and mask_bool is not None:
            field = np.where(mask_bool, field, np.nan)

        display_fields[sigma] = field

    finite = np.concatenate([x[np.isfinite(x)] for x in display_fields.values() if np.isfinite(x).any()])

    if vmax is None:
        vmax = float(np.nanpercentile(finite, percentile))

    vmax = max(vmax, 1e-6)

    cmap = get_cmap_for_variable("prcp")

    im = None

    for ax, sigma in zip(axes, sigma_values):
        field = display_fields[sigma]

        im = ax.imshow(
            field,
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )

        # Derive coastline directly from the saved LSM.
        if show_outline and mask_bool is not None:
            ax.contour(
                mask_bool.astype(float),
                levels=[0.5],
                colors="0.15",
                linewidths=outline_lw,
                alpha=outline_alpha,
                origin="lower",
                zorder=10,
            )

        ax.set_title(rf"$\sigma^*={sigma:.2f}$")
        ax.set_xticks([])
        ax.set_yticks([])

    return im