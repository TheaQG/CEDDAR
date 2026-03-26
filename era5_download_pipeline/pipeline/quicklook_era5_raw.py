"""
Quick sanity check for ERA5 raw NetCDF files.

For each variable directory under RAW_BASE:
- Find the first .nc file, also inside one extra subdirectory level (e.g. pressure-level folders like /850/)
- Open the first .nc file
- Extract first time step (and first pressure level if present)
- Report NaN/Inf stats and min/max/mean
- Save a quicklook plot as PNG using project colormaps / unit corrections where possible

Prefers xarray if available, but falls back to netCDF4 if xarray is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# Make project root importable so we can reuse plotting/unit helpers
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sbgm.variable_utils import (  # type: ignore
    get_cmap_for_variable,
    get_unit_for_variable,
    correct_variable_units,
)

# Optional: prefer xarray if available
try:
    import xarray as xr  # type: ignore
except Exception:
    xr = None

# Fallback: netCDF4 reader
try:
    from netCDF4 import Dataset  # type: ignore
except Exception:
    Dataset = None


def find_data_var(ds, preferred: str | None = None) -> str:
    """Pick the main data variable, ignoring coord-like vars."""
    if xr is not None and hasattr(ds, "data_vars"):
        coordish = set(ds.coords.keys()) | {"time", "latitude", "longitude", "lat", "lon", "valid_time"}
        if preferred and preferred in ds.data_vars:
            return preferred

        best, best_score = None, -1
        for name, da in ds.data_vars.items():
            if name in coordish:
                continue
            dims = set(da.dims)
            has_time = any(d in dims for d in ["time", "valid_time"])
            has_lat = any(d in dims for d in ["lat", "latitude", "y"])
            has_lon = any(d in dims for d in ["lon", "longitude", "x"])
            score = int(has_time) + int(has_lat) + int(has_lon) + len(da.dims)
            if score > best_score:
                best, best_score = name, score

        if best is None:
            raise ValueError(f"No suitable data variable found. data_vars={list(ds.data_vars)}")
        return best

    # netCDF4 fallback
    if preferred and preferred in ds.variables:
        return preferred

    coordish = {"time", "latitude", "longitude", "lat", "lon", "valid_time", "x", "y"}
    best, best_score = None, -1
    for name, v in ds.variables.items():
        if name in coordish:
            continue
        dims = list(getattr(v, "dimensions", ()))
        if len(dims) < 2:
            continue
        has_time = any(d in dims for d in ["time", "valid_time"])
        has_lat = any(d in dims for d in ["lat", "latitude", "y"])
        has_lon = any(d in dims for d in ["lon", "longitude", "x"])
        score = int(has_time) + int(has_lat) + int(has_lon) + len(dims)
        if score > best_score:
            best, best_score = name, score

    if best is None:
        raise ValueError(f"No suitable data variable found. variables={list(ds.variables.keys())}")
    return best



def select_first_n_fields(ds, varname: str, n_steps: int = 5):
    """Select the first n sequential time steps and first pressure level if present.

    Returns:
        arrays: list[np.ndarray] of 2D fields
        dims: tuple of dims for xarray or inferred dims for netCDF4
        labels: list[str] of readable time labels if available
    """
    if xr is not None and hasattr(ds, "__getitem__") and hasattr(ds, "data_vars"):
        da = ds[varname]
        for plev_dim in ["pressure_level", "level", "plev", "isobaricInhPa"]:
            if plev_dim in da.dims:
                da = da.isel({plev_dim: 0})

        time_dim = None
        for cand in ["time", "valid_time"]:
            if cand in da.dims:
                time_dim = cand
                break

        if time_dim is not None:
            n_avail = min(n_steps, int(da.sizes[time_dim]))
            da_sel = da.isel({time_dim: slice(0, n_avail)})
            arrays = [np.asarray(da_sel.isel({time_dim: i}).values) for i in range(n_avail)]
            dims = getattr(da_sel, "dims", ())
            labels = []
            try:
                coord_vals = da_sel[time_dim].values
                for v in coord_vals:
                    labels.append(str(np.datetime_as_string(v, unit="D")))
            except Exception:
                labels = [f"step {i}" for i in range(n_avail)]
            return arrays, dims, labels

        arr = np.asarray(da.values)
        dims = getattr(da, "dims", ())
        return [arr], dims, ["field"]

    # netCDF4 fallback
    v = ds.variables[varname]
    dims = list(v.dimensions)

    time_dim = None
    for cand in ["time", "valid_time"]:
        if cand in dims:
            time_dim = cand
            break

    plev_dims = ["pressure_level", "level", "plev", "isobaricInhPa"]

    if time_dim is not None:
        time_axis = dims.index(time_dim)
        n_avail = min(n_steps, int(v.shape[time_axis]))
        arrays = []
        labels = [f"step {i}" for i in range(n_avail)]
        for i in range(n_avail):
            idx = []
            for d in dims:
                if d == time_dim:
                    idx.append(i)
                elif d in plev_dims:
                    idx.append(0)
                else:
                    idx.append(slice(None))
            arrays.append(np.asarray(v[tuple(idx)]))
        return arrays, tuple(dims), labels

    idx = []
    for d in dims:
        if d in plev_dims:
            idx.append(0)
        else:
            idx.append(slice(None))
    arr = np.asarray(v[tuple(idx)])
    return [arr], tuple(dims), ["field"]



def nan_stats(a: np.ndarray) -> dict:
    finite = np.isfinite(a)
    n = a.size
    n_finite = int(finite.sum())
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    out = {
        "count": int(n),
        "finite": n_finite,
        "nan": n_nan,
        "inf": n_inf,
        "finite_frac": (n_finite / n) if n else 0.0,
    }
    if n_finite > 0:
        af = a[finite]
        out.update({"min": float(np.min(af)), "max": float(np.max(af)), "mean": float(np.mean(af))})
    else:
        out.update({"min": float("nan"), "max": float("nan"), "mean": float("nan")})
    return out


# === Variable normalization, pretty names, unit/cmap helpers, recursive file search ===

def normalize_var_key(var_dir_name: str, nc_file: Path) -> str:
    """Map raw directory / filename patterns to project variable keys used in variable_utils."""
    name = var_dir_name.lower()
    stem = nc_file.stem.lower()
    parent = nc_file.parent.name.lower()

    # direct single-level aliases
    direct = {
        "t2m": "temp",
        "tp": "prcp",
        "cape": "cape",
        "wvf_east": "ewvf",
        "wvf_north": "nwvf",
        "msl": "msl",
        "pev": "pev",
    }
    if name in direct:
        return direct[name]

    # pressure-level folders / filenames
    if name == "z_pl":
        if parent.isdigit():
            return f"z_pl_{parent}"
        for lev in ["250", "500", "850", "1000"]:
            if lev in stem:
                return f"z_pl_{lev}"
    if name == "t_pl":
        if parent.isdigit():
            return f"t_pl_{parent}"
        for lev in ["250", "500", "850", "1000"]:
            if lev in stem:
                return f"t_pl_{lev}"
    if name == "q_pl":
        if parent.isdigit():
            return f"q_pl_{parent}"
        for lev in ["250", "500", "850", "1000"]:
            if lev in stem:
                return f"q_pl_{lev}"
    if name == "thetae_pl":
        if parent.isdigit():
            return f"thetae_pl_{parent}"
        for lev in ["250", "500", "850", "1000"]:
            if lev in stem:
                return f"thetae_pl_{lev}"

    return name


PRETTY_NAMES = {
    "temp": "2 m temperature",
    "prcp": "Total precipitation",
    "cape": "CAPE",
    "ewvf": "Eastward water vapour flux",
    "nwvf": "Northward water vapour flux",
    "msl": "Mean sea-level pressure",
    "pev": "Potential evaporation",
    "z_pl_250": "Geopotential height 250 hPa",
    "z_pl_500": "Geopotential height 500 hPa",
    "z_pl_850": "Geopotential height 850 hPa",
    "z_pl_1000": "Geopotential height 1000 hPa",
    "t_pl_250": "Temperature 250 hPa",
    "t_pl_500": "Temperature 500 hPa",
    "t_pl_850": "Temperature 850 hPa",
    "t_pl_1000": "Temperature 1000 hPa",
    "q_pl_250": "Specific humidity 250 hPa",
    "q_pl_500": "Specific humidity 500 hPa",
    "q_pl_850": "Specific humidity 850 hPa",
    "q_pl_1000": "Specific humidity 1000 hPa",
    "thetae_pl_250": "Equivalent potential temperature 250 hPa",
    "thetae_pl_500": "Equivalent potential temperature 500 hPa",
    "thetae_pl_850": "Equivalent potential temperature 850 hPa",
    "thetae_pl_1000": "Equivalent potential temperature 1000 hPa",
}


FALLBACK_UNITS = {
    "pev": "m",
    "q_pl_250": "kg/kg",
    "q_pl_500": "kg/kg",
    "q_pl_850": "kg/kg",
    "q_pl_1000": "kg/kg",
    "t_pl_250": r"$^\circ$C",
    "t_pl_500": r"$^\circ$C",
    "t_pl_850": r"$^\circ$C",
    "t_pl_1000": r"$^\circ$C",
    "thetae_pl_250": r"$^\circ$C",
    "thetae_pl_500": r"$^\circ$C",
    "thetae_pl_850": r"$^\circ$C",
    "thetae_pl_1000": r"$^\circ$C",
}


def get_pretty_name(var_key: str) -> str:
    return PRETTY_NAMES.get(var_key, var_key)


def get_unit_safe(var_key: str) -> str:
    try:
        return get_unit_for_variable(var_key)
    except Exception:
        return FALLBACK_UNITS.get(var_key, "")


def get_cmap_safe(var_key: str) -> str:
    # Reuse project colormaps where possible. Fall back sensibly for extra pressure-level vars.
    if var_key.startswith("t_pl") or var_key.startswith("thetae_pl"):
        return "plasma"
    if var_key.startswith("q_pl"):
        return "viridis"
    return get_cmap_for_variable(var_key)


def apply_unit_correction(var_key: str, arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).copy()
    try:
        return correct_variable_units(var_key, "ERA5", arr)
    except Exception:
        return arr



def find_first_nc_file(vdir: Path) -> Path | None:
    """Find the first .nc file either directly in vdir or one level below."""
    direct = sorted(vdir.glob("*.nc"))
    if direct:
        return direct[0]
    nested = sorted(vdir.glob("*/*.nc"))
    if nested:
        return nested[0]
    return None


# Helper to extract readable date label from filename
def extract_date_label(nc_file: Path) -> str:
    """Extract a readable date label from filenames like *_1991.nc or *_19910101.nc."""
    stem = nc_file.stem
    parts = stem.split("_")
    if not parts:
        return stem
    last = parts[-1]
    if len(last) == 8 and last.isdigit():
        return f"{last[:4]}-{last[4:6]}-{last[6:8]}"
    if len(last) == 6 and last.isdigit():
        return f"{last[:4]}-{last[4:6]}"
    if len(last) == 4 and last.isdigit():
        return last
    return stem


def shared_finite_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    """Compute shared vmin/vmax across multiple arrays using only finite values."""
    finite_vals = []
    for a in arrays:
        a = np.asarray(a)
        m = np.isfinite(a)
        if np.any(m):
            finite_vals.append(a[m])
    if not finite_vals:
        return 0.0, 1.0
    vals = np.concatenate(finite_vals)
    return float(np.min(vals)), float(np.max(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_base", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefer_varname", default="")
    args = ap.parse_args()

    raw_base = Path(args.raw_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    var_dirs = sorted([p for p in raw_base.iterdir() if p.is_dir()])

    for vdir in var_dirs:
        f0_path = find_first_nc_file(vdir)
        if f0_path is None:
            print(f"[SKIP] {vdir.name}: no .nc files")
            continue

        f0 = str(f0_path)
        print(f"\n=== {vdir.name} ===\nFile: {f0}")

        try:
            if xr is not None:
                ds = xr.open_dataset(f0, decode_times=True)
            else:
                if Dataset is None:
                    raise ImportError("Neither xarray nor netCDF4 is available in this environment.")
                ds = Dataset(f0, mode="r")
        except Exception as e:
            print(f"[ERROR] Could not open {f0}: {e}")
            summary_lines.append(f"{vdir.name}\tOPEN_FAIL\t{f0}\t{e}")
            continue

        try:
            varname = find_data_var(ds, preferred=args.prefer_varname or None)
            arrays, dims, labels = select_first_n_fields(ds, varname, n_steps=5)
            arrays = [np.asarray(a) for a in arrays]

            var_key = normalize_var_key(vdir.name, f0_path)
            arrays = [apply_unit_correction(var_key, a) for a in arrays]

            st = nan_stats(arrays[0])
            unit = get_unit_safe(var_key)
            cmap = get_cmap_safe(var_key)
            pretty_name = get_pretty_name(var_key)
            vmin, vmax = shared_finite_limits(arrays)

            print(f"Data var: {varname} dims={dims} n_fields={len(arrays)} first_shape={arrays[0].shape} var_key={var_key}")
            print(f"First field stats -> NaNs: {st['nan']} | Infs: {st['inf']} | finite_frac={st['finite_frac']:.6f}")
            print(f"Shared color limits -> min={vmin} max={vmax} {unit}".strip())

            date_label = extract_date_label(f0_path)

            ncols = len(arrays)
            fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 5.8), squeeze=False)
            axes = axes[0]

            im = None
            for i, (ax, arr_i, label_i) in enumerate(zip(axes, arrays, labels)):
                im = ax.imshow(arr_i, origin="upper", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
                ax.set_title(label_i, fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f"{pretty_name}\n{date_label} (first {len(arrays)} sequential steps)", fontsize=15)

            # One shared colorbar for all panels.
            cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
            if unit:
                cbar.set_label(unit)

            png = out_dir / f"quicklook_{vdir.name}_{f0_path.stem}.png"
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            fig.savefig(png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {png}")

            summary_lines.append(
                f"{vdir.name}\tOK\t{f0_path.name}\tvar={varname}\tvar_key={var_key}\tn_fields={len(arrays)}\tshape={arrays[0].shape}\t"
                f"nan={st['nan']}\tinf={st['inf']}\tfinite_frac={st['finite_frac']:.6f}\t"
                f"shared_min={vmin}\tshared_max={vmax}\tunit={unit}"
            )
        except Exception as e:
            print(f"[ERROR] Processing failed for {f0}: {e}")
            summary_lines.append(f"{vdir.name}\tPROC_FAIL\t{Path(f0).name}\t{e}")
        finally:
            try:
                ds.close()
            except Exception:
                pass

    (out_dir / "quicklook_summary.tsv").write_text("\n".join(summary_lines) + "\n")
    print(f"\nWrote summary: {out_dir / 'quicklook_summary.tsv'}")


if __name__ == "__main__":
    main()