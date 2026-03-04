"""
Quick sanity check for ERA5 raw NetCDF files.

For each variable directory under RAW_BASE:
- Open the first .nc file
- Extract first time step (and first pressure level if present)
- Report NaN/Inf stats and min/max/mean
- Save a quicklook plot as PNG

Prefers xarray if available, but falls back to netCDF4 if xarray is missing.
"""

from __future__ import annotations

import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


def select_first_field(ds, varname: str):
    """Select first time step and first pressure level if present."""
    if xr is not None and hasattr(ds, "__getitem__") and hasattr(ds, "data_vars"):
        da = ds[varname]
        if "time" in da.dims:
            da = da.isel(time=0)
        elif "valid_time" in da.dims:
            da = da.isel(valid_time=0)
        for plev_dim in ["pressure_level", "level", "plev", "isobaricInhPa"]:
            if plev_dim in da.dims:
                da = da.isel({plev_dim: 0})
        return da

    # netCDF4
    v = ds.variables[varname]
    dims = list(v.dimensions)
    idx = []
    for d in dims:
        if d in ["time", "valid_time", "pressure_level", "level", "plev", "isobaricInhPa"]:
            idx.append(0)
        else:
            idx.append(slice(None))
    return np.asarray(v[tuple(idx)])


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
        nc_files = sorted(glob.glob(str(vdir / "*.nc")))
        if not nc_files:
            print(f"[SKIP] {vdir.name}: no .nc files")
            continue

        f0 = nc_files[0]
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
            field = select_first_field(ds, varname)

            if hasattr(field, "values"):
                arr = np.asarray(field.values)
                dims = getattr(field, "dims", ())
            else:
                arr = np.asarray(field)
                dims = ("y", "x") if arr.ndim == 2 else tuple([f"dim{i}" for i in range(arr.ndim)])

            st = nan_stats(arr)
            print(f"Data var: {varname} dims={dims} shape={arr.shape}")
            print(f"NaNs: {st['nan']} | Infs: {st['inf']} | finite_frac={st['finite_frac']:.6f}")
            print(f"min={st['min']} max={st['max']} mean={st['mean']}")

            plt.figure()
            im = plt.imshow(arr, origin="lower")
            plt.title(f"{vdir.name} (first step)\n{Path(f0).name}")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            png = out_dir / f"quicklook_{vdir.name}_{Path(f0).stem}.png"
            plt.tight_layout()
            plt.savefig(png, dpi=150)
            plt.close()
            print(f"Saved: {png}")

            summary_lines.append(
                f"{vdir.name}\tOK\t{Path(f0).name}\tvar={varname}\tshape={arr.shape}\t"
                f"nan={st['nan']}\tinf={st['inf']}\tfinite_frac={st['finite_frac']:.6f}\t"
                f"min={st['min']}\tmax={st['max']}\tmean={st['mean']}"
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