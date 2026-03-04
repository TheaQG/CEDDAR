import os
import sys
import glob
import argparse
import hashlib
import datetime as dt
import numpy as np
# New imports for optional zarr check
from typing import Optional, Tuple
 # -------------------------
# Zarr helper (for already regridded ERA5-on-DANRA stores)
# -------------------------

def _try_import_zarr():
    try:
        import zarr  # noqa
        return zarr
    except Exception as e:
        raise RuntimeError(
            "Could not import 'zarr'. Install it in the container/env used for this check."
        ) from e


def _list_all_zarr_arrays(g) -> list:
    """Return all array paths (recursive) for a zarr Group."""
    paths = []
    # robust across zarr versions
    try:
        # zarr>=2.15: visititems
        def _visit(name, obj):
            try:
                import zarr as _z
                if isinstance(obj, _z.Array):
                    paths.append(name)
            except Exception:
                # fallback: identify by duck-typing
                if hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "__getitem__"):
                    paths.append(name)
        g.visititems(_visit)
    except Exception:
        # very old zarr: brute-force walk
        def _walk(prefix, grp):
            for k in grp.keys():
                obj = grp[k]
                name = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    paths.append(name)
                else:
                    _walk(name, obj)
        _walk("", g)

    return sorted(set(paths))


def summarize_zarr_store(zarr_root: str, label: str, n_samples: int = 3, array_name_hint: Optional[str] = None) -> None:
    """Open a Zarr store and inspect a few arrays for shape + NaNs.

    The store in this project typically contains arrays under paths like:
      <file_stem>/<dataset_key>
    e.g. 'tp_19940101/tp_589x789'.

    If `array_name_hint` is provided, we prefer arrays whose *path contains* that substring.
    """
    zarr = _try_import_zarr()

    if not os.path.exists(zarr_root):
        raise FileNotFoundError(zarr_root)

    g = zarr.open_group(zarr_root, mode="r")
    arrays = _list_all_zarr_arrays(g)
    print(f"[{label}] zarr_root={zarr_root}")
    print(f"[{label}] discovered {len(arrays)} arrays (recursive)")
    if len(arrays) == 0:
        print(f"[{label}] No arrays found in zarr store.")
        return

    # Choose candidate arrays
    chosen = arrays
    if array_name_hint:
        chosen = [p for p in arrays if array_name_hint in p]
        if len(chosen) == 0:
            print(f"[{label}] No arrays matched hint '{array_name_hint}'. Falling back to first arrays.")
            chosen = arrays

    chosen = chosen[: max(1, int(n_samples))]

    for i, p in enumerate(chosen):
        a = g[p]
        # load whole array (these are cutouts on DANRA domain; should be manageable)
        arr = np.asarray(a[...])
        finite = np.isfinite(arr)
        n = arr.size
        n_bad = int((~finite).sum())
        frac_bad = n_bad / max(n, 1)
        print(f"[{label}] sample[{i}] path='{p}' dtype={arr.dtype} shape={arr.shape}")
        print(f"[{label}] sample[{i}] nonfinite={n_bad}/{n} ({frac_bad:.6%})")
        if finite.any():
            print(f"[{label}] sample[{i}] finite min/max = {float(arr[finite].min()):.6g} / {float(arr[finite].max()):.6g}")
# -------------------------
# NetCDF open helper (tries multiple backends)
# -------------------------
def open_netcdf(path):
    """Open a NetCDF file (weights or data).

    NOTE: The DANRA grid is provided as a *CDO grid description* text file (NOT NetCDF).
    For that, use `parse_cdo_grid_file()`.

    Returns (backend_name, dataset_handle).
    We try xarray, netCDF4, h5netcdf, scipy.io.netcdf (in that order).
    """
    # xarray
    try:
        import xarray as xr  # noqa
        ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
        return ("xarray", ds)
    except Exception:
        pass

    # netCDF4
    try:
        import netCDF4 as nc  # noqa
        ds = nc.Dataset(path, mode="r")
        return ("netCDF4", ds)
    except Exception:
        pass

    # h5netcdf (netcdf4-on-hdf5)
    try:
        import h5netcdf  # noqa
        ds = h5netcdf.File(path, mode="r")
        return ("h5netcdf", ds)
    except Exception:
        pass

    # scipy classic netcdf
    try:
        from scipy.io import netcdf  # noqa
        ds = netcdf.netcdf_file(path, mode="r")
        return ("scipy", ds)
    except Exception:
        pass

    raise RuntimeError(
        "Could not open NetCDF (weights or data). None of these worked: xarray, netCDF4, h5netcdf, scipy.\n"
        "Fix: use a container that includes ONE of them (xarray is easiest)."
    )

def close_netcdf(backend, ds):
    try:
        if backend == "xarray":
            ds.close()
        else:
            ds.close()
    except Exception:
        pass

# -------------------------
# CDO grid-file parser (text grid description)
# -------------------------

def parse_cdo_grid_file(path: str) -> dict:
    """Parse a CDO grid description text file into a dict.

    Example lines:
      xsize = 789
      ysize = 589
      xfirst = 0
      xinc = 2500

    Values can be numbers or quoted strings.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    d: dict = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # drop trailing comments
            if "#" in v:
                v = v.split("#", 1)[0].strip()
            # strip quotes
            if (len(v) >= 2) and ((v[0] == v[-1]) and v[0] in ('"', "'")):
                v2 = v[1:-1]
                d[k] = v2
                continue
            # try int, then float
            try:
                d[k] = int(v)
                continue
            except Exception:
                pass
            try:
                d[k] = float(v)
                continue
            except Exception:
                pass
            d[k] = v

    return d


def summarize_cdo_grid_file(grid: dict, label: str = "grid") -> None:
    """Print a compact summary of a parsed CDO grid description."""
    # common keys
    gridtype = grid.get("gridtype", None)
    gridsize = grid.get("gridsize", None)
    xsize = grid.get("xsize", None)
    ysize = grid.get("ysize", None)

    print(f"[{label}] CDO grid file summary")
    if gridtype is not None:
        print(f"[{label}] gridtype={gridtype}")
    if xsize is not None and ysize is not None:
        print(f"[{label}] xsize={xsize}  ysize={ysize}  (H,W)={(int(ysize), int(xsize))}")
    if gridsize is not None:
        print(f"[{label}] gridsize={gridsize}")
        if xsize is not None and ysize is not None:
            try:
                prod = int(xsize) * int(ysize)
                ok = (int(gridsize) == prod)
                print(f"[{label}] gridsize==xsize*ysize ? {ok}  ({gridsize} vs {prod})")
            except Exception:
                pass

    # show some projection / georef keys if present
    for k in [
        "grid_mapping",
        "grid_mapping_name",
        "standard_parallel",
        "longitude_of_central_meridian",
        "latitude_of_projection_origin",
        "earth_radius",
        "false_easting",
        "false_northing",
        "longitudeOfFirstGridPointInDegrees",
        "latitudeOfFirstGridPointInDegrees",
    ]:
        if k in grid:
            print(f"[{label}] {k} = {grid[k]}")

    # spacing / origin
    for k in ["xfirst", "xinc", "yfirst", "yinc", "xunits", "yunits", "xname", "yname"]:
        if k in grid:
            print(f"[{label}] {k} = {grid[k]}")


def grid_bbox_from_firstpoint(grid: dict):
    """Best-effort bbox from 'longitudeOfFirstGridPointInDegrees'/'latitudeOfFirstGridPointInDegrees' if present.

    CDO grid description for projected grids often only contains the *first* point lat/lon.
    We return None unless both are present.
    """
    lon0 = grid.get("longitudeOfFirstGridPointInDegrees", None)
    lat0 = grid.get("latitudeOfFirstGridPointInDegrees", None)
    if lon0 is None or lat0 is None:
        return None
    try:
        return (float(lat0), float(lat0), float(lon0), float(lon0))
    except Exception:
        return None

def file_fingerprint(path, blocksize=2**20, max_dir_entries_for_hash=20000):
    """
    Fingerprint a file OR a directory (e.g. a .zarr store).

    - For files: md5 over file bytes.
    - For dirs: md5 over a stable manifest (relative path + size + mtime),
      and we also report total bytes + file count.
      (We do NOT hash file contents inside the dir.)
    """
    st = os.stat(path)

    # ---- Regular file: hash contents ----
    if os.path.isfile(path):
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                b = f.read(blocksize)
                if not b:
                    break
                h.update(b)
        return {
            "path": path,
            "type": "file",
            "bytes": st.st_size,
            "nfiles": None,
            "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "md5": h.hexdigest(),
        }

    # ---- Directory (e.g. .zarr): hash manifest ----
    if os.path.isdir(path):
        total_bytes = 0
        nfiles = 0
        h = hashlib.md5()

        # Walk deterministically
        for root, dirs, files in os.walk(path):
            dirs.sort()
            files.sort()
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    fst = os.stat(fp)
                except FileNotFoundError:
                    # file vanished mid-walk; skip
                    continue

                rel = os.path.relpath(fp, path)
                total_bytes += fst.st_size
                nfiles += 1

                # stable manifest line
                h.update(f"{rel}\t{fst.st_size}\t{int(fst.st_mtime)}\n".encode("utf-8"))

                # safety brake for insane inode counts
                if nfiles >= max_dir_entries_for_hash:
                    h.update(b"...TRUNCATED...\n")
                    return {
                        "path": path,
                        "type": "dir",
                        "bytes": total_bytes,
                        "nfiles": nfiles,
                        "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                        "md5": h.hexdigest(),
                        "note": f"manifest hash truncated at {max_dir_entries_for_hash} files",
                    }

        return {
            "path": path,
            "type": "dir",
            "bytes": total_bytes,
            "nfiles": nfiles,
            "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "md5": h.hexdigest(),
        }

    raise RuntimeError(f"Unsupported path type (not file/dir): {path}")

def _as_array(var):
    # var can be xarray.DataArray, netCDF4.Variable, h5netcdf.Variable, numpy array, etc.
    try:
        return np.asarray(var[:])
    except Exception:
        try:
            return np.asarray(var.values)
        except Exception:
            return np.asarray(var)

def list_vars_dims(backend, ds, max_vars=60):
    if backend == "xarray":
        dims = dict(ds.dims)
        vars_ = list(ds.variables.keys())
        return dims, vars_[:max_vars]
    else:
        dims = {k: len(v) for k, v in ds.dimensions.items()}
        vars_ = list(ds.variables.keys())
        return dims, vars_[:max_vars]

def get_var(ds_backend, ds, name_candidates):
    """Return first variable matching any candidate name (exact)."""
    if ds_backend == "xarray":
        for n in name_candidates:
            if n in ds.variables:
                return ds.variables[n]
        return None
    else:
        for n in name_candidates:
            if n in ds.variables:
                return ds.variables[n]
        return None
    

def summarize_grid_like(ds_backend, ds, label, bbox=None):
    """
    Try to find lat/lon arrays and print their ranges + shape.
    bbox: (lat_min, lat_max, lon_min, lon_max) in degrees (lon can be -180..180 or 0..360)
    """
    lat = get_var(ds_backend, ds, ["lat", "latitude", "nav_lat", "y"])
    lon = get_var(ds_backend, ds, ["lon", "longitude", "nav_lon", "x"])
    if lat is None or lon is None:
        print(f"[{label}] Could not find obvious lat/lon variables. (Checked lat/latitude/nav_lat/y and lon/longitude/nav_lon/x)")
        return

    lat_a = _as_array(lat)
    lon_a = _as_array(lon)

    def rng(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan)
        return (float(a.min()), float(a.max()))

    lat_min, lat_max = rng(lat_a)
    lon_min, lon_max = rng(lon_a)

    print(f"[{label}] lat shape={lat_a.shape} range=[{lat_min:.6f}, {lat_max:.6f}]")
    print(f"[{label}] lon shape={lon_a.shape} range=[{lon_min:.6f}, {lon_max:.6f}]")

    if bbox is not None:
        b_lat_min, b_lat_max, b_lon_min, b_lon_max = bbox
        # allow lon wrap (0..360 vs -180..180) by checking both representations
        def wrap360(x):
            return (x + 360.0) % 360.0

        lon_min_360, lon_max_360 = wrap360(lon_min), wrap360(lon_max)
        b_lon_min_360, b_lon_max_360 = wrap360(b_lon_min), wrap360(b_lon_max)

        covers_lat = (lat_min <= b_lat_min) and (lat_max >= b_lat_max)
        covers_lon = ((lon_min <= b_lon_min) and (lon_max >= b_lon_max)) or (
            (lon_min_360 <= b_lon_min_360) and (lon_max_360 >= b_lon_max_360)
        )

        print(f"[{label}] bbox coverage check:")
        print(f"  - covers lat? {covers_lat}   (grid [{lat_min:.3f},{lat_max:.3f}] vs bbox [{b_lat_min:.3f},{b_lat_max:.3f}])")
        print(f"  - covers lon? {covers_lon}   (grid [{lon_min:.3f},{lon_max:.3f}] vs bbox [{b_lon_min:.3f},{b_lon_max:.3f}])")

def summarize_time(ds_backend, ds, label):
    t = get_var(ds_backend, ds, ["time", "Times", "valid_time"])
    if t is None:
        print(f"[{label}] No obvious time variable found.")
        return

    t_a = _as_array(t)
    print(f"[{label}] time shape={t_a.shape} dtype={t_a.dtype}")
    # Try to print units if present
    units = None
    try:
        units = t.attrs.get("units", None) if ds_backend == "xarray" else getattr(t, "units", None)
    except Exception:
        units = None
    if units is not None:
        print(f"[{label}] time units: {units}")

    # show first few numeric time values
    flat = t_a.ravel()
    n = min(5, flat.size)
    print(f"[{label}] time first {n}: {flat[:n].tolist()}")
    print(f"[{label}] time last {n}: {flat[-n:].tolist()}")


def summarize_weights(ds_backend, ds, label):
    """
    CDO/SCRIP weights typically have dims like:
      - src_grid_size, dst_grid_size, num_links, num_wgts
    and variables like:
      - src_address, dst_address, remap_matrix
    """
    dims, vars_ = list_vars_dims(ds_backend, ds, max_vars=200)
    print(f"[{label}] dims (subset): {list(dims.items())[:20]}")
    print(f"[{label}] vars (subset): {vars_[:40]}")

    rm = get_var(ds_backend, ds, ["remap_matrix", "S", "weights"])
    if rm is not None:
        rm_a = _as_array(rm)
        finite = np.isfinite(rm_a)
        n_bad = int((~finite).sum())
        print(f"[{label}] remap_matrix shape={rm_a.shape} nonfinite={n_bad}")
        if finite.any():
            print(f"[{label}] remap_matrix finite range=[{float(rm_a[finite].min()):.6g}, {float(rm_a[finite].max()):.6g}]")
    else:
        print(f"[{label}] No remap_matrix variable found (checked remap_matrix/S/weights).")

def summarize_data_field(ds_backend, ds, label, var_name=None):
    """
    Pick a likely data variable and report NaN stats for first timestep/slice.
    """
    dims, vars_ = list_vars_dims(ds_backend, ds, max_vars=500)

    # Choose candidate variable:
    candidates = []
    if var_name:
        candidates.append(var_name)

    # common ERA5 names + your naming patterns
    candidates += ["tp", "t2m", "msl", "cape", "pev", "q", "t", "z", "u", "v",
                   "tp_tot", "tp_589x789", "t2m_589x789", "msl_589x789", "cape_589x789"]

    chosen = None
    for c in candidates:
        v = get_var(ds_backend, ds, [c])
        if v is not None:
            chosen = (c, v)
            break

    # fallback: pick first non-coordinate-ish 3D variable
    if chosen is None:
        skip = {"lat", "latitude", "lon", "longitude", "time", "valid_time", "x", "y"}
        for name in vars_:
            if name in skip:
                continue
            v = get_var(ds_backend, ds, [name])
            try:
                a = _as_array(v)
                if a.ndim >= 2:
                    chosen = (name, v)
                    break
            except Exception:
                continue

    if chosen is None:
        print(f"[{label}] Could not identify a data variable to inspect.")
        return

    name, v = chosen
    a = _as_array(v)

    # take first time index if present
    if a.ndim >= 3:
        a0 = a[0]
    else:
        a0 = a

    finite = np.isfinite(a0)
    n = a0.size
    n_bad = int((~finite).sum())
    frac_bad = n_bad / max(n, 1)

    print(f"[{label}] data var='{name}' full_shape={a.shape} slice_shape={a0.shape}")
    print(f"[{label}] first-slice nonfinite={n_bad}/{n} ({frac_bad:.6%})")
    if finite.any():
        print(f"[{label}] first-slice finite min/max = {float(a0[finite].min()):.6g} / {float(a0[finite].max()):.6g}")

def pick_first_nc(path_or_glob):
    if os.path.isdir(path_or_glob):
        files = sorted(glob.glob(os.path.join(path_or_glob, "**", "*.nc"), recursive=True))
    else:
        files = sorted(glob.glob(path_or_glob))
    return files[0] if files else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_file", required=True, help="Path to DANRA grid file (CDO grid description text file)")
    ap.add_argument("--weights_file", required=True, help="Path to remap weights file (NetCDF)")
    ap.add_argument("--era5_sample", required=True, help="A raw ERA5 .nc file OR a dir/glob to pick the first .nc from")
    ap.add_argument("--era5_var", default=None, help="Optional: variable name to inspect inside ERA5 sample (e.g. tp, t2m)")
    ap.add_argument("--bbox", default=None, help="Optional bbox 'latmin,latmax,lonmin,lonmax' to test coverage")
    ap.add_argument("--zarr_sample", default=None, help="Optional: path to a .zarr store to inspect (e.g. .../zarr_files/train.zarr)")
    ap.add_argument("--zarr_hint", default=None, help="Optional: substring to prefer when picking arrays (e.g. 'tp_589x789' or 'prcp_589x789')")
    ap.add_argument("--zarr_n", type=int, default=3, help="Optional: number of arrays to inspect in the zarr store")
    args = ap.parse_args()

    bbox = None
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("--bbox must be 'latmin,latmax,lonmin,lonmax'")
        bbox = tuple(parts)

    era5_path = args.era5_sample
    if not os.path.isfile(era5_path):
        era5_path = pick_first_nc(era5_path)
        if era5_path is None:
            raise FileNotFoundError(f"Could not find any .nc under: {args.era5_sample}")

    print("============================================================")
    print("[FILES] Fingerprints")
    fp_list = [args.grid_file, args.weights_file, era5_path]
    if args.zarr_sample:
        fp_list.append(args.zarr_sample)

    for p in fp_list:
        fp = file_fingerprint(p)
        print(f"- {fp['path']}")
        extra = f"  nfiles={fp['nfiles']}" if fp.get("nfiles") is not None else ""
        note = f"  note={fp['note']}" if fp.get("note") else ""
        print(f"  type={fp.get('type','?')}  size={fp['bytes']/1e9:.3f} GB{extra}  mtime={fp['mtime']}  md5={fp['md5']}{note}")
    print("============================================================")

    # --- Grid file (CDO grid description text file) ---
    print("\n==================== DANRA GRID FILE (CDO) ===================")
    grid = parse_cdo_grid_file(args.grid_file)
    summarize_cdo_grid_file(grid, label="grid")
    # Optional: best-effort coverage check only if we have usable lat/lon bounds
    if bbox is not None:
        # For projected grids, CDO grid files typically don't contain full lat/lon arrays.
        # We can only do a weak check if a first-point lat/lon exists.
        fb = grid_bbox_from_firstpoint(grid)
        if fb is None:
            print("[grid] bbox coverage check: SKIP (CDO grid file has no full lat/lon arrays)")
        else:
            g_lat_min, g_lat_max, g_lon_min, g_lon_max = fb
            b_lat_min, b_lat_max, b_lon_min, b_lon_max = bbox
            covers_lat = (g_lat_min <= b_lat_min) and (g_lat_max >= b_lat_max)
            covers_lon = (g_lon_min <= b_lon_min) and (g_lon_max >= b_lon_max)
            print("[grid] bbox coverage check (weak, first-point only):")
            print(f"  - covers lat? {covers_lat}   (grid [{g_lat_min:.3f},{g_lat_max:.3f}] vs bbox [{b_lat_min:.3f},{b_lat_max:.3f}])")
            print(f"  - covers lon? {covers_lon}   (grid [{g_lon_min:.3f},{g_lon_max:.3f}] vs bbox [{b_lon_min:.3f},{b_lon_max:.3f}])")

    # --- Weights file ---
    print("\n==================== WEIGHTS FILE ============================")
    b, ds = open_netcdf(args.weights_file)
    print(f"[weights] backend={b}")
    summarize_weights(b, ds, "weights")
    # Best-effort dimension consistency checks
    try:
        # dst_grid_size should match danra xsize*ysize
        xsize = int(grid.get("xsize")) if grid.get("xsize") is not None else None
        ysize = int(grid.get("ysize")) if grid.get("ysize") is not None else None
        if xsize is not None and ysize is not None:
            dst_expected = xsize * ysize
            dst_dim = None
            try:
                dst_dim = int(ds.dimensions["dst_grid_size"].size) if hasattr(ds.dimensions["dst_grid_size"], "size") else int(len(ds.dimensions["dst_grid_size"]))
            except Exception:
                try:
                    dst_dim = int(len(ds.dimensions["dst_grid_size"]))
                except Exception:
                    dst_dim = None
            if dst_dim is not None:
                print(f"[weights] check: dst_grid_size dim={dst_dim} expected={dst_expected} -> {dst_dim == dst_expected}")

        # src_grid_size should match ERA5 lat*lon for the raw sample
        try:
            if b == "xarray":
                lat_len = int(ds.dims.get("latitude", ds.dims.get("lat", 0)))
                lon_len = int(ds.dims.get("longitude", ds.dims.get("lon", 0)))
            else:
                lat_len = int(len(ds.dimensions.get("latitude"))) if "latitude" in ds.dimensions else int(len(ds.dimensions.get("lat")))
                lon_len = int(len(ds.dimensions.get("longitude"))) if "longitude" in ds.dimensions else int(len(ds.dimensions.get("lon")))
            src_expected = lat_len * lon_len
        except Exception:
            src_expected = None

        src_dim = None
        try:
            src_dim = int(ds.dimensions["src_grid_size"].size) if hasattr(ds.dimensions["src_grid_size"], "size") else int(len(ds.dimensions["src_grid_size"]))
        except Exception:
            try:
                src_dim = int(len(ds.dimensions["src_grid_size"]))
            except Exception:
                src_dim = None

        if src_dim is not None and src_expected is not None:
            print(f"[weights] check: src_grid_size dim={src_dim} expected(lat*lon)={src_expected} -> {src_dim == src_expected}")
    except Exception as e:
        print(f"[weights] check: SKIP (failed with {type(e).__name__}: {e})")
    close_netcdf(b, ds)

    # --- ERA5 sample ---
    print("\n==================== ERA5 RAW SAMPLE =========================")
    b, ds = open_netcdf(era5_path)
    print(f"[era5] backend={b}")
    dims, vars_ = list_vars_dims(b, ds)
    print(f"[era5] file={era5_path}")
    print(f"[era5] dims: {list(dims.items())[:30]}")
    print(f"[era5] vars (first 80): {vars_[:80]}")
    summarize_grid_like(b, ds, "era5", bbox=bbox)
    summarize_time(b, ds, "era5")
    summarize_data_field(b, ds, "era5", var_name=args.era5_var)
    close_netcdf(b, ds)

    # --- Optional: Zarr sample (already regridded ERA5-on-DANRA store) ---
    if args.zarr_sample:
        print("\n==================== ERA5 ZARR SAMPLE (regridded output) =====")
        summarize_zarr_store(
            args.zarr_sample,
            label="era5_zarr",
            n_samples=args.zarr_n,
            array_name_hint=args.zarr_hint,
        )

    print("\n[OK] Done.")

if __name__ == "__main__":
    main()