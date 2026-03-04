#!/usr/bin/env python3
"""
Test ERA5 → DANRA regridding using CDO.

Produces useful artifacts even if matplotlib is not available:
- day-0 NPZ (data array)
- text summary (min/max/mean + NaN fraction)
- PGM image (portable graymap) as a lightweight "plot" substitute
"""

import os
import subprocess
import pathlib
import numpy as np
import netCDF4 as nc
import re
from typing import Optional, Tuple, Dict

# Optional plotting: matplotlib might not be installed in the LUMI micromamba env.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

ERA5_TMP_DIR = pathlib.Path(os.environ["ERA5_TMP_DIR"])

VAR = os.environ.get("TEST_VAR", "cape")
YEAR = os.environ.get("TEST_YEAR", "1994")

RAW_FILE = ERA5_TMP_DIR / "raw" / VAR / f"{VAR}_{YEAR}.nc"
GRID_FILE = ERA5_TMP_DIR / "grid" / "mygrid_danra_small"
WEIGHTS_FILE = ERA5_TMP_DIR / "weights" / "NorthAtlantic_ERA5_to_DANRA_bil_weights.nc"

TEST_DIR = ERA5_TMP_DIR / "test_regrid"
TEST_DIR.mkdir(exist_ok=True)

DAILY_FILE = TEST_DIR / f"{VAR}_{YEAR}_daily_test.nc"
REGRID_FILE = TEST_DIR / f"{VAR}_{YEAR}_DANRA_test.nc"

# Outputs
FIG_FILE = TEST_DIR / f"{VAR}_{YEAR}_DANRA_test.png"
NPZ_FILE = TEST_DIR / f"{VAR}_{YEAR}_DANRA_test_day0.npz"
PGM_FILE = TEST_DIR / f"{VAR}_{YEAR}_DANRA_test_day0.pgm"
TXT_SUMMARY = TEST_DIR / f"{VAR}_{YEAR}_DANRA_test_summary.txt"


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)
def run_capture(cmd) -> str:
    """Run command and capture stdout as text (stderr merged)."""
    print("\n>>>", " ".join(map(str, cmd)))
    p = subprocess.run(
        cmd, check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return p.stdout


def cdo_griddes(path: pathlib.Path) -> str:
    return run_capture(["cdo", "-s", "griddes", str(path)])


def parse_griddes(griddes_txt: str) -> Dict[str, str]:
    """
    Parse key = value lines from `cdo griddes` into a dict (strings).
    """
    d: Dict[str, str] = {}
    for line in griddes_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def lonlat_bounds_from_griddes(d: Dict[str, str]) -> Optional[Tuple[float, float, float, float]]:
    """
    If gridtype=lonlat and xfirst/xinc/xsize + yfirst/yinc/ysize exist,
    return (lon_min, lon_max, lat_min, lat_max). Else None.
    """
    if d.get("gridtype") != "lonlat":
        return None
    needed = ["xfirst", "xinc", "xsize", "yfirst", "yinc", "ysize"]
    if not all(k in d for k in needed):
        return None

    xfirst = float(d["xfirst"])
    xinc = float(d["xinc"])
    xsize = int(d["xsize"])
    yfirst = float(d["yfirst"])
    yinc = float(d["yinc"])
    ysize = int(d["ysize"])

    lon0 = xfirst
    lon1 = xfirst + xinc * (xsize - 1)
    lat0 = yfirst
    lat1 = yfirst + yinc * (ysize - 1)

    lon_min, lon_max = (min(lon0, lon1), max(lon0, lon1))
    lat_min, lat_max = (min(lat0, lat1), max(lat0, lat1))
    return lon_min, lon_max, lat_min, lat_max


def xy_bounds_from_griddes(d: Dict[str, str]) -> Optional[Tuple[float, float, float, float]]:
    """
    For projected grids (often gridtype=projection), best-effort x/y bounds
    using xfirst/xinc/xsize and yfirst/yinc/ysize if present.
    Returns (x_min, x_max, y_min, y_max) or None.
    """
    needed = ["xfirst", "xinc", "xsize", "yfirst", "yinc", "ysize"]
    if not all(k in d for k in needed):
        return None

    xfirst = float(d["xfirst"])
    xinc = float(d["xinc"])
    xsize = int(d["xsize"])
    yfirst = float(d["yfirst"])
    yinc = float(d["yinc"])
    ysize = int(d["ysize"])

    x0 = xfirst
    x1 = xfirst + xinc * (xsize - 1)
    y0 = yfirst
    y1 = yfirst + yinc * (ysize - 1)

    x_min, x_max = (min(x0, x1), max(x0, x1))
    y_min, y_max = (min(y0, y1), max(y0, y1))
    return x_min, x_max, y_min, y_max


def fmt_lonlat(bounds: Optional[Tuple[float, float, float, float]]) -> str:
    if bounds is None:
        return "(lon/lat bounds unavailable)"
    lo0, lo1, la0, la1 = bounds
    return f"lon=[{lo0:.3f}, {lo1:.3f}]  lat=[{la0:.3f}, {la1:.3f}]"


def fmt_xy(bounds: Optional[Tuple[float, float, float, float]]) -> str:
    if bounds is None:
        return "(x/y bounds unavailable)"
    x0, x1, y0, y1 = bounds
    return f"x=[{x0:.3f}, {x1:.3f}]  y=[{y0:.3f}, {y1:.3f}]"


def check_grid_coverage():
    """
    Print grid diagnostics for ERA5 source and DANRA target grid.
    If both grids are lonlat, warn if target is outside source coverage.
    """
    print("\n==== GRID COVERAGE CHECKS ====")

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"RAW_FILE does not exist: {RAW_FILE}")
    if not GRID_FILE.exists():
        raise FileNotFoundError(f"GRID_FILE does not exist: {GRID_FILE}")

    era5_txt = cdo_griddes(RAW_FILE)
    tgt_txt = cdo_griddes(GRID_FILE)

    era5 = parse_griddes(era5_txt)
    tgt = parse_griddes(tgt_txt)

    era5_ll = lonlat_bounds_from_griddes(era5)
    tgt_ll = lonlat_bounds_from_griddes(tgt)

    print("\n[ERA5] gridtype:", era5.get("gridtype"))
    print("[ERA5] size:", era5.get("xsize"), "x", era5.get("ysize"))
    print("[ERA5] lon/lat:", fmt_lonlat(era5_ll))

    print("\n[DANRA target] gridtype:", tgt.get("gridtype"))
    print("[DANRA target] size:", tgt.get("xsize"), "x", tgt.get("ysize"))
    if tgt_ll is not None:
        print("[DANRA target] lon/lat:", fmt_lonlat(tgt_ll))
    else:
        # target is likely projection; show x/y extents instead
        print("[DANRA target] lon/lat: (not lonlat; likely projected grid)")
        print("[DANRA target] x/y:", fmt_xy(xy_bounds_from_griddes(tgt)))

    # Only do strict lon/lat overlap check when both are lonlat
    if era5_ll is not None and tgt_ll is not None:
        e_lo0, e_lo1, e_la0, e_la1 = era5_ll
        t_lo0, t_lo1, t_la0, t_la1 = tgt_ll

        lon_overlap = not (t_lo1 < e_lo0 or t_lo0 > e_lo1)
        lat_overlap = not (t_la1 < e_la0 or t_la0 > e_la1)

        if not (lon_overlap and lat_overlap):
            print("\nWARNING: target lon/lat bounds do NOT overlap ERA5 bounds.")
            print("         This will produce lots of NaNs after regridding.")
        else:
            extends = []
            if t_lo0 < e_lo0: extends.append("west")
            if t_lo1 > e_lo1: extends.append("east")
            if t_la0 < e_la0: extends.append("south")
            if t_la1 > e_la1: extends.append("north")
            if extends:
                print("\nWARNING: target grid extends beyond ERA5 coverage to the:", ", ".join(extends))
                print("         Expect NaNs near edges.")
            else:
                print("\nOK: target bounds are fully within ERA5 bounds (lonlat check).")
    else:
        print("\nNOTE: strict lon/lat overlap check not possible because at least one grid is not lonlat.")
        print("      If target is projected, NaNs can still indicate insufficient ERA5 spatial coverage.")

    print("==== END GRID COVERAGE CHECKS ====\n")

def find_main_variable(ds):
    # Find a “main field” variable (time,y,x) or (time,level,y,x)
    for name, var in ds.variables.items():
        if len(var.dimensions) >= 3:
            return name
    raise RuntimeError("No data variable found")

def _as_float_array(a):
    # netCDF4 may return masked arrays; make a plain float array with NaNs where masked
    try:
        a = np.asarray(a.filled(np.nan), dtype=np.float32)
    except Exception:
        a = np.asarray(a, dtype=np.float32)
    return a

def write_summary(var_name: str, data2d: np.ndarray, out_path: pathlib.Path):
    nonfinite = int(np.sum(~np.isfinite(data2d)))
    frac = float(nonfinite / data2d.size)
    finite = data2d[np.isfinite(data2d)]
    vmin = float(np.min(finite)) if finite.size else float("nan")
    vmax = float(np.max(finite)) if finite.size else float("nan")
    mean = float(np.mean(finite)) if finite.size else float("nan")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"var={var_name}\n")
        f.write(f"shape={data2d.shape}\n")
        f.write(f"nonfinite={nonfinite}\n")
        f.write(f"fraction_nonfinite={frac}\n")
        f.write(f"min={vmin}\n")
        f.write(f"max={vmax}\n")
        f.write(f"mean={mean}\n")

def write_pgm(data2d: np.ndarray, out_path: pathlib.Path):
    """
    Write a portable graymap (PGM, P5) image without external deps.
    Non-finite values become black. Finite values are linearly normalized to 0..255.
    """
    a = data2d.copy()
    mask = ~np.isfinite(a)
    finite = a[~mask]
    if finite.size == 0:
        img = np.zeros(a.shape, dtype=np.uint8)
    else:
        lo = np.percentile(finite, 1)
        hi = np.percentile(finite, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite)) if float(np.max(finite)) > float(np.min(finite)) else float(np.min(finite)) + 1.0
        a = np.clip(a, lo, hi)
        a = (a - lo) / (hi - lo)
        a[mask] = 0.0
        img = (a * 255.0).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = img.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(img.tobytes(order="C"))


# --------------------------------------------------
# 1 generate weights
# --------------------------------------------------

def generate_weights():
    if WEIGHTS_FILE.exists():
        print("Using existing weights:", WEIGHTS_FILE)
        return

    print("Generating weights file")
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    run([
        "cdo",
        f"genbil,{GRID_FILE}",
        str(RAW_FILE),
        str(WEIGHTS_FILE)
    ])


# --------------------------------------------------
# 2 convert hourly → daily
# --------------------------------------------------

def compute_daily():
    if DAILY_FILE.exists():
        print("Daily test file already exists:", DAILY_FILE)
        return

    print("Computing daily statistic (daymean)")
    run([
        "cdo",
        "daymean",
        str(RAW_FILE),
        str(DAILY_FILE)
    ])


# --------------------------------------------------
# 3 regrid
# --------------------------------------------------

def regrid():
    print("Regridding")
    run([
        "cdo",
        f"remap,{GRID_FILE},{WEIGHTS_FILE}",
        str(DAILY_FILE),
        str(REGRID_FILE)
    ])


# --------------------------------------------------
# 4 check NaNs + save artifacts
# --------------------------------------------------

def check_data_and_save():
    ds = nc.Dataset(REGRID_FILE)
    var_name = find_main_variable(ds)
    data = _as_float_array(ds.variables[var_name][0])

    nonfinite = np.sum(~np.isfinite(data))
    frac = nonfinite / data.size

    print("\nVariable:", var_name)
    print("Shape:", data.shape)
    print("Nonfinite:", int(nonfinite))
    print("Fraction:", float(frac))

    if frac > 0.01:
        print("\nWARNING: large number of NaNs")
        print("         This often means the ERA5 source area does not fully cover the DANRA target grid")
        print("         (or the target grid/projection extends beyond the ERA5 domain).")
        print("         See GRID COVERAGE CHECKS above.")
    else:
        print("\nNaN check passed")

    # Save artifacts
    np.savez_compressed(NPZ_FILE, data=data)
    write_summary(var_name, data, TXT_SUMMARY)

    print("Saved day-0 NPZ :", NPZ_FILE)
    print("Saved summary   :", TXT_SUMMARY)

    ds.close()
    return var_name, data


# --------------------------------------------------
# 5 “plot”: PNG if possible, else PGM
# --------------------------------------------------

def make_plot(var_name: str, data: np.ndarray):
    if HAVE_MPL:
        plt.figure(figsize=(8, 6))
        plt.imshow(data, origin="lower")
        plt.colorbar(label=var_name)
        plt.title(f"{VAR} ERA5 → DANRA (day 0)")
        plt.tight_layout()
        plt.savefig(FIG_FILE, dpi=200)
        print("Saved PNG figure:", FIG_FILE)
    else:
        write_pgm(data, PGM_FILE)
        print("matplotlib not available; saved PGM image:", PGM_FILE)


def main():
    print("\n==== ERA5 REGRID TEST ====")
    print("ERA5_TMP_DIR:", ERA5_TMP_DIR)
    print("Variable:", VAR)
    print("Year:", YEAR)
    print("Matplotlib available:", HAVE_MPL)
    check_grid_coverage()

    generate_weights()
    compute_daily()
    regrid()
    var_name, data = check_data_and_save()
    make_plot(var_name, data)

    print("\nOutputs:")
    print("  Regridded NC :", REGRID_FILE)
    print("  Day-0 NPZ    :", NPZ_FILE)
    print("  Summary      :", TXT_SUMMARY)
    if HAVE_MPL:
        print("  PNG figure   :", FIG_FILE)
    else:
        print("  PGM image    :", PGM_FILE)

    print("\nTest finished")


if __name__ == "__main__":
    main()