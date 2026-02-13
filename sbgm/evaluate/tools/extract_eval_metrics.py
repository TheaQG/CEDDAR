#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import csv
from typing import Any, Iterable


# -----------------------------
# helpers
# -----------------------------
def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row dicts (all values as strings)."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]

def _csv_columns(rows: list[dict[str, str]]) -> set[str]:
    cols: set[str] = set()
    for r in rows:
        cols.update(r.keys())
    return cols


def _safe_float(x):
    try:
        if x is None:
            return math.nan
        if isinstance(x, float):
            return x
        return float(x)
    except Exception:
        return math.nan


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return math.nan
    return float(sum(xs) / len(xs))


def _std(xs: Iterable[float], ddof: int = 1) -> float:
    xs = list(xs)
    n = len(xs)
    if n == 0:
        return math.nan
    if n - ddof <= 0:
        return math.nan
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / (n - ddof)
    return float(math.sqrt(var))


def _unique_count(rows: list[dict[str, str]], key: str) -> int:
    return len({r.get(key, "") for r in rows if r.get(key, "") != ""})


def _float_col(rows: list[dict[str, str]], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        if key in r:
            v = _safe_float(r.get(key))
            if not (isinstance(v, float) and math.isnan(v)):
                out.append(v)
    return out


def _filter_rows(rows: list[dict[str, str]], **eq: str) -> list[dict[str, str]]:
    """Return rows where each provided column equals the given string value."""
    out = []
    for r in rows:
        ok = True
        for k, v in eq.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _find_model_dirs(eval_root: Path, *, ignore_names: set[str]) -> list[Path]:
    """
    Discover model directories by looking for '<dir>/prcp' under eval_root/*.
    Skips non-directories and any name in ignore_names.
    """
    out = []
    for p in sorted(eval_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in ignore_names:
            continue
        # "real" model folders have prcp/ directly under them
        if (p / "prcp").is_dir():
            out.append(p)
    return out

# -----------------------------
# metric extractors
# -----------------------------
def extract_psd_highk(model_dir: Path) -> dict:
    """Extract PSD slope+intercept for the high-k range with mean+std across days.

    We only use tables that actually contain columns ['slope','intercept'].
    Your `scale_psd_summary.csv` may contain PSD *power* summaries (hr_highk, gen_highk, ...),
    which is NOT what we need for slope/intercept, so we ignore it unless it has slope/intercept.
    """
    tables_dir = model_dir / "prcp" / "scale" / "tables"

    # Candidate files, ordered by preference (per-day -> avg)
    candidates = [
        tables_dir / "scale_psd_slopes_summary.csv",
        tables_dir / "scale_psd_slopes_daily.csv",
        tables_dir / "scale_psd_slopes.csv",
        tables_dir / "scale_psd_summary.csv",     # only if it *actually* has slope/intercept
        tables_dir / "scale_psd_slopes_avg.csv",  # fallback
    ]

    path = None
    rows = None
    for p in candidates:
        if not p.exists():
            continue
        tmp_rows = _read_csv_rows(p)
        cols = _csv_columns(tmp_rows)
        if ("slope" in cols) and ("intercept" in cols):
            path = p
            rows = tmp_rows
            break

    if rows is None or path is None:
        raise FileNotFoundError(
            f"Could not find a PSD slopes table with columns ['slope','intercept'] under {tables_dir}. "
            f"Checked: {[c.name for c in candidates]}"
        )

    # Prefer ensemble-based series if available (B)
    series_preferred = ["GEN_ens", "GEN_ENS", "GEN_ENS_MEAN", "GEN", "ensmean"]
    available_series = {str(r.get("series", "")) for r in rows if r.get("series", "") != ""}

    series = None
    for s in series_preferred:
        if s in available_series:
            series = s
            break
    if series is None:
        gen_like = [s for s in available_series if "GEN" in s.upper()]
        series = gen_like[0] if gen_like else None
    if series is None:
        raise ValueError(f"No GEN-like series found in {path}. Available series={sorted(available_series)}")

    if "range" not in _csv_columns(rows):
        raise ValueError(f"Expected a 'range' column in {path} to select 'high-k' rows.")

    hk_rows = [r for r in rows if str(r.get("series", "")) == series and str(r.get("range", "")) == "high-k"]
    if not hk_rows:
        raise ValueError(f"No rows for series={series}, range='high-k' in {path}")

    if "date" in _csv_columns(hk_rows):
        slopes = [_safe_float(r.get("slope")) for r in hk_rows]
        icpts = [_safe_float(r.get("intercept")) for r in hk_rows]
        slope_mean = _mean(slopes)
        slope_std = _std(slopes, ddof=1)
        icpt_mean = _mean(icpts)
        icpt_std = _std(icpts, ddof=1)
        n = _unique_count(hk_rows, "date")
    else:
        r0 = hk_rows[0]
        slope_mean = _safe_float(r0.get("slope"))
        slope_std = _safe_float(r0.get("slope_std", math.nan))
        icpt_mean = _safe_float(r0.get("intercept"))
        icpt_std = _safe_float(r0.get("intercept_std", math.nan))
        n = len(hk_rows)

    return {
        "psd_slopes_file": str(path.name),
        "psd_series_used": series,
        "psd_slope_highk_mean": _safe_float(slope_mean),
        "psd_slope_highk_std": _safe_float(slope_std),
        "psd_intercept_highk_mean": _safe_float(icpt_mean),
        "psd_intercept_highk_std": _safe_float(icpt_std),
        "psd_n": n,
    }


def extract_iss_20km(model_dir: Path, thr_mm: float) -> dict:
    """
    From: prcp/scale/tables/scale_iss_ens_summary_uncertainty.csv
    We take mean+std at 20km for given threshold.
    """
    path = model_dir / "prcp" / "scale" / "tables" / "scale_iss_ens_summary_uncertainty.csv"
    rows = _read_csv_rows(path)

    # match threshold robustly
    thr_key = None
    for r in rows:
        v = _safe_float(r.get("thr_mm"))
        if abs(v - thr_mm) < 1e-6:
            thr_key = r
            break
    if thr_key is None:
        avail = sorted({_safe_float(r.get("thr_mm")) for r in rows})
        raise ValueError(f"Threshold {thr_mm} not found in {path}. Available={avail}")

    r = thr_key
    out = {
        f"iss20km_thr{thr_mm:g}_mean": _safe_float(r.get("iss_20km_mean")),
        f"iss20km_thr{thr_mm:g}_std": _safe_float(r.get("iss_20km_std")),
    }
    if "iss_20km_n" in r:
        out[f"iss20km_thr{thr_mm:g}_n"] = int(_safe_float(r.get("iss_20km_n")))
    else:
        out[f"iss20km_thr{thr_mm:g}_n"] = math.nan
    return out

def extract_crps(model_dir: Path) -> dict:
    """
    From: prcp/probabilistic/tables/prob_crps_daily.csv
    Need mean+std across days.
    """
    path = model_dir / "prcp" / "probabilistic" / "tables" / "prob_crps_daily.csv"
    rows = _read_csv_rows(path)
    cols = _csv_columns(rows)
    if "crps" not in cols:
        raise ValueError(f"'crps' column missing in {path}. Columns={sorted(cols)}")
    crps_vals = [_safe_float(r.get("crps")) for r in rows]
    return {
        "crps_mean": _safe_float(_mean(crps_vals)),
        "crps_std": _safe_float(_std(crps_vals, ddof=1)),
        "crps_n": int(len(crps_vals)),
    }


def extract_tails(model_dir: Path) -> dict:
    """
    From:
      prcp/extremes/tables/ext_tails.csv  (levels: HR/GEN/LR/GEN_ENS)
      prcp/probabilistic/tables/ext_tails_uncertainty.csv (std for GEN_ENS)
    You said: already calculated, just extract.
    We'll extract GEN_ENS if present, else GEN.

    Outputs:
      p95,p99,p99.9,p99.99,wet_freq,wet_hit_rate
      and (if available) stds for those.
    """
    tails_path = model_dir / "prcp" / "extremes" / "tables" / "ext_tails.csv"
    unc_path = model_dir / "prcp" / "probabilistic" / "tables" / "ext_tails_uncertainty.csv"

    rows = _read_csv_rows(tails_path)
    which_candidates = ["GEN_ENS", "GEN_ens", "GEN_ENS_MEAN", "GEN"]
    available = {str(r.get("which", "")) for r in rows if r.get("which", "") != ""}

    which = None
    for w in which_candidates:
        if w in available:
            which = w
            break
    if which is None:
        # fall back to any GEN-like row
        gen_like = [w for w in available if "GEN" in w.upper()]
        which = gen_like[0] if gen_like else None
    if which is None:
        raise ValueError(f"No GEN-like 'which' found in {tails_path}. Available={sorted(available)}")

    matches = [r for r in rows if str(r.get("which", "")) == which]
    if not matches:
        raise ValueError(f"No row for which='{which}' found in {tails_path}")
    r = matches[0]
    out = {
        "tails_which_used": which,
        "p95": _safe_float(r.get("P95")),
        "p99": _safe_float(r.get("P99")),
        "p999": _safe_float(r.get("P99.9")),
        "p9999": _safe_float(r.get("P99.99")),
        "wet_freq": _safe_float(r.get("wet_freq")),
        "wet_hit_rate": _safe_float(r.get("wet_hit_rate", math.nan)),
        "tails_n_days": int(_safe_float(r.get("n_days", math.nan))) if "n_days" in r else math.nan,
    }

    if unc_path.exists():
        du = _read_csv_rows(unc_path)
        # Try to find same `which`, else fall back to GEN_ENS if present
        du_row = [rr for rr in du if str(rr.get("which", "")) == which]
        if not du_row:
            du_row = [rr for rr in du if str(rr.get("which", "")) == "GEN_ENS"]
        if du_row:
            u = du_row[0]
            out.update({
                "p95_std": _safe_float(u.get("P95_std", math.nan)),
                "p99_std": _safe_float(u.get("P99_std", math.nan)),
                "p999_std": _safe_float(u.get("P99.9_std", math.nan)),
                "p9999_std": _safe_float(u.get("P99.99_std", math.nan)),
                "wet_freq_std": _safe_float(u.get("wet_freq_std", math.nan)),
                "wet_hit_rate_std": _safe_float(u.get("wet_hit_rate_std", math.nan)),
                "tails_n_members": int(_safe_float(u.get("n_members", math.nan))) if "n_members" in u else math.nan,
            })

    return out


def extract_yearly_sum(model_dir: Path, year: int) -> dict:
    """
    From: prcp/spatial/tables/spatial_summary.csv
    You want "Yearly sum (2017)".
    In your example, for group==2017:
      source == 'hr' has sum_mean and sum_std and sum_total
      source == 'ensmean' has sum_mean etc.

    Usually, for “model performance”, you want ensmean_vs_hr row:
      group==2017, source=='ensmean_vs_hr' and use sum_mean (bias) or sum_rmse etc.
    BUT your table column in the screenshot says "Yearly sum (2017)" (looks like absolute total?).

    So: I will extract:
      - hr_sum_total_2017  (sum_total for hr)
      - gen_sum_total_2017 (sum_total for ensmean if available else gen-like)
      - lr_sum_total_2017  (sum_total for lr)
      - gen_sum_bias_2017  (sum_bias for ensmean_vs_hr, if present)
    You can drop what you don’t want in post-processing.
    """
    path = model_dir / "prcp" / "spatial" / "tables" / "spatial_summary.csv"
    rows = _read_csv_rows(path)
    # group can be int-like; compare as string
    year_rows = [r for r in rows if str(r.get("group", "")) == str(year)]
    if not year_rows:
        raise ValueError(f"No group=={year} rows found in {path}")

    def _row(source: str):
        ms = [r for r in year_rows if str(r.get("source", "")) == source]
        return ms[0] if ms else None

    hr = _row("hr")
    lr = _row("lr")
    ensmean = _row("ensmean")
    ensmean_vs_hr = _row("ensmean_vs_hr")

    out = {}
    if hr is not None:
        out[f"hr_sum_total_{year}"] = _safe_float(hr.get("sum_total", math.nan))
        out[f"hr_sum_mean_{year}"] = _safe_float(hr.get("sum_mean", math.nan))
        out[f"hr_sum_std_{year}"] = _safe_float(hr.get("sum_std", math.nan))
    if lr is not None:
        out[f"lr_sum_total_{year}"] = _safe_float(lr.get("sum_total", math.nan))
        out[f"lr_sum_mean_{year}"] = _safe_float(lr.get("sum_mean", math.nan))
        out[f"lr_sum_std_{year}"] = _safe_float(lr.get("sum_std", math.nan))
    if ensmean is not None:
        out[f"gen_sum_total_{year}"] = _safe_float(ensmean.get("sum_total", math.nan))
        out[f"gen_sum_mean_{year}"] = _safe_float(ensmean.get("sum_mean", math.nan))
        out[f"gen_sum_std_{year}"] = _safe_float(ensmean.get("sum_std", math.nan))
    if ensmean_vs_hr is not None:
        out[f"gen_vs_hr_sum_bias_{year}"] = _safe_float(ensmean_vs_hr.get("sum_bias", math.nan))
        out[f"gen_vs_hr_sum_rmse_{year}"] = _safe_float(ensmean_vs_hr.get("sum_rmse", math.nan))
        out[f"gen_vs_hr_sum_ratio_{year}"] = _safe_float(ensmean_vs_hr.get("sum_ratio", math.nan))
    return out


def extract_all_metrics(model_dir: Path, year: int = 2017) -> dict:
    """
    Pull everything you listed, with minimal assumptions.
    """
    d = {"model": model_dir.name}

    # PSD slope/intercept high-k
    d.update(extract_psd_highk(model_dir))

    # ISS @20km for 1mm/day and 10mm/day (you asked specifically)
    d.update(extract_iss_20km(model_dir, thr_mm=1.0))
    d.update(extract_iss_20km(model_dir, thr_mm=10.0))

    # CRPS daily mean/std
    d.update(extract_crps(model_dir))

    # Percentiles + wet-day
    d.update(extract_tails(model_dir))

    # Yearly sum (2017)
    d.update(extract_yearly_sum(model_dir, year=year))

    return d


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", type=str, required=True,
                    help="Path to .../generated_samples/evaluation/")
    ap.add_argument("--model", type=str, default=None,
                    help="If set: only process this exact model folder name under eval_root.")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Where to write the consolidated metrics CSV.")
    ap.add_argument("--ignore", type=str, nargs="*", default=["baselines", "ablations1", "ablations2", "ablations3"],
                    help="Folder names under eval_root to ignore (only applies in --model=None scan mode).")
    ap.add_argument("--strict", action="store_true",
                    help="If set: raise on missing files. Otherwise: fill NaN and keep going.")
    ap.add_argument("--year", type=int, default=2017,
                help="Year to extract from spatial_summary.csv (group column).")
    args = ap.parse_args()

    eval_root = Path(args.eval_root).expanduser().resolve()
    if not eval_root.is_dir():
        raise FileNotFoundError(f"--eval_root not found: {eval_root}")

    ignore_names = set(args.ignore or [])

    if args.model is not None:
        model_dirs = [eval_root / args.model]
        if not model_dirs[0].is_dir():
            raise FileNotFoundError(f"Model folder not found: {model_dirs[0]}")
        if not (model_dirs[0] / "prcp").is_dir():
            raise FileNotFoundError(f"Model folder has no prcp/: {model_dirs[0]}")
    else:
        model_dirs = _find_model_dirs(eval_root, ignore_names=ignore_names)

    rows = []
    errors = []
    for md in model_dirs:
        try:
            rows.append(extract_all_metrics(md, year=int(args.year)))
        except Exception as e:
            msg = f"{md.name}: {type(e).__name__}: {e}"
            if args.strict:
                raise
            errors.append(msg)
            rows.append({"model": md.name, "error": msg})

    # Build union of keys across rows for a stable CSV header
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())

    # Put model first, then the rest sorted for stability
    header = ["model"] + sorted([k for k in keys if k != "model"])

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            # Ensure missing keys are blank
            w.writerow({k: r.get(k, "") for k in header})

    if errors:
        err_path = out_csv.with_suffix(".errors.txt")
        err_path.write_text("\n".join(errors) + "\n")
        print(f"[WARN] Completed with {len(errors)} model(s) having missing/failed extractions.")
        print(f"[WARN] Wrote error log to: {err_path}")

    print(f"[OK] Wrote metrics table to: {out_csv}")
    print(f"[OK] Processed {len(model_dirs)} model(s).")


if __name__ == "__main__":
    main()