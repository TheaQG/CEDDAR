from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import csv
import json
import math
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compact sampler-grid summarizer: ONLY the metrics you use in the paper.
# ---------------------------------------------------------------------------

@dataclass
class SamplerGridRow:
    sampler_id: str
    rho: float
    Schurn: float
    sigma_scale: float

    # Structure / scale
    psd_slope_highk: Optional[float] = None
    psd_intercept_highk: Optional[float] = None
    iss20km_thr10: Optional[float] = None

    # Probabilistic
    crps_mean: Optional[float] = None

    # Tails
    p99: Optional[float] = None
    p999: Optional[float] = None
    wet_freq: Optional[float] = None

    # Spatial yearly-sum stats
    yearly_sum_mean: Optional[float] = None
    yearly_sum_std: Optional[float] = None


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return math.nan
        if isinstance(x, float):
            return x
        s = str(x).strip()
        if s == "":
            return math.nan
        return float(s)
    except Exception:
        return math.nan


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _cols(rows: List[Dict[str, str]]) -> set[str]:
    c: set[str] = set()
    for r in rows:
        c.update(r.keys())
    return c


def _first(rows: List[Dict[str, str]], **eq: str) -> Optional[Dict[str, str]]:
    for r in rows:
        ok = True
        for k, v in eq.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            return r
    return None


def _mean(vals: List[float]) -> float:
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def _parse_sampler_name(name: str) -> Tuple[float, float, float]:
    """Folder name: rho=5.00_Schurn=2.00_sigscale=0.90"""
    rho = schurn = sigscale = math.nan
    for part in name.split("_"):
        if part.startswith("rho="):
            rho = _safe_float(part.split("=", 1)[1])
        elif part.startswith("Schurn="):
            schurn = _safe_float(part.split("=", 1)[1])
        elif part.startswith("sigscale=") or part.startswith("sigma_scale="):
            sigscale = _safe_float(part.split("=", 1)[1])
    return rho, schurn, sigscale


def _pick_gen_label(available: set[str], preferred: List[str]) -> Optional[str]:
    for p in preferred:
        if p in available:
            return p
    # fallback: anything containing GEN
    gen_like = [x for x in available if "GEN" in x.upper()]
    return gen_like[0] if gen_like else None


# -----------------------------
# Metric extractors (GEN/ENS only)
# -----------------------------
def extract_psd_highk(prcp_root: Path) -> Tuple[Optional[float], Optional[float]]:
    tables_dir = prcp_root / "scale" / "tables"
    candidates = [
        tables_dir / "scale_psd_slopes_summary.csv",
        tables_dir / "scale_psd_slopes_daily.csv",
        tables_dir / "scale_psd_slopes.csv",
        tables_dir / "scale_psd_summary.csv",      # only if has slope/intercept
        tables_dir / "scale_psd_slopes_avg.csv",
    ]

    rows: List[Dict[str, str]] = []
    used = None
    for p in candidates:
        rr = _read_csv(p)
        if not rr:
            continue
        if {"slope", "intercept"}.issubset(_cols(rr)):
            rows = rr
            used = p
            break
    if not rows:
        logger.warning(f"[core] PSD slopes not found under {tables_dir}")
        return None, None

    if "range" not in _cols(rows):
        logger.warning(f"[core] PSD table missing 'range': {used}")
        return None, None

    avail = {str(r.get("series", "")) for r in rows if r.get("series", "")}
    series = _pick_gen_label(avail, ["GEN_ENS", "GEN_ens", "GEN_ENS_MEAN", "GEN", "ensmean"])
    if series is None:
        logger.warning(f"[core] PSD table: no GEN-like series in {used}")
        return None, None

    hk = [r for r in rows if str(r.get("series", "")) == series and str(r.get("range", "")) == "high-k"]
    if not hk:
        logger.warning(f"[core] PSD table: no high-k rows for series={series} in {used}")
        return None, None

    r0 = hk[0]
    return _safe_float(r0.get("slope")), _safe_float(r0.get("intercept"))


def extract_iss20km_thr10(prcp_root: Path) -> Optional[float]:
    path = prcp_root / "scale" / "tables" / "scale_iss_ens_summary_uncertainty.csv"
    rows = _read_csv(path)
    if not rows:
        logger.warning(f"[core] Missing ISS table: {path}")
        return None
    for r in rows:
        if abs(_safe_float(r.get("thr_mm")) - 10.0) < 1e-6:
            return _safe_float(r.get("iss_20km_mean"))
    logger.warning(f"[core] thr_mm=10 not found in {path}")
    return None


def extract_crps_mean(prcp_root: Path) -> Optional[float]:
    path = prcp_root / "probabilistic" / "tables" / "prob_crps_daily.csv"
    rows = _read_csv(path)
    if not rows:
        logger.warning(f"[core] Missing CRPS table: {path}")
        return None
    if "crps" not in _cols(rows):
        logger.warning(f"[core] CRPS table missing 'crps' column: {path}")
        return None
    vals = [_safe_float(r.get("crps")) for r in rows]
    m = _mean(vals)
    return None if math.isnan(m) else m


def extract_tails(prcp_root: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    path = prcp_root / "extremes" / "tables" / "ext_tails.csv"
    rows = _read_csv(path)
    if not rows:
        logger.warning(f"[core] Missing tails table: {path}")
        return None, None, None

    avail = {str(r.get("which", "")).upper() for r in rows if r.get("which", "")}
    which = _pick_gen_label(avail, ["GEN_ENS", "GEN_ENS_MEAN", "GEN"])
    if which is None:
        logger.warning(f"[core] No GEN-like 'which' found in {path}")
        return None, None, None

    r0 = None
    for r in rows:
        if str(r.get("which", "")).upper() == which:
            r0 = r
            break
    if r0 is None:
        return None, None, None

    return _safe_float(r0.get("P99")), _safe_float(r0.get("P99.9")), _safe_float(r0.get("wet_freq"))


def extract_yearly_sum(prcp_root: Path, year: int) -> Tuple[Optional[float], Optional[float]]:
    path = prcp_root / "spatial" / "tables" / "spatial_summary.csv"
    rows = _read_csv(path)
    if not rows:
        logger.warning(f"[core] Missing spatial_summary.csv: {path}")
        return None, None

    year_rows = [r for r in rows if str(r.get("group", "")) == str(year)]
    if not year_rows:
        logger.warning(f"[core] No group=={year} rows in {path}")
        return None, None

    r_ens = _first(year_rows, source="ensmean")
    if r_ens is None:
        logger.warning(f"[core] No source='ensmean' row for year={year} in {path}")
        return None, None

    return _safe_float(r_ens.get("sum_mean")), _safe_float(r_ens.get("sum_std"))


# -----------------------------
# Main driver
# -----------------------------
def summarize_sampler_grid(
    sampler_grid_root: str | Path,
    *,
    year: int = 2017,
    out_csv: Optional[str | Path] = None,
    out_json: Optional[str | Path] = None,
) -> tuple[Path, Path]:
    sampler_grid_root = Path(sampler_grid_root)

    out_csv = Path(out_csv) if out_csv else sampler_grid_root / "sampler_grid_core_metrics.csv"
    out_json = Path(out_json) if out_json else sampler_grid_root / "sampler_grid_core_metrics.json"

    rows_out: List[SamplerGridRow] = []

    for combo_dir in sorted(sampler_grid_root.iterdir()):
        if not combo_dir.is_dir():
            continue
        if combo_dir.name.startswith(".") or combo_dir.name.lower() in {"old"}:
            continue

        prcp_root = combo_dir / "prcp"
        if not prcp_root.exists():
            logger.warning(f"[core] No prcp/ in {combo_dir}, skipping")
            continue

        rho, schurn, sigscale = _parse_sampler_name(combo_dir.name)
        row = SamplerGridRow(
            sampler_id=combo_dir.name,
            rho=rho,
            Schurn=schurn,
            sigma_scale=sigscale,
        )

        row.psd_slope_highk, row.psd_intercept_highk = extract_psd_highk(prcp_root)
        row.iss20km_thr10 = extract_iss20km_thr10(prcp_root)
        row.crps_mean = extract_crps_mean(prcp_root)
        row.p99, row.p999, row.wet_freq = extract_tails(prcp_root)
        row.yearly_sum_mean, row.yearly_sum_std = extract_yearly_sum(prcp_root, year=year)

        rows_out.append(row)

    if not rows_out:
        raise RuntimeError(f"[core] No sampler combos found under {sampler_grid_root}")

    records: List[Dict[str, Any]] = [asdict(r) for r in rows_out]

    fieldnames = [
        "sampler_id",
        "rho",
        "Schurn",
        "sigma_scale",
        "psd_slope_highk",
        "psd_intercept_highk",
        "iss20km_thr10",
        "crps_mean",
        "p99",
        "p999",
        "wet_freq",
        "yearly_sum_mean",
        "yearly_sum_std",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            w.writerow(rec)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"[core] Wrote CSV : {out_csv}")
    logger.info(f"[core] Wrote JSON: {out_json}")
    return out_csv, out_json


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser(description="Summarize sampler-grid core metrics into a compact table.")
    ap.add_argument("--sampler_grid_root", required=True,
                    help="Path to .../generated_samples/evaluation/<MODEL_KEY>/sampler_grid")
    ap.add_argument("--year", type=int, default=2017,
                    help="Year (spatial_summary.csv group) to use for annual-sum mean/std.")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional output CSV path (default in sampler_grid_root).")
    ap.add_argument("--out_json", type=str, default=None,
                    help="Optional output JSON path (default in sampler_grid_root).")

    args = ap.parse_args()
    summarize_sampler_grid(
        sampler_grid_root=args.sampler_grid_root,
        year=int(args.year),
        out_csv=args.out_csv,
        out_json=args.out_json,
    )