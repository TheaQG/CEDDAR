

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class TargetSpec:
    kind: str  # zero | one | hr
    ref_metric: Optional[str] = None
    spread_metric: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Paper 2 summary figures from evaluation summary JSON files.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--eval-root", type=str, default=None, help="Override evaluation root from config.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir from config.")
    parser.add_argument("--model-prefixes", nargs="*", default=None, help="Optional override for model prefixes.")
    parser.add_argument("--baseline-prefix", type=str, default=None, help="Optional override for baseline prefix.")
    parser.add_argument("--baseline-seed-prefixes", nargs="*", default=None, help="Optional override for baseline seed prefixes.")
    return parser.parse_args()


def _mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return cfg


def _flatten_dict(d: Mapping[str, Any], parent_key: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out


def _to_numeric_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _sanitize_name(name: str) -> str:
    return str(name).replace(".", "__").replace("/", "_").replace(" ", "_")


def _short_run_label(run_id: str) -> str:
    rid = str(run_id)
    marker = "__HR_"
    if marker in rid:
        return rid.split(marker, 1)[0]
    return rid


def _is_seed_token(token: str) -> bool:
    tok = str(token).strip().lower()
    return tok.startswith("seed")


def _extract_run_signature(run_id: str) -> str:
    rid = str(run_id).strip()
    if not rid:
        return rid
    parts = rid.split("__")
    if len(parts) >= 2 and _is_seed_token(parts[1]):
        return "__".join([parts[0]] + parts[2:])
    return rid


def _normalize_per_metric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = _to_numeric_series(out[col])
        finite = s.dropna()
        if finite.empty:
            out[col] = s
            continue
        vmin = float(finite.min())
        vmax = float(finite.max())
        if math.isclose(vmin, vmax):
            out[col] = 0.5
        else:
            out[col] = (s - vmin) / (vmax - vmin)
    return out


# --- Additional helpers for better-than-baseline heatmap extensions ---

def _symmetric_scale_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    finite = s.dropna()
    if finite.empty:
        return s
    max_abs = float(np.nanmax(np.abs(finite.values)))
    if max_abs <= 0.0 or not math.isfinite(max_abs):
        return pd.Series(0.5, index=s.index, dtype=float)
    return 0.5 + 0.5 * (s / max_abs)


def _pillar_for_metric(metric_name: str, cfg: Mapping[str, Any]) -> Optional[str]:
    m = str(metric_name)
    for pillar_name in _pillar_names(cfg):
        if m in _metrics_for_pillar(cfg, pillar_name):
            return pillar_name
    return None

def _heatmap_cmap(cfg: Mapping[str, Any], key: str, fallback: str) -> str:
    cmaps = ((cfg.get("plots", {}) or {}).get("colormaps", {}) or {})
    return str(cmaps.get(key, fallback))


def _pillar_figsize(cfg: Mapping[str, Any], n_panels: int, ncols: int = 2) -> Tuple[float, float]:
    base = tuple(((cfg.get("plots", {}) or {}).get("figsize", {}) or {}).get("pillar_panel", [14, 9]))
    base_w, base_h = float(base[0]), float(base[1])
    nrows = max(1, math.ceil(max(1, n_panels) / max(1, ncols)))
    width = max(base_w, 6.2 * ncols)
    height = max(4.2 * nrows, base_h * (nrows / 2.0))
    return width, height

def build_target_registry(cfg: Mapping[str, Any]) -> Dict[str, TargetSpec]:
    registry: Dict[str, TargetSpec] = {}
    metrics_cfg = cfg.get("metrics", {}) or {}
    target_rules = metrics_cfg.get("target_rules", {}) or {}
    hr_map = metrics_cfg.get("hr_reference_map", {}) or {}

    for metric in target_rules.get("zero", []) or []:
        registry[str(metric)] = TargetSpec(kind="zero")
    for metric in target_rules.get("one", []) or []:
        registry[str(metric)] = TargetSpec(kind="one")
    for metric in target_rules.get("hr_reference", []) or []:
        registry[str(metric)] = TargetSpec(kind="hr", ref_metric=str(hr_map.get(str(metric), "") or ""))
    return registry


def metric_display_name(metric: str) -> str:
    m = str(metric)
    return m.split(".", 1)[1] if "." in m else m


def find_summary_files(eval_root: str | Path, model_prefixes: Optional[Sequence[str]], recursive: bool) -> List[Path]:
    root = Path(eval_root)
    if not root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {root}")

    pattern = "**/summary/evaluation_summary.json" if recursive else "*/summary/evaluation_summary.json"
    files = sorted(root.glob(pattern))
    if not model_prefixes:
        return files

    prefixes = [str(p) for p in model_prefixes]
    selected: List[Path] = []
    for fp in files:
        model_dir = fp.parent.parent.name
        if any(model_dir.startswith(p) for p in prefixes):
            selected.append(fp)
    return selected


def load_summary_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_summary(summary_json: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_id": summary_json.get("run_id"),
        "model_key": summary_json.get("model_key"),
        "eval_root": summary_json.get("eval_root"),
        "generated_root": summary_json.get("generated_root"),
        "config_path": summary_json.get("config_path"),
    }
    metadata = summary_json.get("metadata", {}) or {}
    for key, value in metadata.items():
        row[f"metadata.{key}"] = value

    pillars = summary_json.get("pillars", {}) or {}
    for pillar_name, pillar_payload in pillars.items():
        metrics = (pillar_payload or {}).get("metrics", {}) or {}
        row.update(_flatten_dict(metrics, pillar_name))
        notes = (pillar_payload or {}).get("notes", []) or []
        row[f"{pillar_name}.notes"] = " | ".join(str(x) for x in notes)

    rid = str(row.get("run_id") or row.get("model_key") or "")
    row["run_signature"] = _extract_run_signature(rid)
    row["short_label"] = _short_run_label(rid)
    return row


def is_physics_no_prcp(run_id: str) -> bool:
    rid = str(run_id)
    short = _short_run_label(rid)
    if not short.startswith("D_"):
        return False
    if "__HR_" in rid:
        left = rid.split("__HR_", 1)[0]
    else:
        left = short
    if not left.startswith("D_"):
        return False
    between = left[len("D_"):]
    return "P" not in between


def classify_group(run_id: str) -> str:
    rid = str(run_id)
    short = _short_run_label(rid)

    if rid == "baseline_mean" or short == "V0_mean":
        return "baseline_mean"
    if short == "V0":
        return "baseline"
    if short.startswith("V0__"):
        return "baseline"
    if short.startswith("C_"):
        return "context_only"
    if short.startswith("V0_") and short.endswith("_0"):
        return "single_var_small"
    if short.startswith("V0_") and short.endswith("_1"):
        return "single_var_large"
    if is_physics_no_prcp(run_id):
        return "physics_no_prcp"
    if short.startswith("D_"):
        return "physics"
    return "other"


def build_group_order_map(cfg: Mapping[str, Any]) -> Dict[str, int]:
    order = list((cfg.get("grouping", {}) or {}).get("order", []) or [])
    return {name: i for i, name in enumerate(order)}


def build_within_group_order_map(cfg: Mapping[str, Any]) -> Dict[str, Dict[str, int]]:
    ordering_cfg = cfg.get("ordering", {}) or {}
    out: Dict[str, Dict[str, int]] = {}
    for group_name, values in ordering_cfg.items():
        vals = [str(v) for v in (values or [])]
        out[str(group_name)] = {name: i for i, name in enumerate(vals)}
    return out


def sort_rows_by_group(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    group_order = build_group_order_map(cfg)
    within_order = build_within_group_order_map(cfg)
    tmp = df.copy()
    tmp["_group_rank"] = tmp["model_group"].astype(str).map(lambda g: group_order.get(g, 999))
    tmp["_within_rank"] = [
        within_order.get(str(g), {}).get(str(s), 999)
        for g, s in zip(tmp["model_group"].astype(str), tmp["short_label"].astype(str))
    ]
    tmp["_within_label"] = tmp["short_label"].astype(str)
    tmp = tmp.sort_values(["_group_rank", "_within_rank", "_within_label"], kind="stable")
    return tmp.drop(columns=["_group_rank", "_within_rank", "_within_label"])


def attach_grouping(flat_df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    df = flat_df.copy()
    df["model_group"] = df["run_id"].astype(str).map(classify_group)

    exclude_groups = {
        str(x) for x in ((cfg.get("grouping", {}) or {}).get("exclude_groups", []) or [])
    }
    if exclude_groups:
        df = df.loc[~df["model_group"].astype(str).isin(exclude_groups)].copy()

    return sort_rows_by_group(df, cfg)


# --- Seed aggregation: aggregate multiple seeds per run_signature if enabled in config ---
def aggregate_seeds(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """
    Aggregate multiple seeds per run_signature into mean (+ optional std columns).
    Controlled via cfg["seeds"]["mode"]: "none" | "mean".
    """
    seeds_cfg = cfg.get("seeds", {}) or {}
    mode = str(seeds_cfg.get("mode", "none"))

    if mode != "mean":
        return df

    group_cols = ["run_signature"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols = [c for c in df.columns if c not in numeric_cols and c not in group_cols]

    grouped = df.groupby(group_cols, dropna=False)

    mean_df = grouped[numeric_cols].mean().reset_index()
    std_df = grouped[numeric_cols].std().add_suffix("_std").reset_index()

    # Keep representative metadata (first occurrence)
    meta_df = grouped[meta_cols].first().reset_index()

    out = meta_df.merge(mean_df, on="run_signature", how="left")
    out = out.merge(std_df, on="run_signature", how="left")

    # Recompute short_label to reflect aggregation
    out["short_label"] = out["run_signature"].astype(str).map(_short_run_label)

    return out


def identify_baseline_runs(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    baseline_cfg = cfg.get("baseline", {}) or {}
    prefix = str(baseline_cfg.get("prefix", "") or "").strip()
    seed_prefixes = [str(x).strip() for x in baseline_cfg.get("seed_prefixes", []) or [] if str(x).strip()]
    selectors = list(dict.fromkeys(([prefix] if prefix else []) + seed_prefixes))
    if not selectors:
        return df.iloc[0:0].copy()

    sig = df["run_signature"].astype(str)
    rid = df["run_id"].astype(str)
    mask = pd.Series(False, index=df.index)
    for sel in selectors:
        mask = mask | sig.str.startswith(sel) | rid.str.startswith(sel)
    out = df.loc[mask].copy()

    baseline_order = build_within_group_order_map(cfg).get("baseline", {})
    if not out.empty and baseline_order:
        out["_baseline_rank"] = out["short_label"].astype(str).map(lambda x: baseline_order.get(x, 999))
        out = out.sort_values(["_baseline_rank", "short_label"], kind="stable").drop(columns=["_baseline_rank"])
    return out

def identify_primary_baseline_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    mask = df["short_label"].astype(str) == "V0"
    return df.loc[mask].copy()


def identify_v0_seed_family(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.iloc[0:0].copy()
    mask = df["short_label"].astype(str) == "V0"
    if not mask.any():
        return df.iloc[0:0].copy()
    canonical_sig = str(df.loc[mask, "run_signature"].iloc[0])
    return df.loc[df["run_signature"].astype(str) == canonical_sig].copy()


def _build_overall_improvement_columns_from_impr_df(
    impr_df: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    out = pd.DataFrame(index=impr_df.index)
    if impr_df.empty:
        out["combined_improvement"] = np.nan
        out["multi_pillar_binary"] = np.nan
        out["supportive_rule"] = np.nan
        out["strict_rule"] = np.nan
        return out

    out["combined_improvement"] = impr_df.mean(axis=1, skipna=True)

    pillar_positive_counts: List[int] = []
    for row_idx in impr_df.index:
        positive_pillars = set()
        for metric in impr_df.columns:
            val = impr_df.loc[row_idx, metric]
            if pd.notna(val) and float(val) > 0.0:
                pillar = _pillar_for_metric(metric, cfg)
                if pillar is not None:
                    positive_pillars.add(pillar)
        pillar_positive_counts.append(len(positive_pillars))

    out["multi_pillar_binary"] = pd.Series(
        [1.0 if n >= 2 else 0.0 for n in pillar_positive_counts],
        index=impr_df.index,
        dtype=float,
    )

    supportive_flags: List[float] = []
    for row_idx in impr_df.index:
        row = pd.to_numeric(impr_df.loc[row_idx], errors="coerce")
        supportive_pillars = set()
        supportive_degrade_count = 0
        for metric_name, val in row.items():
            if pd.isna(val):
                continue
            if float(val) >= 0.65:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    supportive_pillars.add(pillar)
            if float(val) <= -0.65:
                supportive_degrade_count += 1
        supportive_flags.append(
            1.0 if (len(supportive_pillars) >= 2 and supportive_degrade_count == 0) else 0.0
        )
    out["supportive_rule"] = pd.Series(supportive_flags, index=impr_df.index, dtype=float)

    robust_flags: List[float] = []
    for row_idx in impr_df.index:
        row = pd.to_numeric(impr_df.loc[row_idx], errors="coerce")
        strong_degrade = int((row <= -0.60).sum())
        strong_improve_pillars = set()
        for metric_name, val in row.items():
            if pd.notna(val) and float(val) >= 0.75:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    strong_improve_pillars.add(pillar)
        robust_flags.append(
            1.0 if (len(strong_improve_pillars) >= 2 and strong_degrade == 0) else 0.0
        )
    out["strict_rule"] = pd.Series(robust_flags, index=impr_df.index, dtype=float)
    return out


def _seed_aligned_v0_improvement_df(
    seed_df: pd.DataFrame,
    v0_seed_family: pd.DataFrame,
    metric_columns: Sequence[str],
    cfg: Mapping[str, Any],
    registry: Mapping[str, TargetSpec],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if seed_df.empty or v0_seed_family.empty:
        return seed_df.iloc[0:0].copy(), pd.DataFrame()

    tmp = seed_df.copy()
    tmp["_seed_id"] = tmp["run_id"].astype(str).map(_seed_id_from_run_id)

    ref = v0_seed_family.copy()
    ref["_seed_id"] = ref["run_id"].astype(str).map(_seed_id_from_run_id)
    ref_by_seed = {str(seed_id): idx for idx, seed_id in zip(ref.index, ref["_seed_id"].astype(str))}

    raw_impr: Dict[str, pd.Series] = {}
    ordered_metrics: List[str] = []

    for metric in metric_columns:
        if metric not in tmp.columns:
            continue

        vals: List[float] = []
        for idx in tmp.index:
            seed_id = str(tmp.at[idx, "_seed_id"])
            ref_idx = ref_by_seed.get(seed_id, ref_by_seed.get("seed1"))
            if ref_idx is None:
                vals.append(np.nan)
                continue

            s = _improvement_vs_baseline_series(
                tmp.loc[[idx]],
                ref.loc[[ref_idx]],
                metric,
                registry,
            )
            if s is None or s.empty:
                vals.append(np.nan)
            else:
                vals.append(float(pd.to_numeric(s.iloc[0], errors="coerce")))

        raw_impr[str(metric)] = pd.Series(vals, index=tmp.index, dtype=float)
        ordered_metrics.append(str(metric))

    if not raw_impr:
        return _signature_plot_df(seed_df, cfg), pd.DataFrame()

    raw_impr_df = pd.DataFrame(raw_impr, index=tmp.index)
    grouped_impr = raw_impr_df.groupby(tmp["run_signature"].astype(str), dropna=False).mean()

    plot_df = _signature_plot_df(seed_df, cfg)
    aligned_impr = pd.DataFrame(index=plot_df.index)
    for metric in ordered_metrics:
        aligned_impr[metric] = plot_df["run_signature"].astype(str).map(grouped_impr[metric])

    return plot_df, aligned_impr


def compute_baseline_mean_row(df: pd.DataFrame, baseline_runs: pd.DataFrame, cfg: Mapping[str, Any]) -> Optional[pd.Series]:
    include = bool((cfg.get("baseline", {}) or {}).get("include_baseline_mean_row", True))
    if not include or baseline_runs.empty:
        return None

    row: Dict[str, Any] = {
        "run_id": "baseline_mean",
        "model_key": "baseline_mean",
        "run_signature": "baseline_mean",
        "short_label": "V0_mean",
        "model_group": "baseline_mean",
        "eval_root": None,
        "generated_root": None,
        "config_path": None,
    }
    for col in baseline_runs.columns:
        if col in row:
            continue
        s = _to_numeric_series(baseline_runs[col])
        if s.notna().any():
            row[col] = float(s.mean())
        else:
            row[col] = baseline_runs[col].iloc[0] if len(baseline_runs[col]) > 0 else None
    return pd.Series(row)


def _target_value_for_metric(df: pd.DataFrame, metric_name: str, registry: Mapping[str, TargetSpec]) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[str]]:
    spec = registry.get(str(metric_name))
    if spec is None:
        return None, None, None

    if spec.kind == "zero":
        return 0.0, None, "target=0"
    if spec.kind == "one":
        return 1.0, None, "target=1"
    if spec.kind == "hr":
        ref_metric = str(spec.ref_metric or "")
        if not ref_metric or ref_metric not in df.columns:
            return None, None, None
        s = _to_numeric_series(df[ref_metric]).dropna()
        if s.empty:
            return None, None, None
        center = float(s.mean())
        spread_metric = str(spec.spread_metric or "") if spec.spread_metric else ""
        if spread_metric and spread_metric in df.columns:
            ss = _to_numeric_series(df[spread_metric]).dropna()
            if not ss.empty:
                spread_mean = float(ss.mean())
                return center, (center - spread_mean, center + spread_mean), "target=HR"
        return center, None, "target=HR"
    return None, None, None


def compute_distance_to_target(df: pd.DataFrame, metric_columns: Sequence[str], registry: Mapping[str, TargetSpec]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for metric in metric_columns:
        if metric not in df.columns:
            continue
        center, _, _ = _target_value_for_metric(df, metric, registry)
        if center is None or not math.isfinite(center):
            continue
        vals = _to_numeric_series(df[metric])
        out[metric] = (vals - float(center)).abs()
    return out


def compute_delta_to_baseline(df: pd.DataFrame, baseline_runs: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    if baseline_runs.empty:
        return out
    for metric in metric_columns:
        if metric not in baseline_runs.columns:
            continue
        s = _to_numeric_series(baseline_runs[metric]).dropna()
        if s.empty:
            continue
        out[f"delta::{metric}"] = _to_numeric_series(out[metric]) - float(s.mean())
    return out


def compute_delta_to_target_distance(df: pd.DataFrame, baseline_runs: pd.DataFrame, metric_columns: Sequence[str], registry: Mapping[str, TargetSpec]) -> pd.DataFrame:
    out = df.copy()
    dist_df = compute_distance_to_target(df, metric_columns, registry)
    if baseline_runs.empty or dist_df.empty:
        return out
    base_dist = compute_distance_to_target(baseline_runs, metric_columns, registry)
    for metric in dist_df.columns:
        s = _to_numeric_series(base_dist[metric]).dropna()
        if s.empty:
            continue
        out[f"delta_target::{metric}"] = _to_numeric_series(dist_df[metric]) - float(s.mean())
    return out


def baseline_minmax_for_metric(baseline_runs: pd.DataFrame, metric_name: str) -> Optional[Tuple[float, float, float]]:
    if baseline_runs.empty or metric_name not in baseline_runs.columns:
        return None
    s = _to_numeric_series(baseline_runs[metric_name]).dropna()
    if s.empty:
        return None
    mean = float(s.mean())
    return mean, float(s.min()), float(s.max())

def _target_center_only(df: pd.DataFrame, metric_name: str, registry: Mapping[str, TargetSpec]) -> Optional[float]:
    center, _, _ = _target_value_for_metric(df, metric_name, registry)
    if center is None or not math.isfinite(float(center)):
        return None
    return float(center)


def _improvement_vs_baseline_series(
    df: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    metric_name: str,
    registry: Mapping[str, TargetSpec],
) -> Optional[pd.Series]:
    if metric_name not in df.columns or baseline_runs.empty or metric_name not in baseline_runs.columns:
        return None

    vals = _to_numeric_series(df[metric_name])
    base_vals = _to_numeric_series(baseline_runs[metric_name]).dropna()
    if base_vals.empty:
        return None

    spec = registry.get(str(metric_name))
    baseline_mean = float(base_vals.mean())

    if spec is None:
        return vals - baseline_mean

    if spec.kind == "zero":
        return baseline_mean - vals

    if spec.kind == "one":
        return vals - baseline_mean

    if spec.kind == "hr":
        target = _target_center_only(df, metric_name, registry)
        if target is None:
            return vals - baseline_mean
        baseline_dist_mean = float((base_vals - target).abs().mean())
        return baseline_dist_mean - (vals - target).abs()

    return vals - baseline_mean


def _baseline_improvement_band(
    baseline_runs: pd.DataFrame,
    metric_name: str,
    registry: Mapping[str, TargetSpec],
) -> Optional[Tuple[float, float]]:
    if baseline_runs.empty or metric_name not in baseline_runs.columns:
        return None

    base_vals = _to_numeric_series(baseline_runs[metric_name]).dropna()
    if base_vals.empty:
        return None

    spec = registry.get(str(metric_name))
    baseline_mean = float(base_vals.mean())

    if spec is None:
        band_vals = base_vals - baseline_mean
    elif spec.kind == "zero":
        band_vals = baseline_mean - base_vals
    elif spec.kind == "one":
        band_vals = base_vals - baseline_mean
    elif spec.kind == "hr":
        target = _target_center_only(baseline_runs, metric_name, registry)
        if target is None:
            band_vals = base_vals - baseline_mean
        else:
            baseline_dist_mean = float((base_vals - target).abs().mean())
            band_vals = baseline_dist_mean - (base_vals - target).abs()
    else:
        band_vals = base_vals - baseline_mean

    band_vals = _to_numeric_series(pd.Series(band_vals)).dropna()
    if band_vals.empty:
        return None
    return float(band_vals.min()), float(band_vals.max())

def get_group_colors(cfg: Mapping[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in ((cfg.get("grouping", {}) or {}).get("colors", {}) or {}).items()}


# --- Additional helpers for better-than-baseline heatmap extensions ---

def _baseline_seed_improvement_series(
    baseline_runs: pd.DataFrame,
    metric_name: str,
    registry: Mapping[str, TargetSpec],
) -> Optional[pd.Series]:
    if baseline_runs.empty or metric_name not in baseline_runs.columns:
        return None

    base_vals = _to_numeric_series(baseline_runs[metric_name]).dropna()
    if base_vals.empty:
        return None

    spec = registry.get(str(metric_name))
    baseline_mean = float(base_vals.mean())

    if spec is None:
        vals = base_vals - baseline_mean
    elif spec.kind == "zero":
        vals = baseline_mean - base_vals
    elif spec.kind == "one":
        vals = base_vals - baseline_mean
    elif spec.kind == "hr":
        target = _target_center_only(baseline_runs, metric_name, registry)
        if target is None:
            vals = base_vals - baseline_mean
        else:
            baseline_dist_mean = float((base_vals - target).abs().mean())
            vals = baseline_dist_mean - (base_vals - target).abs()
    else:
        vals = base_vals - baseline_mean

    return _to_numeric_series(pd.Series(vals)).dropna()



def _improvement_vs_baseline_spread_series(
    df: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    metric_name: str,
    registry: Mapping[str, TargetSpec],
) -> Optional[pd.Series]:
    model_impr = _improvement_vs_baseline_series(df, baseline_runs, metric_name, registry)
    base_impr = _baseline_seed_improvement_series(baseline_runs, metric_name, registry)
    if model_impr is None or base_impr is None or base_impr.empty:
        return None

    lo = float(base_impr.min())
    hi = float(base_impr.max())
    vals = _to_numeric_series(model_impr)

    out = pd.Series(index=vals.index, dtype=float)
    for idx, v in vals.items():
        if pd.isna(v):
            out.loc[idx] = np.nan
        elif v > hi:
            out.loc[idx] = float(v) - hi
        elif v < lo:
            out.loc[idx] = float(v) - lo
        else:
            out.loc[idx] = 0.0
    return out



def _build_overall_improvement_columns(
    df: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    cfg: Mapping[str, Any],
    metric_columns: Sequence[str],
    registry: Mapping[str, TargetSpec],
) -> pd.DataFrame:
    improvement_table: Dict[str, pd.Series] = {}
    for metric in metric_columns:
        s = _improvement_vs_baseline_series(df, baseline_runs, metric, registry)
        if s is not None:
            improvement_table[str(metric)] = _to_numeric_series(s)

    if not improvement_table:
        out = pd.DataFrame(index=df.index)
        out["combined_improvement"] = np.nan
        out["multi_pillar_binary"] = np.nan
        out["supportive_rule"] = np.nan
        out["strict_rule"] = np.nan
        return out

    impr_df = pd.DataFrame(improvement_table, index=df.index)
    return _build_overall_improvement_columns_from_impr_df(impr_df, cfg)
# def _build_overall_improvement_columns(
#     df: pd.DataFrame,
#     baseline_runs: pd.DataFrame,
#     cfg: Mapping[str, Any],
#     metric_columns: Sequence[str],
#     registry: Mapping[str, TargetSpec],
# ) -> pd.DataFrame:
#     out = pd.DataFrame(index=df.index)

#     improvement_table: Dict[str, pd.Series] = {}
#     for metric in metric_columns:
#         s = _improvement_vs_baseline_series(df, baseline_runs, metric, registry)
#         if s is not None:
#             improvement_table[str(metric)] = _to_numeric_series(s)

#     if not improvement_table:
#         out["combined_improvement"] = np.nan
#         out["multi_pillar_binary"] = np.nan
#         return out

#     impr_df = pd.DataFrame(improvement_table, index=df.index)
#     out["combined_improvement"] = impr_df.mean(axis=1, skipna=True)

#     pillar_positive_counts: List[int] = []
#     for row_idx in impr_df.index:
#         positive_pillars = set()
#         for metric in impr_df.columns:
#             val = impr_df.loc[row_idx, metric]
#             if pd.notna(val) and float(val) > 0.0:
#                 pillar = _pillar_for_metric(metric, cfg)
#                 if pillar is not None:
#                     positive_pillars.add(pillar)
#         pillar_positive_counts.append(len(positive_pillars))

#     out["multi_pillar_binary"] = pd.Series(
#         [1.0 if n >= 2 else 0.0 for n in pillar_positive_counts],
#         index=df.index,
#         dtype=float,
#     )

#     # Supportive rule: milder than strict_rule, but still requires support across at least two distinct pillars
#     supportive_flags: List[float] = []
#     for row_idx in impr_df.index:
#         row = pd.to_numeric(impr_df.loc[row_idx], errors="coerce")
#         supportive_pillars = set()
#         supportive_degrade_count = 0
#         for metric_name, val in row.items():
#             if pd.isna(val):
#                 continue
#             if float(val) >= 0.65:
#                 pillar = _pillar_for_metric(metric_name, cfg)
#                 if pillar is not None:
#                     supportive_pillars.add(pillar)
#             if float(val) <= -0.55:
#                 supportive_degrade_count += 1
#         supportive_flags.append(1.0 if (len(supportive_pillars) >= 2 and supportive_degrade_count == 0) else 0.0)
#     out["supportive_rule"] = pd.Series(supportive_flags, index=df.index, dtype=float)

#     # Stricter robustness rule for interpretation:
#     # no metric may be strongly degraded, and strong improvements must occur in at least two distinct pillars.
#     robust_flags: List[float] = []
#     for row_idx in impr_df.index:
#         row = pd.to_numeric(impr_df.loc[row_idx], errors="coerce")
#         strong_degrade = int((row <= -0.60).sum())
#         strong_improve_pillars = set()
#         for metric_name, val in row.items():
#             if pd.notna(val) and float(val) >= 0.75:
#                 pillar = _pillar_for_metric(metric_name, cfg)
#                 if pillar is not None:
#                     strong_improve_pillars.add(pillar)
#         robust_flags.append(1.0 if (len(strong_improve_pillars) >= 2 and strong_degrade == 0) else 0.0)
#     out["strict_rule"] = pd.Series(robust_flags, index=df.index, dtype=float)
#     return out


def _add_heatmap_pillar_annotations(ax, scaled: pd.DataFrame, metric_names_only: Sequence[str], cfg: Mapping[str, Any]) -> None:
    pillar_boundaries: List[int] = []
    pillar_label_positions: List[Tuple[float, str]] = []
    current = 0
    for pillar in _pillar_names(cfg):
        metrics = [m for m in _metrics_for_pillar(cfg, pillar) if m in metric_names_only]
        if not metrics:
            continue
        start = current
        end = current + len(metrics) - 1
        pillar_label_positions.append(((start + end) / 2.0, pillar))
        pillar_boundaries.append(end)
        current += len(metrics)

    for b in pillar_boundaries[:-1]:
        ax.axvline(b + 0.5, linestyle="--", linewidth=1.2, color="black", alpha=0.6)

    short_labels = [c.split(".")[-1] for c in scaled.columns]
    ax.set_xticklabels(short_labels, rotation=60, ha="right")

    for i, (pos, pillar) in enumerate(pillar_label_positions):
        y = -1.35 if (i % 2 == 0) else -1.85
        ax.text(pos, y, pillar, ha="center", va="bottom", fontsize=11, fontweight="bold")


def get_group_labels(cfg: Mapping[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in ((cfg.get("grouping", {}) or {}).get("labels", {}) or {}).items()}


def _metric_list_from_cfg(cfg: Mapping[str, Any]) -> List[str]:
    metrics_cfg = cfg.get("metrics", {}) or {}
    pillars = metrics_cfg.get("pillars", {}) or {}
    out: List[str] = []
    for _, values in pillars.items():
        for metric in values or []:
            m = str(metric)
            if m not in out:
                out.append(m)
    return out


def _metrics_for_pillar(cfg: Mapping[str, Any], pillar_name: str) -> List[str]:
    pillars = (cfg.get("metrics", {}) or {}).get("pillars", {}) or {}
    return [str(x) for x in (pillars.get(str(pillar_name), []) or [])]


def _pillar_names(cfg: Mapping[str, Any]) -> List[str]:
    return [str(k) for k in ((cfg.get("metrics", {}) or {}).get("pillars", {}) or {}).keys()]


def _plot_dir(cfg: Mapping[str, Any]) -> Path:
    return _mkdir(Path((cfg.get("paths", {}) or {}).get("output_dir", ".")) / "figures")


def _table_dir(cfg: Mapping[str, Any]) -> Path:
    return _mkdir(Path((cfg.get("paths", {}) or {}).get("output_dir", ".")) / "tables")


def _heatmap_row_blocks(df: pd.DataFrame, cfg: Mapping[str, Any]) -> List[Tuple[str, int, int]]:
    labels = get_group_labels(cfg)
    blocks: List[Tuple[str, int, int]] = []
    if df.empty:
        return blocks
    current_group = None
    start = 0
    for i, g in enumerate(df["model_group"].astype(str).tolist()):
        if current_group is None:
            current_group = g
            start = i
            continue
        if g != current_group:
            blocks.append((labels.get(current_group, current_group), start, i - 1))
            current_group = g
            start = i
    if current_group is not None:
        blocks.append((labels.get(current_group, current_group), start, len(df) - 1))
    return blocks

def plot_better_than_baseline_heatmap(
    df: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    cfg: Mapping[str, Any],
    metric_columns: Sequence[str],
    registry: Mapping[str, TargetSpec],
) -> None:
    if df.empty or baseline_runs.empty:
        return

    plot_df = sort_rows_by_group(df[["run_id", "short_label", "model_group"] + [m for m in metric_columns if m in df.columns]].copy(), cfg)
    if plot_df.empty:
        return

    improvement_cols: Dict[str, pd.Series] = {}
    ordered_metrics: List[str] = []
    for metric in metric_columns:
        if metric not in plot_df.columns:
            continue
        s = _improvement_vs_baseline_series(plot_df, baseline_runs, metric, registry)
        if s is not None:
            improvement_cols[str(metric)] = _to_numeric_series(s)
            ordered_metrics.append(str(metric))

    if not improvement_cols:
        return

    improv_df = pd.DataFrame(improvement_cols, index=plot_df.index)
    summary_df = _build_overall_improvement_columns(plot_df, baseline_runs, cfg, ordered_metrics, registry)
    full_df = pd.concat([improv_df, summary_df], axis=1)

    scaled = pd.DataFrame(index=full_df.index)
    for col in ordered_metrics:
        scaled[col] = _symmetric_scale_series(full_df[col])

    combined_raw = pd.to_numeric(full_df["combined_improvement"], errors="coerce")
    scaled["combined_improvement"] = _symmetric_scale_series(combined_raw)

    binary_raw = pd.to_numeric(full_df["multi_pillar_binary"], errors="coerce")
    scaled["multi_pillar_binary"] = binary_raw.fillna(0.0)

    supportive_raw = pd.to_numeric(full_df["supportive_rule"], errors="coerce")
    scaled["supportive_rule"] = supportive_raw.fillna(0.0)

    strict_raw = pd.to_numeric(full_df["strict_rule"], errors="coerce")
    scaled["strict_rule"] = strict_raw.fillna(0.0)

    display_names = ordered_metrics + ["combined_improvement", "multi_pillar_binary", "supportive_rule", "strict_rule"]
    scaled = scaled[display_names]
    scaled = scaled.rename(columns={
        "combined_improvement": "combined",
        "multi_pillar_binary": ">=2 pillars",
        "supportive_rule": "supportive",
        "strict_rule": "robust",
    })

    heatmap_size = tuple(((cfg.get("plots", {}) or {}).get("figsize", {}) or {}).get("heatmap", [14, 10]))
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    colors = get_group_colors(cfg)

    fig, ax = plt.subplots(figsize=heatmap_size)
    im = ax.imshow(
        scaled.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=_heatmap_cmap(cfg, "better_than_baseline_heatmap", "RdYlBu_r"),
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short_label"].astype(str).tolist(), fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(range(len(scaled.columns)))
    ax.set_title("Better-than-baseline heatmap (towards target)", fontsize=15)

    for group_name, start, end in _heatmap_row_blocks(plot_df, cfg):
        group_key = plot_df.iloc[start]["model_group"]
        edge_color = colors.get(str(group_key), "black")
        rect = patches.Rectangle(
            (-0.5, start - 0.5),
            width=len(scaled.columns),
            height=(end - start + 1),
            fill=False,
            linewidth=2.2,
            edgecolor=edge_color,
        )
        ax.add_patch(rect)
        ax.hlines(end + 0.5, xmin=-0.5, xmax=len(scaled.columns) - 0.5, colors=edge_color, linewidth=1.8)
        if str(group_key) != "baseline_mean":
            ax.text(
                -3.15,
                (start + end) / 2.0,
                group_name,
                rotation=90,
                va="center",
                ha="right",
                fontsize=12,
                color=edge_color,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor=edge_color, linewidth=1.4),
            )

    # Pillar separators and cleaner metric labels.
    metric_names_only = ordered_metrics
    _add_heatmap_pillar_annotations(ax, scaled, metric_names_only, cfg)


    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("0.5 = baseline, >0.5 better, <0.5 worse", fontsize=12)
    fig.savefig(_plot_dir(cfg) / "better_than_baseline_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --- New function: better-than-baseline spread heatmap ---

def plot_better_than_baseline_spread_heatmap(
    df: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    cfg: Mapping[str, Any],
    metric_columns: Sequence[str],
    registry: Mapping[str, TargetSpec],
) -> None:
    if df.empty or baseline_runs.empty:
        return

    plot_df = sort_rows_by_group(df[["run_id", "short_label", "model_group"] + [m for m in metric_columns if m in df.columns]].copy(), cfg)
    if plot_df.empty:
        return

    spread_cols: Dict[str, pd.Series] = {}
    ordered_metrics: List[str] = []
    for metric in metric_columns:
        if metric not in plot_df.columns:
            continue
        s = _improvement_vs_baseline_spread_series(plot_df, baseline_runs, metric, registry)
        if s is not None:
            spread_cols[str(metric)] = _to_numeric_series(s)
            ordered_metrics.append(str(metric))

    if not spread_cols:
        return

    spread_df = pd.DataFrame(spread_cols, index=plot_df.index)

    # Summary columns derived only from improvements outside the baseline seed spread.
    combined_vals = spread_df.mean(axis=1, skipna=True)
    multi_pillar_vals: List[float] = []
    robust_vals: List[float] = []
    supportive_vals: List[float] = []
    for row_idx in spread_df.index:
        row = pd.to_numeric(spread_df.loc[row_idx], errors="coerce")

        positive_pillars = set()
        for metric_name, val in row.items():
            if pd.notna(val) and float(val) > 0.0:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    positive_pillars.add(pillar)
        multi_pillar_vals.append(1.0 if len(positive_pillars) >= 2 else 0.0)
        supportive_pillars = set()
        supportive_degrade_count = 0
        for metric_name, val in row.items():
            if pd.notna(val) and float(val) >= 0.65:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    supportive_pillars.add(pillar)
            if pd.notna(val) and float(val) <= -0.55:
                supportive_degrade_count += 1

        strong_degrade = int((row <= -0.60).sum())
        strong_improve_pillars = set()
        for metric_name, val in row.items():
            if pd.notna(val) and float(val) >= 0.75:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    strong_improve_pillars.add(pillar)
        supportive_vals.append(1.0 if (len(supportive_pillars) >= 2 and supportive_degrade_count == 0) else 0.0)
        robust_vals.append(1.0 if (len(strong_improve_pillars) >= 2 and strong_degrade == 0) else 0.0)

    spread_df["combined_improvement"] = combined_vals
    spread_df["multi_pillar_binary"] = pd.Series(multi_pillar_vals, index=spread_df.index, dtype=float)
    spread_df["supportive_rule"] = pd.Series(supportive_vals, index=spread_df.index, dtype=float)
    spread_df["strict_rule"] = pd.Series(robust_vals, index=spread_df.index, dtype=float)

    scaled = pd.DataFrame(index=spread_df.index)
    for col in ordered_metrics:
        scaled[col] = _symmetric_scale_series(spread_df[col])

    scaled["combined_improvement"] = _symmetric_scale_series(pd.to_numeric(spread_df["combined_improvement"], errors="coerce"))
    scaled["multi_pillar_binary"] = pd.to_numeric(spread_df["multi_pillar_binary"], errors="coerce").fillna(0.0)
    scaled["supportive_rule"] = pd.to_numeric(spread_df["supportive_rule"], errors="coerce").fillna(0.0)
    scaled["strict_rule"] = pd.to_numeric(spread_df["strict_rule"], errors="coerce").fillna(0.0)

    scaled = scaled.rename(columns={
        "combined_improvement": "combined",
        "multi_pillar_binary": ">=2 pillars",
        "supportive_rule": "supportive",
        "strict_rule": "robust",
    })

    heatmap_size = tuple(((cfg.get("plots", {}) or {}).get("figsize", {}) or {}).get("heatmap", [14, 10]))
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    colors = get_group_colors(cfg)

    fig, ax = plt.subplots(figsize=heatmap_size)
    im = ax.imshow(
        scaled.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=_heatmap_cmap(cfg, "better_than_baseline_spread_heatmap", "RdYlBu_r"),
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short_label"].astype(str).tolist(), fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(range(len(scaled.columns)))
    ax.set_title("Better-than-baseline heatmap outside seed spread", fontsize=15)

    for group_name, start, end in _heatmap_row_blocks(plot_df, cfg):
        group_key = plot_df.iloc[start]["model_group"]
        edge_color = colors.get(str(group_key), "black")
        rect = patches.Rectangle(
            (-0.5, start - 0.5),
            width=len(scaled.columns),
            height=(end - start + 1),
            fill=False,
            linewidth=2.2,
            edgecolor=edge_color,
        )
        ax.add_patch(rect)
        ax.hlines(end + 0.5, xmin=-0.5, xmax=len(scaled.columns) - 0.5, colors=edge_color, linewidth=1.8)

        if str(group_key) != "baseline_mean":
            ax.text(
                -3.15,
                (start + end) / 2.0,
                group_name,
                rotation=90,
                va="center",
                ha="right",
                fontsize=12,
                color=edge_color,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor=edge_color, linewidth=1.4),
            )

    # Pillar separators and cleaner metric labels.
    metric_names_only = ordered_metrics
    _add_heatmap_pillar_annotations(ax, scaled, metric_names_only, cfg)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("0.5 = inside seed spread, >0.5 better outside, <0.5 worse outside", fontsize=12)
    fig.savefig(_plot_dir(cfg) / "better_than_baseline_spread_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_better_than_v0_heatmap(
    df: pd.DataFrame,
    v0_run: pd.DataFrame,
    cfg: Mapping[str, Any],
    metric_columns: Sequence[str],
    registry: Mapping[str, TargetSpec],
    seed_df: Optional[pd.DataFrame] = None,
    v0_seed_family: Optional[pd.DataFrame] = None,
) -> None:
    if df.empty or v0_run.empty:
        return

    use_seed_aligned = (
        _seed_mode(cfg) == "mean"
        and seed_df is not None
        and v0_seed_family is not None
        and not seed_df.empty
        and not v0_seed_family.empty
    )

    if use_seed_aligned:
        plot_df, improv_df = _seed_aligned_v0_improvement_df(
            seed_df, v0_seed_family, metric_columns, cfg, registry
        )
        if plot_df.empty or improv_df.empty:
            return
        ordered_metrics = [str(c) for c in improv_df.columns]
        summary_df = _build_overall_improvement_columns_from_impr_df(improv_df, cfg)
        full_df = pd.concat([improv_df, summary_df], axis=1)
    else:
        plot_df = sort_rows_by_group(
            df[["run_id", "short_label", "model_group"] + [m for m in metric_columns if m in df.columns]].copy(),
            cfg,
        )
        if plot_df.empty:
            return

        improvement_cols: Dict[str, pd.Series] = {}
        ordered_metrics: List[str] = []
        for metric in metric_columns:
            if metric not in plot_df.columns:
                continue
            s = _improvement_vs_baseline_series(plot_df, v0_run, metric, registry)
            if s is not None:
                improvement_cols[str(metric)] = _to_numeric_series(s)
                ordered_metrics.append(str(metric))

        if not improvement_cols:
            return

        improv_df = pd.DataFrame(improvement_cols, index=plot_df.index)
        summary_df = _build_overall_improvement_columns(plot_df, v0_run, cfg, ordered_metrics, registry)
        full_df = pd.concat([improv_df, summary_df], axis=1)

    scaled = pd.DataFrame(index=full_df.index)
    for col in ordered_metrics:
        scaled[col] = _symmetric_scale_series(full_df[col])

    scaled["combined_improvement"] = _symmetric_scale_series(pd.to_numeric(full_df["combined_improvement"], errors="coerce"))
    scaled["multi_pillar_binary"] = pd.to_numeric(full_df["multi_pillar_binary"], errors="coerce").fillna(0.0)
    scaled["supportive_rule"] = pd.to_numeric(full_df["supportive_rule"], errors="coerce").fillna(0.0)
    scaled["strict_rule"] = pd.to_numeric(full_df["strict_rule"], errors="coerce").fillna(0.0)

    display_names = ordered_metrics + ["combined_improvement", "multi_pillar_binary", "supportive_rule", "strict_rule"]
    scaled = scaled[display_names]
    scaled = scaled.rename(columns={
        "combined_improvement": "combined",
        "multi_pillar_binary": ">=2 pillars",
        "supportive_rule": "supportive",
        "strict_rule": "robust",
    })

    heatmap_size = tuple(((cfg.get("plots", {}) or {}).get("figsize", {}) or {}).get("heatmap", [14, 10]))
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    colors = get_group_colors(cfg)

    fig, ax = plt.subplots(figsize=heatmap_size)
    im = ax.imshow(
        scaled.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=_heatmap_cmap(cfg, "better_than_v0_heatmap", "RdYlBu_r"),
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short_label"].astype(str).tolist(), fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(range(len(scaled.columns)))
    ax.set_title("Better-than-V0 heatmap (towards target)", fontsize=15)

    for group_name, start, end in _heatmap_row_blocks(plot_df, cfg):
        group_key = plot_df.iloc[start]["model_group"]
        edge_color = colors.get(str(group_key), "black")
        rect = patches.Rectangle(
            (-0.5, start - 0.5),
            width=len(scaled.columns),
            height=(end - start + 1),
            fill=False,
            linewidth=2.2,
            edgecolor=edge_color,
        )
        ax.add_patch(rect)
        ax.hlines(end + 0.5, xmin=-0.5, xmax=len(scaled.columns) - 0.5, colors=edge_color, linewidth=1.8)
        if str(group_key) != "baseline_mean":
            ax.text(
                -3.15,
                (start + end) / 2.0,
                group_name,
                rotation=90,
                va="center",
                ha="right",
                fontsize=12,
                color=edge_color,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor=edge_color, linewidth=1.4),
            )

    metric_names_only = ordered_metrics
    _add_heatmap_pillar_annotations(ax, scaled, metric_names_only, cfg)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("0.5 = V0, >0.5 better, <0.5 worse", fontsize=12)
    fig.savefig(_plot_dir(cfg) / "better_than_v0_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_better_than_v0_spread_heatmap(
    df: pd.DataFrame,
    v0_run: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    cfg: Mapping[str, Any],
    metric_columns: Sequence[str],
    registry: Mapping[str, TargetSpec],
    seed_df: Optional[pd.DataFrame] = None,
    v0_seed_family: Optional[pd.DataFrame] = None,
) -> None:
    if df.empty or v0_run.empty or baseline_runs.empty:
        return

    use_seed_aligned = (
        _seed_mode(cfg) == "mean"
        and seed_df is not None
        and v0_seed_family is not None
        and not seed_df.empty
        and not v0_seed_family.empty
    )

    if use_seed_aligned:
        plot_df, aligned_impr = _seed_aligned_v0_improvement_df(
            seed_df, v0_seed_family, metric_columns, cfg, registry
        )
        if plot_df.empty or aligned_impr.empty:
            return

        spread_cols: Dict[str, pd.Series] = {}
        ordered_metrics: List[str] = []
        for metric in aligned_impr.columns:
            base_spread = _baseline_seed_improvement_series(baseline_runs, metric, registry)
            if base_spread is None or base_spread.empty:
                continue

            lo = float(base_spread.min())
            hi = float(base_spread.max())
            vals = _to_numeric_series(aligned_impr[metric])

            out = pd.Series(index=vals.index, dtype=float)
            for idx, v in vals.items():
                if pd.isna(v):
                    out.loc[idx] = np.nan
                elif v > hi:
                    out.loc[idx] = float(v) - hi
                elif v < lo:
                    out.loc[idx] = float(v) - lo
                else:
                    out.loc[idx] = 0.0

            spread_cols[str(metric)] = out
            ordered_metrics.append(str(metric))
    else:
        plot_df = sort_rows_by_group(
            df[["run_id", "short_label", "model_group"] + [m for m in metric_columns if m in df.columns]].copy(),
            cfg,
        )
        if plot_df.empty:
            return

        spread_cols: Dict[str, pd.Series] = {}
        ordered_metrics: List[str] = []
        for metric in metric_columns:
            if metric not in plot_df.columns:
                continue

            v0_impr = _improvement_vs_baseline_series(plot_df, v0_run, metric, registry)
            base_spread = _baseline_seed_improvement_series(baseline_runs, metric, registry)
            if v0_impr is None or base_spread is None or base_spread.empty:
                continue

            lo = float(base_spread.min())
            hi = float(base_spread.max())
            vals = _to_numeric_series(v0_impr)

            out = pd.Series(index=vals.index, dtype=float)
            for idx, v in vals.items():
                if pd.isna(v):
                    out.loc[idx] = np.nan
                elif v > hi:
                    out.loc[idx] = float(v) - hi
                elif v < lo:
                    out.loc[idx] = float(v) - lo
                else:
                    out.loc[idx] = 0.0

            spread_cols[str(metric)] = out
            ordered_metrics.append(str(metric))

    if not spread_cols:
        return

    spread_df = pd.DataFrame(spread_cols, index=plot_df.index)

    combined_vals = spread_df.mean(axis=1, skipna=True)
    multi_pillar_vals: List[float] = []
    robust_vals: List[float] = []
    supportive_vals: List[float] = []
    for row_idx in spread_df.index:
        row = pd.to_numeric(spread_df.loc[row_idx], errors="coerce")
        positive_pillars = set()
        for metric_name, val in row.items():
            if pd.notna(val) and float(val) > 0.0:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    positive_pillars.add(pillar)
        multi_pillar_vals.append(1.0 if len(positive_pillars) >= 2 else 0.0)

        supportive_pillars = set()
        supportive_degrade_count = 0
        for metric_name, val in row.items():
            if pd.notna(val) and float(val) >= 0.65:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    supportive_pillars.add(pillar)
            if pd.notna(val) and float(val) <= -0.55:
                supportive_degrade_count += 1

        strong_degrade = int((row <= -0.60).sum())
        strong_improve_pillars = set()
        for metric_name, val in row.items():
            if pd.notna(val) and float(val) >= 0.75:
                pillar = _pillar_for_metric(metric_name, cfg)
                if pillar is not None:
                    strong_improve_pillars.add(pillar)

        supportive_vals.append(1.0 if (len(supportive_pillars) >= 2 and supportive_degrade_count == 0) else 0.0)
        robust_vals.append(1.0 if (len(strong_improve_pillars) >= 2 and strong_degrade == 0) else 0.0)

    spread_df["combined_improvement"] = combined_vals
    spread_df["multi_pillar_binary"] = pd.Series(multi_pillar_vals, index=spread_df.index, dtype=float)
    spread_df["supportive_rule"] = pd.Series(supportive_vals, index=spread_df.index, dtype=float)
    spread_df["strict_rule"] = pd.Series(robust_vals, index=spread_df.index, dtype=float)

    scaled = pd.DataFrame(index=spread_df.index)
    for col in ordered_metrics:
        scaled[col] = _symmetric_scale_series(spread_df[col])

    scaled["combined_improvement"] = _symmetric_scale_series(pd.to_numeric(spread_df["combined_improvement"], errors="coerce"))
    scaled["multi_pillar_binary"] = pd.to_numeric(spread_df["multi_pillar_binary"], errors="coerce").fillna(0.0)
    scaled["supportive_rule"] = pd.to_numeric(spread_df["supportive_rule"], errors="coerce").fillna(0.0)
    scaled["strict_rule"] = pd.to_numeric(spread_df["strict_rule"], errors="coerce").fillna(0.0)

    scaled = scaled.rename(columns={
        "combined_improvement": "combined",
        "multi_pillar_binary": ">=2 pillars",
        "supportive_rule": "supportive",
        "strict_rule": "robust",
    })

    heatmap_size = tuple(((cfg.get("plots", {}) or {}).get("figsize", {}) or {}).get("heatmap", [14, 10]))
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    colors = get_group_colors(cfg)

    fig, ax = plt.subplots(figsize=heatmap_size)
    im = ax.imshow(
        scaled.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=_heatmap_cmap(cfg, "better_than_v0_spread_heatmap", "RdYlBu_r"),
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short_label"].astype(str).tolist(), fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xticks(range(len(scaled.columns)))
    ax.set_title("Better-than-V0 heatmap outside baseline seed spread", fontsize=15)

    for group_name, start, end in _heatmap_row_blocks(plot_df, cfg):
        group_key = plot_df.iloc[start]["model_group"]
        edge_color = colors.get(str(group_key), "black")
        rect = patches.Rectangle(
            (-0.5, start - 0.5),
            width=len(scaled.columns),
            height=(end - start + 1),
            fill=False,
            linewidth=2.2,
            edgecolor=edge_color,
        )
        ax.add_patch(rect)
        ax.hlines(end + 0.5, xmin=-0.5, xmax=len(scaled.columns) - 0.5, colors=edge_color, linewidth=1.8)
        if str(group_key) != "baseline_mean":
            ax.text(
                -3.15,
                (start + end) / 2.0,
                group_name,
                rotation=90,
                va="center",
                ha="right",
                fontsize=12,
                color=edge_color,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor=edge_color, linewidth=1.4),
            )

    metric_names_only = ordered_metrics
    _add_heatmap_pillar_annotations(ax, scaled, metric_names_only, cfg)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("0.5 = inside seed spread around V0, >0.5 better outside, <0.5 worse outside", fontsize=12)
    fig.savefig(_plot_dir(cfg) / "better_than_v0_spread_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_absolute_heatmap(df: pd.DataFrame, cfg: Mapping[str, Any], metric_columns: Sequence[str]) -> None:
    if df.empty:
        return
    cols = [m for m in metric_columns if m in df.columns]
    if not cols:
        return

    plot_df = sort_rows_by_group(df[["run_id", "short_label", "model_group"] + cols].copy(), cfg)
    value_df = _normalize_per_metric(plot_df[cols].apply(pd.to_numeric, errors="coerce"))

    heatmap_size = tuple(((cfg.get("plots", {}) or {}).get("figsize", {}) or {}).get("heatmap", [14, 10]))
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    colors = get_group_colors(cfg)

    fig, ax = plt.subplots(figsize=heatmap_size)
    im = ax.imshow(
        value_df.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=_heatmap_cmap(cfg, "absolute_heatmap", "cividis"),
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short_label"].astype(str).tolist(), fontsize=12)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right")
    ax.set_title("Absolute metric heatmap (per-metric normalized)")

    for group_name, start, end in _heatmap_row_blocks(plot_df, cfg):
        group_key = plot_df.iloc[start]["model_group"]
        edge_color = colors.get(str(group_key), "black")
        rect = patches.Rectangle(
            (-0.5, start - 0.5),
            width=len(cols),
            height=(end - start + 1),
            fill=False,
            linewidth=2.2,
            edgecolor=edge_color,
        )
        ax.add_patch(rect)
        ax.hlines(end + 0.5, xmin=-0.5, xmax=len(cols) - 0.5, colors=edge_color, linewidth=1.8)
        ax.text(
            -3.15,
            (start + end) / 2.0,
            group_name,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            color=edge_color,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor=edge_color, linewidth=1.4),
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative position within metric [0, 1]")
    fig.savefig(_plot_dir(cfg) / "absolute_metric_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_target_heatmap(df: pd.DataFrame, cfg: Mapping[str, Any], metric_columns: Sequence[str], registry: Mapping[str, TargetSpec]) -> None:
    if df.empty:
        return
    dist_df = compute_distance_to_target(df, metric_columns, registry)
    if dist_df.empty:
        return

    plot_df = pd.concat([df[["run_id", "short_label", "model_group"]].copy(), dist_df], axis=1)
    plot_df = sort_rows_by_group(plot_df, cfg)
    value_df = _normalize_per_metric(plot_df[dist_df.columns].apply(pd.to_numeric, errors="coerce"))

    heatmap_size = tuple(((cfg.get("plots", {}) or {}).get("figsize", {}) or {}).get("heatmap", [14, 10]))
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    colors = get_group_colors(cfg)

    fig, ax = plt.subplots(figsize=heatmap_size)
    im = ax.imshow(
        value_df.values,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=_heatmap_cmap(cfg, "target_heatmap", "cividis"),
    )
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["short_label"].astype(str).tolist())
    ax.set_xticks(range(len(value_df.columns)))
    ax.set_xticklabels(value_df.columns.tolist(), rotation=60, ha="right")
    ax.set_title("Target-closeness heatmap (per-metric normalized)")

    for group_name, start, end in _heatmap_row_blocks(plot_df, cfg):
        group_key = plot_df.iloc[start]["model_group"]
        edge_color = colors.get(str(group_key), "black")
        rect = patches.Rectangle(
            (-0.5, start - 0.5),
            width=len(value_df.columns),
            height=(end - start + 1),
            fill=False,
            linewidth=2.2,
            edgecolor=edge_color,
        )
        ax.add_patch(rect)
        ax.hlines(end + 0.5, xmin=-0.5, xmax=len(value_df.columns) - 0.5, colors=edge_color, linewidth=1.8)
        ax.text(
            -3.15,
            (start + end) / 2.0,
            group_name,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            color=edge_color,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor=edge_color, linewidth=1.4),
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative distance to target [0, 1]")
    fig.savefig(_plot_dir(cfg) / "target_metric_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _group_spans(df: pd.DataFrame) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    if df.empty:
        return out
    groups = df["model_group"].astype(str).tolist()
    start = 0
    current = groups[0]
    for i, g in enumerate(groups[1:], start=1):
        if g != current:
            out.append((current, start, i - 1))
            current = g
            start = i
    out.append((current, start, len(groups) - 1))
    return out


def _apply_group_backgrounds(ax, df: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    colors = get_group_colors(cfg)
    for group_name, start, end in _group_spans(df):
        c = colors.get(str(group_name))
        if c is None:
            continue
        ax.axvspan(start - 0.5, end + 0.5, facecolor=c, alpha=0.14, linewidth=0)


def _mark_baseline_ticks(ax, df: pd.DataFrame) -> None:
    for i, g in enumerate(df["model_group"].astype(str).tolist()):
        if g == "baseline":
            ax.axvspan(i - 0.5, i + 0.5, facecolor="none", hatch="///", edgecolor="0.5", linewidth=0.0, alpha=0.0)


def _plot_target_line(ax, center: Optional[float], band: Optional[Tuple[float, float]], label: Optional[str]) -> None:
    if center is None or not math.isfinite(float(center)):
        return
    if band is not None and all(math.isfinite(float(x)) for x in band):
        lo, hi = float(band[0]), float(band[1])
        if hi < lo:
            lo, hi = hi, lo
        ax.axhspan(lo, hi, alpha=0.12, color="black")
    ax.axhline(float(center), linestyle="--", linewidth=1.2, color="black", alpha=0.8)
    if label:
        ax.text(
            0.98,
            0.97,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
        )


def _per_metric_y_limits(values: Sequence[float], yerr: Sequence[float], ref_center: Optional[float], ref_band: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    errs = [abs(float(e)) for e in yerr if e is not None and math.isfinite(float(e))]
    if not vals and ref_center is None:
        return None

    lows: List[float] = []
    highs: List[float] = []
    for i, v in enumerate(vals):
        e = errs[i] if i < len(errs) else 0.0
        lows.append(v - e)
        highs.append(v + e)

    if ref_center is not None and math.isfinite(float(ref_center)):
        lows.append(float(ref_center))
        highs.append(float(ref_center))
    if ref_band is not None and all(math.isfinite(float(x)) for x in ref_band):
        lows.append(min(ref_band))
        highs.append(max(ref_band))

    ymin = min(lows) if lows else float(ref_center)
    ymax = max(highs) if highs else float(ref_center)
    pad = 0.06 * max(1e-12, ymax - ymin)
    return ymin - pad, ymax + pad


def _std_column_for_metric(df: pd.DataFrame, metric: str) -> Optional[str]:
    candidates = [
        f"{metric}_std",
        metric.replace("_mean", "_std"),
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _seed_mode(cfg: Mapping[str, Any]) -> str:
    return str(((cfg.get("seeds", {}) or {}).get("mode", "none") or "none")).lower()



def _seed_id_from_run_id(run_id: str) -> str:
    rid = str(run_id)
    for part in rid.split("__"):
        if _is_seed_token(part):
            return part.lower()
    return "seed1"



def _seed_sort_key(seed_id: str) -> Tuple[int, str]:
    sid = str(seed_id).lower()
    if sid == "seed1":
        return (0, sid)
    if sid.startswith("seed"):
        try:
            return (int(sid.replace("seed", "")), sid)
        except Exception:
            return (99, sid)
    return (99, sid)




def _seed_colors(cfg: Mapping[str, Any]) -> Dict[str, str]:
    seeds_cfg = cfg.get("seeds", {}) or {}
    user_colors = seeds_cfg.get("colors", {}) or {}
    colors = {
        "seed1": "#1f77b4",
        "seed2": "#ff7f0e",
        "seed3": "#2ca02c",
    }
    for k, v in user_colors.items():
        colors[str(k).lower()] = str(v)
    return colors


def _seed_alpha(cfg: Mapping[str, Any], key: str, default: float) -> float:
    seeds_cfg = cfg.get("seeds", {}) or {}
    try:
        return float(seeds_cfg.get(key, default))
    except Exception:
        return float(default)



def _apply_panel_xticks(ax, labels: Sequence[str], row_idx: int, nrows: int) -> None:
    x = np.arange(len(labels))
    ax.set_xticks(x)
    if row_idx == nrows - 1:
        ax.set_xticklabels(list(labels), rotation=60, ha="right")
    else:
        ax.set_xticklabels([])



def _seed_offsets(seed_ids: Sequence[str]) -> Dict[str, float]:
    ordered = sorted({str(s).lower() for s in seed_ids}, key=_seed_sort_key)
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: 0.0}
    span = 0.28
    xs = np.linspace(-span, span, num=len(ordered))
    return {sid: float(x) for sid, x in zip(ordered, xs)}



def _signature_plot_df(seed_df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if seed_df.empty:
        return seed_df.copy()
    tmp = seed_df.drop_duplicates(subset=["run_signature"]).copy()
    tmp["short_label"] = tmp["run_signature"].astype(str).map(_short_run_label)
    return sort_rows_by_group(tmp, cfg)




def _plot_seed_points(
    ax,
    signature_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    values: pd.Series,
    yerr: Optional[pd.Series],
    cfg: Mapping[str, Any],
    show_errorbars: bool,
    use_offsets: bool = True,
    alpha: Optional[float] = None,
) -> List[Line2D]:
    if seed_df.empty:
        return []

    xmap = {str(sig): i for i, sig in enumerate(signature_df["run_signature"].astype(str).tolist())}
    tmp = seed_df.copy()
    tmp["_seed_id"] = tmp["run_id"].astype(str).map(_seed_id_from_run_id)
    offsets = _seed_offsets(tmp["_seed_id"].astype(str).tolist()) if use_offsets else {str(s).lower(): 0.0 for s in tmp["_seed_id"].astype(str).unique().tolist()}
    colors = _seed_colors(cfg)
    handles: List[Line2D] = []
    alpha_val = _seed_alpha(cfg, "point_alpha", 0.60) if alpha is None else float(alpha)

    ordered_seed_ids = sorted(tmp["_seed_id"].astype(str).unique().tolist(), key=_seed_sort_key)
    for seed_id in ordered_seed_ids:
        sub = tmp.loc[tmp["_seed_id"].astype(str) == seed_id].copy()
        if sub.empty:
            continue
        xs = np.array([xmap.get(str(sig), np.nan) + offsets.get(str(seed_id).lower(), 0.0) for sig in sub["run_signature"].astype(str)])
        ys = pd.to_numeric(values.loc[sub.index], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not mask.any():
            continue
        xs = xs[mask]
        ys = ys[mask]
        color = colors.get(str(seed_id).lower(), "#1f77b4")
        label = str(seed_id).replace("seed", "seed ")

        if show_errorbars and yerr is not None:
            err = pd.to_numeric(yerr.loc[sub.index], errors="coerce").fillna(0.0).to_numpy(dtype=float)[mask]
            ax.errorbar(xs, ys, yerr=np.abs(err), fmt="o", color=color, alpha=alpha_val, capsize=2, linestyle="none")
        else:
            ax.scatter(xs, ys, color=color, alpha=alpha_val, s=24)

        handles.append(Line2D([0], [0], marker="o", linestyle="none", color=color, label=label))

    return handles



def _plot_seed_bars(
    ax,
    signature_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    values: pd.Series,
    yerr: Optional[pd.Series],
    cfg: Mapping[str, Any],
    show_errorbars: bool,
    alpha: Optional[float] = None,
) -> List[Line2D]:
    if seed_df.empty:
        return []

    xmap = {str(sig): i for i, sig in enumerate(signature_df["run_signature"].astype(str).tolist())}
    tmp = seed_df.copy()
    tmp["_seed_id"] = tmp["run_id"].astype(str).map(_seed_id_from_run_id)
    colors = _seed_colors(cfg)
    handles: List[Line2D] = []
    alpha_val = _seed_alpha(cfg, "bar_alpha", 0.45) if alpha is None else float(alpha)

    ordered_seed_ids = sorted(tmp["_seed_id"].astype(str).unique().tolist(), key=_seed_sort_key)
    for seed_id in ordered_seed_ids:
        sub = tmp.loc[tmp["_seed_id"].astype(str) == seed_id].copy()
        if sub.empty:
            continue
        xs = np.array([xmap.get(str(sig), np.nan) for sig in sub["run_signature"].astype(str)])
        ys = pd.to_numeric(values.loc[sub.index], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not mask.any():
            continue
        xs = xs[mask]
        ys = ys[mask]
        color = colors.get(str(seed_id).lower(), "#1f77b4")
        label = str(seed_id).replace("seed", "seed ")

        if show_errorbars and yerr is not None:
            err = pd.to_numeric(yerr.loc[sub.index], errors="coerce").fillna(0.0).to_numpy(dtype=float)[mask]
            ax.bar(xs, ys, width=0.72, color=color, alpha=alpha_val, yerr=np.abs(err), capsize=2)
        else:
            ax.bar(xs, ys, width=0.72, color=color, alpha=alpha_val)

        handles.append(Line2D([0], [0], marker="s", linestyle="none", color=color, label=label))

    return handles



def plot_absolute_pillar_panels(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    registry: Mapping[str, TargetSpec],
    seed_df: Optional[pd.DataFrame] = None,
) -> None:
    if df.empty:
        return

    use_seed_points = (_seed_mode(cfg) == "mean") and (seed_df is not None) and (not seed_df.empty)
    plot_df = sort_rows_by_group(df.copy(), cfg)
    signature_df = _signature_plot_df(seed_df, cfg) if use_seed_points else plot_df
    ncols = 2
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    show_seed_errorbars = bool((((cfg.get("plots", {}) or {}).get("per_model_panels", {}) or {}).get("show_seed_errorbars", False)))

    for pillar_name in _pillar_names(cfg):
        metrics = [m for m in _metrics_for_pillar(cfg, pillar_name) if m in (seed_df.columns if use_seed_points else plot_df.columns)]
        if not metrics:
            continue
        nrows = math.ceil(len(metrics) / ncols)
        figsize = _pillar_figsize(cfg, len(metrics), ncols=ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel()
        fig_handles: List[Line2D] = []

        for panel_idx, (ax, metric) in enumerate(zip(axes_flat, metrics)):
            row_idx = panel_idx // ncols
            if use_seed_points:
                values = _to_numeric_series(seed_df[metric])
                std_col = _std_column_for_metric(seed_df, metric)
                yerr = _to_numeric_series(seed_df[std_col]).abs().fillna(0.0) if std_col else None
                _apply_group_backgrounds(ax, signature_df, cfg)
                handles = _plot_seed_points(ax, signature_df, seed_df, values, yerr, cfg, show_seed_errorbars, use_offsets=False)
                if handles and not fig_handles:
                    fig_handles = handles
                _apply_panel_xticks(ax, signature_df["short_label"].astype(str).tolist(), row_idx, nrows)
                center, band, label = _target_value_for_metric(seed_df, metric, registry)
                ylim = _per_metric_y_limits(values.tolist(), yerr.tolist() if yerr is not None else [0.0] * len(values), center, band)
            else:
                values = _to_numeric_series(plot_df[metric])
                std_col = _std_column_for_metric(plot_df, metric)
                yerr = _to_numeric_series(plot_df[std_col]).abs().fillna(0.0) if std_col else pd.Series(0.0, index=plot_df.index)
                x = np.arange(len(plot_df))
                _apply_group_backgrounds(ax, plot_df, cfg)
                ax.errorbar(x, values, yerr=yerr, fmt="o", capsize=3)
                _apply_panel_xticks(ax, plot_df["short_label"].astype(str).tolist(), row_idx, nrows)
                center, band, label = _target_value_for_metric(plot_df, metric, registry)
                ylim = _per_metric_y_limits(values.tolist(), yerr.tolist(), center, band)

            ax.set_title(metric_display_name(metric))
            ax.grid(True, axis="y", alpha=0.3)
            _plot_target_line(ax, center, band, label)
            if ylim is not None:
                ax.set_ylim(*ylim)

        for ax in axes_flat[len(metrics):]:
            ax.axis("off")

        if fig_handles:
            fig.legend(handles=fig_handles, loc="upper right", frameon=False, ncol=max(1, len(fig_handles)))
        fig.subplots_adjust(hspace=0.38)
        fig.suptitle(f"Absolute per-model results by pillar: {pillar_name}", y=0.995)
        fig.savefig(_plot_dir(cfg) / f"absolute_pillar_panel__{_sanitize_name(pillar_name)}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def _baseline_band_for_delta(baseline_runs: pd.DataFrame, metric: str) -> Optional[Tuple[float, float]]:
    mm = baseline_minmax_for_metric(baseline_runs, metric)
    if mm is None:
        return None
    mean, vmin, vmax = mm
    return vmin - mean, vmax - mean



def plot_delta_pillar_panels(
    df: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    cfg: Mapping[str, Any],
    registry: Mapping[str, TargetSpec],
    seed_df: Optional[pd.DataFrame] = None,
) -> None:
    if df.empty or baseline_runs.empty:
        return

    plot_df = sort_rows_by_group(df.copy(), cfg)
    use_seed_points = (_seed_mode(cfg) == "mean") and (seed_df is not None) and (not seed_df.empty)
    signature_df = _signature_plot_df(seed_df, cfg) if use_seed_points else plot_df
    ncols = 2
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    show_seed_errorbars = bool((((cfg.get("plots", {}) or {}).get("delta_panels", {}) or {}).get("show_model_spread_if_available", False)))

    for pillar_name in _pillar_names(cfg):
        base_metrics = [m for m in _metrics_for_pillar(cfg, pillar_name) if m in (seed_df.columns if use_seed_points else plot_df.columns)]
        if not base_metrics:
            continue
        nrows = math.ceil(len(base_metrics) / ncols)
        figsize = _pillar_figsize(cfg, len(base_metrics), ncols=ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel()
        fig_handles: List[Line2D] = []

        for panel_idx, (ax, metric) in enumerate(zip(axes_flat, base_metrics)):
            row_idx = panel_idx // ncols
            if use_seed_points:
                values = _improvement_vs_baseline_series(seed_df, baseline_runs, metric, registry)
                if values is None:
                    ax.axis("off")
                    continue
                values = _to_numeric_series(values)
                std_col = _std_column_for_metric(seed_df, metric)
                yerr = _to_numeric_series(seed_df[std_col]).abs().fillna(0.0) if std_col else None
                _apply_group_backgrounds(ax, signature_df, cfg)
                handles = _plot_seed_bars(ax, signature_df, seed_df, values, yerr, cfg, show_seed_errorbars)
                if handles and not fig_handles:
                    fig_handles = handles
                _apply_panel_xticks(ax, signature_df["short_label"].astype(str).tolist(), row_idx, nrows)
            else:
                values = _improvement_vs_baseline_series(plot_df, baseline_runs, metric, registry)
                if values is None:
                    ax.axis("off")
                    continue
                values = _to_numeric_series(values)
                x = np.arange(len(plot_df))
                _apply_group_backgrounds(ax, plot_df, cfg)
                ax.bar(x, values)
                _mark_baseline_ticks(ax, plot_df)
                _apply_panel_xticks(ax, plot_df["short_label"].astype(str).tolist(), row_idx, nrows)

            ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
            band = _baseline_improvement_band(baseline_runs, metric, registry)
            if band is not None and all(math.isfinite(float(v)) for v in band):
                ax.axhspan(float(band[0]), float(band[1]), alpha=0.14, color="black")

            ax.set_title(f"{metric_display_name(metric)} (positive = better)")
            ax.grid(True, axis="y", alpha=0.3)

        for ax in axes_flat[len(base_metrics):]:
            ax.axis("off")

        if fig_handles:
            fig.legend(handles=fig_handles, loc="upper right", frameon=False, ncol=max(1, len(fig_handles)))
        fig.subplots_adjust(hspace=0.38)
        fig.suptitle(f"Delta vs baseline by pillar: {pillar_name}", y=0.995)
        fig.savefig(_plot_dir(cfg) / f"delta_pillar_panel__{_sanitize_name(pillar_name)}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)



def plot_delta_target_pillar_panels(
    df: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    cfg: Mapping[str, Any],
    registry: Mapping[str, TargetSpec],
    seed_df: Optional[pd.DataFrame] = None,
) -> None:
    if df.empty or baseline_runs.empty:
        return

    metric_columns = _metric_list_from_cfg(cfg)
    use_seed_points = (_seed_mode(cfg) == "mean") and (seed_df is not None) and (not seed_df.empty)
    if use_seed_points:
        delta_df = compute_delta_to_target_distance(seed_df, baseline_runs, metric_columns, registry)
        plot_df = sort_rows_by_group(delta_df.copy(), cfg)
        signature_df = _signature_plot_df(seed_df, cfg)
    else:
        delta_df = compute_delta_to_target_distance(df, baseline_runs, metric_columns, registry)
        plot_df = sort_rows_by_group(delta_df.copy(), cfg)
        signature_df = plot_df

    ncols = 2
    dpi = int((cfg.get("plots", {}) or {}).get("dpi", 300))
    show_seed_errorbars = bool((((cfg.get("plots", {}) or {}).get("delta_panels", {}) or {}).get("show_model_spread_if_available", False)))

    for pillar_name in _pillar_names(cfg):
        base_metrics = [m for m in _metrics_for_pillar(cfg, pillar_name) if f"delta_target::{m}" in plot_df.columns]
        if not base_metrics:
            continue
        nrows = math.ceil(len(base_metrics) / ncols)
        figsize = _pillar_figsize(cfg, len(base_metrics), ncols=ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.ravel()
        fig_handles: List[Line2D] = []

        for panel_idx, (ax, metric) in enumerate(zip(axes_flat, base_metrics)):
            row_idx = panel_idx // ncols
            dcol = f"delta_target::{metric}"
            if use_seed_points:
                values = _to_numeric_series(plot_df[dcol]) * -1.0
                std_col = _std_column_for_metric(seed_df, metric)
                yerr = _to_numeric_series(seed_df[std_col]).abs().fillna(0.0) if std_col else None
                _apply_group_backgrounds(ax, signature_df, cfg)
                handles = _plot_seed_bars(ax, signature_df, plot_df, values, yerr, cfg, show_seed_errorbars)
                if handles and not fig_handles:
                    fig_handles = handles
                _apply_panel_xticks(ax, signature_df["short_label"].astype(str).tolist(), row_idx, nrows)
            else:
                values = _to_numeric_series(plot_df[dcol]) * -1.0
                x = np.arange(len(plot_df))
                _apply_group_backgrounds(ax, plot_df, cfg)
                ax.bar(x, values)
                _mark_baseline_ticks(ax, plot_df)
                _apply_panel_xticks(ax, plot_df["short_label"].astype(str).tolist(), row_idx, nrows)

            ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black", alpha=0.8)
            base_dist = compute_distance_to_target(baseline_runs, [metric], registry)
            if metric in base_dist.columns:
                s = _to_numeric_series(base_dist[metric]).dropna()
                if not s.empty:
                    mean = float(s.mean())
                    band_lo = mean - float(s.max())
                    band_hi = mean - float(s.min())
                    ax.axhspan(band_lo, band_hi, alpha=0.14, color="black")

            ax.set_title(f"{metric_display_name(metric)} (positive = better)")
            ax.grid(True, axis="y", alpha=0.3)

        for ax in axes_flat[len(base_metrics):]:
            ax.axis("off")

        if fig_handles:
            fig.legend(handles=fig_handles, loc="upper right", frameon=False, ncol=max(1, len(fig_handles)))
        fig.subplots_adjust(hspace=0.38)
        fig.suptitle(f"Delta in target-closeness by pillar: {pillar_name}", y=0.995)
        fig.savefig(_plot_dir(cfg) / f"delta_target_pillar_panel__{_sanitize_name(pillar_name)}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def write_diagnostics(df: pd.DataFrame, baseline_runs: pd.DataFrame, cfg: Mapping[str, Any]) -> None:
    table_dir = _table_dir(cfg)
    df.to_csv(table_dir / "master_metrics_flat.csv", index=False)
    baseline_runs.to_csv(table_dir / "baseline_runs_selected.csv", index=False)


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)

    if args.eval_root is not None:
        cfg.setdefault("paths", {})["evaluation_root"] = args.eval_root
    if args.output_dir is not None:
        cfg.setdefault("paths", {})["output_dir"] = args.output_dir
    if args.model_prefixes is not None:
        cfg.setdefault("selection", {})["model_prefixes"] = list(args.model_prefixes)
    if args.baseline_prefix is not None:
        cfg.setdefault("baseline", {})["prefix"] = args.baseline_prefix
    if args.baseline_seed_prefixes is not None:
        cfg.setdefault("baseline", {})["seed_prefixes"] = list(args.baseline_seed_prefixes)

    eval_root = str((cfg.get("paths", {}) or {}).get("evaluation_root"))
    output_dir = str((cfg.get("paths", {}) or {}).get("output_dir"))
    model_prefixes = list(((cfg.get("selection", {}) or {}).get("model_prefixes", []) or []))
    recursive = bool((cfg.get("selection", {}) or {}).get("recursive", False))

    _mkdir(output_dir)
    _plot_dir(cfg)
    _table_dir(cfg)

    summary_files = find_summary_files(eval_root, model_prefixes, recursive)
    if not summary_files:
        raise ValueError(f"No evaluation_summary.json files found under {eval_root} for prefixes {model_prefixes!r}.")

    flattened = [flatten_summary(load_summary_json(p)) for p in summary_files]
    master_df = pd.DataFrame(flattened)
    if master_df.empty:
        raise ValueError("Loaded summary files but produced an empty master dataframe.")

    raw_master_df = attach_grouping(master_df, cfg)
    baseline_runs = identify_baseline_runs(raw_master_df, cfg)
    v0_run = identify_primary_baseline_run(raw_master_df)
    v0_seed_family = identify_v0_seed_family(raw_master_df)

    master_df = aggregate_seeds(raw_master_df, cfg)
    master_df = attach_grouping(master_df, cfg)

    baseline_mean_row = compute_baseline_mean_row(raw_master_df, baseline_runs, cfg)
    if baseline_mean_row is not None and _seed_mode(cfg) != "mean":
        master_df = pd.concat([master_df, baseline_mean_row.to_frame().T], ignore_index=True)
        master_df = attach_grouping(master_df, cfg)

    registry = build_target_registry(cfg)
    metric_columns = _metric_list_from_cfg(cfg)

    write_diagnostics(master_df, baseline_runs, cfg)

    plot_kinds = set((((cfg.get("plots", {}) or {}).get("make", [])) or []))
    if "absolute_heatmap" in plot_kinds:
        plot_absolute_heatmap(master_df, cfg, metric_columns)
    if "target_heatmap" in plot_kinds:
        plot_target_heatmap(master_df, cfg, metric_columns, registry)
    if "better_than_baseline_heatmap" in plot_kinds:
        plot_better_than_baseline_heatmap(master_df, baseline_runs, cfg, metric_columns, registry)
    if "better_than_baseline_spread_heatmap" in plot_kinds:
        plot_better_than_baseline_spread_heatmap(master_df, baseline_runs, cfg, metric_columns, registry)
    if "better_than_v0_heatmap" in plot_kinds:
        plot_better_than_v0_heatmap(
            master_df,
            v0_run,
            cfg,
            metric_columns,
            registry,
            seed_df=raw_master_df,
            v0_seed_family=v0_seed_family,
        )
    if "better_than_v0_spread_heatmap" in plot_kinds:
        plot_better_than_v0_spread_heatmap(
            master_df,
            v0_run,
            baseline_runs,
            cfg,
            metric_columns,
            registry,
            seed_df=raw_master_df,
            v0_seed_family=v0_seed_family,
        )
    if "absolute_pillar_panels" in plot_kinds:
        plot_absolute_pillar_panels(master_df, cfg, registry, seed_df=raw_master_df)
    if "delta_pillar_panels" in plot_kinds:
        plot_delta_pillar_panels(master_df, baseline_runs, cfg, registry, seed_df=raw_master_df)
    if "delta_target_pillar_panels" in plot_kinds:
        plot_delta_target_pillar_panels(master_df, baseline_runs, cfg, registry, seed_df=raw_master_df)

    print("[paper2_figures_new] Done.")
    print(f"[paper2_figures_new] Loaded {len(summary_files)} summary files from: {eval_root}")
    print(f"[paper2_figures_new] Output directory: {output_dir}")
    if baseline_runs.empty:
        print("[paper2_figures_new] WARNING: No baseline runs matched. Delta plots may be absent.")
    else:
        print(f"[paper2_figures_new] Baseline runs used: {len(baseline_runs)}")


if __name__ == "__main__":
    main()