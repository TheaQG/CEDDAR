from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import math

import pandas as pd
import yaml
import matplotlib.pyplot as plt


MODEL_GROUPS = {
    "V0__": "baseline",
    "C_": "context_only",
    "D_": "physics",
}

PREFIX_GROUPS = {
    "V0__": "baseline",
    "V0_": "single_variable",
    "C_": "context_only",
    "D_": "physics",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "evaluation_root": "/scratch/project_465002493/quistgaa/Code/CEDDAR/models_and_samples/generated_samples/evaluation",
        "output_dir": "./paper2_summary_outputs",
    },
    "selection": {
        "model_prefixes": [],
        "group_presets": [],
        "pool_mode": "all", # all | by_group | by_prefix
        "recursive": False,
    },
    "baseline": {
        "baseline_prefix": "V0__",
        "baseline_seed_prefixes": ["V0__", "V0__seed"],
        "matching_mode": "signature",  # signature | prefix
    },
    "metrics": {
        "core": [
            "distributional.wasserstein",
            "distributional.ks_stat",
            "probabilistic.crps_mean",
            "probabilistic.pit_ks_D",
            "scale.iss_primary",
            "scale.slope_low",
            "scale.slope_mid",
            "scale.slope_high",
            "scale.corr_gen_lr_low",
            "scale.corr_gen_lr_mid",
            "scale.corr_gen_lr_high",
            "extremes.p99",
            "extremes.p99_9",
            "extremes.rx1day",
            "extremes.rx5day",
            "features.gen_ens_SAL",
            "climatological.annual_sum_mean",
            "climatological.annual_sum_std",
            "temporal.lag1_gen",
            "temporal.wet_mean_length_gen",
            "temporal.dry_mean_length_gen",
        ],
        "lower_is_better": [
            "distributional.wasserstein",
            "distributional.ks_stat",
            "probabilistic.crps_mean",
            "probabilistic.pit_ks_D",
        ],
    },
    "plots": {
        "make": ["baseline_variance", "delta_bars", "heatmap", "group_comparisons", "pillar_panels"],
        "dpi": 300,
        "figsize": [12,6],
        "share_y_within_metric": True,
        "pillar_panel_columns": 2,
    },
    "group_presets": {
        "baseline": ["V0__"],
        "context_only": ["C_"],
        "physics": ["D_"],
        "all": [],
    },
}

# Pillar-to-metric mapping for pillar panel plots

PILLAR_METRIC_GROUPS: Dict[str, List[str]] = {
    "distributional": [
        "distributional.wasserstein",
        "distributional.ks_stat",
    ],
    "probabilistic": [
        "probabilistic.crps_mean",
        "probabilistic.pit_ks_D",
    ],
    "scale": [
        "scale.iss_primary",
        "scale.slope_low",
        "scale.slope_mid",
        "scale.slope_high",
        "scale.corr_gen_lr_low",
        "scale.corr_gen_lr_mid",
        "scale.corr_gen_lr_high",
    ],
    "extremes": [
        "extremes.p99",
        "extremes.p99_9",
        "extremes.rx1day",
        "extremes.rx5day",
    ],
    "features": [
        "features.gen_ens_SAL",
    ],
    "climatological": [
        "climatological.annual_sum_mean",
        "climatological.annual_sum_std",
    ],
    "temporal": [
        "temporal.lag1_gen",
        "temporal.wet_mean_length_gen",
        "temporal.dry_mean_length_gen",
    ],
}

# --------------------------------------------------------------------------------
# Config / CLI helpers
# --------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Paper 2 summary tables and figures from evaluation summaries.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--eval-root", type=str, default=None, help="Root directory containing per-model evaluation folders.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for summary tables and figures.")
    parser.add_argument("--model-prefixes", nargs="*", default=None, help="Model ID prefixes to include.")
    parser.add_argument("--group-presets", nargs="*", default=None, help="Preset names defined in config/group_presets.")
    parser.add_argument("--baseline-prefix", type=str, default=None, help="Primary baseline prefix.")
    parser.add_argument("--baseline-seed-prefixes", nargs="*", default=None, help="Prefixes identifying baseline seed runs.")
    parser.add_argument("--pool-mode", type=str, default=None, choices=["all", "by_group", "by_prefix"], help="Pooling mode.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Override selected metric paths.")
    parser.add_argument("--write-example-config", type=str, default=None, help="Write an example YAML config to this path and exit.")
    return parser.parse_args()


def _sanitize_metric_name(metric: str) -> str:
    return str(metric).replace(".", "__").replace("::", "__").replace("/", "_")


# ------------------- Plotting helpers for run labels and error bars --------------------

def _short_run_label(run_id: str) -> str:
    """
    Keep only the experiment prefix / run-specific part of the model ID and drop
    the long shared experiment-definition suffix starting at `__HR_...`.

    Examples:
      V0_C_0__HR_prcp_DANRA__SIZE_128x128__... -> V0_C_0
      D_IP_1__HR_prcp_DANRA__SIZE_128x128__... -> D_IP_1
      V0__seed1__HR_prcp_DANRA__SIZE_128x128__... -> V0__seed1
    """
    rid = str(run_id)
    marker = "__HR_"
    if marker in rid:
        return rid.split(marker, 1)[0]
    return rid


def _safe_nonnegative_yerr(yerr: Optional[pd.Series | pd.DataFrame | Any]) -> Optional[Any]:
    if yerr is None:
        return None
    arr = pd.to_numeric(yerr, errors="coerce")
    if hasattr(arr, "abs"):
        arr = arr.abs()
    return arr


def _get_plot_dir(cfg: Dict[str, Any]) -> Path:
    out = Path(cfg["paths"]["output_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_metric_columns(dataframe: pd.DataFrame, metric_columns: List[str]) -> List[str]:
    return [m for m in metric_columns if m in dataframe.columns]


def _numeric_metric_frame(dataframe: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
    cols = _resolve_metric_columns(dataframe, metric_columns)
    if not cols:
        return pd.DataFrame(index=dataframe.index)
    return dataframe[cols].apply(pd.to_numeric, errors="coerce")

# --- Helper for pillar display ---

def _metric_display_name(metric_name: str) -> str:
    m = str(metric_name)
    if "." in m:
        return m.split(".", 1)[1]
    return m


def _metrics_for_pillar(metric_columns: List[str], pillar_name: str) -> List[str]:
    requested = PILLAR_METRIC_GROUPS.get(str(pillar_name), [])
    return [m for m in requested if m in metric_columns]


# ------------------------------------------------------------------------------
# Metric helpers for plotting targets and error bars
# ------------------------------------------------------------------------------

def _metric_target_and_label(metric_name: str, dataframe: Optional[pd.DataFrame] = None) -> tuple[Optional[float], Optional[str]]:
    m = str(metric_name)

    # Metrics with obvious ideal values.
    if m.endswith(".ks_stat") or m.endswith(".pit_ks_D"):
        return 0.0, "target=0"
    if m.endswith(".crps_mean") or m.endswith(".wasserstein"):
        return 0.0, "target=0"
    if m.endswith(".gen_ens_SAL") or m.endswith(".gen_SAL"):
        return 0.0, "target=0"
    if m.endswith(".iss_primary"):
        return 1.0, "target=1"
    if m.endswith(".corr_gen_lr_low") or m.endswith(".corr_gen_lr_mid") or m.endswith(".corr_gen_lr_high"):
        return 1.0, "target=1"

    # Metrics where the target is the HR / DANRA reference in the summary itself.
    reference_map = {
        "extremes.p99": "extremes.hr_p99",
        "extremes.p99_9": "extremes.hr_p99_9",
        "extremes.rx1day": "extremes.hr_rx1day",
        "extremes.rx5day": "extremes.hr_rx5day",
        "climatological.annual_sum_mean": "climatological.hr_annual_sum_mean",
        "climatological.annual_sum_std": "climatological.hr_annual_sum_std",
        "temporal.wet_mean_length_gen": "temporal.wet_mean_length_hr",
        "temporal.dry_mean_length_gen": "temporal.dry_mean_length_hr",
        "temporal.lag1_gen": "temporal.lag1_hr",
        "scale.slope_low": "scale.hr_slope_low",
        "scale.slope_mid": "scale.hr_slope_mid",
        "scale.slope_high": "scale.hr_slope_high",
    }
    ref_col = reference_map.get(m)
    if dataframe is not None and ref_col is not None and ref_col in dataframe.columns:
        s = pd.to_numeric(dataframe[ref_col], errors="coerce").dropna()
        if not s.empty:
            return float(s.mean()), "target=HR"

    return None, None


def _metric_std_column(metric_name: str, dataframe: pd.DataFrame) -> Optional[str]:
    candidates = [
        f"{metric_name}_std",
        metric_name.replace("_mean", "_std"),
        metric_name.replace("_gen", "_gen_std"),
    ]
    for c in candidates:
        if c in dataframe.columns:
            return c
    return None


def _plot_target_line(ax, target_value: Optional[float], target_label: Optional[str]) -> None:
    if target_value is None or not math.isfinite(float(target_value)):
        return
    ax.axhline(float(target_value), linestyle="--", linewidth=1.2, color="black", alpha=0.8)
    if target_label:
        ax.text(
            0.98,
            0.98,
            target_label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path is None:
        return cfg
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Config at {config_path} must contain a mapping at top level.")
    return deep_update(cfg, user_cfg)


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.eval_root is not None:
        cfg["paths"]["evaluation_root"] = args.eval_root
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = args.output_dir
    if args.model_prefixes is not None:
        cfg["selection"]["model_prefixes"] = list(args.model_prefixes)
    if args.group_presets is not None:
        cfg["selection"]["group_presets"] = list(args.group_presets)
    if args.pool_mode is not None:
        cfg["selection"]["pool_mode"] = args.pool_mode
    if args.baseline_prefix is not None:
        cfg["baseline"]["baseline_prefix"] = args.baseline_prefix
    if args.baseline_seed_prefixes is not None:
        cfg["baseline"]["baseline_seed_prefixes"] = list(args.baseline_seed_prefixes)
    # Normalize baseline selectors to stripped strings and ensure baseline_prefix is also represented.
    bp = str(cfg.get("baseline", {}).get("baseline_prefix", "") or "").strip()
    bsp = [str(x).strip() for x in cfg.get("baseline", {}).get("baseline_seed_prefixes", []) or [] if str(x).strip()]
    if bp and bp not in bsp:
        bsp.insert(0, bp)
    cfg["baseline"]["baseline_seed_prefixes"] = bsp
    if args.metrics is not None:
        cfg["metrics"]["core"] = list(args.metrics)
    return cfg


def write_example_config(path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)



# --------------------------------------------------------------------------------
# A. Discovery / loading of summary json files
# --------------------------------------------------------------------------------

def expand_requested_prefixes(cfg: Dict[str, Any]) -> List[str]:
    prefixes: List[str] = list(cfg["selection"].get("model_prefixes", []))
    presets: Iterable[str] = cfg["selection"].get("group_presets", []) or []
    preset_map: Dict[str, List[str]] = cfg.get("group_presets", {}) or {}
    for preset in presets:
        prefixes.extend(preset_map.get(preset, []))
    out: List[str] = []
    seen = set()
    for p in prefixes:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def find_summary_files(eval_root: str | Path, requested_prefixes: Optional[List[str]] = None, recursive: bool = False) -> List[Path]:
    root = Path(eval_root)
    if not root.exists():
        raise FileNotFoundError(f"Evaluation root does not exist: {root}")
    
    pattern = "**/summary/evaluation_summary.json" if recursive else "*/summary/evaluation_summary.json"
    candidates = sorted(root.glob(pattern))
    if not requested_prefixes:
        return candidates

    selected: List[Path] = []
    for file_path in candidates:
        model_dir = file_path.parent.parent.name
        if match_model_prefix(model_dir, requested_prefixes) is not None:
            selected.append(file_path)
    return selected


def load_summary_json(file_path: str | Path) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_model_prefix(file_name: str, requested_prefixes: List[str]) -> Optional[str]:
    for prefix in requested_prefixes:
        if file_name.startswith(prefix):
            return prefix
    return None


SEED_TOKEN_PREFIXES = ["seed", "Seed", "SEED"]


def _is_seed_token(token: str) -> bool:
    tok = str(token).strip()
    if not tok:
        return False
    return tok.startswith(tuple(SEED_TOKEN_PREFIXES))


def extract_run_signature(run_id: str) -> str:
    """
    Return a seed-invariant run signature.

    Expected naming examples:
      - V0__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5...
      - V0__seed1__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5...
      - V0__seed2__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5...

    The signature should remove the optional seed token after the first model token,
    but otherwise keep the full experiment definition intact.
    """
    rid = str(run_id).strip()
    if not rid:
        return rid
    parts = rid.split("__")
    if len(parts) >= 2 and _is_seed_token(parts[1]):
        return "__".join([parts[0]] + parts[2:])
    return rid


def _matches_baseline_prefix(run_id: str, baseline_seed_prefixes: List[str]) -> bool:
    rid = str(run_id)
    sig = extract_run_signature(rid)
    for prefix in baseline_seed_prefixes:
        p = str(prefix)
        if rid.startswith(p) or sig.startswith(p):
            return True
    return False



# --------------------------------------------------------------------------------
# B. Parsing / flattening
# --------------------------------------------------------------------------------

def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}.{key}" if parent_key else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, new_key))
        else:
            out[new_key] = value
    return out


def flatten_summary(summary_json: Dict[str, Any]) -> Dict[str, Any]:
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
        metrics = pillar_payload.get("metrics", {}) or {}
        row.update(_flatten_dict(metrics, pillar_name))
        notes = pillar_payload.get("notes", []) or []
        row[f"{pillar_name}.notes"] = " | ".join(str(x) for x in notes)

    row["model_group"] = infer_model_group(str(row.get("run_id") or row.get("model_key") or ""))
    row["run_signature"] = extract_run_signature(str(row.get("run_id") or row.get("model_key") or ""))
    row["baseline_family"] = str(row["run_signature"]).split("__")[0] if str(row.get("run_signature", "")) else ""
    return row


def build_master_dataframe(flattened_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    if not flattened_summaries:
        return pd.DataFrame()
    df = pd.DataFrame(flattened_summaries)
    preferred_cols = [
        "run_id",
        "run_signature",
        "baseline_family",
        "model_key",
        "model_group",
        "eval_root",
        "generated_root",
        "config_path",
    ]
    other_cols = [c for c in df.columns if c not in preferred_cols]
    return df[preferred_cols + sorted(other_cols)]


def infer_model_group(model_name: str) -> str:
    name = str(model_name)

    # More specific prefixes first
    for prefix, group in PREFIX_GROUPS.items():
        if name.startswith(prefix):
            return group
    for exact_name, group in MODEL_GROUPS.items():
        if name.startswith(exact_name):
            return group
    return "other"


# -------------------------------------------------------------------------------- #
# C. Baseline handling
# -------------------------------------------------------------------------------- #

def identify_baseline_runs(
    dataframe: pd.DataFrame,
    baseline_seed_prefixes: List[str],
    matching_mode: str = "signature",
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    prefixes = [str(p).strip() for p in (baseline_seed_prefixes or []) if str(p).strip()]
    if not prefixes:
        return dataframe.iloc[0:0].copy()

    mode = str(matching_mode).strip().lower()
    run_ids = dataframe["run_id"].astype(str) if "run_id" in dataframe.columns else pd.Series("", index=dataframe.index)
    model_keys = dataframe["model_key"].astype(str) if "model_key" in dataframe.columns else pd.Series("", index=dataframe.index)
    signatures = (
        dataframe["run_signature"].astype(str)
        if "run_signature" in dataframe.columns
        else run_ids.map(extract_run_signature)
    )

    mask = pd.Series(False, index=dataframe.index)

    if mode == "prefix":
        for p in prefixes:
            mask = mask | run_ids.str.startswith(p) | model_keys.str.startswith(p)
        return dataframe.loc[mask].copy()

    # Default and recommended: signature-based matching.
    # Prefer the seed-invariant signature, but also allow direct run_id/model_key prefix matches.
    for p in prefixes:
        mask = (
            mask
            | signatures.str.startswith(p)
            | run_ids.str.startswith(p)
            | model_keys.str.startswith(p)
        )
    return dataframe.loc[mask].copy()


def compute_baseline_reference(dataframe: pd.DataFrame, baseline_runs: pd.DataFrame, metric_columns: List[str]) -> pd.Series:
    if baseline_runs.empty:
        raise ValueError("No baseline runs found; cannot compute baseline reference.")
    cols = [c for c in metric_columns if c in baseline_runs.columns]
    return baseline_runs[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0)


def compute_baseline_seed_variance(dataframe: pd.DataFrame, baseline_runs: pd.DataFrame, metric_columns: List[str]) -> pd.DataFrame:
    if baseline_runs.empty:
        return pd.DataFrame(columns=["metric", "mean", "std", "min", "max", "n"])

    cols = [c for c in metric_columns if c in baseline_runs.columns]
    numeric = baseline_runs[cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "metric": col,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
                "n": int(len(s)),
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------- #
# D. Derived metrics
# -------------------------------------------------------------------------------- #

def compute_delta_table(dataframe: pd.DataFrame, baseline_reference: pd.Series, metric_columns: List[str]) -> pd.DataFrame:
    df = dataframe.copy()
    for col in metric_columns:
        if col in df.columns and col in baseline_reference.index:
            df[f"delta::{col}"] = pd.to_numeric(df[col], errors="coerce") - float(baseline_reference[col])
    return df


def flag_within_baseline_variance(delta_table: pd.DataFrame, baseline_seed_variance: pd.DataFrame) -> pd.DataFrame:
    out = delta_table.copy()
    if baseline_seed_variance.empty:
        return out
    spread_map = baseline_seed_variance.set_index("metric")["std"].to_dict()
    for metric, std_val in spread_map.items():
        delta_col = f"delta::{metric}"
        flag_col = f"within_seed_var::{metric}"
        if delta_col in out.columns:
            out[flag_col] = pd.to_numeric(out[delta_col], errors="coerce").abs() <= float(std_val)
    return out


# -------------------------------------------------------------------------------- #
# E. Plotting (stub design only, to be implemented with actual plotting code later)
# -------------------------------------------------------------------------------- #

def plot_baseline_variance(baseline_seed_variance: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    if baseline_seed_variance is None or baseline_seed_variance.empty:
        return None

    dpi = int(cfg.get("plots", {}).get("dpi", 200))
    figsize = tuple(cfg.get("plots", {}).get("figsize", [12, 6]))
    metrics = baseline_seed_variance["metric"].astype(str).tolist()
    means = pd.to_numeric(baseline_seed_variance["mean"], errors="coerce")
    stds = pd.to_numeric(baseline_seed_variance["std"], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(metrics))
    ax.errorbar(x, means, yerr=stds, fmt="o", capsize=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics, rotation=60, ha="right")
    ax.set_title("Baseline seed variance")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)

    out_path = _get_plot_dir(cfg) / "baseline_seed_variance.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return None


def plot_delta_bars(delta_table: pd.DataFrame, within_baseline_variance: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    metric_columns = list(cfg.get("metrics", {}).get("core", []))
    delta_cols = [f"delta::{m}" for m in metric_columns if f"delta::{m}" in delta_table.columns]
    if delta_table.empty or not delta_cols:
        return None

    dpi = int(cfg.get("plots", {}).get("dpi", 200))
    plot_dir = _get_plot_dir(cfg)

    run_ids = [_short_run_label(x) for x in delta_table["run_id"].astype(str).tolist()]
    for delta_col in delta_cols:
        vals = pd.to_numeric(delta_table[delta_col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        metric_name = delta_col.replace("delta::", "")
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(run_ids)), 5))
        ax.bar(run_ids, vals)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_title(f"Delta vs baseline: {metric_name}")
        ax.set_ylabel("Delta")
        ax.tick_params(axis="x", rotation=60)
        ax.grid(True, axis="y", alpha=0.3)
        out_path = plot_dir / f"delta_bar__{_sanitize_metric_name(metric_name)}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return None


def plot_metric_heatmap(delta_table: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    metric_columns = list(cfg.get("metrics", {}).get("core", []))
    if delta_table.empty:
        return None

    metric_df = _numeric_metric_frame(delta_table, metric_columns)
    if metric_df.empty:
        return None

    plot_df = metric_df.copy()
    plot_df.index = [_short_run_label(x) for x in delta_table["run_id"].astype(str).tolist()]
    plot_df = plot_df.dropna(axis=1, how="all")
    if plot_df.empty:
        return None

    # Normalize each metric column independently to make cross-model differences visible.
    norm_df = plot_df.copy()
    for col in norm_df.columns:
        s = pd.to_numeric(norm_df[col], errors="coerce")
        finite = s.dropna()
        if finite.empty:
            norm_df[col] = s
            continue
        vmin = float(finite.min())
        vmax = float(finite.max())
        if math.isclose(vmax, vmin):
            norm_df[col] = 0.5
        else:
            norm_df[col] = (s - vmin) / (vmax - vmin)

    dpi = int(cfg.get("plots", {}).get("dpi", 200))
    fig, ax = plt.subplots(figsize=(max(10, 0.6 * norm_df.shape[1]), max(6, 0.35 * norm_df.shape[0])))
    im = ax.imshow(norm_df.values, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(norm_df.shape[1]))
    ax.set_xticklabels(norm_df.columns.tolist(), rotation=60, ha="right")
    ax.set_yticks(range(norm_df.shape[0]))
    ax.set_yticklabels(norm_df.index.tolist())
    ax.set_title("Absolute metric heatmap (per-metric normalized)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative position within metric [0, 1]")

    out_path = _get_plot_dir(cfg) / "absolute_metric_heatmap.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return None


def plot_group_comparisons(delta_table: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    if delta_table.empty or "model_group" not in delta_table.columns or "run_id" not in delta_table.columns:
        return None

    metric_columns = list(cfg.get("metrics", {}).get("core", []))
    metric_df = _numeric_metric_frame(delta_table, metric_columns)
    if metric_df.empty:
        return None

    plot_dir = _get_plot_dir(cfg)
    dpi = int(cfg.get("plots", {}).get("dpi", 200))

    tmp = pd.concat([
        delta_table[["run_id", "model_group"]].copy(),
        metric_df,
    ], axis=1)

    groups = [g for g in sorted(tmp["model_group"].dropna().astype(str).unique().tolist())]
    if not groups:
        return None

    for metric_name in metric_columns:
        if metric_name not in tmp.columns:
            continue

        group_frames = []
        for g in groups:
            sub = tmp.loc[tmp["model_group"].astype(str) == g, ["run_id", "model_group", metric_name]].copy()
            if sub.empty:
                continue
            sub[metric_name] = pd.to_numeric(sub[metric_name], errors="coerce")
            sub = sub.loc[sub[metric_name].notna()].copy()
            if sub.empty:
                continue
            std_col = _metric_std_column(metric_name, delta_table)
            if std_col is not None and std_col in delta_table.columns:
                sub[std_col] = pd.to_numeric(
                    delta_table.loc[sub.index, std_col], errors="coerce"
                )
            group_frames.append((g, sub, std_col))

        if not group_frames:
            continue

        ncols = 1 if len(group_frames) == 1 else 2
        nrows = math.ceil(len(group_frames) / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(max(10, 6 * ncols), max(4, 4 * nrows)),
            squeeze=False,
            sharey=bool(cfg.get("plots", {}).get("share_y_within_metric", True)),
        )
        axes_flat = axes.ravel()

        target_value, target_label = _metric_target_and_label(metric_name, delta_table)

        # Compute a common y-range across groups for this metric to improve comparability.
        all_y = []
        all_yerr = []
        for _, sub, std_col in group_frames:
            yy = pd.to_numeric(sub[metric_name], errors="coerce").dropna()
            if not yy.empty:
                all_y.extend(yy.tolist())
            if std_col is not None and std_col in sub.columns:
                ee = _safe_nonnegative_yerr(sub[std_col])
                ee = pd.to_numeric(ee, errors="coerce").dropna()
                if not ee.empty:
                    all_yerr.extend(ee.tolist())
        common_ymin = None
        common_ymax = None
        if all_y:
            ymin = min(all_y)
            ymax = max(all_y)
            extra = max(all_yerr) if all_yerr else 0.0
            pad = 0.05 * max(1e-12, ymax - ymin + extra)
            common_ymin = ymin - extra - pad
            common_ymax = ymax + extra + pad
            if target_value is not None and math.isfinite(float(target_value)):
                common_ymin = min(common_ymin, float(target_value) - pad)
                common_ymax = max(common_ymax, float(target_value) + pad)

        for ax, (group_name, sub, std_col) in zip(axes_flat, group_frames):
            x = list(range(len(sub)))
            y = pd.to_numeric(sub[metric_name], errors="coerce")
            yerr = None
            if std_col is not None and std_col in sub.columns:
                yerr = _safe_nonnegative_yerr(sub[std_col])
            ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels([_short_run_label(r) for r in sub["run_id"].astype(str).tolist()], rotation=60, ha="right")
            ax.set_title(group_name)
            ax.set_ylabel(metric_name)
            ax.grid(True, axis="y", alpha=0.3)
            _plot_target_line(ax, target_value, target_label)
            if common_ymin is not None and common_ymax is not None:
                ax.set_ylim(common_ymin, common_ymax)
# ------------------------------------------------------------------------------
# Pillar panel plots: one figure per evaluation pillar, pooling metrics
# ------------------------------------------------------------------------------
def plot_pillar_panels(delta_table: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    if delta_table.empty or "model_group" not in delta_table.columns or "run_id" not in delta_table.columns:
        return None

    metric_columns = list(cfg.get("metrics", {}).get("core", []))
    metric_df = _numeric_metric_frame(delta_table, metric_columns)
    if metric_df.empty:
        return None

    plot_dir = _get_plot_dir(cfg)
    dpi = int(cfg.get("plots", {}).get("dpi", 200))
    ncols = int(cfg.get("plots", {}).get("pillar_panel_columns", 2))
    groups = [g for g in sorted(delta_table["model_group"].dropna().astype(str).unique().tolist())]
    if not groups:
        return None

    tmp = pd.concat([
        delta_table[["run_id", "model_group"]].copy(),
        metric_df,
    ], axis=1)

    for pillar_name in PILLAR_METRIC_GROUPS.keys():
        pillar_metrics = _metrics_for_pillar(metric_columns, pillar_name)
        if not pillar_metrics:
            continue

        n_panels = len(pillar_metrics)
        nrows = math.ceil(n_panels / ncols)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(max(10, 6 * ncols), max(4, 4 * nrows)),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for ax, metric_name in zip(axes_flat, pillar_metrics):
            pillar_group_stats = []
            for g in groups:
                sub = tmp.loc[tmp["model_group"].astype(str) == g, ["run_id", "model_group", metric_name]].copy()
                if sub.empty:
                    continue
                vals = pd.to_numeric(sub[metric_name], errors="coerce").dropna()
                if vals.empty:
                    continue
                pillar_group_stats.append((g, float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, int(len(vals))))

            if not pillar_group_stats:
                ax.axis("off")
                continue

            x = list(range(len(pillar_group_stats)))
            means = [t[1] for t in pillar_group_stats]
            errs = [abs(t[2]) for t in pillar_group_stats]
            labels = [t[0] for t in pillar_group_stats]
            ns = [t[3] for t in pillar_group_stats]

            ax.bar(x, means, yerr=errs, capsize=5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, ns)], rotation=20, ha="right")
            ax.set_title(_metric_display_name(metric_name))
            ax.set_ylabel("metric value")
            ax.grid(True, axis="y", alpha=0.3)

            target_value, target_label = _metric_target_and_label(metric_name, delta_table)
            _plot_target_line(ax, target_value, target_label)

        for ax in axes_flat[n_panels:]:
            ax.axis("off")

        fig.suptitle(f"Grouped results by pillar: {pillar_name}", y=0.995)
        out_path = plot_dir / f"pillar_panel__{_sanitize_metric_name(pillar_name)}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return None

    #     for ax in axes_flat[len(group_frames):]:
    #         ax.axis("off")

    #     fig.suptitle(f"Per-model results by group: {metric_name}", y=0.995)
    #     out_path = plot_dir / f"group_subplots__{_sanitize_metric_name(metric_name)}.png"
    #     fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    #     plt.close(fig)
    # return None


# ------------------------------------------------------------------------------
# Additional group mean + spread plot
# ------------------------------------------------------------------------------

def plot_group_mean_with_spread(delta_table: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    if delta_table.empty or "model_group" not in delta_table.columns:
        return None

    metric_columns = list(cfg.get("metrics", {}).get("core", []))
    metric_df = _numeric_metric_frame(delta_table, metric_columns)
    if metric_df.empty:
        return None

    tmp = pd.concat([delta_table[["model_group"]].copy(), metric_df], axis=1)
    mean_df = tmp.groupby("model_group", dropna=False).mean(numeric_only=True)
    std_df = tmp.groupby("model_group", dropna=False).std(numeric_only=True)
    if mean_df.empty:
        return None

    plot_dir = _get_plot_dir(cfg)
    dpi = int(cfg.get("plots", {}).get("dpi", 200))

    for metric_name in mean_df.columns:
        vals = pd.to_numeric(mean_df[metric_name], errors="coerce")
        errs = _safe_nonnegative_yerr(pd.to_numeric(std_df[metric_name], errors="coerce").fillna(0.0))
        valid = vals.notna()
        if valid.sum() == 0:
            continue

        vals = vals.loc[valid]
        errs = errs.loc[valid]
        groups = vals.index.astype(str).tolist()

        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(groups)), 5))
        x = list(range(len(groups)))
        ax.bar(x, vals.values, yerr=errs.values, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_title(f"Group mean ± spread: {metric_name}")
        ax.set_ylabel("Mean metric value")
        ax.grid(True, axis="y", alpha=0.3)

        target_value, target_label = _metric_target_and_label(metric_name, delta_table)
        _plot_target_line(ax, target_value, target_label)

        out_path = plot_dir / f"group_mean_spread__{_sanitize_metric_name(metric_name)}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return None


# -------------------------------------------------------------------------------- #
# F. Output
# -------------------------------------------------------------------------------- #

def write_master_tables(dataframe: pd.DataFrame, delta_table: pd.DataFrame, baseline_seed_variance: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(out / "master_metrics.csv", index=False)
    delta_table.to_csv(out / "delta_vs_baseline.csv", index=False)
    baseline_seed_variance.to_csv(out / "baseline_seed_variance.csv", index=False)
    if "model_group" in dataframe.columns:
        group_summary = dataframe.groupby("model_group", dropna=False).size().reset_index(name="n_runs")
        group_summary.to_csv(out / "group_counts.csv", index=False)


def write_latex_tables(delta_table: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "delta_vs_baseline.tex", "w", encoding="utf-8") as f:
        f.write(delta_table.to_latex(index=False))


def save_plot(plot, file_name: str | Path):
    # Placeholder for future figure objects.
    return None


# -------------------------------------------------------------------------------- #
# Main entry point
# -------------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()

    if args.write_example_config is not None:
        write_example_config(args.write_example_config)
        return

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    eval_root = cfg["paths"]["evaluation_root"]
    output_dir = cfg["paths"]["output_dir"]
    requested_prefixes = expand_requested_prefixes(cfg)
    recursive = bool(cfg["selection"].get("recursive", False))
    metric_columns = list(cfg["metrics"].get("core", []))
    baseline_prefix = str(cfg["baseline"].get("baseline_prefix", "") or "").strip()
    baseline_seed_prefixes = [str(x).strip() for x in cfg["baseline"].get("baseline_seed_prefixes", []) or [] if str(x).strip()]
    if baseline_prefix and baseline_prefix not in baseline_seed_prefixes:
        baseline_seed_prefixes.insert(0, baseline_prefix)
    baseline_matching_mode = str(cfg["baseline"].get("matching_mode", "signature"))

    summary_files = find_summary_files(eval_root, requested_prefixes=requested_prefixes, recursive=recursive)
    flattened = [flatten_summary(load_summary_json(path)) for path in summary_files]
    master_df = build_master_dataframe(flattened)

    if master_df.empty:
        raise ValueError("No matching evaluation_summary.json files were found.")

    baseline_runs = identify_baseline_runs(
        master_df,
        baseline_seed_prefixes=baseline_seed_prefixes,
        matching_mode=baseline_matching_mode,
    )
    # Diagnostic export: helps verify which runs were selected as baseline candidates.
    diag_dir = Path(output_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    master_df[[c for c in ["run_id", "run_signature", "model_key", "model_group"] if c in master_df.columns]].to_csv(
        diag_dir / "baseline_diagnostic_all_runs.csv", index=False
    )
    baseline_runs[[c for c in ["run_id", "run_signature", "model_key", "model_group"] if c in baseline_runs.columns]].to_csv(
        diag_dir / "baseline_diagnostic_selected_runs.csv", index=False
    )
    if baseline_runs.empty:
        print(
            "[paper2_figures] No baseline runs found yet. Continuing with absolute-result outputs only. "
            f"baseline_prefix={baseline_prefix!r}, baseline_seed_prefixes={baseline_seed_prefixes!r}, "
            f"matching_mode={baseline_matching_mode!r}."
        )
        baseline_seed_variance = pd.DataFrame(columns=["metric", "mean", "std", "min", "max", "n"])
        delta_table = master_df.copy()
    else:
        baseline_reference = compute_baseline_reference(master_df, baseline_runs, metric_columns=metric_columns)
        baseline_seed_variance = compute_baseline_seed_variance(master_df, baseline_runs, metric_columns=metric_columns)
        delta_table = compute_delta_table(master_df, baseline_reference, metric_columns=metric_columns)
        delta_table = flag_within_baseline_variance(delta_table, baseline_seed_variance)

    write_master_tables(master_df, delta_table, baseline_seed_variance, output_dir)

    plot_names = set(cfg.get("plots", {}).get("make", []))
    if "baseline_variance" in plot_names:
        plot_baseline_variance(baseline_seed_variance, cfg)
    if "delta_bars" in plot_names:
        plot_delta_bars(delta_table, delta_table, cfg)
    if "heatmap" in plot_names:
        plot_metric_heatmap(delta_table, cfg)
    if "group_comparisons" in plot_names:
        plot_group_comparisons(delta_table, cfg)
    if "group_comparisons" in plot_names:
        plot_group_mean_with_spread(delta_table, cfg)
    if "pillar_panels" in plot_names:
        plot_pillar_panels(delta_table, cfg)

if __name__ == "__main__":
    main()