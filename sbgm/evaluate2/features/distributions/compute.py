from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sbgm.evaluate2.config import Eval2Plan
from sbgm.evaluate2.data_resolver import EvalDataResolver
from sbgm.evaluate2.store import FeatureStore
from sbgm.evaluate2.features.distributions.feature import DistributionsConfig

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _atomic_write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    tmp.replace(path)


def _atomic_write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _flatten_masked(x: torch.Tensor, mask: Optional[torch.Tensor]) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=np.float64)
    if x.ndim != 2:
        x = x.squeeze()
    if mask is not None:
        if mask.shape != x.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} != x shape {tuple(x.shape)}")
        x = x[mask]
    return x.detach().cpu().numpy().astype(np.float64).ravel()


def _choose_bins_from_hr(
    plan: Eval2Plan,
    resolver: EvalDataResolver,
    cfg: DistributionsConfig,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Choose histogram bin edges.

    Policy:
      - range_policy='hr_percentile': determine upper bound from HR across dates
        using cfg.hr_percentile, but never clip true max.
      - range_policy='fixed': use cfg.value_min/cfg.value_max directly.

    Implementation notes:
      - We stream over dates and sample HR pixels; we do NOT store full fields.
      - To stay light, we subsample pixels per day.
    """

    if cfg.range_policy == "fixed":
        if cfg.value_max is None:
            raise ValueError("range_policy='fixed' requires value_max")
        vmin = float(cfg.value_min) if cfg.value_min is not None else 0.0
        vmax = float(cfg.value_max)
        edges = np.linspace(vmin, vmax, int(cfg.n_bins) + 1, dtype=np.float64)
        meta = {
            "range_policy": cfg.range_policy,
            "value_min": vmin,
            "value_max": vmax,
            "hr_percentile": cfg.hr_percentile,
            "pixel_subsample": None,
            "n_dates_seen": 0,
        }
        return edges, meta

    # hr_percentile
    per_date = 50000
    rng = np.random.default_rng(int(plan.seed))

    samples: List[np.ndarray] = []
    global_max = -np.inf
    n_seen = 0

    for date in plan.dates:
        s = resolver.fetch(date, want_ensemble=False, n_members=None, seed=int(plan.ensemble_member_seed))
        if s.hr is None:
            continue
        x = _flatten_masked(s.hr, s.mask if plan.eval_land_only else None)
        if x.size == 0:
            continue
        n_seen += 1
        global_max = max(global_max, float(np.nanmax(x)))

        if x.size > per_date:
            idx = rng.choice(x.size, size=per_date, replace=False)
            x = x[idx]
        samples.append(x)

        if n_seen >= 50:
            break

    if not samples:
        raise RuntimeError("Could not determine bins: no HR data found")

    all_samp = np.concatenate(samples)
    all_samp = all_samp[np.isfinite(all_samp)]
    if all_samp.size == 0:
        raise RuntimeError("Could not determine bins: HR samples are all non-finite")

    p = float(np.nanpercentile(all_samp, cfg.hr_percentile))
    vmax = float(max(p, global_max))
    vmin = float(cfg.value_min) if cfg.value_min is not None else float(np.nanmin(all_samp))

    if vmax <= vmin:
        vmax = vmin + 1e-6

    edges = np.linspace(vmin, vmax, int(cfg.n_bins) + 1, dtype=np.float64)
    meta = {
        "range_policy": cfg.range_policy,
        "value_min": vmin,
        "value_max": vmax,
        "hr_percentile": cfg.hr_percentile,
        "pixel_subsample": per_date,
        "n_dates_seen": n_seen,
    }
    return edges, meta


def _hist_counts(x: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, int]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.int64), 0
    x = np.clip(x, edges[0], edges[-1])
    c, _ = np.histogram(x, bins=edges)
    return c.astype(np.int64), int(x.size)


def _safe_pdf(counts: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts.astype(np.float64) / float(n)


def _kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return float(np.sum(p * np.log(p / q)))


def _cdf(p: np.ndarray) -> np.ndarray:
    s = float(p.sum())
    if s <= 0:
        return np.zeros_like(p, dtype=np.float64)
    return np.cumsum(p / s)


def _ks_from_hist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.max(np.abs(_cdf(p) - _cdf(q))))


def _w1_from_hist(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    return float(np.sum(np.abs(_cdf(p) - _cdf(q))) * float(dx))


def compute_distributions(plan: Eval2Plan, resolver: EvalDataResolver, store: FeatureStore, cfg: DistributionsConfig) -> None:
    """Compute distribution tables (eval2)."""

    if not cfg.compute_pooled:
        logger.info("[eval2:distributions] compute_pooled=False; nothing to do")
        return

    edges, bin_meta = _choose_bins_from_hr(plan=plan, resolver=resolver, cfg=cfg)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])

    _atomic_write_npz(
        store.path_table("bins.npz"),
        edges=edges,
        centers=centers,
        dx=np.asarray(dx),
        meta=np.asarray([bin_meta], dtype=object),
    )

    B = len(edges) - 1
    pooled: Dict[str, np.ndarray] = {"hr": np.zeros(B, dtype=np.int64), "gen": np.zeros(B, dtype=np.int64)}
    pooled_n: Dict[str, int] = {"hr": 0, "gen": 0}

    if plan.include_lr:
        pooled["lr"] = np.zeros(B, dtype=np.int64)
        pooled_n["lr"] = 0

    # daily (optional)
    dates_used: List[str] = []
    daily_counts: Dict[str, List[np.ndarray]] = {}
    daily_n: Dict[str, List[int]] = {}
    if cfg.compute_daily:
        for k in pooled.keys():
            daily_counts[k] = []
            daily_n[k] = []

    n_dates_loaded = 0

    for date in plan.dates:
        s = resolver.fetch(
            date,
            want_ensemble=bool(plan.use_ensemble and cfg.compute_ensemble),
            n_members=plan.ensemble_n_members,
            seed=int(plan.ensemble_member_seed),
        )

        if s.hr is None:
            continue

        mask = s.mask if plan.eval_land_only else None

        x_hr = _flatten_masked(s.hr, mask)
        c_hr, n_hr = _hist_counts(x_hr, edges)
        pooled["hr"] += c_hr
        pooled_n["hr"] += n_hr

        # GEN: prefer deterministic generated field if present (pmm); else first ensemble member
        x_gen = None
        if getattr(s, "pmm", None) is not None:
            x_gen = _flatten_masked(s.pmm, mask)
        elif getattr(s, "ens", None) is not None and s.ens.ndim == 3 and s.ens.shape[0] > 0:
            x_gen = _flatten_masked(s.ens[0], mask)

        if x_gen is not None:
            c_gen, n_gen = _hist_counts(x_gen, edges)
            pooled["gen"] += c_gen
            pooled_n["gen"] += n_gen
        else:
            c_gen = np.zeros(B, dtype=np.int64)
            n_gen = 0

        if plan.include_lr and getattr(s, "lr", None) is not None:
            lr2 = s.lr.squeeze(0)
            x_lr = _flatten_masked(lr2, None)
            c_lr, n_lr = _hist_counts(x_lr, edges)
            pooled["lr"] += c_lr
            pooled_n["lr"] += n_lr
        else:
            c_lr = np.zeros(B, dtype=np.int64)
            n_lr = 0

        if cfg.compute_daily:
            dates_used.append(date)
            daily_counts["hr"].append(c_hr)
            daily_n["hr"].append(n_hr)
            daily_counts["gen"].append(c_gen)
            daily_n["gen"].append(n_gen)
            if plan.include_lr:
                daily_counts["lr"].append(c_lr)
                daily_n["lr"].append(n_lr)

        n_dates_loaded += 1

    pooled_npz = {}
    for k, c in pooled.items():
        pooled_npz[f"counts_{k}"] = c
        pooled_npz[f"n_{k}"] = np.asarray(pooled_n[k], dtype=np.int64)

    _atomic_write_npz(store.path_table("pooled_counts.npz"), **pooled_npz)
    logger.info("[eval2:distributions] wrote pooled_counts.npz (%d dates loaded)", n_dates_loaded)

    if cfg.compute_daily:
        daily_npz = {"dates": np.asarray(dates_used, dtype=str)}
        for k in daily_counts.keys():
            daily_npz[f"counts_{k}"] = (
                np.stack(daily_counts[k], axis=0) if len(daily_counts[k]) > 0 else np.zeros((0, B), dtype=np.int64)
            )
            daily_npz[f"n_{k}"] = np.asarray(daily_n[k], dtype=np.int64)
        _atomic_write_npz(store.path_table("daily_counts.npz"), **daily_npz)
        logger.info("[eval2:distributions] wrote daily_counts.npz (D=%d)", len(dates_used))

    if cfg.compute_metrics:
        p_hr = _safe_pdf(pooled["hr"], pooled_n["hr"])
        rows: List[List[object]] = []
        header = ["ref", "comp", "kl", "ks", "w1"]

        for comp in [k for k in pooled.keys() if k != "hr"]:
            p_c = _safe_pdf(pooled[comp], pooled_n[comp])
            rows.append([
                "hr",
                comp,
                _kl_div(p_hr, p_c),
                _ks_from_hist(p_hr, p_c),
                _w1_from_hist(p_hr, p_c, dx=dx),
            ])

        _atomic_write_csv(store.path_table("metrics.csv"), header=header, rows=rows)
        logger.info("[eval2:distributions] wrote metrics.csv")

    manifest = {
        "feature": "distributions",
        "n_dates_plan": len(plan.dates),
        "n_dates_loaded": n_dates_loaded,
        "bin_meta": bin_meta,
        "config": asdict(cfg),
        "notes": {
            "baselines": "not supported in eval2",
            "metrics": "KL/KS/W1 computed from histograms only",
        },
    }
    _atomic_write_json(store.path_table("run_manifest.json"), manifest)
