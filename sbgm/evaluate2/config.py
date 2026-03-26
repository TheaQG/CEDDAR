from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RunMode(str, Enum):
    MINIMAL = "minimal"
    COMPUTE = "compute"
    PLOT = "plot"
    BOTH = "both"


CANONICAL_FEATURES = (
    "dates",
    "distributions",
    "extremes",
    "probabilistic",
    "scale",
    "spatial",
    "temporal",
    "sal",
)


@dataclass
class Eval2Config:
    enabled: bool = True
    run_mode: RunMode = RunMode.COMPUTE
    seed: int = 1234

    # optional overrides
    gen_dir: Optional[str] = None
    out_dir: Optional[str] = None

    # data policy
    max_dates: Optional[int] = None
    eval_land_only: bool = True
    prefer_phys: bool = True
    lr_key: Optional[str] = "lr"
    region_mask_path: Optional[str] = None

    # inclusion / overlays
    include_lr: bool = True
    include_pmm: bool = True

    # ensemble policy
    use_ensemble: bool = True
    ensemble_n_members: Optional[int] = None
    ensemble_member_seed: int = 1234

    # plot gating
    make_plots: bool = True

    # execution policy
    fail_fast: bool = False

    # feature selection
    features: List[str] = field(default_factory=list)

    # per-feature settings (opaque to runner)
    feature_cfg: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Eval2Plan:
    run_mode: RunMode
    features: List[str]
    gen_root: str
    out_root: str
    seed: int

    dates: List[str]
    max_dates: Optional[int]

    eval_land_only: bool
    prefer_phys: bool
    lr_key: Optional[str]
    region_mask_path: Optional[str]

    use_ensemble: bool
    ensemble_n_members: Optional[int]
    ensemble_member_seed: int

    include_lr: bool
    include_pmm: bool

    make_plots: bool


def parse_eval2_config(cfg: dict) -> Eval2Config:
    """
    Priority:
      1) cfg['eval2'] if present
      2) derive from legacy cfg['full_gen_eval'] (so P0_eval_quick works immediately)
    """
    if isinstance(cfg.get("eval2", None), dict):
        return _parse_eval2_block(cfg["eval2"])

    fe = cfg.get("full_gen_eval", {})

    plot_only = bool(fe.get("plot_only", False))
    make_plots = bool(fe.get("make_plots", True))
    if plot_only:
        run_mode = RunMode.PLOT
    else:
        run_mode = RunMode.BOTH if make_plots else RunMode.COMPUTE

    # feature-level only (legacy flags)
    legacy_map = {
        "do_dates": "dates",
        "do_dist": "distributions",
        "do_ext": "extremes",
        "do_prob": "probabilistic",
        "do_scale": "scale",
        "do_spat": "spatial",
        "do_temp": "temporal",
        "do_feat": "sal",
    }
    feats: List[str] = [name for k, name in legacy_map.items() if bool(fe.get(k, False))]

    # Prefer families.* enabled if present
    fams = fe.get("families", None)
    feature_cfg: Dict[str, Dict[str, Any]] = {}
    if isinstance(fams, dict) and len(fams) > 0:
        feats = []
        for fam_key, fam_cfg in fams.items():
            if not isinstance(fam_cfg, dict) or not bool(fam_cfg.get("enabled", False)):
                continue
            fk = str(fam_key).lower()
            tgt = None
            if "dates" in fk:
                tgt = "dates"
            elif "distribution" in fk:
                tgt = "distributions"
            elif "extreme" in fk:
                tgt = "extremes"
            elif "prob" in fk:
                tgt = "probabilistic"
            elif "scale" in fk or "psd" in fk:
                tgt = "scale"
            elif "spatial" in fk:
                tgt = "spatial"
            elif "temporal" in fk:
                tgt = "temporal"
            elif "feature" in fk or "sal" in fk:
                tgt = "sal"

            if tgt is not None:
                feats.append(tgt)
                feature_cfg[tgt] = dict(fam_cfg)

        # de-dup preserving order
        seen = set()
        feats = [f for f in feats if not (f in seen or seen.add(f))]

    if not feats:
        feats = ["distributions"]  # sensible default

    max_dates = fe.get("max_dates", None)
    if max_dates in (-1, None):
        max_dates = None
    else:
        max_dates = int(max_dates)

    return Eval2Config(
        enabled=True,
        run_mode=run_mode,
        seed=int(fe.get("seed", 1234)),
        gen_dir=fe.get("gen_dir", None),
        out_dir=fe.get("eval_dir", None),
        max_dates=max_dates,
        eval_land_only=bool(fe.get("eval_land_only", True)),
        prefer_phys=bool(fe.get("prefer_phys", True)),
        lr_key=str(fe.get("lr_key", "lr")) if fe.get("lr_key", None) is not None else None,
        region_mask_path=fe.get("region_mask_path", None),
        include_lr=bool(fe.get("pixel_dist_include_lr", True)),
        include_pmm=True,
        use_ensemble=bool(fe.get("use_ensemble", True)),
        ensemble_n_members=fe.get("ensemble_n_members", None),
        ensemble_member_seed=int(fe.get("ensemble_member_seed", 1234)),
        make_plots=make_plots,
        fail_fast=False,
        features=feats,
        feature_cfg=feature_cfg,
    )


def _parse_eval2_block(c: dict) -> Eval2Config:
    rm = str(c.get("run_mode", "compute")).strip().lower()
    run_mode = RunMode(rm)  # raises if invalid

    feats_raw = c.get("features", [])
    feats = list(feats_raw) if isinstance(feats_raw, list) else []
    feats = [str(f).strip().lower() for f in feats]
    if not feats:
        feats = ["distributions"]

    # per-feature blocks: any dict under a canonical name
    reserved = {
        "enabled", "run_mode", "seed", "gen_dir", "out_dir", "max_dates",
        "eval_land_only", "prefer_phys", "lr_key", "region_mask_path",
        "include_lr", "include_pmm",
        "use_ensemble", "ensemble_n_members", "ensemble_member_seed",
        "make_plots", "fail_fast", "features",
    }
    feature_cfg: Dict[str, Dict[str, Any]] = {}
    for k, v in c.items():
        kk = str(k).strip().lower()
        if kk in reserved:
            continue
        if kk in CANONICAL_FEATURES and isinstance(v, dict):
            feature_cfg[kk] = dict(v)

    max_dates = c.get("max_dates", None)
    if max_dates in (-1, None):
        max_dates = None
    else:
        max_dates = int(max_dates)

    return Eval2Config(
        enabled=bool(c.get("enabled", True)),
        run_mode=run_mode,
        seed=int(c.get("seed", 1234)),
        gen_dir=c.get("gen_dir", None),
        out_dir=c.get("out_dir", None),
        max_dates=max_dates,
        eval_land_only=bool(c.get("eval_land_only", True)),
        prefer_phys=bool(c.get("prefer_phys", True)),
        lr_key=str(c.get("lr_key", "lr")) if c.get("lr_key", None) is not None else None,
        region_mask_path=c.get("region_mask_path", None),
        include_lr=bool(c.get("include_lr", True)),
        include_pmm=bool(c.get("include_pmm", True)),
        use_ensemble=bool(c.get("use_ensemble", True)),
        ensemble_n_members=c.get("ensemble_n_members", None),
        ensemble_member_seed=int(c.get("ensemble_member_seed", 1234)),
        make_plots=bool(c.get("make_plots", True)),
        fail_fast=bool(c.get("fail_fast", False)),
        features=feats,
        feature_cfg=feature_cfg,
    )