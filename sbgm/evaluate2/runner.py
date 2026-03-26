from __future__ import annotations

from pathlib import Path
import importlib
import logging

import numpy as np
import torch

from sbgm.evaluate2.config import Eval2Config, Eval2Plan, RunMode
from sbgm.evaluate2.data_resolver import EvalDataResolver
from sbgm.evaluate2.registry import FEATURE_REGISTRY
from sbgm.evaluate2.store import ArtifactStore

logger = logging.getLogger(__name__)


class Eval2Runner:
    def __init__(self, cfg_yaml: dict, ev2_cfg: Eval2Config, gen_root: Path, out_root: Path):
        self.cfg_yaml = cfg_yaml
        self.ev2_cfg = ev2_cfg
        self.gen_root = Path(gen_root)
        self.out_root = Path(out_root)

        seed = int(ev2_cfg.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

        self.resolver = EvalDataResolver(
            gen_root=self.gen_root,
            eval_land_only=bool(ev2_cfg.eval_land_only),
            roi_mask_path=ev2_cfg.region_mask_path,
            prefer_phys=bool(ev2_cfg.prefer_phys),
            lr_phys_key=ev2_cfg.lr_key,
        )

        self.store = ArtifactStore(self.out_root)

    def build_plan(self) -> Eval2Plan:
        cfg = self.ev2_cfg
        features = list(cfg.features)

        all_dates = self.resolver.list_dates()
        dates = all_dates
        if cfg.max_dates is not None and int(cfg.max_dates) > 0:
            dates = dates[: int(cfg.max_dates)]

        plan = Eval2Plan(
            run_mode=cfg.run_mode,
            features=features,
            gen_root=str(self.gen_root),
            out_root=str(self.out_root),
            seed=int(cfg.seed),
            dates=dates,
            max_dates=int(cfg.max_dates) if cfg.max_dates is not None else None,
            eval_land_only=bool(cfg.eval_land_only),
            prefer_phys=bool(cfg.prefer_phys),
            lr_key=cfg.lr_key,
            region_mask_path=cfg.region_mask_path,
            use_ensemble=bool(cfg.use_ensemble),
            ensemble_n_members=cfg.ensemble_n_members,
            ensemble_member_seed=int(cfg.ensemble_member_seed),
            include_lr=bool(cfg.include_lr),
            include_pmm=bool(cfg.include_pmm),
            make_plots=bool(cfg.make_plots),
        )
        self._validate_plan(plan)
        return plan

    def _validate_plan(self, plan: Eval2Plan) -> None:
        if plan.run_mode not in (RunMode.MINIMAL, RunMode.COMPUTE, RunMode.PLOT, RunMode.BOTH):
            raise ValueError(f"Invalid run_mode: {plan.run_mode}")

        unknown = [f for f in plan.features if f not in FEATURE_REGISTRY]
        if unknown:
            raise ValueError(
                "Unknown eval2 features: " + ", ".join(unknown) +
                ". Allowed: " + ", ".join(sorted(FEATURE_REGISTRY.keys()))
            )

        # Guardrail: features must not be internal steps
        for f in plan.features:
            if any(tok in f for tok in ("pool", "daily", "metrics", "plot")):
                raise ValueError(
                    f"Feature name '{f}' looks like an internal step. "
                    "Eval2 features must be feature-level only."
                )

        if plan.run_mode == RunMode.PLOT:
            logger.info("[eval2] run_mode=plot: will not compute tables; features must find existing tables")

    def run(self) -> None:
        plan = self.build_plan()
        self.store.write_run_metadata(cfg_yaml=self.cfg_yaml, ev2_cfg=self.ev2_cfg, plan=plan)

        logger.info("[eval2] Running with run_mode=%s features=%s n_dates=%d",
                    plan.run_mode.value, plan.features, len(plan.dates))

        for feat_name in plan.features:
            self._run_feature(feat_name, plan)

    def _run_feature(self, feat_name: str, plan: Eval2Plan) -> None:
        spec = FEATURE_REGISTRY[feat_name]
        mod_path = spec["module"]
        cls_name = spec["class"]

        logger.info("[eval2] Feature '%s' -> %s:%s", feat_name, mod_path, cls_name)

        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:
            logger.warning("[eval2] Could not import feature module %s for '%s': %s", mod_path, feat_name, e)
            return

        if not hasattr(mod, cls_name):
            logger.warning("[eval2] Feature module %s does not define %s; skipping '%s'", mod_path, cls_name, feat_name)
            return

        feature = getattr(mod, cls_name)()
        feat_store = self.store.for_feature(feat_name)

        do_compute = plan.run_mode in (RunMode.MINIMAL, RunMode.COMPUTE, RunMode.BOTH)
        do_plot = plan.run_mode in (RunMode.PLOT, RunMode.BOTH) and bool(plan.make_plots)

        try:
            feature.run(
                plan=plan,
                resolver=self.resolver,
                store=feat_store,
                do_compute=do_compute,
                do_plot=do_plot,
                feature_cfg=self.ev2_cfg.feature_cfg.get(feat_name, {}),
            )
        except Exception as e:
            logger.exception("[eval2] Feature '%s' failed: %s", feat_name, e)
            if bool(self.ev2_cfg.fail_fast):
                raise