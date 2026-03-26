from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import logging

from sbgm.evaluate2.config import Eval2Config, Eval2Plan

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.tables = self.root / "tables"
        self.figures = self.root / "figures"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)

    def path_table(self, name: str) -> Path:
        return self.tables / name

    def path_figure(self, name: str) -> Path:
        return self.figures / name


class ArtifactStore:
    def __init__(self, out_root: Path):
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        (self.out_root / "features").mkdir(parents=True, exist_ok=True)

    def for_feature(self, feature_name: str) -> FeatureStore:
        return FeatureStore(self.out_root / "features" / feature_name)

    def write_run_metadata(self, cfg_yaml: dict, ev2_cfg: Eval2Config, plan: Eval2Plan) -> None:
        (self.out_root / "config_snapshot.json").write_text(json.dumps(cfg_yaml, indent=2, default=str))
        (self.out_root / "eval2_config.json").write_text(json.dumps(asdict(ev2_cfg), indent=2, default=str))
        (self.out_root / "manifest.json").write_text(json.dumps(asdict(plan), indent=2, default=str))
        logger.info("[eval2] Wrote config_snapshot.json, eval2_config.json, manifest.json to %s", self.out_root)