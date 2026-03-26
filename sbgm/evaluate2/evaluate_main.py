from __future__ import annotations

from pathlib import Path
import logging

from sbgm.utils import get_model_string
from sbgm.evaluate2.config import parse_eval2_config
from sbgm.evaluate2.runner import Eval2Runner

logger = logging.getLogger(__name__)


def _default_gen_dir(cfg: dict) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "generation" / model_str


def _default_eval_dir(cfg: dict) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "evaluation" / model_str / "eval2"


def evaluation_main(cfg: dict):
    ev2 = parse_eval2_config(cfg)

    gen_root = Path(ev2.gen_dir) if ev2.gen_dir is not None else _default_gen_dir(cfg)
    out_root = Path(ev2.out_dir) if ev2.out_dir is not None else _default_eval_dir(cfg)
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("[eval2.evaluation_main] gen_root: %s", gen_root)
    logger.info("[eval2.evaluation_main] out_root: %s", out_root)

    runner = Eval2Runner(cfg_yaml=cfg, ev2_cfg=ev2, gen_root=gen_root, out_root=out_root)
    runner.run()

    logger.info("[eval2.evaluation_main] Done. Outputs at: %s", out_root)
    return out_root