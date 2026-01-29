# baseline/launch_baseline_evaluation.py
from __future__ import annotations
import argparse, json, yaml, logging
from pathlib import Path
from typing import Any, Dict
from baselines.evaluate_baselines.evaluation_baselines import run_baseline_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

def _load_cfg(p: str | Path) -> Dict[str, Any]:
    p = Path(p)
    if p.suffix.lower() in {".yml", ".yaml"}:
        return yaml.safe_load(p.read_text())
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text())
    raise ValueError(f"Unsupported config format: {p}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="YAML/JSON config for baseline evaluation")
    ap.add_argument("--baseline", required=True, help="baseline type: bilinear | qmap | unet_sr | ...")
    ap.add_argument("--no-plots", action="store_true", help="compute metrics only")
    args = ap.parse_args()

    cfg = _load_cfg(args.cfg)
    run_baseline_evaluation(cfg, baseline_type=args.baseline, make_plots=not args.no_plots)

if __name__ == "__main__":
    main()