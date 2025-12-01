import argparse
import logging
from typing import Any, MutableMapping, cast

from data_analysis_pipeline.cli.launch_split_creation import run as run_split
from data_analysis_pipeline.cli.launch_statistics import run as run_statistics
from data_analysis_pipeline.cli.launch_comparison import run as run_comparison
from data_analysis_pipeline.cli.launch_small_batches_creation import run as run_small_batches
from data_analysis_pipeline.cli.launch_correlations import run as run_correlation
from sbgm.utils import load_config


print('>>> Entered main_data_app.py')

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    logger.info("In main data app: Setting up to run...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["create_splits", "run_statistics", "run_comparison", "create_small_batches", "run_correlation"], required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip heavy computations and only re-plot from cached results where available.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # Ensure cfg is treated as a mutable mapping so string-key assignment is accepted by type checkers
    cfg = cast(MutableMapping[str, Any], cfg)

    # Wire global knobs down the pipeline
    if 'global' not in cfg:
        cfg['global'] = {}
    cfg['global']['plot_only'] = bool(args.plot_only)

    if args.mode == "create_splits":
        logger.info("\t\tIn main data app: Creating data splits%s..." % (" [PLOT-ONLY]" if args.plot_only else ""))
        run_split(cfg)
    elif args.mode == "run_statistics":
        logger.info("\t\tIn main data app: Running data statistics%s..." % (" [PLOT-ONLY]" if args.plot_only else ""))
        run_statistics(cfg)
    elif args.mode == "run_comparison":
        logger.info("\t\tIn main data app: Running data comparison%s..." % (" [PLOT-ONLY]" if args.plot_only else ""))
        run_comparison(cfg)
    elif args.mode == "create_small_batches":
        logger.info("\t\tIn main data app: Creating small data batches%s..." % (" [PLOT-ONLY]" if args.plot_only else ""))
        run_small_batches(cfg)
    elif args.mode == "run_correlation":
        logger.info("\t\tIn main data app: Running data correlation%s..." % (" [PLOT-ONLY]" if args.plot_only else ""))
        run_correlation(cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("Finished run...")

if __name__ == "__main__":
    main()