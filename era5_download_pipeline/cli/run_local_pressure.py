'''
    This script is used to run the ERA5 download pipeline locally.
'''

import argparse
import pathlib
import shutil
import logging
import yaml

from era5_download_pipeline.pipeline import download, transfer, stream
from era5_download_pipeline.utils.logging_utils import setup_logging

cfg_path = pathlib.Path(__file__).resolve().parents[1] / "cfg/era5_pressure_pipeline.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

parser = argparse.ArgumentParser(description="Run the ERA5 download pipeline locally.")
parser.add_argument("--log", default="era5_logs/era5_download.log",
                    help="Path to the log file. Default: era5_logs/era5_download.log")
parser.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging level. Default: INFO")
parser.add_argument("--mode", choices=["bulk", "stream"], default="bulk",
                    help="Mode of operation. 'bulk': download everything first, 'stream': download->rsyng->delete per file.")
parser.add_argument("--workers", type=int, default=2,
                    help="Threads for stream mode.")
args = parser.parse_args()


# Set up logging
setup_logging(args.log, args.log_level)

log = logging.getLogger(__name__)
log.debug("Configuration loaded from %s", cfg_path)

# Check whether data is single level or pressure level data
pressure_levels = cfg.get('pressure_levels', None)
if pressure_levels is not None:
    log.info("Running in pressure level mode with levels: %s", pressure_levels)
else:
    log.info("Running in single-level mode (no pressure levels specified).")


if args.mode == "bulk":
    # Bulk mode: download all data first, then transfer
    download.pull_all(cfg)
    for var_long, vinfo in cfg['variables'].items():
        vshort = vinfo['short']
        tmp_dir = pathlib.Path(cfg['tmp_dir']) / vshort
        remote_dir = cfg['lumi']['raw_dir'].format(var=vshort)
        transfer.rsync_push(tmp_dir, remote_dir, cfg)
        shutil.rmtree(tmp_dir)
elif args.mode == "stream":
    # Streaming mode: download and transfer each file immediately
    stream.download_transfer_delete(cfg, n_workers=args.workers)
else:
    raise ValueError(f"Unknown mode: {args.mode}. Use 'bulk' or 'stream'.")