'''
    This script is used to run the ERA5 download pipeline locally.
'''

import argparse
import pathlib
import shutil
import logging
import yaml
import os
from copy import deepcopy

def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{var_name}' is not set. "
            f"Export it before running the pipeline."
        )
    return value


def _resolve_env_placeholders(obj):
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        result = obj
        while "${env:" in result:
            start = result.index("${env:")
            end = result.index("}", start)
            env_name = result[start + 6:end]
            env_value = _require_env(env_name)
            result = result[:start] + env_value + result[end + 1:]
        return result
    return obj

from era5_download_pipeline.pipeline import download, transfer, stream
from era5_download_pipeline.utils.logging_utils import setup_logging

cfg_path = pathlib.Path(__file__).resolve().parents[1] / "cfg/era5_pressure_pipeline.yaml"
cfg = _resolve_env_placeholders(deepcopy(yaml.safe_load(cfg_path.read_text())))

parser = argparse.ArgumentParser(description="Run the ERA5 download pipeline locally.")
parser.add_argument("--log", default="era5_logs/era5_download.log",
                    help="Path to the log file. Default: era5_logs/era5_download.log")
parser.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging level. Default: INFO")
parser.add_argument("--mode", choices=["bulk", "stream"], default="bulk",
                    help="Mode of operation. 'bulk': download everything first, 'stream': download->rsyng->delete per file.")
parser.add_argument("--workers", type=int, default=2,
                                        help="Threads for download work in stream mode.")
parser.add_argument("--rsync-workers", type=int, default=1,
                    help="Parallel rsync transfers in stream mode. Default: 1")
args = parser.parse_args()


# Set up logging
setup_logging(args.log, args.log_level)

log = logging.getLogger(__name__)
log.debug("Configuration loaded from %s", cfg_path)
log.debug("Resolved tmp_dir: %s", cfg['tmp_dir'])
log.debug("Resolved remote raw_dir template: %s", cfg['lumi']['raw_dir'])

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
        raw_dir_tmpl = cfg["lumi"]["raw_dir"]

        if pressure_levels:
            for plev in pressure_levels:
                remote_dir = raw_dir_tmpl.format(var=vshort, plev=plev)
                transfer.rsync_push(tmp_dir, remote_dir, cfg)
            shutil.rmtree(tmp_dir)
            continue

        remote_dir = raw_dir_tmpl.format(var=vshort, plev="").rstrip("/")
        transfer.rsync_push(tmp_dir, remote_dir, cfg)
        shutil.rmtree(tmp_dir)
elif args.mode == "stream":
    # Streaming mode: download and transfer each file immediately
    stream.download_transfer_delete(
        cfg,
        n_workers=args.workers,
        rsync_workers=args.rsync_workers,
    )
else:
    raise ValueError(f"Unknown mode: {args.mode}. Use 'bulk' or 'stream'.")