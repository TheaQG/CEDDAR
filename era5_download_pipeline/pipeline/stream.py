"""
    Download --> rsync to LUMI --> delete in a streaming fashion.
    Each worker handles exactly one <variable, year> combination.
"""

import pathlib # For file paths
import logging
import concurrent.futures # For parallel execution
from . import download, transfer # Import download and transfer functions
from .remote_utils import remote_years_present # Import utility to check remote years


logger = logging.getLogger(__name__)

def _job(args):
    """
    Worker function to download and transfer data for a specific variable and year.
    """
    var_long, vshort, yr, cfg = args
    tmp_dir = pathlib.Path(cfg['tmp_dir']) / vshort
    tmp_dir.mkdir(parents=True, exist_ok=True)  # Ensure the temporary directory exists

    out_nc = tmp_dir / f"{vshort}_{yr}.nc"
    remote_dir = cfg['lumi']['raw_dir'].format(var=vshort)

    logger.info("Starting job for %s %s to %s", var_long, yr, out_nc)
    # 1) Download
    download.download_year(var_long, yr, out_nc, cfg)
    logger.info("Downloaded %s for %s to %s", var_long, yr, out_nc)

    # 2) Transfer and auto-delete
    logger.info("Transferring %s to %s", out_nc, remote_dir)
    try:
        transfer.rsync_push(out_nc, remote_dir, cfg)
    except Exception as e:
        logger.error("Failed to transfer %s to %s: %s", out_nc, remote_dir, e)
        raise
    # transfer.rsync_push(out_nc, remote_dir, cfg)
    logger.info("Transferred %s to %s", out_nc, remote_dir)

    # 3) Extra safety: Remove the local file after transfer
    if out_nc.exists():
        out_nc.unlink()

    # Remove empty directory if it exists
    if not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()

def _job_pressure(args):
    """
    Worker function to download and transfer data for a specific variable, year, and pressure level.
    """
    var_long, vshort, yr, pressure_level, cfg = args
    tmp_dir = pathlib.Path(cfg['tmp_dir']) / vshort
    tmp_dir.mkdir(parents=True, exist_ok=True)  # Ensure the temporary directory exists

    out_nc = tmp_dir / f"{vshort}_{pressure_level}_{yr}.nc"
    remote_dir = cfg['lumi']['raw_dir'].format(var=vshort, plev=pressure_level)

    logger.info("Starting job for %s %s at %s hPa to %s", var_long, yr, pressure_level, out_nc)
    # 1) Download
    download.download_year_pressure(var_long, yr, pressure_level, out_nc, cfg)
    logger.info("Downloaded %s for %s at %s hPa to %s", var_long, yr, pressure_level, out_nc)

    # 2) Transfer and auto-delete
    logger.info("Transferring %s to %s", out_nc, remote_dir)
    try:
        transfer.rsync_push(out_nc, remote_dir, cfg)
    except Exception as e:
        logger.error("Failed to transfer %s to %s: %s", out_nc, remote_dir, e)
        raise
    
    logger.info("Transferred %s to %s", out_nc, remote_dir)

    # 3) Extra safety: Remove the local file after transfer
    if out_nc.exists():
        out_nc.unlink()

    # Remove empty directory if it exists
    if not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()


def download_transfer_delete(cfg, n_workers=2):
    """
    Download ERA5 data, transfer it to LUMI, and delete the local copy.
    
    Args:
        cfg (dict): Configuration dictionary containing settings for download and transfer.
        n_workers (int): Number of parallel workers to use for downloading and transferring data.
    """
    # Check if pressure levels are specified
    pressure_levels = cfg.get('pressure_levels')            # None --> Single level run (empty list)
    
    jobs = []

    for var_long, vinfo in cfg['variables'].items():
        vshort = vinfo['short']
        
        if pressure_levels:                                 # --- Pressure level case
            for plev in pressure_levels:
                done = remote_years_present(vshort, cfg, plev=plev)
                logger.info("Remote has %d years for %s at %d hPa: %s", 
                            len(done), vshort, plev, sorted(done) if done else 'None')
                
                for year in range(cfg['years'][0], cfg['years'][1] + 1):
                    # Re-do the MAX year to catch partial transfers (unless all years are present)
                    if year in done and year != max(done): # Skip if year is already present remotely
                        logger.info("Skipping %s for %s at %s hPa as it is already present remotely.", 
                                    vshort, year, plev)
                        continue
                    jobs.append((var_long, vshort, year, plev, cfg))
                    logger.info("Scheduled job for %s %s at %s hPa for year %s",
                                var_long, vshort, plev, year)
        else:                                               # --- Single level case
            done = remote_years_present(vshort, cfg)
            logger.info("Remote has %d years for %s: %s", len(done), vshort, sorted(done) if done else 'None')
            for year in range(cfg['years'][0], cfg['years'][1] + 1):
                # Re-do the MAX year to catch partial transfers (unless all years are present)
                if year in done and year != max(done):
                    logger.info("Skipping %s for %d as it is already present remotely.", vshort, year)
                    continue
                jobs.append((var_long, vshort, year, cfg))

        if not jobs:
            logger.info("No jobs to process for %s. All requested data is already present remotely.", vshort)
            continue

        # I/O bound -> threads are fine
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Every worker builds its own cdsapi.Client instance, so no cross-thread sharing
            if pressure_levels:
                # Use _job_pressure for pressure level data
                executor.map(_job_pressure, jobs)
            else:
                # Use _job for single-level data
                executor.map(_job, jobs)
            # Log the number of jobs processed
            logger.info("Processed %d jobs for variable %s", len(jobs), vshort)

        logger.info("All downloads and transfers completed for variable %s.", vshort)
