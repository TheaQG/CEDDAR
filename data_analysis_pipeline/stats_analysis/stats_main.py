import logging
from data_analysis_pipeline.stats_analysis.data_stats_pipeline import run_data_statistics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def statistics_main(cfg):
    """
        Entry point for data split + Zarr conversion. Run this from launch_splits.py
    """

    logger.info("[INFO] Running main data statistics")
    run_data_statistics(cfg)