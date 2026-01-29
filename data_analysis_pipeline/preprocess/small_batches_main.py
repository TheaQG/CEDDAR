import logging
from data_analysis_pipeline.preprocess.create_small_data_batches import run_small_data_batch_creation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def small_batches_main(cfg):
    """
        Entry point for small data split + Zarr conversion. Run this from launch_small_batches.py
    """

    logger.info("[INFO] Running main data statistics")
    run_small_data_batch_creation(cfg)
