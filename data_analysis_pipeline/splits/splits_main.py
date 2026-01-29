import logging
from data_analysis_pipeline.splits.create_train_valid_test import create_data_splits, convert_splits_to_zarr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def splits_main(cfg):
    """
        Entry point for data split + Zarr conversion. Run this from launch_splits.py
    """

    if cfg.split_params.run_split:
        logger.info("Splitting .npz files to train/valid/test")
        create_data_splits(cfg)

    if cfg.split_params.run_convert_to_zarr:
        logger.info("Saving data splits to Zarr archives")
        convert_splits_to_zarr(cfg)