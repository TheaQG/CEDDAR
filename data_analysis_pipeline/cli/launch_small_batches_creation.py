import logging

from data_analysis_pipeline.preprocess.small_batches_main import small_batches_main


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run(cfg):
    logger.info("Launching small data batches creation")
    small_batches_main(cfg)
    logger.info("Done.")