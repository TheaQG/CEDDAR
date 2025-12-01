import logging

from data_analysis_pipeline.stats_analysis.stats_main import statistics_main


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run(cfg):
    logger.info("Launching data statistics")
    statistics_main(cfg)
    logger.info("Done.")