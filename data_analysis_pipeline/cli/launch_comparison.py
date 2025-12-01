import logging

from data_analysis_pipeline.comparison.comparison_main import comparison_main


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def run(cfg):
    logger.info("\t\t\tIn launch file: Launching comparison...")
    comparison_main(cfg)
    logger.info("Done.")