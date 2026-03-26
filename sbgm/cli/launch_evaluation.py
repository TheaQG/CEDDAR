import logging

from sbgm.evaluate_sbgm.evaluation_main import evaluation_main as evaluation_main_legacy
from sbgm.evaluate_sbgm.plot_utils import make_publication_outputs

from sbgm.evaluate.evaluate_main import evaluation_main as evaluation_main_v1
from sbgm.evaluate2.evaluate_main import evaluation_main as evaluation_main_v2

logger = logging.getLogger(__name__)


def run_evaluation(cfg, make_plots: bool = True, force_eval2: bool = False):
    """Launch evaluation.

    Resolution order:
      1) If `force_eval2` is True OR cfg.eval2.enabled is True -> eval2.
      2) Else if cfg.full_gen_eval.use_new_eval is True -> v1 (current 'new').
      3) Else -> legacy evaluation + optional publication plots.
    """
    fe = cfg.get("full_gen_eval", {})
    use_v1 = bool(fe.get("use_new_eval", False))
    use_v2 = bool(cfg.get("eval2", {}).get("enabled", False))

    if force_eval2:
        use_v2 = True

    if use_v2:
        logger.info("[launch_evaluation] Using EVAL2 evaluation main.")
        evaluation_main_v2(cfg)
        return

    if use_v1:
        logger.info("[launch_evaluation] Using NEW evaluation main (v1).")
        evaluation_main_v1(cfg)
    else:
        logger.info("[launch_evaluation] Using LEGACY evaluation main.")
        evaluation_main_legacy(cfg)

        # Make publication-ready plots (legacy only)
        if make_plots:
            make_publication_outputs(cfg)
