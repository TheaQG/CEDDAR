
import os

from pathlib import Path


LEGACY_ROOT = Path(
    os.environ.get(
        "CEDDAR_PAPER1_ORIGINAL",
        "/home/theaqg/CEDDAR_runs/paper1_original"
    )
)


REVISION_ROOT = Path(
    os.environ.get(
        "CEDDAR_PAPER1_REVISION",
        "/home/theaqg/CEDDAR_runs/paper1_revision"
    )
)