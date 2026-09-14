"""Machine-specific paths, kept separate from scientific configuration."""
from datetime import datetime, timezone
import os
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def external_output(path):
    """Reject accidental writes into source, including through symlinks."""
    resolved = Path(path).expanduser().resolve()
    if resolved == SOURCE_ROOT or SOURCE_ROOT in resolved.parents:
        raise ValueError(f"Output must be outside the source repository: {resolved}")
    return resolved


def setup_environment():
    """Set portable defaults; explicit environment overrides always take precedence."""
    runs = external_output(os.environ.get("CEDDAR_RUNS", SOURCE_ROOT.parent / "CEDDAR_runs"))
    defaults = {
        "CEDDAR_RUNS": runs,
        "DATA_DIR": SOURCE_ROOT.parent / "Data_DiffMod_small",
        "STATS_LOAD_DIR": SOURCE_ROOT / "repro/assets/stats/statistics_run/stats",
        "CKPT_DIR": runs / "checkpoints",
        "SAMPLE_DIR": runs / "samples",
        "LOG_DIR": runs / "logs",
        "TMPDIR": runs / "tmp",
        "XDG_CACHE_HOME": runs / "cache",
        "EXP_DATE": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "SLURM_CPUS_PER_TASK": "1",
        "DEVICE": "cpu",
        "MPLBACKEND": "Agg",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))
    os.environ.setdefault("EVAL_DIR", str(Path(os.environ["SAMPLE_DIR"]) / "evaluation"))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ["XDG_CACHE_HOME"]) / "matplotlib"))
    os.environ.setdefault("TORCH_HOME", str(Path(os.environ["XDG_CACHE_HOME"]) / "torch"))
    for key in ("CKPT_DIR", "SAMPLE_DIR", "EVAL_DIR", "LOG_DIR", "TMPDIR",
                "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME"):
        path = external_output(os.environ[key])
        os.environ[key] = str(path)
        path.mkdir(parents=True, exist_ok=True)


def validate_output_paths(cfg):
    """Check the model pipeline's resolved output roots; inputs remain read-only."""
    paths = cfg.get("paths", {})
    for key in ("checkpoint_dir", "sample_dir", "evaluation_dir", "log_dir", "path_save", "root"):
        if paths.get(key):
            paths[key] = str(external_output(paths[key]))
    for section, keys in {
        "diagnostics": ("histogram_path",),
        "full_gen_eval": ("gen_dir", "eval_dir"),
    }.items():
        for key in keys:
            if cfg.get(section, {}).get(key):
                cfg[section][key] = str(external_output(cfg[section][key]))
