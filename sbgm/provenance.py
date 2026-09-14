"""Record observed runtime facts without modifying model state or random streams."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import uuid

from omegaconf import OmegaConf
import torch
import yaml

from sbgm.runtime import SOURCE_ROOT, external_output


def checkpoint_info(path, weights_key="network_params"):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "weights_key": weights_key}


def git_info(root=SOURCE_ROOT):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()
    try:
        return {"branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, subprocess.CalledProcessError):
        return {"branch": None, "commit": None, "dirty": None}


def effective_sampler_settings(sampler, kwargs):
    """Bind the SAME kwargs used for inference; omitted arguments use Python defaults."""
    bound = inspect.signature(sampler).bind(**kwargs)
    bound.apply_defaults()
    tensor_keys = {"score_model", "y", "cond_img", "lsm_cond", "topo_cond", "lr_ups"}
    settings = {k: v for k, v in bound.arguments.items() if k not in tensor_keys}
    settings["device"] = str(settings["device"])
    settings["name"] = sampler.__name__
    return OmegaConf.to_container(OmegaConf.create(settings), resolve=True)


def write_provenance(directory, cfg, *, stage, device=None, checkpoint=None, sampler=None, **details):
    """Write a unique YAML per invocation/stage. Null means not observed, not inferred."""
    directory = external_output(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    config = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    full = config.get("full_gen_eval", {})
    state = torch.get_rng_state().numpy().tobytes()
    manifest = {
        "timestamp": now.isoformat(), "stage": stage, "hostname": socket.gethostname(),
        "git": git_info(), "command": sys.argv, "cwd": str(Path.cwd()),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "torch": {"version": str(torch.__version__), "device": str(device) if device is not None else None,
                  "cuda": torch.version.cuda, "hip": torch.version.hip,
                  "threads": torch.get_num_threads(),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "checkpoint": checkpoint or {"path": None, "sha256": None, "weights_key": None},
        "data": {"root": config.get("paths", {}).get("data_dir"),
                 "split": config.get("data_handling", {}).get("split", full.get("split"))},
        "config": config, "sampler": sampler,
        "rng": {"cpu_state_sha256": hashlib.sha256(state).hexdigest(),
                "generation_seed_requested": full.get("seed"),
                "scope": "process/sweep; GenerationRunner does not reset the seed"},
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
        **details,
    }
    filename = f"{now:%Y%m%dT%H%M%S.%fZ}_{stage}_{os.getpid()}_{uuid.uuid4().hex[:8]}.yaml"
    path = directory / filename
    with path.open("x") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return path
