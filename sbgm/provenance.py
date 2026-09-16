"""Record observed runtime facts without modifying model state or random streams."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
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
from sbgm.sigma_control import build_edm_schedule
from sbgm.sampling_noise import PROTOCOL, sampling_noise_mode


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
    tensor_keys = {"score_model", "y", "cond_img", "lsm_cond", "topo_cond", "lr_ups", "noise_audit"}
    settings = {k: v for k, v in bound.arguments.items() if k not in tensor_keys}
    settings["device"] = str(settings["device"])
    settings["name"] = sampler.__name__
    if sampler.__name__ == "edm_sampler":
        schedule_kwargs = {key: settings[key] for key in inspect.signature(build_edm_schedule).parameters}
        _, settings["schedule"] = build_edm_schedule(**schedule_kwargs)
    return OmegaConf.to_container(OmegaConf.create(settings), resolve=True)


def write_provenance(directory, cfg, *, stage, device=None, checkpoint=None, sampler=None, **details):
    """Write a unique YAML per invocation/stage. Null means not observed, not inferred."""
    directory = external_output(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    config = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    full = config.get("full_gen_eval", {})
    state = torch.get_rng_state().numpy().tobytes()
    noise_mode = sampling_noise_mode(config)
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
                "noise_mode": noise_mode,
                "protocol": PROTOCOL if noise_mode == 'paired' else None,
                "scope": ("seed/date/role/step; see meta/noise/<date>.json for actual draw hashes"
                          if noise_mode == 'paired' else "process/sweep; GenerationRunner does not reset the seed")},
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
        **details,
    }
    filename = f"{now:%Y%m%dT%H%M%S.%fZ}_{stage}_{os.getpid()}_{uuid.uuid4().hex[:8]}.yaml"
    path = directory / filename
    with path.open("x") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return path


def sigma_generation_metadata(base_dir, grid, cfg):
    """Read observed sampler calls; config-derived plot labels are not evidence."""
    from sbgm.sigma_control import sigma_star_kwargs
    full = cfg.get("full_gen_eval", {})
    expected = sigma_star_kwargs(cfg.get("edm", {}), full.get("sigma_control", {}))
    records = []
    for value in grid:
        paths = sorted((Path(base_dir) / f"sigma_star={float(value):.2f}" / "meta").glob("*_generation_*.yaml"))
        if not paths:
            if full.get("sigma_control", {}).get("require_generation_manifest", False):
                raise FileNotFoundError(f"Missing generation provenance for sigma*={value}")
            records.append({"sigma_star": float(value), "sampler": None})
            continue
        if len(paths) != 1:
            raise ValueError(f"Ambiguous generation provenance for sigma*={value}: {paths}")
        if full.get("sigma_control", {}).get("require_generation_manifest", False):
            completion = paths[0].parent / "manifest.json"
            if not completion.is_file() or json.loads(completion.read_text()).get("n_days", 0) < 1:
                raise ValueError(f"Generation did not complete for sigma*={value}")
        manifest = yaml.safe_load(paths[0].read_text())
        sampler = manifest["sampler"]
        actual_mode = manifest.get('rng', {}).get('noise_mode', 'sequential')
        if actual_mode != sampling_noise_mode(cfg):
            raise ValueError(f"Generation/evaluation noise_mode mismatch: {actual_mode}")
        if actual_mode == 'paired' and manifest['rng']['generation_seed_requested'] != full.get('seed'):
            raise ValueError('Generation/evaluation paired seed mismatch')
        wanted = {**expected, "sigma_star": float(value)}
        wanted.update({k: cfg['edm'][k] for k in
                       ('sigma_min', 'sigma_max', 'rho', 'S_churn', 'S_min', 'S_max', 'S_noise')
                       if k in cfg.get('edm', {})})
        wanted['num_steps'] = int(cfg.get('edm', {}).get('sampling_steps', 40))
        for key, value_expected in wanted.items():
            if sampler.get(key) != value_expected:
                raise ValueError(f"Generation/evaluation mismatch for {key}: {sampler.get(key)} != {value_expected}")
        records.append({"sigma_star": float(value), "manifest": str(paths[0]), "sampler": sampler})
    return records
