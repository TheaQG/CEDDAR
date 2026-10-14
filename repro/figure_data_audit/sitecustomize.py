"""Opt-in file-read audit for manuscript figure subprocesses; no plotting changes."""
import json
import os
import sys
from pathlib import Path


def install():
    config_path = os.environ.get("CEDDAR_FIGURE_AUDIT_CONFIG")
    if not config_path:
        return
    config = json.loads(Path(config_path).read_text())
    root = Path(config["data_root"]).resolve()
    forbidden = [Path(p).resolve() for p in config["forbidden_roots"]]
    log = open(Path(config["audit_dir"]) / f"{os.getpid()}.jsonl", "x", buffering=1)
    seen = set()
    argv = list(sys.orig_argv)
    module = argv[argv.index("-m") + 1] if "-m" in argv else "unknown"
    output = argv[argv.index("--output") + 1] if "--output" in argv else ""
    log.write(json.dumps({"event": "start", "module": module, "output": output}) + "\n")

    def audit(event, args):
        if event not in ("open", "os.listdir", "os.scandir"):
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.path.realpath(os.fsdecode(args[0])))
        denied = any(path.is_relative_to(p) for p in forbidden)
        writing = event == "open" and bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if writing and path.is_relative_to(root):
            denied = True
        key = (event, str(path), writing, denied)
        if key not in seen:
            seen.add(key)
            log.write(json.dumps({"event": event, "path": str(path), "writing": writing,
                                  "denied": denied}) + "\n")
        if denied:
            raise PermissionError(f"Figure-data audit blocked {event}: {path}")

    sys.addaudithook(audit)


install()
