'''
    Transfer data to a remote server using rsync.
'''

import subprocess
import pathlib
import shlex
import time

import logging
logger = logging.getLogger(__name__)

def rsync_push(local_dir:pathlib.Path,
               remote_dir:str,
               cfg,
               delete=True
               ):
    """    
        Transfer files from a local directory to a remote directory using rsync over SSH.
    """

    # Expand user and resolve to absolute path
    local_path = pathlib.Path(local_dir).expanduser().resolve()
    ssh_target = f"{cfg['lumi']['user']}@{cfg['lumi']['host']}"
    ssh_cmd = [
        "ssh",
        "-o", "IdentitiesOnly=yes",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=10",
        "-o", "TCPKeepAlive=yes",
        "-o", "ConnectTimeout=30",
    ]
    if cfg.get("lumi_key"):
        ssh_cmd += ["-i", cfg['lumi_key']]

    # 0. Ensure the remote directory exists, using shlex
    mkdir_cmd = ssh_cmd + [ssh_target, f"mkdir -p {shlex.quote(remote_dir)}"]
    mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True)
    if mkdir_result.returncode != 0:
        logger.error("Remote mkdir failed with return code: %d", mkdir_result.returncode)
        logger.error("Remote mkdir stderr: %s", mkdir_result.stderr.strip())
        raise RuntimeError(mkdir_result.stderr.strip())

    # 1. rsync (remove local file on success)
    rsync_cmd = [
        "rsync",
        "-av",
        "--partial",
        "--append",
        "--contimeout=30",
        "--timeout=600",
        "--progress",
        "-e", " ".join(ssh_cmd),
        str(local_path) + ("/" if local_path.is_dir() else ""),
        f"{ssh_target}:{remote_dir}/"
    ]
    if delete:
        rsync_cmd.insert(3, "--remove-source-files")

    max_attempts = 3
    last_error = None
    for attempt in range(1, max_attempts + 1):
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("Rsync completed successfully.")
            logger.debug("Rsync output: %s", result.stdout.strip())
            return

        last_error = result.stderr.strip() or result.stdout.strip()
        logger.error("Rsync failed with return code: %d", result.returncode)
        logger.error("Error output: %s", last_error)

        if attempt < max_attempts:
            backoff = 10 * attempt
            logger.info("Retrying rsync in %d seconds...", backoff)
            time.sleep(backoff)

    raise RuntimeError(last_error or "rsync failed without error output")