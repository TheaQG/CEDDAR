'''
    Transfer data to a remote server using rsync.
'''

import subprocess
import pathlib
import shlex

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
    ssh_cmd = ["ssh", "-o", "IdentitiesOnly=yes"]
    if cfg.get("lumi_key"):
        ssh_cmd += ["-i", cfg['lumi_key']]

    # 0. Ensure the remote directory exists, using shlex
    mkdir_cmd = ssh_cmd + [ssh_target, f"mkdir -p {shlex.quote(remote_dir)}"]
    subprocess.run(mkdir_cmd, check=True) # Ensure the remote directory exists

    # 1. rsync (remove local file on success)
    rsync_cmd = [
        "rsync",
        "-avz",  # Archive mode, verbose, compress files during transfer
        "--remove-source-files" if delete else "",  # Remove source files after transfer if delete is True
        "--progress",  # Show progress during transfer
        "-e", " ".join(ssh_cmd),  # Use SSH for the transfer
        str(local_path) + ("/" if local_path.is_dir() else ""),  # Local directory to sync (ensure trailing slash for directory)
        f"{ssh_target}:{remote_dir}/"  # Remote directory to sync to (ensure trailing slash for directory)
    ]

    result = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # If the command failed, log the error and raise an exception
        logger.error("Rsync failed with return code: %d", result.returncode)
        logger.error("Error output: %s", result.stderr.strip())
        raise RuntimeError(result.stderr.strip())

    else:
        # If the command succeeded, log the output
        logger.info("Rsync completed successfully.")
        logger.debug("Rsync output: %s", result.stdout.strip())