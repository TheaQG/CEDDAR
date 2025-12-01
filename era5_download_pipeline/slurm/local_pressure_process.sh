#!/usr/bin/env bash
# local_process.sh - start an SSH agent and run the ERA5 download pipeline locally then clean up

set -euo pipefail # Fail on any error, treat unset variables as an error, and fail on any command in a pipeline that fails

eval "$(/usr/bin/ssh-agent -s)" # Start the ssh-agent (using full path to bypass aliases)
ssh-add ~/.ssh/id_ed25519 # Add the SSH key to the agent (will prompt for passphrase once)

# Run the Python script with the provided configuration
python3 -m era5_download_pipeline.cli.run_local_pressure \
        --mode stream --workers 3 \
        --log era5_logs/era5_run_restartable.log

# kill the agent
ssh-agent -k