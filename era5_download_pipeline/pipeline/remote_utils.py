import subprocess
import re # Regular expressions for parsing CDO output
import logging
import shlex

logger = logging.getLogger(__name__)
# Regular expression to match "..._YYYY.nc" or "..._YYY_850.nc" (if pressure level is included)
YEAR_RE = re.compile(r"_(\d{4})\.nc$")

def remote_years_present(vshort:str,
                         cfg: dict,
                         plev: int | None = None) -> set[int]:
    """
        Return a set of integer years already present in the 
        LUMI remote directory for the given variable `vshort`.
        If `plev` is provided, it will check the directory for that pressure level.
        Assumes files are named in the format "<vshort>_YYYY.nc".
        (e.g., "pev_1991.nc", "t2m_2020.nc").
    """

    if plev is not None: # If a pressure level is specified, include it in the directory path
        remote_dir = cfg["lumi"]["raw_dir"].format(var=vshort, plev=plev)
    else:
        remote_dir = cfg["lumi"]["raw_dir"].format(var=vshort)
        
    ssh_target = f'{cfg["lumi"]["user"]}@{cfg["lumi"]["host"]}'
    cmd = ["ssh", ssh_target, f"ls -1 {remote_dir} 2>/dev/null"] # List files in the remote directory. "2>/dev/null" suppresses error messages if the directory does not exist.

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Could not list %s: %s", remote_dir, result.stderr.strip())
        return set()
    
    years = set()
    for line in result.stdout.splitlines():
        m = YEAR_RE.search(line)
        if m:
            years.add(int(m.group(1)))

    if not years:
        logger.info("No years found in %s", remote_dir)
    else:
        logger.info("Found years in %s: %s", remote_dir, sorted(years))
    
    return years

    