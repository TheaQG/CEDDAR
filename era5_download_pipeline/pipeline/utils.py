'''
    Utility functions for the ERA5 download pipeline.
'''
import pathlib

def ensure_dir(path: pathlib.Path):
    """Ensure that the directory exists."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def hours():
    """Return a list of hours in the format required by the CDS API."""
    return [f"{str(i).zfill(2)}:00" for i in range(24)]

def months():
    """Return a list of months in the format required by the CDS API."""
    return [str(i).zfill(2) for i in range(1, 13)]

def days():
    """Return a list of days in the format required by the CDS API."""
    return [str(i).zfill(2) for i in range(1, 32)]