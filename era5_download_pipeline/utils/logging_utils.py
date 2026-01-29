import logging
import sys
import pathlib
from typing import Union
from logging.handlers import RotatingFileHandler

def setup_logging(log_file: Union[pathlib.Path, str],
                    log_level: str = "INFO",
                    max_mb: int = 10, # 10 * 1024 * 1024 bytes = 10 MB
                    backup_count: int = 3):
        """
        Set up logging configuration.
    
        Args:
            log_file (pathlib.Path | str): Path to the log file.
            log_level (int): Logging level (default: logging.INFO).
            max_bytes (int): Maximum size of the log file before rotation.
            backup_count (int): Number of backup files to keep.
        """
        log_file = pathlib.Path(log_file).expanduser().resolve()  # Resolve to absolute path and expand user directory
        log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        level = getattr(logging, log_level.upper(), logging.INFO)
        fmt = '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
        date = '%Y-%m-%d %H:%M:%S'

        # File handler (rotate), logging to a file
        file_hdlr = RotatingFileHandler(log_file,
                                        maxBytes=max_mb * 1024 * 1024,  # Convert MB to bytes
                                        backupCount=backup_count,
                                        encoding='utf-8'  # Ensure UTF-8 encoding
                                        )
        file_hdlr.setFormatter(logging.Formatter(fmt, date))
    
        # Console handler, also log to console 
        console_hdlr = logging.StreamHandler(sys.stdout)
        console_hdlr.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        
        logging.basicConfig(level=level,
                            handlers=[file_hdlr, console_hdlr])

        logging.getLogger("cdsapi").setLevel(logging.WARNING)  # Reduce verbosity of cdsapi logs
        logging.getLogger("urllib3").setLevel(logging.WARNING)  # Reduce verbosity of urllib3 logs
        logging.info(f"Logging started; file -> {log_file}, level -> {log_level.upper()}")