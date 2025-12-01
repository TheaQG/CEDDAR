import os 
import re
import zarr
import logging
import numpy as np
from datetime import datetime
from glob import glob
from datetime import datetime

from data_analysis_pipeline.stats_analysis.path_utils import build_data_path
from sbgm.variable_utils import crop_to_region, get_var_name_short, correct_variable_units
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def _open_zarr_group(root: str):
    """
        Opens a Zarr group from the given root path.
    """
    try:
        return zarr.open_consolidated(root, mode='r') 
    except Exception:
        return zarr.open_group(root, mode='r')
    
def _list_all_zarr_arrays(root: str):
    """
        Return RELATIVE array keys inside the store e.g.:
        ['t2m_ave/t2m_ave_20200101', 't2m_ave/20201211/t2m_ave', t2m_ave/]
    """
    g = _open_zarr_group(root)
    rel_paths = []
    def _visit(path, obj):
        # robust across zarr versions
        if getattr(obj, "ndim", None) is not None and hasattr(obj, "dtype") and hasattr(obj, "shape"):
            rel_paths.append(path.replace("\\", "/"))  # normalize slashes
    g.visititems(_visit) # type: ignore
    return rel_paths

def _extract_date_from_relkey(rel_key: str, var_short: str):
    m = re.search(fr'(?<!\d)(\d{{8}})(?!\d)', rel_key)
    if m:
        return m.group(1)
    base = os.path.basename(rel_key)
    m2 = re.search(fr'{re.escape(var_short)}_(\d{{8}})', base)
    return m2.group(1) if m2 else None
            

def _load_one_entry(entry_path: str, var_name_short: str):
    """
        I/O helper that loads a single entry from either a .npz file or a Zarr store path
        of the form '.../<split>.zarr/<relative/key>'. Returns (data: np.ndarray, timestamp: datetime)
    """
    # === .npz case ===
    if entry_path.endswith(".npz"):
        with np.load(entry_path) as npz:
            # Take the first available key if neither 'data' nor 'arr_0' exists
            key = 'data' if 'data' in npz else ('arr_0' if 'arr_0' in npz else list(npz.keys())[0])
            data = np.array(npz[key])  # Ensure it's a numpy array 

            # Extract date from filename like 'prcp_20200101.npz'
            timestamp = get_date_from_filename(entry_path, 'npz', var_name_short)

        return data, timestamp
    
    # === Zarr case: must contain .zarr in the path and a relative key after it ===
    if ".zarr" in entry_path:
        zarr_root = entry_path[:entry_path.index(".zarr") + len(".zarr")]
        rel_key = os.path.relpath(entry_path, start=zarr_root).replace("\\", "/")  # keep nested paths, normalize slashes
        
        g = _open_zarr_group(zarr_root)
        target_key = None

        # Try the rel_key directly
        if rel_key in g:
            target_key = rel_key
        else:
            # Try <var_name_short>/<basename> if var_name_short is a group
            basename = os.path.basename(rel_key)
            candidate = f"{var_name_short}/{basename}" if var_name_short in g.group_keys() else None # type: ignore
            if candidate and candidate in g:
                target_key = candidate

        if target_key is None:
            raise ValueError(
                f"Key '{rel_key}' not found in Zarr group at {zarr_root}. "
                f"Groups: {list(g.group_keys())[:20]}." # type: ignore
                f"(tip: we expect entry_path to include the full relative key inside the store)"
            )
        
        arr = g[target_key][...]
        date_str = _extract_date_from_relkey(target_key, var_name_short)
        if not date_str:
            raise ValueError(f"Could not extract an 8-digit date from Zarr key: {target_key}")
        
        timestamp = datetime.strptime(date_str, "%Y%m%d")
        
        return arr, timestamp
    
    raise ValueError(f"Unsupported file or path: {entry_path}")

def get_date_from_filename(file_path, data_type, var_name_short):
    """
        Extract the date from the filename.
    """

    # Get filename from file path (basename means removing the directory)
    filename = os.path.basename(file_path)

    if data_type == 'npz':
        # Extract date from filename like varname_YYYYMMDD.npz
         match = re.search(r"(\d{8})\.npz$", filename) # match = re.search(rf"{re.escape(var_name_short)}_(\d{8})\.npz$", filename)        
    elif data_type == 'zarr':
        # For zarr, filepath is like {split}.zarr/varname_YYYYMMDD
        filename = os.path.basename(file_path)  # get the last part after the last '/'
        match = re.search(r"(\d{8})$", filename)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    if match:
        date_str = match.group(1)
        timestamp = datetime.strptime(date_str, "%Y%m%d")
    else:
        raise ValueError(f"Date not found in filename: {filename}")

    return timestamp

def process_single_file(file_path, variable, model, var_name_short, crop_region):
    """
        Reads a single entry (npz or zarr), applies variable-specific unit corrections and crops to region if specified.
    """
    data, timestamp = _load_one_entry(file_path, var_name_short)
    
    data = correct_variable_units(variable, model, data)

    if crop_region is not None:
        data = crop_to_region(data, crop_region)

    if data is None:
        raise ValueError(f"Data is None after processing: {file_path}")

    # logger.info(f"[process_single_file] Loaded {file_path} | shape: {data.shape} | date: {timestamp.date()}")

    return data, timestamp

class DataLoader:
    def __init__(self,
                 base_dir: str,
                 n_workers: int,
                 variable: str,
                 model: str,
                 domain_size: list,
                 split: str,
                 crop_region: list,
                 verbose: bool = False):

        self.variable = variable
        self.var_name_short = get_var_name_short(variable, model)
        
        self.domain_size = domain_size
        self.crop_region = crop_region
        self.split = split
        self.zarr = self.split in ["train", "valid", "val", "test"]

        self.model_type = model
        self.data_dir = build_data_path(base_dir, self.model_type, self.variable, self.domain_size, self.split, use_zarr=self.zarr)
        logger.info(f"[DataLoader] Data directory set to: {self.data_dir} (zarr={self.zarr})")
        
        self.n_workers = n_workers
        self.verbose = verbose

    def _get_file_list(self):
        if self.zarr:
            root = self.data_dir # e.g. .../zarr_files/train.zarr
            rel_arrays = _list_all_zarr_arrays(root)

            if not rel_arrays:
                raise FileNotFoundError(f"No arrays discovered under Zarr root: {root}"
                                        f"Check that the store is not empty and has metadata")
            var = self.var_name_short

            # 1) Prefer arrays whose BASENAME equals the variable
            matches = [p for p in rel_arrays if os.path.basename(p) == var]

            # 2) Else, arrays whose BASENAME starts with '<var>_' (flat naming)
            if not matches:
                matches = [p for p in rel_arrays if os.path.basename(p).startswith(f"{var}_")]

            # 3) Else, anywhere in path contains '<var>' (catch-all)
            if not matches:
                matches = [p for p in rel_arrays if var in p]

            if not matches:
                sample = ", ".join(rel_arrays[:25])
                raise FileNotFoundError(f"No arrays for variables '{var}' found in {root}."
                                        f"Sample of discovered arrays: [{sample}]")
            # Return ABSOLUTE entry paths that carry the rel key after the store
            out = [os.path.join(root, m) for m in matches]
            out.sort()
            logger.info(f"Discovered {len(out)} arrays for variable '{var}' under root: {root}")
            return out

        else:
            return sorted(glob(os.path.join(self.data_dir, f"{self.var_name_short}_*.npz")))


    def _process_wrapper(self, file_path):
        cutout, timestamp = process_single_file(file_path, self.variable, self.model_type, self.var_name_short, self.crop_region)
        return cutout, timestamp

    def load(self):
        file_list = self._get_file_list()
        logger.info(f"[DataLoader] Found {len(file_list)} files for variable '{self.variable}' (model:{self.model_type}, split: {self.split})")

        if not file_list:
            raise FileNotFoundError(f"No input files found in {self.data_dir} for variable {self.variable} with short name {self.var_name_short}")
        


        results = []
        if self.n_workers > 1:
            # Verbose logging of progress
            if self.verbose:
                # Set up parallel processing
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:    
                    # Create a dictionary to map futures to file names
                    futures = {executor.submit(self._process_wrapper, f): f for f in file_list}
                    
                    # Use as_completed to get results as they finish 
                    for i, future in enumerate(as_completed(futures), 1):
                        try:
                            # Issue: files
                            results.append(future.result())
                        except Exception as e:
                            logger.error(f"Error processing file {futures[future]}: {e}")
                        if i % 500 == 0 or i == len(file_list):
                            logger.info(f"[DataLoader] Processed {i}/{len(file_list)} files for {self.variable}...")
            # Non-verbose (speed optimized)
            else:
                # Set up parallel processing
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:    
                    results = list(executor.map(self._process_wrapper, file_list))
        
        else:
            for i, f in enumerate(file_list, 1):
                results.append(self._process_wrapper(f))
                if i % 500 == 0 or i == len(file_list):
                    logger.info(f"[DataLoader] Processed {i}/{len(file_list)} files for {self.variable}...")


        # Get the data sorted
        cutouts, timestamps = zip(*results)
        sorted_pairs = sorted(zip(timestamps, cutouts))
        timestamps, cutouts = zip(*sorted_pairs)

        return {
            "cutouts": list(cutouts),
            "timestamps": list(timestamps)
        }
    def load_single_day(self, date_str: str):
        """
            Load data for a single specified date (YYYYMMDD)
        """
        file_list = self._get_file_list()
        match_file = None
        for f in file_list:
            if date_str in f:
                match_file = f
                break
        if not match_file:
            raise FileNotFoundError(f"No file found for date {date_str} in {self.data_dir}")

        data, timestamp = self._process_wrapper(match_file)
        return {
            "cutouts": data,
            "timestamps": timestamp
        }

    def load_multi(self, dates_or_n):
        file_list = self._get_file_list()
        if isinstance(dates_or_n, int):
            selected_files = np.random.choice(file_list, size=dates_or_n, replace=False)
        elif isinstance(dates_or_n, (list, tuple)):
            selected_files = [f for f in file_list if any(d in f for d in dates_or_n)]
        else:
            raise ValueError(f"dates_or_n must be an int or a list/tuple of date strings. Got {dates_or_n}")

        results = [self._process_wrapper(f) for f in selected_files]
        cutouts, timestamps = zip(*results)

        return {
            "cutouts": list(cutouts),
            "timestamps": list(timestamps)
        }





