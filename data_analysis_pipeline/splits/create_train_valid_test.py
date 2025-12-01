import os
import shutil
import zarr
import logging
import re
from glob import glob
from tqdm import tqdm
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def extract_date(filename):
    """ Extract the date string (YYYYMMDD) from filename like 'data_YYYYMMDD.npz' """
    match = re.search(r'_(\d{8})\.npz$', filename)
    return match.group(1) if match else None

def create_data_splits(cfg):
    """
        Copies .npz files from 'all/' to 'train/', 'valid/' and 'test/' folders, based on split ratios. 
        Keeps the original files in 'all/' and supports multiple variables.
    """
    data_dir = cfg.paths.data_dir
    hr_vars = cfg.highres.variables
    lr_vars = cfg.lowres.condition_variables
    hr_model = cfg.highres.model
    lr_model = cfg.lowres.model
    hr_domain_size = cfg.highres.domain_size
    hr_domain_size_str = "x".join(map(str, hr_domain_size))
    lr_domain_size = cfg.lowres.domain_size
    lr_domain_size_str = "x".join(map(str, lr_domain_size))
    data_split_type = cfg.split_params.split_type
    data_split_params = cfg.split_params.data_split_params

    overwrite = cfg.split_params.get('overwrite', False)
    seed = cfg.split_params.get('seed', 42)

    if cfg.highres.filtered:
        all_data_str_hr = cfg.highres.filter_str
    else:
        all_data_str_hr = 'all'
    # Set paths to where HR and LR data lives
    HR_DIRS = {
        hr_var: os.path.join(data_dir, f'data_{hr_model}', 'size_' + hr_domain_size_str, f"{hr_var}_{hr_domain_size_str}", all_data_str_hr)
        for hr_var in hr_vars
    }
    LR_DIRS = {
        lr_var: os.path.join(data_dir, f'data_{lr_model}', 'size_' + lr_domain_size_str, f"{lr_var}_{lr_domain_size_str}", 'all')
        for lr_var in lr_vars
    }

    # Collect all available HR files per date 
    date_to_hr_files = {}
    for hr_var, hr_dir in HR_DIRS.items():
        files = sorted(glob(os.path.join(hr_dir, '*.npz')))
        logger.info(f"[HR - {hr_var}] Found {len(files)} files in {hr_dir}")
        for f in files:
            date = extract_date(os.path.basename(f))
            if date:
                date_to_hr_files.setdefault(date, {})[hr_var] = f

    # === SAFETY CHECK ===
    # === STEP 1: Only retain dates where all HR vars are available ===
    valid_dates = [date for date, d in date_to_hr_files.items() if len(d) == len(hr_vars)]
    logger.info(f"Valid dates with all HR vars: {len(valid_dates)}")

    # === STEP 2: Further filter valid_dates based on LR variable availability ===
    for lr_var, lr_dir in LR_DIRS.items():
        lr_dates = set()
        for f in glob(os.path.join(lr_dir, "*npz")):
            date = extract_date(os.path.basename(f))
            if date:
                lr_dates.add(date)

        # Identify which dates are missing
        current_valid_dates = set(valid_dates)
        missing_dates = current_valid_dates - lr_dates

        if missing_dates:
            logger.warning(
                f"[{lr_var}] Missing {len(missing_dates)} dates from LR data."
                f"Examples: {sorted(list(missing_dates))[:5]}..."
            )

        # Restrict to dates that exist for the LR variable
        valid_dates = [d for d in valid_dates if d in lr_dates]

    logger.info(f"Final valid dates after HR+LR check: {len(valid_dates)}")

    # Determine splits
    if data_split_type == "Time":
        train_years = set(map(str, np.arange(*data_split_params['train_years'])))
        valid_years = set(map(str, np.arange(*data_split_params['valid_years'])))
        test_years = set(map(str, np.arange(*data_split_params['test_years'])))

        split_dates = {
            'train': [d for d in valid_dates if d[:4] in train_years],
            'valid': [d for d in valid_dates if d[:4] in valid_years],
            'test':  [d for d in valid_dates if d[:4] in test_years],
        }

    elif data_split_type == "Random":
        np.random.seed(seed)
        np.random.shuffle(valid_dates)
        total = len(valid_dates)

        n_train = int(data_split_params['train_frac'] * total)
        n_valid = int(data_split_params['valid_frac'] * total)

        split_dates = {
            'train': valid_dates[:n_train],
            'valid': valid_dates[n_train: n_train+n_valid],
            'test':  valid_dates[n_train + n_valid:],
        }

    else:
        raise ValueError(f"Unknown split type: {data_split_type}")


    for split, dates in split_dates.items():
        logger.info(f"{split.capitalize()} files: {len(dates)}")


        # HR copy
        for hr_var in hr_vars:
            logger.info(f"Copying HR variable: {hr_var} for split: {split}")
            # HR destination
            dest_dir = os.path.join(data_dir, f"data_{hr_model}", f"size_{hr_domain_size_str}", f"{hr_var}_{hr_domain_size_str}", split)
            os.makedirs(dest_dir, exist_ok=True)
            for d in tqdm(dates, desc=f"[{split.upper()}] Copying HR {hr_var}"):
                f = date_to_hr_files[d][hr_var]
                dst = os.path.join(dest_dir, os.path.basename(f))
                if not os.path.exists(dst) or overwrite:
                    shutil.copy(f, dst)

        # Copy corresponding LR files for each var
        for lr_var in lr_vars:
            logger.info(f"Copying LR variable: {lr_var} for split: {split}")
            lr_dir = LR_DIRS[lr_var]
            lr_dest = os.path.join(data_dir, f'data_{lr_model}', f'size_{lr_domain_size_str}', f"{lr_var}_{lr_domain_size_str}", split)
            os.makedirs(lr_dest, exist_ok=True)

            for d in tqdm(dates, desc=f"[{split.upper()}] Copying LR {lr_var}"):
                matching_file = sorted(glob(os.path.join(lr_dir, f"*{d}*.npz")))
                if matching_file:
                    dst = os.path.join(lr_dest, os.path.basename(matching_file[0]))
                    if not os.path.exists(dst) or overwrite:
                        shutil.copy(matching_file[0], dst)
                else:
                    logger.warning(f"No matching LR file found for date {d} in {lr_dir}")

    logger.info("Data splitting complete.")




def convert_splits_to_zarr(cfg):
    """
        Converts .npz split folders (train/valid/test) into .zarr archives for both HR and LR variabeles
        Each zarr group is saved as: <split>.zarr inside the variable's folder
    """
    data_dir = cfg.paths.data_dir
    hr_vars = cfg.highres.variables
    lr_vars = cfg.lowres.condition_variables
    hr_model = cfg.highres.model
    lr_model = cfg.lowres.model
    hr_domain_size = cfg.highres.domain_size
    hr_domain_size_str = "x".join(map(str, hr_domain_size))
    lr_domain_size = cfg.lowres.domain_size
    lr_domain_size_str = "x".join(map(str, lr_domain_size))
    overwrite = cfg.split_params.get("overwrite_zarr", True)
    keep_npz = cfg.split_params.get("keep_npz_after_zarr", False)


    for split in ["train", "valid", "test"]:
        logger.info(f"\n=== Converting split: {split} ===")

        # ---- High-resolution (DANRA) ----
        for hr_var in hr_vars:
            hr_base = os.path.join(data_dir, f"data_{hr_model}", f"size_{hr_domain_size_str}", f"{hr_var}_{hr_domain_size_str}")
            split_dir = os.path.join(hr_base, split)
            zarr_dir = os.path.join(hr_base, "zarr_files")
            os.makedirs(zarr_dir, exist_ok=True)
            zarr_path = os.path.join(zarr_dir, f"{split}.zarr")

            if overwrite and os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
                logger.info(f"[{hr_model} - {hr_var}] Overwriting {zarr_path}")

            logger.info(f"[{hr_model} - {hr_var}] Writing Zarr file: {zarr_path}")
            zarr_group = zarr.open_group(zarr_path, mode="w")

            npz_files = sorted(glob(os.path.join(split_dir, "*.npz")))
            for f in tqdm(npz_files, desc=f"{hr_var} {split}"):
                fname = os.path.splitext(os.path.basename(f))[0]
                with np.load(f) as data:
                    for key in data:
                        zarr_group.array(f"{fname}/{key}", data[key], chunks=True, dtype=np.float32)

            if not keep_npz: 
                logger.info(f"Deleting HR {hr_var} .npz files from {split_dir}")
                for f in npz_files:
                    os.remove(f)
        
        
        # ---- Low-res (ERA5) ----
        for lr_var in lr_vars:
            lr_base = os.path.join(data_dir, f"data_{lr_model}", f"size_{lr_domain_size_str}", f"{lr_var}_{lr_domain_size_str}")
            split_dir = os.path.join(lr_base, split)
            zarr_dir = os.path.join(lr_base, "zarr_files")
            os.makedirs(zarr_dir, exist_ok=True)
            zarr_path = os.path.join(zarr_dir, f"{split}.zarr")

            if overwrite and os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
                logger.info(f"[{lr_model} - {lr_var}] Overwriting: {zarr_path}")
            
            logger.info(f"[{lr_model} - {lr_var}] Writing zarr file: {zarr_path}")
            zarr_group = zarr.open_group(zarr_path, mode="w")

            npz_files = sorted(glob(os.path.join(split_dir, "*.npz")))
            for f in tqdm(npz_files, desc=f"{lr_var} {split}"):
                fname = os.path.splitext(os.path.basename(f))[0]
                with np.load(f) as data:
                    for key in data:
                        zarr_group.array(f"{fname}/{key}", data[key], chunks=True, dtype=np.float32)
        
            if not keep_npz:    
                logger.info(f"Deletin LR {lr_var} .npz files from {split_dir}")
                for f in npz_files:
                    os.remove(f) # Delete after writing
                


    logger.info("Zarr conversion complete.")
    