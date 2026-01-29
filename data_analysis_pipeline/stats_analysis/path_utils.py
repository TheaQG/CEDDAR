import os

def build_data_path(base_dir, model_type, variable, domain_size, split=None, use_zarr=False):
    """
    Constructs the full path to the data directory.

    Args:
        base_dir (str): Root data directory, e.g., ${DATA_DIR}
        model_type (str): e.g., 'ERA5' or 'DANRA'
        variable (str): e.g., 'prcp'
        domain_size (list or tuple): e.g., [589, 789]

    Returns:
        str: Full path to data directory.
    """
    size_str = f"{domain_size[0]}x{domain_size[1]}"
    path = os.path.join(
        base_dir,
        f"data_{model_type}",
        f"size_{size_str}",
        f"{variable}_{size_str}"
    )

    if not use_zarr:
        # .npz case (e.g. split == 'all')
        return os.path.join(path, 'all')
    
    # zarr case
    split_dict = {'train': 'train', 'val': 'valid', 'valid': 'valid', 'test': 'test', 'all': 'all'}
    split_norm = split_dict.get(split, split) if split else 'all'
    return os.path.join(path, 'zarr_files', f"{split_norm}.zarr")

