"""
This script creates a small data batch for testing purposes.
It takes a number of .npz or .nc files from all variable directories
and creates a small data batch with a specified number of files.
The small data batch is then saved under e.g. 'Data_DiffMod/data_DANRA/size_589x789/all_small/'
which can then be used to generate 'train_small', 'valid_small' and 'test_small' datasets.
The small data batch is created by taking a random sample of files from the original data batch
and saving them in a new directory, then converting them to zarr format.

Functions:
    - select_sample_files: Selects a random sample of files from the input directory (all samples dir).
        Input:  model_list              [list]  : List of models to create small data batch for
                var_list                [list]  : List of variables to create small data batch for
                dir_all_data__name      [str]   : Name of the directory containing the full dataset
                n_samples               [int]   : Number of files to select for the small data batch
                sample_method           [str]   : Method to select the sample files (random or sequential)
        Output: file_names              [list]  : List of file names selected for the small data batch
    - copy_to_dirs: Copies the selected files to the small data batch directory and splits them into train, val and test directories.
        Input:  model_list              [list]  : List of models to create small data batch for
                var_list                [list]  : List of variables to create small data batch for
                file_names              [list]  : List of file names selected for the small data batch
        Output: small_data_dir          [str]   : Directory containing the small data batch
    - dirs_to_zarr: Converts the small data batch directories to zarr format.
        Input:  model_list              [list]  : List of models to create small data batch for
                var_list                [list]  : List of variables to create small data batch for
                small_data_dirs         [str]   : Directories containing the small data batches (for each model and variable)
        Output: None
"""

import os
import shutil
import random
import logging
from data_analysis_pipeline.preprocess.daily_files_to_zarr import convert_npz_to_zarr
from sbgm.utils import build_data_path
from sbgm.variable_utils import get_var_name_short

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def select_sample_files(model1,
                        model2,
                        full_domain_model1,
                        full_domain_model2,
                        vars_model1,
                        vars_model2,
                        n_samples_total=100,
                        sample_method='random',
                        data_dir='Data/Data_DiffMod/',
                        all_data__dir_name='all_filtered',
                        all_data__dir_name_small='all_small',
                        ):
    """
        Selects a random sample of files from the input directory (all samples dir).
        Check that the selected files exist in all model and variable directories.
        Input:      model1                  [str]   : First model to create small data batch for
                    model2                  [str]   : Second model to create small data batch for
                    vars_model1             [list]  : List of variables for the first model
                    vars_model2             [list]  : List of variables for the second model
                    dir_all_data__name      [str]   : Name of the directory containing the full dataset. Needs to be added to a full data path specifying model and variable.
                    n_samples               [int]   : Number of files to select for the small data batch
                    sample_method           [str]   : Method to select the sample files (random or sequential)
    """

    # Check that the sample method is valid
    if sample_method not in ['random', 'sequential']:
        raise ValueError("sample_method must be either 'random' or 'sequential'")

    # Create a list to store the file names
    file_names = []

    # Get the path to the input directory (using the first model and variable as reference)
    input_dir = build_data_path(base_path=data_dir, model=model1, var=vars_model1[0], full_domain_dims=full_domain_model1, split=all_data__dir_name, zarr_file=False)
    # Get the list of files in the input directory
    all_files = os.listdir(input_dir)
    # Check if the input directory is empty
    if len(all_files) == 0:
        raise ValueError(f"Input directory {input_dir} is empty")
    # Check if the number of samples is greater than the number of files
    if n_samples_total > len(all_files):
        raise ValueError(f"Number of samples {n_samples_total} is greater than the number of files {len(all_files)}")
    # Select the sample files
    if sample_method == 'random':
        file_names = random.sample(all_files, n_samples_total)
    elif sample_method == 'sequential':
        file_names = all_files[:n_samples_total]

    logger.info(f"      Selected {len(file_names)} files from {input_dir} using {sample_method} method.\n")
    # Strip the date from the file names and remove the file extension
    file_dates = [file_name.split('_')[-1].split('.')[0] for file_name in file_names]

    for var in vars_model1:
        input_dir = build_data_path(base_path=data_dir, model=model1, var=var, full_domain_dims=full_domain_model1, split=all_data__dir_name, zarr_file=False)
        all_dates_tmp = os.listdir(input_dir)
        all_dates_tmp = [file_name.split('_')[-1].split('.')[0] for file_name in all_dates_tmp]
        for file_date, file_name in zip(file_dates, file_names):
            if file_date not in all_dates_tmp:
                logger.warning(f"File {file_name} does not exist in {input_dir}, removing from selection.")
                # Remove the file name from the list
                file_names.remove(file_name)
                
    for var in vars_model2:
        input_dir = build_data_path(base_path=data_dir, model=model2, var=var, full_domain_dims=full_domain_model2, split=all_data__dir_name, zarr_file=False)
        all_dates_tmp = os.listdir(input_dir)
        all_dates_tmp = [file_name.split('_')[-1].split('.')[0] for file_name in all_dates_tmp]
        for file_date, file_name in zip(file_dates, file_names):
            if file_date not in all_dates_tmp:
                logger.warning(f"File {file_name} does not exist in {input_dir}, removing from selection.")
                # Remove the file name from the list
                file_names.remove(file_name)
                    
    # Print the selected file names
    logger.info(f"      Selected {len(file_names)} files for the small data batch:")
    # for file_name in file_names:
    #     logger.info(file_name)

    # Return the list of file names
    return file_names

def split_and_copy_to_dirs(model1='DANRA',
                    model2='ERA5',
                    vars_model1=['temp', 'prcp'],
                    vars_model2=['temp', 'prcp'],
                    full_domain_model1=[589, 789],
                    full_domain_model2=[589, 789],
                    file_names=[],
                    n_samples_total=100,
                    sample_method='random',
                    data_path_full='Data/Data_DiffMod/',
                    data_path_small='Data/Data_DiffMod_small/',
                    all_data__dir_name='all_filtered',
                    all_data__dir_name_small='all_small',
                    split_ratio=[0.7, 0.15, 0.15],
                 ):
    """
    Copies the selected files to the small data batch directory and splits
    them into train, val and test directories.
    Runs the function select_sample_files to get the selected file names if not provided.
    Input:  model1                    [str]   : First model to create small data batch for
            model2                    [str]   : Second model to create small data batch for
            vars_model1               [list]  : List of variables for the first model
            vars_model2               [list]  : List of variables for the second model
            n_samples_total          [int]   : Number of files to select for the small data batch
            sample_method            [str]   : Method to select the sample files (random or sequential)
            data_path                [str]   : Path to the data directory
            all_data__dir_name       [str]   : Name of the directory containing the full dataset
            all_data__dir_name_small [str]   : Name of the directory containing the small dataset
            split_ratio              [list]  : Split ratio for train, val and test
    Output: small_data_dirs             [str]   : Directory containing the small data batch
    """

    # Check if the file names are provided, if not run select_sample_files
    if len(file_names) == 0:
        logger.info(f'      No file names provided, selecting {n_samples_total} files with {sample_method} method')
        logger.info(f"      Selecting files from {data_path_full} using models {model1} and {model2}")
        file_names = select_sample_files(model1=model1,
                                         model2=model2,
                                         full_domain_model1=[589, 789],
                                         full_domain_model2=[589, 789],
                                         vars_model1=vars_model1,
                                         vars_model2=vars_model2,
                                         n_samples_total=n_samples_total,
                                         sample_method=sample_method,
                                         data_dir=data_path_full,
                                         all_data__dir_name=all_data__dir_name,
                                         all_data__dir_name_small=all_data__dir_name_small)
    
    # Get the file dates from the file names
    file_dates = [file_name.split('_')[-1].split('.')[0] for file_name in file_names]

    # Split the file names into train, val and test sets based on the split ratio
    n_train = int(n_samples_total * split_ratio[0])
    n_val = int(n_samples_total * split_ratio[1])
    n_test = n_samples_total - n_train - n_val
    logger.info(f"      Splitting {n_samples_total} files into {n_train} train, {n_val} val and {n_test} test sets")

    train_files = file_names[:n_train]
    val_files = file_names[n_train:n_train + n_val]
    test_files = file_names[n_train + n_val:]

    train_dates = file_dates[:n_train]
    val_dates = file_dates[n_train:n_train + n_val]
    test_dates = file_dates[n_train + n_val:]
    
    
    # Create an empty dictionary to store dictionaries of models and variables
    # small_data_dirs = {'DANRA': {'temp': {}, 'prcp': {'train': "..."}},
    #                   'ERA5': {}}

    small_data_dirs = {}

    # Copy to new (empty) directories
    model_list = [model1, model2]
    for model in model_list:
        if model == model1:
            var_list = vars_model1
            full_domain = full_domain_model1
        elif model == model2:
            var_list = vars_model2
            full_domain = full_domain_model2
        else:
            raise ValueError(f"Model {model} not recognized. Should be either {model1} or {model2}.")
        
        small_data_dirs[model] = {}
        for var in var_list:
            # Set the prescript of the file names (depending on the model and variable)
            file_prefix = get_var_name_short(var, model, domain_size=full_domain)

            small_data_dirs[model][var] = {}
            # Create the small data batch directory 
            small_data_dir = build_data_path(base_path=data_path_small, model=model, var=var, full_domain_dims=full_domain, split='', zarr_file=False)

            # Create the train, val and test directories
            os.makedirs(small_data_dir, exist_ok=True)
            logger.info(f"      Created small data batch directory: {small_data_dir}")
            

            train_dir = os.path.join(small_data_dir, 'train')
            val_dir = os.path.join(small_data_dir, 'valid')
            test_dir = os.path.join(small_data_dir, 'test')
            all_dir = os.path.join(small_data_dir, 'all')

            # Create and clear the train, val and test directories
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            os.makedirs(train_dir, exist_ok=True)
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
            os.makedirs(val_dir, exist_ok=True)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
            os.makedirs(test_dir, exist_ok=True)
            if os.path.exists(all_dir):
                shutil.rmtree(all_dir)
            os.makedirs(all_dir, exist_ok=True)
            

            # Copy the files to the new directories (with prefix and date)
            src_dir = build_data_path(base_path=data_path_full, model=model, var=var, full_domain_dims=full_domain, split=all_data__dir_name, zarr_file=False)
            for file_date in train_dates:
                file_name = f"{file_prefix}_{file_date}.npz"
                src = os.path.join(src_dir, file_name)
                dst = os.path.join(train_dir, file_name)
                if not os.path.exists(dst):
                    # Copy the file to the new directory NOT move it
                    shutil.copyfile(src, dst)

                # also copy to the 'all' directory
                all_dst = os.path.join(all_dir, file_name)
                if not os.path.exists(all_dst):
                    shutil.copyfile(src, all_dst)
            for file_date in val_dates:
                file_name = f"{file_prefix}_{file_date}.npz"
                src = os.path.join(src_dir, file_name)
                dst = os.path.join(val_dir, file_name)
                if not os.path.exists(dst):
                    # Copy the file to the new directory NOT move it
                    shutil.copyfile(src, dst)

                # also copy to the 'all' directory
                all_dst = os.path.join(all_dir, file_name)
                if not os.path.exists(all_dst):
                    shutil.copyfile(src, all_dst)
            for file_date in test_dates:
                file_name = f"{file_prefix}_{file_date}.npz"
                src = os.path.join(src_dir, file_name)
                dst = os.path.join(test_dir, file_name)
                if not os.path.exists(dst):
                    # Copy the file to the new directory NOT move it
                    shutil.copyfile(src, dst)

                # also copy to the 'all' directory
                all_dst = os.path.join(all_dir, file_name)
                if not os.path.exists(all_dst):
                    shutil.copyfile(src, all_dst)
            small_data_dirs[model][var]['train'] = train_dir
            small_data_dirs[model][var]['valid'] = val_dir
            small_data_dirs[model][var]['test'] = test_dir

    # logger.info the small data batch directories
    logger.info(f"      Small data batch directories:")
    for model in small_data_dirs.keys():
        logger.info(f"        Model: {model}")
        for var in small_data_dirs[model].keys():
            logger.info(f"          Variable: {var}")
            logger.info(f"            Train dir: {small_data_dirs[model][var]['train']}")
            logger.info(f"            Val dir:   {small_data_dirs[model][var]['valid']}")
            logger.info(f"            Test dir:  {small_data_dirs[model][var]['test']}")

    # Return the small data batch directories
    return small_data_dirs

def dirs_to_zarr(data_path,
                small_data_dirs,
                all_data__dir_name_small='all_small',
                ):
    """
    Converts the small data batch directories to zarr format.
    Input:  small_data_dirs         [str]   : Directories containing the small data batches (for each model and variable)
    Output: None
    """
    
    # Loop through all models and variables
    for model in small_data_dirs.keys():
        logger.info(f"      Converting small data batch directories for model: {model}")
        for var in small_data_dirs[model].keys():
            logger.info(f"          Variable: {var}")
            small_data_dir = build_data_path(base_path=data_path, model=model, var=var, full_domain_dims=[589, 789], split='', zarr_file=False) # Empty split to get the base path
            small_data_dir_zarr = os.path.join(small_data_dir, 'zarr_files') # Directory to save the zarr files
            os.makedirs(small_data_dir_zarr, exist_ok=True)
            # Get the train, val and test directories
            train_dir = small_data_dirs[model][var]['train']
            val_dir = small_data_dirs[model][var]['valid']
            test_dir = small_data_dirs[model][var]['test']
            # Convert the directories to zarr format
            logger.info(f"      Converting {train_dir} to zarr format...")
            convert_npz_to_zarr(train_dir, os.path.join(small_data_dir_zarr, 'train.zarr'), VERBOSE=False)
            logger.info(f"      Converting {val_dir} to zarr format...")
            convert_npz_to_zarr(val_dir, os.path.join(small_data_dir_zarr, 'valid.zarr'), VERBOSE=False)
            logger.info(f"      Converting {test_dir} to zarr format...")
            convert_npz_to_zarr(test_dir, os.path.join(small_data_dir_zarr, 'test.zarr'), VERBOSE=False)

    logger.info("   Converted all small data batch directories to zarr format")


def run_small_data_batch_creation(cfg):
    """
    Runs the small data batch creation process:
        1. Selects a random sample of files from the input directory (all samples dir).
        2. Copies the selected files to the small data batch directory and splits them into train, val and test directories.
        3. Converts the small data batch directories to zarr format.
    Input:  model1                    [str]   : First model to create small data batch for
            model2                    [str]   : Second model to create small data batch for
            vars_model1               [list]  : List of variables for the first model
            vars_model2               [list]  : List of variables for the second model
            n_samples_total          [int]   : Number of files to select for the small data batch
            sample_method            [str]   : Method to select the sample files (random or sequential)
            data_path                [str]   : Path to the data directory
            all_data__dir_name       [str]   : Name of the directory containing the full dataset
            all_data__dir_name_small [str]   : Name of the directory containing the small dataset
            split_ratio              [list]  : Split ratio for train, val and test
    Output: None
    """
    hr_model_cfg = cfg.get("highres", {})
    model1 = hr_model_cfg.get("model", "DANRA")
    vars_model1 = hr_model_cfg.get("variables", ["temp", "prcp"])
    full_domain_model1 = hr_model_cfg.get("full_domain", [589, 789])

    lr_model_cfg = cfg.get("lowres", {})
    model2 = lr_model_cfg.get("model", "ERA5")
    vars_model2 = lr_model_cfg.get("variables", ["temp", "prcp"])
    full_domain_model2 = lr_model_cfg.get("full_domain", [589, 789])

    data_cfg = cfg.get("data", {})
    n_samples_total = data_cfg.get("n_samples", 10)
    sample_method = data_cfg.get("sample_method", "random")
    data_path_full = data_cfg.get("data_path_full", "./data/full")
    data_path_small = data_cfg.get("data_path_small", "./data/small")
    all_data__dir_name = data_cfg.get("all_data__dir_name", "all")
    all_data__dir_name_small = data_cfg.get("all_data__dir_name_small", "all_data_small")
    split_ratio = data_cfg.get("split_ratio", [0.7, 0.15, 0.15])

    # Step 1: Select a random sample of files from the input directory (all samples dir).
    logger.info("\n\n############ Step 1: Selecting sample files... ############\n")
    file_names = select_sample_files(model1=model1,
                                     model2=model2,
                                     full_domain_model1=full_domain_model1,
                                     full_domain_model2=full_domain_model2,
                                     vars_model1=vars_model1,
                                     vars_model2=vars_model2,
                                     n_samples_total=n_samples_total,
                                     sample_method=sample_method,
                                     data_dir=data_path_full,
                                     all_data__dir_name=all_data__dir_name,
                                     all_data__dir_name_small=all_data__dir_name_small)
    # Step 2: Copy the selected files to the small data batch directory and split them into train, val and test directories.
    logger.info("\n\n############ Step 2: Copying files to small data batch directories and splitting into train, val and test... ############\n")
    small_data_dirs = split_and_copy_to_dirs(model1=model1,
                                             model2=model2,
                                             vars_model1=vars_model1,
                                             vars_model2=vars_model2,
                                             full_domain_model1=full_domain_model1,
                                             full_domain_model2=full_domain_model2,
                                             file_names=file_names,
                                             n_samples_total=n_samples_total,
                                             sample_method=sample_method,
                                             data_path_full=data_path_full,
                                             data_path_small=data_path_small,
                                             all_data__dir_name=all_data__dir_name,
                                             all_data__dir_name_small=all_data__dir_name_small,
                                             split_ratio=split_ratio)
    # Step 3: Convert the small data batch directories to zarr format.
    logger.info("\n\n############ Step 3: Converting small data batch directories to zarr format... ############\n")
    dirs_to_zarr(data_path=data_path_small,
                 small_data_dirs=small_data_dirs,
                 all_data__dir_name_small=all_data__dir_name_small) 
    logger.info("\n\n############ Small data batch creation process completed. ############\n")


