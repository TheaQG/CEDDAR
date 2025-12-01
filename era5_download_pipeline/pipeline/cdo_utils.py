'''
    CDO utilities for regridding and converting NetCDF files to daily statistics.
    This module provides functions to convert NetCDF files to daily statistics,
    regrid data to a specific grid, generate regridding weights, and convert daily
    NetCDF data to .npz files for each day.
'''


from datetime import datetime, timedelta

import os
import subprocess
import pathlib
import calendar
from pathlib import Path

import netCDF4 as nc
import numpy as np


import logging
logger = logging.getLogger(__name__)

def convert_to_daily_stat(input_file,
                          stat,
                          output_file):
    '''
        Convert the input NetCDF file to a daily statistic (e.g., sum, mean, max).
    '''

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Converting %s to daily %s -> %s", input_file, stat, output_file)

    cdo_command = ["cdo", stat, input_file, output_file]

    subprocess.run(cdo_command, check=True)

def regrid_to_danra(input_nc,
                    output_nc,
                    weights_file=None,
                    grid_file="/path/to/mygrid_danra_small",
                    interpolation_method='bilinear'):
    '''
        Regrid the input NetCDF file to the DANRA grid using bilinear interpolation.
    '''
    
    logger.info("Regridding %s to %s using %s interpolation...", input_nc, output_nc, interpolation_method)
    if weights_file is not None:
        logger.info("Using weights file: %s", weights_file)

    # If weights file is not provided, just remap directly
    if weights_file is None:
        if interpolation_method == 'bilinear':
            # Use the grid file directly for bilinear interpolation
            cdo_command = [
                "cdo", f"remap,{grid_file}", str(input_nc), str(output_nc)
            ]
        elif interpolation_method == 'nearest':
            # Use the grid file directly for nearest neighbor interpolation
            cdo_command = [
                "cdo", f"remapnn,{grid_file}", input_nc, output_nc
            ]
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")
    # If weights file is provided, use it for regridding
    else:
        if interpolation_method == 'bilinear':
            cdo_command = [
                "cdo", f"remap,{grid_file},{weights_file}", str(input_nc), str(output_nc)
            ]
        elif interpolation_method == 'nearest':
            cdo_command = [
                "cdo", f"remap,{grid_file},{weights_file}", str(input_nc), str(output_nc)
            ]
        else:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}")

    subprocess.run(cdo_command, check=True)


def generate_regridding_weights(reference_file,
                                weights_file,
                                grid_file="/path/to/mygrid_danra_small"):
    '''
        Generates regridding weights for converting data from a reference grid to a target grid.
        Uses CDO to create bilinear interpolation weights.
        One-time step to create bilinear weights
    '''
    # Make sure the path exists
    weights_file = pathlib.Path(weights_file)
    weights_file.parent.mkdir(parents=True, exist_ok=True)

    grid_file = str(grid_file)

    cdo_command = ["cdo", f"genbil,{grid_file}", str(reference_file), str(weights_file)]        
    logger.info("Creating weights file: %s", weights_file)
    subprocess.run(cdo_command, check=True)

def _find_data_var(ds, preferred_name=None):
    """
        Return the name of the main data variable in *ds*.
        Accept either (time, y, x) **or** (time, level, y, x) variables
        where the level dimension has lenght 1
        Fall back to a simple scan if preferred_name is not suitable
    """
    TIME_DIMS = {"time", "valid_time", "forecast_time", "step", "t"}
    LAT_DIMS = {"lat", "latitude", "y"}
    LON_DIMS = {"lon", "longitude", "x"}

    def looks_like_field(var) -> bool:
        """
            Check if the variable has dimensions that look like a 3D field.
        """
        dims = var.dimensions
        has_time = any(d in TIME_DIMS for d in dims)
        has_lat = any(d in LAT_DIMS for d in dims)
        has_lon = any(d in LON_DIMS for d in dims)
        if not (has_time and has_lat and has_lon):
            return False
        
        if len(dims) == 3:
            # Normal, single-level file
            return True
        if len(dims) == 4 and "pressure_level" in dims:
            # Pressure level file with a single level
            return var.shape[dims.index("pressure_level")] == 1 # Check if pressure level dimension has length 1
        return False

    # Debug: check existing variables
    logger.info("Available variables in dataset: %s", list(ds.variables.keys()))    
    # 1) Check if exact match
    if preferred_name and preferred_name in ds.variables:
        v = ds.variables[preferred_name]
        if looks_like_field(v):
            return preferred_name
        
    # 2) Otherwise first suitable variable
    for name, var in ds.variables.items():
        if looks_like_field(var):
            return name
        
    raise ValueError("No suitable data variable found")

def convert_daily_to_npz(cfg,
                         input_nc: Path,
                         output_dir: Path,
                         year: int,
                         var_str: str = 'pev',
                         plevel = None):
    '''
        Converts daily NetCDF data to individual .npz files for each day.
        Saves the data in a specified output directory.
    '''
    logger.info("\nConverting %s to daily .npz files in %s\n", input_nc, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(input_nc) as ds: # type: ignore
        # Check if the variable exists in the dataset
        meta = cfg['variables_by_short'][var_str]
        preferred_name = meta.get('nc_var', var_str)  # Use 'nc_var' if available, otherwise use 'vshort'
        # Find the variable name, using the preferred name if provided
        var_name = _find_data_var(ds, preferred_name=var_str)
        var = ds.variables[var_name] # Masked array (T, Y, X)
        # If a pressure level is specified, check if the variable has that dimension
        if "pressure_level" in var.dimensions and var.shape[var.dimensions.index("pressure_level")] == 1:
            var = np.squeeze(var, axis=var.dimensions.index("pressure_level"))  # Remove the pressure level dimension if it has length 1

        t_len = var.shape[0]
        exp_days = 366 if calendar.isleap(year) else 365
        if t_len != exp_days:
            logger.warning("\n !!! Expected %d days, but file contains %d. !!! \n", exp_days, t_len)

        for day_idx in range(t_len):
            day_data = np.asarray(var[day_idx, :, :])  # Convert masked array to regular array
            # Build file name
            date_str = day_to_date(day_idx + 1, year).replace('_', '')
            if plevel is not None:
                # If a pressure level is specified, include it in the filename
                day_fn = f"{var_str}_{plevel}_hPa_589x789_{date_str}.npz"
            else:
                # Otherwise, use the variable string as is
                day_fn = f"{var_str}_589x789_{date_str}.npz"
            fname = output_dir / day_fn

            logger.info("Saving day: %s to %s...", date_str, fname)

            # Save the data to a compressed .npz file
            np.savez_compressed(fname, data=day_data)

    logger.info("\nConversion to .npz files completed.\n")

    # ds = nc.Dataset(input_nc) # type: ignore
    # var_keys = list(ds.variables.keys())
    # # Exclude coordinate variables
    # data_var_key = [k for k in var_keys if k not in ['time', 'latitude', 'longitude']][0]
    # data = ds.variables[data_var_key][:]

    # os.makedirs(output_dir, exist_ok=True)

    # for day_idx in range(data.shape[0]):
    #     day_data = data[day_idx, :, :]
    #     date_str = day_to_date(day_idx + 1, int(year)).replace('_', '')
    #     day_fn = f"{var_str}_589x789_{date_str}.npz"
    #     day_path = os.path.join(output_dir, day_fn)
    #     logger.info(f"Saving day: {date_str}...")

    #     np.savez_compressed(day_path, data=day_data)

    # ds.close()





# Define a function to convert a day number to a date string
def day_to_date(day_number, year_use):
    '''
        Converts a day number to a date string in the format YYYY_MM_DD
        If underscores are unwanted add .replace('_','') to the return statement
    '''
    date_format = '%Y_%m_%d'
    # create a datetime object for January 1st of the given year_use
    start_date = datetime(year_use, 1, 1)
    # add the number of days to the start date
    result_date = start_date + timedelta(days=day_number-1)
    # format the date string using the specified format
    return result_date.strftime(date_format)#.replace('_','')

