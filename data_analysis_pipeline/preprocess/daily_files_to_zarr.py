'''
    File to convert daily ERA5 (.npz)/DANRA (.nc) files to zarr files 
    for better storage and access to prevent memory errors.
'''

import zarr 
import os 
import argparse
import logging

import numpy as np
import netCDF4 as nc

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def convert_npz_to_zarr(npz_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert DANRA .npz files to zarr files
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        zarr_file: str
            Name of zarr file to be created
    '''
    logger.info(f'      Converting {len(os.listdir(npz_directory))} .npz files to zarr file...')
    # Create zarr group (equivalent to a directory) 
    zarr_group = zarr.open_group(zarr_file, mode='w')

    # Loop through all .npz files in the .npz directory
    for npz_file in os.listdir(npz_directory):
        # Check if the file is a .npz file (not dir or .DS_Store)
        if npz_file.endswith('.npz'):
            if VERBOSE:
                logger.info(os.path.join(npz_directory, npz_file))
            # Load the .npz file
            npz_data = np.load(os.path.join(npz_directory, npz_file))
            # Loop through all keys in the .npz file
            for key in npz_data:
                # Save the data as a zarr array
                zarr_group.array(npz_file.replace('.npz', '') + '/' + key, npz_data[key], chunks=True, dtype=np.float32)


def convert_nc_to_zarr(nc_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert ERA5 .nc files to zarr files
        
        Parameters:
        -----------
        nc_directory: str
            Directory containing .nc files
        zarr_file: str
            Name of zarr file to be created
    '''
    logger.info(f'      Converting {len(os.listdir(nc_directory))} .nc files to zarr file...')
    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(zarr_file, mode='w')
    logger.info('       zarr group created')
    
    # Loop through all .nc files in the .nc directory 
    for nc_file in os.listdir(nc_directory):
        # Print the first file name
        if nc_file == os.listdir(nc_directory)[0]:
            logger.info(nc_file)
        # Check if the file is a .nc file (not dir or .DS_Store)
        if nc_file.endswith('.nc'):
            if VERBOSE:
                logger.info(os.path.join(nc_directory, nc_file))
            # Load the .nc file
            nc_data = nc.Dataset(os.path.join(nc_directory, nc_file)) # type: ignore
            # Loop through all variables in the .nc file
            for var in nc_data.variables:
                # Select the data from the variable
                data = nc_data[var][:]
                # Save the data as a zarr array
                zarr_group.array(nc_file.replace('.nc', '') + '/' + var, data, chunks=True, dtype=np.float32)

