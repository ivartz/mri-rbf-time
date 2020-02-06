#import os
#import shutil
import numpy as np
import nibabel as nib
#import pandas as pd
#from pathlib import Path


def getStackUnits(nifti_class):
    return tuple(nifti_class.header["pixdim"][1:4])

"""
def load_nifti(file):
    # Returns 
    # voxel data : numpy array
    # voxel dimensions (x, y, z) : tuple(,,)
    # nifti header : nibabel.nifti1.Nifti1Header
    data_class = nib.load(file)
    return data_class.get_fdata(), tuple(data_class.header["pixdim"][1:4]), data_class.header

def xyzt_to_tzyx(vol_xyzt):
    vol_tyzx = np.swapaxes(vol_xyzt, 0, 3)
    vol_tzyx = np.swapaxes(vol_tyzx, 1, 2)
    
    # Return data with reverse x axis
    #return vol_tzyx[:,:,:,::-1]
    return vol_tzyx

def xyz_to_zyx(vol_xyz):
    vol_zyx = np.swapaxes(vol_xyz, 0, 2)
    
    # Return data with reverse x axis
    #return vol_zyx[:,:,::-1]
    return vol_zyx


"""