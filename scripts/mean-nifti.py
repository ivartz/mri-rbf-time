import sys
import numpy as np
import nibabel as nib

# python scripts/mean-nifti.py <path/to/nifti1.nii> <path/to/nifti2.nii> <path/to/mean-output-nifti.nii>

nifti_file_1=sys.argv[1]
nifti_file_2=sys.argv[2]
nifti_mean_file=sys.argv[3]

n1_img = nib.load(nifti_file_1)
n2_img = nib.load(nifti_file_2)

mean_data = np.mean(np.stack((n1_img.get_fdata(),n2_img.get_fdata())),axis=0)

mean_img = nib.spatialimages.SpatialImage(mean_data, affine=n1_img.affine, header=n1_img.header)

nib.save(mean_img, nifti_mean_file)
