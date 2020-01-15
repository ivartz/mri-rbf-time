import numpy as np
import nibabel as nib

img = nib.load("dilatedmask.nii")

voxelstozero = 50

data = img.get_fdata()

mask = np.array([True]*np.prod(img.shape)).reshape(img.shape)

mask[voxelstozero:-voxelstozero,voxelstozero:-voxelstozero,voxelstozero:-voxelstozero] = False

data[mask] = 0

imgmod = nib.Nifti1Image(data, affine=img.affine, header=img.header)

nib.save(imgmod, "dilatedmask2.nii")