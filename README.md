# Radial Basis Function Interpolation on the time axis

Assumes all volumes are co-rigestered to the first (baseline) volume.

Recommended dependencies:
```cmd
conda create -n interpenv python=3.8
conda activate interpenv
conda install scipy=1.3.2 nibabel nilearn
```

Help:

```cmd
cd scripts
python rbfinterp_mp_large.py -h
```

Example run on Windows
```cmd
python scripts\rbfinterp_mp_large.py --nifti data\input\T2_SPACE_v01.nii data\input\T2_SPACE_v01_to_v02.linear.nii --timeint 7 --mask data\input\masks\T2_SPACE_v01-Custom-ROI-cube.nii --savedir data\output\res_cube
```
The command interpolates between two volumes such that there will be 7 output volumes. The first output volume is the first input volume, the last output volume is the second input volume, but only at binary mask regions. --mask can be ommitted to interpolate between complete volumes.
