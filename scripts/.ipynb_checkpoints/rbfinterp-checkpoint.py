import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from scipy.interpolate import Rbf
from spimagine import volshow, volfig
from spimagine.utils.quaternion import Quaternion

# _t suffix in names means tuple
# _l suffix in names means list

def spimagine_show_volume_numpy(numpy_array, stackUnits=(1, 1, 1), interpolation="nearest", cmap="grays"):
    # Spimagine OpenCL volume renderer.
    volfig()
    spim_widget = \
            volshow(numpy_array, stackUnits=stackUnits, interpolation=interpolation)
    spim_widget.set_colormap(cmap)
    spim_widget.transform.setQuaternion(Quaternion(-0.005634209439510011,0.00790509382124309,-0.0013812284289010514,-0.9999519273706857))

def flatten_tensor(T):
    # Flattens the M x N x ..., x D tensor while preserving original indices
    # to the output shape (MxNx...xD, F, M, N, ..., D)
    # where F is a vector of the values in A, and M, N, ..., D a vector of 
    # original indices in each dimension in A (for the values in F).
    # https://stackoverflow.com/questions/46135070/generalise-slicing-operation-in-a-numpy-array/46135084#46135084
    n = T.ndim
    grid = np.ogrid[tuple(map(slice, T.shape))]
    out = np.empty(T.shape + (n+1,), dtype=T.dtype)
    for i in range(n):
        out[...,i+1] = grid[i]
    out[...,0] = T
    out.shape = (-1,n+1)
    return out

def interpolate_volume_elements_to_linear_time(volumes_niilike_t, \
                                               intervals_between_volumes_days_t, \
                                               z_interval_t, \
                                               y_interval_t, \
                                               x_interval_t):
    num_vols = len(volumes_niilike_t)
    assert num_vols - 1 == len(intervals_between_volumes_days_t), \
    "Non-matching number of volumes and intervals"
    volumes_data_l = [None for _ in range(num_vols)]
    for num in range(num_vols):
        # Get volume element data
        volumes_data_l[num] = \
        volumes_niilike_t[num].get_fdata()[z_interval_t[0]:z_interval_t[1], \
                                           y_interval_t[0]:y_interval_t[1], \
                                           x_interval_t[0]:x_interval_t[1]]
    # Stack all volumes along a new time dimension, creating the data with shape (t, z, y, x)
    data = np.stack(tuple(volumes_data_l))
    
    #spimagine_show_volume_numpy(data)
    
    # Get the resulting dimensions of the stacked data
    # for later use in the grid defition
    tdim, zdim, ydim, xdim = data.shape
    
    # Flatten the stacked data, for use in Rbf
    data_flattened = flatten_tensor(data)
    
    # Get the colums in the flattened data
    # The voxel values
    f = data_flattened[:,0]
    # Time coordinates of the voxel values
    t = data_flattened[:,1]
    # Z coordinates of the voxel values
    z = data_flattened[:,2]
    # Y coordinates of the voxel values
    y = data_flattened[:,3]
    # X coordinates of the voxel values
    x = data_flattened[:,4]
    
    # Make grids of indices with resolutions we want after the interpolation
    grids = [np.mgrid[time_idx:time_idx+1:1/interval_duration_days, 0:zdim, 0:ydim, 0:xdim] for time_idx, interval_duration_days in enumerate(intervals_between_volumes_days_t)]
    
    # Stack all grids
    TI, ZI, YI, XI = np.hstack(tuple(grids))
    
    # Create radial basis functions
    rbf = Rbf(t, z, y, x, f, function="multiquadric", norm='euclidean')
    
    # Interpolate the voxel values f to have values for the indices in the grids,
    # resulting in interpolated voxel values FI
    FI = rbf(TI, ZI, YI, XI)
    
    data_interpolated = FI
    
    #spimagine_show_volume_numpy(data_interpolated)
    
    return data_interpolated

if __name__ == "__main__":
    
    # Assuming all volumes have the same voxel dimensions
    # and spatial dimensions.
    vol1_spatialimg = nib.load("../T2_SPACE_v01.nii")
    vol2_spatialimg = nib.load("../T2_SPACE_v01_to_v02.linear.nii")
    
    # Resample data in volumes to the first volume using affine transforms in the nifti headers.
    vol2_niftiimg_resampled = resample_to_img(vol2_spatialimg, vol1_spatialimg)

    zdim, ydim, xdim = vol1_spatialimg.get_fdata().shape

    #cubedim = 2 # 20 was the maximum on a 32 GB RAM machine before memory error
    
    volumes_niilike_t = (vol1_spatialimg, vol2_niftiimg_resampled)

    intervals_between_volumes_days_t = (7,) # Note , at the end in other to make it iterable when it contains only one value

    stiched_data = np.zeros((np.sum(intervals_between_volumes_days_t), zdim, ydim, xdim))
    
    # Non-overlapping volumes
    vol_dim_z, vol_dim_y, vol_dim_x = 2, 2, 2
    num_vols_z, num_vols_y, num_vols_x = zdim//vol_dim_z-1, ydim//vol_dim_y-1, xdim//vol_dim_x-1
    for vol_num_z in range(num_vols_z):
        for vol_num_y in range(num_vols_y):
            for vol_num_x in range(num_vols_x):
    
                z_interval_t = (zdim//2-vol_dim_z*num_vols_z//2-vol_dim_z//2+vol_dim_z*vol_num_z, \
                                zdim//2-vol_dim_z*num_vols_z//2+vol_dim_z//2+vol_dim_z*vol_num_z) # The patient y axis
        
                y_interval_t = (ydim//2-vol_dim_y*num_vols_y//2-vol_dim_y//2+vol_dim_y*vol_num_y, \
                                ydim//2-vol_dim_y*num_vols_y//2+vol_dim_y//2+vol_dim_y*vol_num_y) # The patient z axis
            
                x_interval_t = (xdim//2-vol_dim_x*num_vols_x//2-vol_dim_x//2+vol_dim_x*vol_num_x, \
                                xdim//2-vol_dim_x*num_vols_x//2+vol_dim_x//2+vol_dim_x*vol_num_x) # The patient x axis


                data_interpolated = interpolate_volume_elements_to_linear_time(volumes_niilike_t, \
                                                                               intervals_between_volumes_days_t, \
                                                                               z_interval_t, \
                                                                               y_interval_t, \
                                                                               x_interval_t)
                stiched_data[:, \
                             z_interval_t[0]:z_interval_t[1], \
                             y_interval_t[0]:y_interval_t[1], \
                             x_interval_t[0]:x_interval_t[1]] = data_interpolated
    
    spimagine_show_volume_numpy(stiched_data)