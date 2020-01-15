import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from scipy.interpolate import Rbf
from spimagine import volshow, volfig
from spimagine.utils.quaternion import Quaternion
import multiprocessing as mp
import copy
import argparse
from time import sleep

# _t suffix in names means tuple
# _l suffix in names means list
# _q suffix in name means a multiprocessing queue

def spimagine_show_volume_numpy(numpy_array, stackUnits=(1, 1, 1), \
                                interpolation="nearest", cmap="grays"):
    # Spimagine OpenCL volume renderer.
    volfig()
    spim_widget = \
            volshow(numpy_array, stackUnits=stackUnits, interpolation=interpolation)
    spim_widget.set_colormap(cmap)
    spim_widget.transform.setQuaternion(Quaternion(-0.005634209439510011,\
                                                    0.00790509382124309,\
                                                   -0.0013812284289010514,\
                                                   -0.9999519273706857))

def spimagine_show_volume_numpy_return_widget(numpy_array, stackUnits=(1, 1, 1), \
                                              interpolation="nearest", cmap="grays"):
    # Spimagine OpenCL volume renderer.
    volfig()
    spim_widget = \
            volshow(numpy_array, stackUnits=stackUnits, interpolation=interpolation)
    spim_widget.set_colormap(cmap)
    spim_widget.transform.setQuaternion(Quaternion(-0.005634209439510011,\
                                                    0.00790509382124309,\
                                                   -0.0013812284289010514,\
                                                   -0.9999519273706857))
    return spim_widget

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

def interpolate_volume_elements_to_linear_time(full_volumes_data_t, \
                                               intervals_between_volumes_days_t, \
                                               z_interval_t, \
                                               y_interval_t, \
                                               x_interval_t):
    num_vols = len(full_volumes_data_t)
    assert num_vols - 1 == len(intervals_between_volumes_days_t), \
    "Non-matching number of volumes and intervals"
    volumes_data_l = [None for _ in range(num_vols)]
    for num in range(num_vols):
        # Get volume element data
        volumes_data_l[num] = \
                       full_volumes_data_t[num][z_interval_t[0]:z_interval_t[1], \
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
    
    #print(intervals_between_volumes_days_t)
    
    # Make grids of indices with resolutions we want after the interpolation
    grids = [np.mgrid[time_idx:time_idx+1:1/interval_duration_days, 0:zdim, 0:ydim, 0:xdim] \
    for time_idx, interval_duration_days in enumerate(intervals_between_volumes_days_t)]
    
    # Stack all grids
    TI, ZI, YI, XI = np.hstack(tuple(grids))
    
    #print("hahaha")
    #print(f.shape)
    #print(t.shape)
    #print(z.shape)
    #print(y.shape)
    #print(x.shape)
    
    # Create radial basis functions
    #rbf_clinst = Rbf(t, z, y, x, f, function="multiquadric", norm='euclidean')
    rbf = Rbf(t, z, y, x, f, function='multiquadric') # If scipy 1.1.0 , only euclidean, default
    
    #print("hehehe")
    
    # Interpolate the voxel values f to have values for the indices in the grids,
    # resulting in interpolated voxel values FI
    FI = rbf(TI, ZI, YI, XI)
    
    data_interpolated = FI
    
    #print("hei")
    #print(data_interpolated)
    
    #spimagine_show_volume_numpy(data_interpolated)
    
    return data_interpolated

def vol_get_subvols_interval_indexes(vol_shape, subvol_shape, stride_shape):
    
    # Orig volume shape
    zdim, ydim, xdim = vol_shape
    
    #
    stride_z, stride_y, stride_x = stride_shape
    
    # Subvol shape
    vol_dim_z, vol_dim_y, vol_dim_x = subvol_shape
    
    # Calculate the number of non-overlapping volumes along each dimension
    num_vols_z, num_vols_y, num_vols_x = zdim//stride_z, ydim//stride_y, xdim//stride_x
    # For testing purposes
    #num_vols_z, num_vols_y, num_vols_x = 5, 5, 5
    
    # A list of tuples containing interval indexes along each dimension
    interval_indexes_l = []
    
    for vol_num_z in range(num_vols_z):
        for vol_num_y in range(num_vols_y):
            for vol_num_x in range(num_vols_x):
                """
                z_interval_t = (zdim//2-vol_dim_z*num_vols_z//2 - vol_dim_z//2+vol_dim_z*vol_num_z, \
                                zdim//2-vol_dim_z*num_vols_z//2 + vol_dim_z//2+vol_dim_z*vol_num_z) # The patient y axis
                
                y_interval_t = (ydim//2-vol_dim_y*num_vols_y//2 - vol_dim_y//2+vol_dim_y*vol_num_y, \
                                ydim//2-vol_dim_y*num_vols_y//2 + vol_dim_y//2+vol_dim_y*vol_num_y) # The patient z axis
                
                x_interval_t = (xdim//2-vol_dim_x*num_vols_x//2 - vol_dim_x//2+vol_dim_x*vol_num_x, \
                                xdim//2-vol_dim_x*num_vols_x//2 + vol_dim_x//2+vol_dim_x*vol_num_x) # The patient x axis
                """
                """
                # Non-stride version
                z_interval_t = (vol_num_z*vol_dim_z, \
                                (vol_num_z+1)*vol_dim_z) # The patient y axis
                
                y_interval_t = (vol_num_y*vol_dim_y, \
                                (vol_num_y+1)*vol_dim_y) # The patient z axis
                
                x_interval_t = (vol_num_x*vol_dim_x, \
                                (vol_num_x+1)*vol_dim_x) # The patient x axis
                """
                # Stride version
                # Check that volume is within the index dimensions
                if vol_num_z*stride_z+vol_dim_z <= zdim \
                and vol_num_y*stride_y+vol_dim_y <= ydim \
                and vol_num_x*stride_x+vol_dim_x <= xdim:
                    z_interval_t = (vol_num_z*stride_z, \
                                    vol_num_z*stride_z+vol_dim_z) # The patient y axis
                    
                    y_interval_t = (vol_num_y*stride_y, \
                                    vol_num_y*stride_y+vol_dim_y) # The patient z axis
                    
                    x_interval_t = (vol_num_x*stride_x, \
                                    vol_num_x*stride_x+vol_dim_x) # The patient x axis
                    
                    
                    # Append volume slice to list
                    interval_indexes_l += [(z_interval_t, y_interval_t, x_interval_t)]
    
    return interval_indexes_l

def vol_get_subvols_interval_indexes2(vol_shape, subvol_shape, stride_shape):
    
    # Orig volume shape
    zdim, ydim, xdim = vol_shape
    
    #
    stride_z, stride_y, stride_x = stride_shape
    
    # Subvol shape
    vol_dim_z, vol_dim_y, vol_dim_x = subvol_shape
    
    # Calculate the number of non-overlapping volumes along each dimension
    #num_vols_z, num_vols_y, num_vols_x = zdim//stride_z, ydim//stride_y, xdim//stride_x
    num_vols_z, num_vols_y, num_vols_x = 2, 2, 2
    
    # A list of tuples containing interval indexes along each dimension
    interval_indexes_l = []
    
    for vol_num_z in range(num_vols_z):
        for vol_num_y in range(num_vols_y):
            for vol_num_x in range(num_vols_x):
                # Non-stride centered version
                z_interval_t = (zdim//2-vol_dim_z*num_vols_z//2 - vol_dim_z//2+vol_dim_z*vol_num_z, \
                                zdim//2-vol_dim_z*num_vols_z//2 + vol_dim_z//2+vol_dim_z*vol_num_z) # The patient y axis
                
                y_interval_t = (ydim//2-vol_dim_y*num_vols_y//2 - vol_dim_y//2+vol_dim_y*vol_num_y, \
                                ydim//2-vol_dim_y*num_vols_y//2 + vol_dim_y//2+vol_dim_y*vol_num_y) # The patient z axis
                
                x_interval_t = (xdim//2-vol_dim_x*num_vols_x//2 - vol_dim_x//2+vol_dim_x*vol_num_x, \
                                xdim//2-vol_dim_x*num_vols_x//2 + vol_dim_x//2+vol_dim_x*vol_num_x) # The patient x axis

                interval_indexes_l += [(z_interval_t, y_interval_t, x_interval_t)]
    
    return interval_indexes_l

def save_nifti(data, file, affine, header):
    img = nib.spatialimages.SpatialImage(data, affine=affine, header=header)
    img.set_data_dtype(np.float32)
    nib.save(img, file)

def stitch_subvols_from_queue_and_save(subvols_q, \
                                       subvol_stride, \
                                       subvol_shape, \
                                       tot_vol_dims, \
                                       export_q, \
                                       save_dir, \
                                       nifti_header, \
                                       nifti_affine, \
                                       num_subvols_save_buffer):
                                       
    current_process = mp.current_process()
    current_process_name = current_process.name
                                       
    #stride_z, stride_y, stride_x = subvol_stride
    #subvol_dim_z, subvol_dim_y, subvol_dim_x = subvol_shape
    
    # Create buffer to avoid saving to disk too often
    subvol_save_data_buffer = np.zeros((num_subvols_save_buffer,tot_vol_dims[0])+subvol_shape)
    #subvol_save_data_buffer = np.empty((2, 7, 2, 2, 2))
    subvol_save_index_buffer = []
    num_subvols_buffered = 0
    
    stitched_data = np.empty(tot_vol_dims)
    stitched_data.fill(np.nan)
    np.savez(save_dir + "/raw_voxels.npz", stitched_data)
    del stitched_data
    #widget = None
    #i = 1
    while True:
        if not subvols_q.empty():
            m = subvols_q.get_nowait()
            print("write %s: getting interpolated subvolumes" % current_process_name)
            #print(m.empty())
            #"""
            #if m.empty():
            #    pass
            if m == "finished":
                # Set remaining nan values to 0, if there
                stitched_data = np.load(save_dir + "/raw_voxels.npz")["arr_0"]
                stitched_data[np.isnan(stitched_data)] = 0
                export_q.put_nowait(stitched_data)
                break
            else:
                if len(subvol_save_index_buffer) < num_subvols_save_buffer:
                    # Continue filling up the buffer
                    print("write %s: adding to buffer; %i" % (current_process_name, num_subvols_buffered+1))
                    slice_indices, subvol_data = m
                    #print(slice_indices)
                    #print(subvol_data)
                    subvol_save_index_buffer += [slice_indices]
                    #print("ja")
                    #print(subvol_save_data_buffer.shape)
                    #print(subvol_data.shape)
                    #print(subvol_save_data_buffer[num_subvols_buffered])
                    #print(subvol_save_data_buffer[num_subvols_buffered].shape)
                    subvol_save_data_buffer[num_subvols_buffered] = subvol_data
                    #print("jo!")
                    num_subvols_buffered += 1
                    #continue
                else:
                    # Buffer is full, time to save to disk
                    # Load already interpolated data from a saved numpy .npz file
                    print("write %s: time to save" % current_process_name)
                    stitched_data = np.load(save_dir + "/raw_voxels.npz")["arr_0"]
                    
                    for i, subvol_data in enumerate(subvol_save_data_buffer):
                        print("write %s: going over subvols; %i" % (current_process_name, i+1))
                        # Get slice indices
                        slice_indices = subvol_save_index_buffer[i]
                        z_interval_t, y_interval_t, x_interval_t = slice_indices
                        
                        
                        # Version that takes the mean of data begin - non-deterministic
                        #"""
                        
                        # Get the existing subvol data previously stored
                        subvol_existing_data = stitched_data[:, \
                                                             z_interval_t[0]:z_interval_t[1], \
                                                             y_interval_t[0]:y_interval_t[1], \
                                                             x_interval_t[0]:x_interval_t[1]]
                        
                        # Make a new subvol data that is going to contain the mean 
                        # of existing and new subvol data if non nan value at 
                        # a voxel location in existing subvol
                        new_subvol_data = subvol_data
                        
                        #print("hei!")
                        
                        # For voxels that are not nan in existing subvol data, 
                        # take the mean of existing and new voxels 
                        # and store the means in new subvol data
                        new_subvol_data[~np.isnan(subvol_existing_data)] = \
                        np.max((subvol_existing_data[~np.isnan(subvol_existing_data)], \
                                                      subvol_data[~np.isnan(subvol_existing_data)]), axis=0)
                        
                        #print("ho!")
                        
                        # Update the stitched_data with the new subvol data, which contains
                        # 1. means of existing and new voxels if existing voxel was non nan
                        # 2. new voxels of exisiting voxel was nan
                        stitched_data[:, \
                                      z_interval_t[0]:z_interval_t[1], \
                                      y_interval_t[0]:y_interval_t[1], \
                                      x_interval_t[0]:x_interval_t[1]] = new_subvol_data
                        
                        #"""
                        # Version that takes the mean of data end
                        
                        
                        
                                      
                        # Version that only replaces existing subvol data with the received subvol data
                        """
                        stitched_data[:, \
                                      z_interval_t[0]:z_interval_t[1], \
                                      y_interval_t[0]:y_interval_t[1], \
                                      x_interval_t[0]:x_interval_t[1]] = subvol_data
                        """
                        
                        # Version that only takes the volume in subvol data overlapping with the stride
                        # and stitches the overlapping subvol data together - deterministic
                        #"""
                        
                        #subvol_one_third_dim = 
                        #z_mid_interval_t 
                        
                        #"""
                        
                        #print("{0:.2f}".format(100*np.sum(~np.isnan(stitched_data))/np.prod(tot_vol_dims)) + " % finished")
                        #print(str(100*np.sum(~np.isnan(stitched_data))/np.prod(tot_vol_dims)) + " % finished")

                        #d = stitched_data.copy()
                        #d[np.isnan(d)] = 0
                        
                        #w = spimagine_show_volume_numpy_return_widget(d)
                        #w.saveFrame(str(i) + ".png")
                        #i += 1
                        #w.closMe()
                        """
                        if widget != None:
                            widget.closeMe()
                            widget = spimagine_show_volume_numpy_return_widget(d)
                        else:
                            widget = spimagine_show_volume_numpy_return_widget(d)
                        """
                    
                    # Empty the subvol save buffer
                    num_subvols_buffered = 0
                    subvol_save_index_buffer = []
                    
                    # Fill in the excess volume in the clean buffer for handling in the nest interation
                    print("write %s: adding to buffer; %i" % (current_process_name, num_subvols_buffered+1))
                    slice_indices, subvol_data = m
                    subvol_save_index_buffer += [slice_indices]
                    subvol_save_data_buffer[num_subvols_buffered] = subvol_data
                    num_subvols_buffered += 1
                    
                    print("write %s: starting to save" % current_process_name)

                    # Save the updated stitched data (overwriting)
                    np.savez(save_dir + "/raw_voxels.npz", stitched_data)
                    
                    # Save NIFTI1 files for each dynamic window of the current stitched data (overwriting)
                    for i, data in enumerate(stitched_data):
                        save_nifti(data, save_dir + "/day_" + str(i+1) + ".nii", nifti_affine, nifti_header)
                    
                    print("write %s: raw + NIFTI1 saved or updated" % current_process_name)
                    #[save_nifti(data, save_dir + "/day_" + str(i+1) + ".nii", \
                    #nifti_affine, nifti_header) for i, data in enumerate(stitched_data)]
                    
                    # Free stitched data
                    del stitched_data
                    #continue

def interpolate_subvol_and_send_to_queue(z_interval_t, \
                                         y_interval_t, \
                                         x_interval_t):
    current_process = mp.current_process()
    current_process_name = current_process.name
    """
    volumes_data_t = \
    interpolate_subvol_and_send_to_queue.volumes_data_t
    intervals_between_volumes_days_t = \
    interpolate_subvol_and_send_to_queue.intervals_between_volumes_days_t
    """
    #print(z_interval_t)
    #print(y_interval_t)
    #print(x_interval_t)
    data_interpolated = \
    interpolate_volume_elements_to_linear_time(interpolate_subvol_and_send_to_queue.full_volumes_data_t, \
                                               interpolate_subvol_and_send_to_queue.intervals_between_volumes_days_t, \
                                               z_interval_t, \
                                               y_interval_t, \
                                               x_interval_t)
    print("interpolate %s: sending interpolated data to queue" % current_process_name)
    #print(data_interpolated.shape)
    interpolate_subvol_and_send_to_queue.subvols_q.put_nowait(((z_interval_t, y_interval_t, x_interval_t), data_interpolated))

def interpolate_subvol_and_send_to_queue_init(subvols_q, \
                                              volumes_data_t, \
                                              intervals_between_volumes_days_t):
    interpolate_subvol_and_send_to_queue.subvols_q = subvols_q
    """
    interpolate_subvol_and_send_to_queue.full_volumes_data_t = \
        copy.deepcopy(tuple([volume_data.get_fdata() for volume_data in volumes_data_t]))
    interpolate_subvol_and_send_to_queue.intervals_between_volumes_days_t = \
        copy.deepcopy(intervals_between_volumes_days_t)
    """
    """
    interpolate_subvol_and_send_to_queue.full_volumes_data_t = \
        copy.deepcopy(volumes_data_t)
    interpolate_subvol_and_send_to_queue.intervals_between_volumes_days_t = \
        copy.deepcopy(intervals_between_volumes_days_t)
    """
    #"""
    #interpolate_subvol_and_send_to_queue.full_volumes_niilike_t = volumes_niilike_t
    interpolate_subvol_and_send_to_queue.full_volumes_data_t = volumes_data_t
    interpolate_subvol_and_send_to_queue.intervals_between_volumes_days_t = intervals_between_volumes_days_t
    #"""

if __name__ == "__main__":
    
    # Define command line options
    # this also generates --help and error handling
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
      "--nifti",  # name on the CLI - drop the `--` for positional/required parameters
      nargs="*",  # 0 or more values expected => creates a list
      help="Takes in the relative file path + \
      file name of two or more co-registrated NIFTI1 files, \
      separated by space",
      type=str,
      default=["1.nii", "2.nii"],
    )
    CLI.add_argument(
      "--timeint",
      nargs="*",
      help="The time interval between the NIFTI1 file scans, in days, separated by space",
      type=int,
      default=[7],
    )
    CLI.add_argument(
      "--savedir",
      help="The directory to save interpolated NIFTI files and raw voxel data",
      type=str,
      default="interpolated",
    )

    # Parse the command line
    args = CLI.parse_args()
    
    # Assuming all volumes have the same voxel dimensions
    # and spatial dimensions.
    #vol1_spatialimg = nib.load("../T2_SPACE_v01.nii")
    #vol2_spatialimg = nib.load("../T2_SPACE_v01_to_v02.linear.nii")
    vols_spatialimg_t = tuple(nib.load(file) for file in args.nifti)
    
    #print("HEI")
    #print(type(vols_spatialimg_t))
    
    # Resample data in volumes to the first volume using affine transforms in the nifti headers.
    #vol2_niftiimg_resampled = resample_to_img(vol2_spatialimg, vol1_spatialimg)
    vols_spatialimg_resampled_t = tuple(resample_to_img(vol_spatialimg, \
                                                        vols_spatialimg_t[0]) \
                                                        for vol_spatialimg in vols_spatialimg_t[1:])
    
    # Constrcut tuple containing the nifti object of each examination volume
    #volumes_data_t = (vol1_spatialimg, vol2_niftiimg_resampled)
    volumes_niilike_t = (vols_spatialimg_t[0],) + vols_spatialimg_resampled_t
    volumes_data_t = np.stack(tuple(volume_niilike.get_fdata() for volume_niilike in volumes_niilike_t))
    
    # Test with random data
    #n = 5
    #volumes_data_t = np.stack((np.random.rand(n,n,n), np.random.rand(n,n,n)))
    
    # Construct tuple containing interval (in days) between each examination
    #intervals_between_volumes_days_t = (7,) # Note , at the end in other to make 
                                            # it iterable when it contains only one value
    intervals_between_volumes_days_t = tuple(args.timeint)
    
    # Get the shape of the first volume for use in the interpolation
    #vol_shape = volumes_data_t[0].get_fdata().shape
    vol_shape = volumes_niilike_t[0].shape
    
    #vol_shape = volumes_niilike_t[0].shape
    
    # Set the shape of each subvolume that is unterpolated over time
    # 20, 20, 20 was the maximum shape on a 32 GB RAM machine before memory error
    #subvol_shape = (9, 9, 9)
    #subvol_stride = (3, 3, 3)
    subvol_shape = (12, 12, 12)
    subvol_stride = (8, 8, 8)

    interval_indexes_l = vol_get_subvols_interval_indexes(vol_shape, subvol_shape, subvol_stride)
    
    #print(interval_indexes_l)
    
    # Multiprocessing manager
    manager = mp.Manager()
    
    # A queue that each interpolation process puts interpolated data on
    subvols_q = manager.Queue()
    
    # 
    export_q = manager.Queue()
        
    # Multiprocessing pool of N workers, N = number of physical CPU cores mp.cpu_count()
    # Main and the volume writing process are in serparate process, thus subtract 2 from mp.cpu_count()
    # to utilize exactly all available cpu cores
    mp_p = mp.Pool(mp.cpu_count()-2, \
                   interpolate_subvol_and_send_to_queue_init, \
                   initargs=(subvols_q, \
                             volumes_data_t, \
                             intervals_between_volumes_days_t), \
                   ) # maxtasksperchild=1
    
    # The final shape of the total volumes interpolated over time (number of days)
    tot_vol_shape = (np.sum(intervals_between_volumes_days_t),) + vol_shape
    
    # Start process that listens for data on subvols_q and stiches the data into
    # the total interpolated volume series
    mp_p.apply_async(stitch_subvols_from_queue_and_save, args=(subvols_q, \
                                                               subvol_stride, \
                                                               subvol_shape, \
                                                               tot_vol_shape, \
                                                               export_q, \
                                                               args.savedir, \
                                                               volumes_niilike_t[0].header, \
                                                               volumes_niilike_t[0].affine, \
                                                               10))
    
    
    #sleep(5)
    #print(interval_indexes_l)
    # Interpolate subvols in paralell on N processes
    mp_p.starmap(interpolate_subvol_and_send_to_queue, interval_indexes_l)
    
    # Interpolation processes finished, so put finish to subvols_q in order to end that process
    subvols_q.put("finished")
    
    # Get final image
    vol_series = export_q.get()
        
    # Close the multiprocessing pool, terminating the script
    mp_p.close()
    
    #
    #w1 = spimagine_show_volume_numpy_return_widget(volumes_data_t)
    w2 = spimagine_show_volume_numpy_return_widget(vol_series)
        
    input('Press any key to quit: ')
    
    #w1.closeMe()
    w2.closeMe()
