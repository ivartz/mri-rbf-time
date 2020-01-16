import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from scipy.interpolate import Rbf
import multiprocessing as mp
import argparse
import copy

# _t suffix in names means tuple
# _l suffix in names means list
# _q suffix in name means a multiprocessing queue

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

def interpolate_volume_elements_to_linear_time(full_volumes_data_arr, \
                                               intervals_between_volumes_t, \
                                               z_interval_t, \
                                               y_interval_t, \
                                               x_interval_t):
    num_vols = len(full_volumes_data_arr)
    assert num_vols - 1 == len(intervals_between_volumes_t), \
    "Non-matching number of volumes and intervals"
    volumes_data_l = [None for _ in range(num_vols)]
    for num in range(num_vols):
        # Get volume element data
        volumes_data_l[num] = \
                       full_volumes_data_arr[num][z_interval_t[0]:z_interval_t[1], \
                                                  y_interval_t[0]:y_interval_t[1], \
                                                  x_interval_t[0]:x_interval_t[1]]
    # Stack all volumes along a new time dimension, creating the data with shape (t, z, y, x)
    data = np.stack(tuple(volumes_data_l))
            
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
    grids = [np.mgrid[time_idx:time_idx+1:1/interval_duration, 0:zdim, 0:ydim, 0:xdim] \
    for time_idx, interval_duration in enumerate(intervals_between_volumes_t)]
    
    # Stack all grids
    TI, ZI, YI, XI = np.hstack(tuple(grids))

    # Create radial basis functions
    #rbf_clinst = Rbf(t, z, y, x, f, function="multiquadric", norm='euclidean')
    rbf = Rbf(t, z, y, x, f, function='multiquadric') # If scipy 1.1.0 , only euclidean, default

    # Interpolate the voxel values f to have values for the indices in the grids,
    # resulting in interpolated voxel values FI
    FI = rbf(TI, ZI, YI, XI)

    data_interpolated = FI
        
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

def save_nifti(data, f, affine, header):
    img = nib.spatialimages.SpatialImage(data, affine=affine, header=header)
    img.set_data_dtype(np.float32)
    nib.save(img, f)

def merge_update_to_disk(save_dir, \
                         subvol_save_index_buffer, \
                         subvol_save_data_buffer, \
                         nifti_affine, \
                         nifti_header, \
                         current_process, \
                         current_process_name, \
                         tot_vol_dims):
    # This function takes time and is a bottleneck when called frequently
    # Merge update files sequentially to avoid loading and all volumes 
    # into in working menory at the same time
    
    for time_point in range(subvol_save_data_buffer.shape[1]):
        print("write %s: for time point %i, reading previously saved voxels from %s" % \
             (current_process_name, time_point+1, save_dir + "/" + "{0:03d}".format(time_point+1) + "_raw_voxels.npz"))
        
        # Load the existing voxel data for time_point
        stitched_data_time_point = np.load(save_dir + "/" + "{0:03d}".format(time_point+1) + "_raw_voxels.npz")["arr_0"]
        
        # Only look at the subvols from the first time point
        subvols_data_buffer_time_point = subvol_save_data_buffer[:, time_point]        
                
        # Iterate through the subvols for the given time_point
        for buffer_index, subvol_data in enumerate(subvols_data_buffer_time_point):
            print("write %s: for time point %i, going over subvol %i" % (current_process_name, time_point+1, buffer_index+1))
            
            # Get slice indices
            slice_indices = subvol_save_index_buffer[buffer_index]
            z_interval_t, y_interval_t, x_interval_t = slice_indices
            
            # Version that takes the min - deterministic
            # an attempt to avoid being sensitiv to interpolation overshooting
            # as of Gibbs phenomenon, etc.
            
            # Get the existing subvol data previously stored
            subvol_existing_data = stitched_data_time_point[z_interval_t[0]:z_interval_t[1], \
                                                            y_interval_t[0]:y_interval_t[1], \
                                                            x_interval_t[0]:x_interval_t[1]]
             
            # Make a new subvol data that is going to contain the mean 
            # of existing and new subvol data if non nan value at 
            # a voxel location in existing subvol
            new_subvol_data = subvol_data
            
            # For voxels that are not nan in existing subvol data, 
            # take the mean of existing and new voxels 
            # and store the means in new subvol data
            new_subvol_data[~np.isnan(subvol_existing_data)] = \
            np.min((subvol_existing_data[~np.isnan(subvol_existing_data)], \
                                          subvol_data[~np.isnan(subvol_existing_data)]), axis=0)
            
            # Update the stitched_data with the new subvol data, which contains
            # 1. means of existing and new voxels if existing voxel was not nan
            # 2. new voxels if exisiting voxel was nan
            stitched_data_time_point[z_interval_t[0]:z_interval_t[1], \
                                     y_interval_t[0]:y_interval_t[1], \
                                     x_interval_t[0]:x_interval_t[1]] = new_subvol_data 
        
        print("write %s: percent complete: " % current_process_name, end="")
        print("{0:.2f}".format(subvol_save_data_buffer.shape[1]*100*np.sum(~np.isnan(stitched_data_time_point))/np.prod(tot_vol_dims)))
        print("write %s: saving updated raw and NIFTI1 to disk" % current_process_name)        
        # Save raw voxel data (overwriting)
        np.savez(save_dir + "/" + "{0:03d}".format(time_point+1) + "_raw_voxels.npz", stitched_data_time_point)
        # Save NIFTI1 file of the current stitched data (overwriting)
        save_nifti(stitched_data_time_point, save_dir + "/" + "{0:03d}".format(time_point+1) + ".nii.gz", nifti_affine, nifti_header)

def stitch_subvols_from_queue_and_save(subvols_q, \
                                       subvol_shape, \
                                       tot_vol_dims, \
                                       save_dir, \
                                       nifti_header, \
                                       nifti_affine, \
                                       num_subvols_save_buffer):
    #                                        
    current_process = mp.current_process()
    current_process_name = current_process.name
                                           
    # Create buffer to avoid saving to disk too often
    subvol_save_data_buffer = np.zeros((num_subvols_save_buffer,tot_vol_dims[0])+subvol_shape)
    subvol_save_index_buffer = []
    num_subvols_buffered = 0
    
    stitched_data = np.empty(tot_vol_dims)
    stitched_data.fill(np.nan)
    [np.savez(save_dir + "/" + "{0:03d}".format(time_point+1) + "_raw_voxels.npz", data) for time_point, data in enumerate(stitched_data)]
    del stitched_data
    while True:
        if not subvols_q.empty():
            m = subvols_q.get_nowait()
            if m == "finished":
                break
            else:
                if len(subvol_save_index_buffer) < num_subvols_save_buffer:
                    # Continue filling up the buffer
                    print("write %s: adding to buffer; %i" % (current_process_name, num_subvols_buffered+1))
                    slice_indices, subvol_data = m
                    subvol_save_index_buffer += [slice_indices]
                    subvol_save_data_buffer[num_subvols_buffered] = subvol_data
                    num_subvols_buffered += 1
                else:
                    # Buffer is full, time to save to disk
                    print("write %s: buffer full, time to save to disk" % current_process_name)
                    
                    # Merge buffered subvols with saved subvols from disk
                    # and save (overwrite) raw and NIFTI1 files to disk.
                    # This takes some time
                    merge_update_to_disk(save_dir, \
                                         subvol_save_index_buffer, \
                                         subvol_save_data_buffer, \
                                         nifti_affine, \
                                         nifti_header, \
                                         current_process, \
                                         current_process_name, \
                                         tot_vol_dims)

                    # Empty the subvol save buffer
                    num_subvols_buffered = 0
                    subvol_save_index_buffer = []
                    
                    # Fill the excess volume into the clean buffer for handling in the nest interation
                    print("write %s: adding to buffer; %i" % (current_process_name, num_subvols_buffered+1))
                    slice_indices, subvol_data = m
                    subvol_save_index_buffer += [slice_indices]
                    subvol_save_data_buffer[num_subvols_buffered] = subvol_data
                    num_subvols_buffered += 1                    

def interpolate_subvol_and_send_to_queue(z_interval_t, \
                                         y_interval_t, \
                                         x_interval_t):
    current_process = mp.current_process()
    current_process_name = current_process.name
    #print("interpolate %s: received intervals for interpolating subvol" % current_process_name)
    data_interpolated = \
    interpolate_volume_elements_to_linear_time(interpolate_subvol_and_send_to_queue.full_volumes_data_arr, \
                                               interpolate_subvol_and_send_to_queue.intervals_between_volumes_t, \
                                               z_interval_t, \
                                               y_interval_t, \
                                               x_interval_t)
    #print("interpolate %s: sending interpolated data to queue" % current_process_name)
    interpolate_subvol_and_send_to_queue.subvols_q.put_nowait(((z_interval_t, y_interval_t, x_interval_t), data_interpolated))

def interpolate_subvol_and_send_to_queue_init(subvols_q, \
                                              volumes_data_arr, \
                                              intervals_between_volumes_t):
    interpolate_subvol_and_send_to_queue.subvols_q = subvols_q
    #"""
    interpolate_subvol_and_send_to_queue.full_volumes_data_arr = volumes_data_arr
    interpolate_subvol_and_send_to_queue.intervals_between_volumes_t = intervals_between_volumes_t
    #"""
    """
    interpolate_subvol_and_send_to_queue.full_volumes_data_arr = copy.deepcopy(volumes_data_arr)
    interpolate_subvol_and_send_to_queue.intervals_between_volumes_t = copy.deepcopy(intervals_between_volumes_t)
    """

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
      help="The time interval between the NIFTI1 file scans, in some time unit, f. ex. days, separated by space",
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
 
    # Set the shape of each subvolume that is unterpolated over time
    # 20, 20, 20 was the maximum shape on a 32 GB RAM machine before memory error
    # This was a fast setup
    # part 1
    subvol_shape = (3, 3, 3)
    subvol_stride = (2, 2, 2)
    
    #subvol_shape = (21, 21, 21)
    #subvol_stride = (14, 14, 14)
   
    # part 2
    num_subvols_save_buffer = 10000
    #num_subvols_save_buffer = 10
    
    # num_workes needs to be > 1
    # to have concurrent worker and writer process 
    #num_workers = mp.cpu_count()-1
    # part 3
    num_workers = 2
   
    print("----------------------------------------------------------------")
    print("Welcome to the Radial Basis Function time interpolation routine!")
    print("                  This will run for a while                     ")
    print("----------------------------------------------------------------")
   
    # Assuming all volumes have the same voxel dimensions
    # and spatial dimensions, and are properly co-registrated.
    vols_spatialimg_t = tuple(nib.load(file) for file in args.nifti)
    
    # Resample data in volumes to the first volume using affine transforms in the nifti headers.
    vols_spatialimg_resampled_t = tuple(resample_to_img(vol_spatialimg, \
                                                        vols_spatialimg_t[0]) \
                                                        for vol_spatialimg in vols_spatialimg_t[1:])
    
    # Constrcut tuples containing the nifti object and data of each examination volume
    volumes_niilike_t = (vols_spatialimg_t[0],) + vols_spatialimg_resampled_t
    volumes_data_arr = np.stack(tuple(volume_niilike.get_fdata() for volume_niilike in volumes_niilike_t))
    
    # Test with random data
    #n = 5
    #volumes_data_arr = np.stack((np.random.rand(n,n,n), np.random.rand(n,n,n)))
    
    # Construct tuple containing interval (in a given time unit, f. ex. days) between each examination
    #intervals_between_volumes_t = (7,) # Note , at the end in other to make 
                                            # it iterable when it contains only one value
    intervals_between_volumes_t = tuple(args.timeint)
    
    # Get the shape of the first volume for use in the interpolation
    vol_shape = volumes_niilike_t[0].shape   

    interval_indexes_l = vol_get_subvols_interval_indexes(vol_shape, subvol_shape, subvol_stride)
    
    # Multiprocessing manager
    manager = mp.Manager()
    
    # A queue that each interpolation process puts interpolated data on
    subvols_q = manager.Queue()
            
    # Multiprocessing pool of N workers, N = number of physical CPU cores mp.cpu_count()
    # Main will run in a serparate process, thus subtract 1 from mp.cpu_count()
    # to utilize exactly all available cpu cores
    mp_p = mp.Pool(num_workers, \
                   interpolate_subvol_and_send_to_queue_init, \
                   initargs=(subvols_q, \
                             volumes_data_arr, \
                             intervals_between_volumes_t), \
                   ) # maxtasksperchild=1
    
    # The final shape of the total volumes interpolated over time (number of time units)
    tot_vol_shape = (np.sum(intervals_between_volumes_t),) + vol_shape
    
    # Start process that listens for data on subvols_q and stiches the data into
    # the total interpolated volume series
    mp_p.apply_async(stitch_subvols_from_queue_and_save, args=(subvols_q, \
                                                               subvol_shape, \
                                                               tot_vol_shape, \
                                                               args.savedir, \
                                                               volumes_niilike_t[0].header, \
                                                               volumes_niilike_t[0].affine, \
                                                               num_subvols_save_buffer))
        
    # Interpolate subvols in paralell on N processes
    mp_p.starmap(interpolate_subvol_and_send_to_queue, interval_indexes_l)
    
    # Interpolation processes finished, so put finish to subvols_q in order to end that process
    subvols_q.put("finished")
    
    # Close the multiprocessing pool, terminating the script
    mp_p.close()

    print("----------------------------------------------------------------")   
    print("                  Finished interpolation                        ")
    print("----------------------------------------------------------------")
 
