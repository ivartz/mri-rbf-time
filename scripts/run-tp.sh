# https://www.grymoire.com/Unix/Sed.html#uh-25

# Usage:
# bash scripts/run-tp.sh \
#   <1 run commands or 0 only print commands>
#   <patient oncohabitats dir> \
#   <path/to/timeintsfile.txt> \
#   <patient sequence results dir> \
#   <path/to/maskfile.nii or 0 for no mask> \
#   <interpolation time step>
#   <max number of specific interval legths in partition>
#   <usegpu>
#   2>&1 | tee <path/to/runlog.txt>

# 1: run script
# 0: dbg, print bash commands
#run_evals=0
run_evals=$1

# TODO
#nifti_dir=

#patient_onco_data_dir=../Elies-longitudinal-data-test
patient_onco_data_dir=$2

# To make a timeints_mod.txt from a .txt with a date format, 
# see scripts/get-timeints-days.py
# A timeints_mod.txt can be calculated 
# based on a file with time points with a 
# given time format (ack-times.txt) by
#python scripts/get-timeints-days.py <pat/to/ack-times.txt> > <path/to/saving/timeints-days.txt>
# F. ex. running:
#python scripts/get-timeints-days.py ../Elies-longitudinal-data-test/ack-times.txt > timeints-days.txt
# ack-times.txt was aquired form a dicom header reader program.

#timeints_days_file=$patient_onco_data_dir/timeints_mod.txt
timeints_days_file=$3

#patient_sequence_output_dir=/media/ivar/SSD700GB/gitprojects/Longitudinal_Study/data/output/sailor-patient/FLAIR
patient_sequence_output_dir=$4

#mask_file=$patient_onco_data_dir/flairbinmask.nii
# Set to use no mask
#mask_file=0
mask_file=$5


# Calculate how to optimal divide the inteprolation problem into partitions
# along the time axis.
#time_step=5
time_step=$6
#max_num_specif_intervals=0 # see timeints-divide-and-partition.py
max_num_specif_intervals=$7
readarray timeints_partitions_arr < <(python scripts/timeints-divide-and-partition.py \
    $timeints_days_file \
    $time_step \
    $max_num_specif_intervals)

usegpu=$8

# Make common folder for png files of inteprolated data
png_folder=$patient_sequence_output_dir/png
# Make directory if not exists
mkdir_command="[ -d $png_folder ] || mkdir -p $png_folder"
echo $mkdir_command
#
if [ $run_evals == 1 ] ; then
    eval $mkdir_command
fi
#

#echo ${#timeints_partitions_arr[@]}

num_processed_intervals=0
num_processed_volumes=0

#for partitions in "${timeints_partitions_arr[@]}"; do
for (( i = 0 ; i < ${#timeints_partitions_arr[@]} ; i++ )) ; do
    timeints=${timeints_partitions_arr[i]}
    #echo $timeints
    #read -a timeints_arr <<< $timeints
    timeints_arr=($timeints)
    num_timeints=${#timeints_arr[@]}
    #echo $num_intervals
    if [ $i == 0 ] ; then
        exam_num_begin=1
        exam_num_end=$(($num_timeints+1))
    else
        prev_exam_num_begin=$exam_num_begin
        exam_num_begin=$(($num_processed_intervals+1))
        exam_num_end=$(($num_processed_intervals+$num_timeints+1))
    fi
    niftis=$(ls $patient_onco_data_dir/*/Flair.nii.gz \
        | sed -n "$exam_num_begin,$exam_num_end p")
        
    savedir=$patient_sequence_output_dir/$exam_num_begin-$exam_num_end
    
    #echo $niftis
    #echo $timeints
    #echo $savedir
    
    # Run RBF interpolation
    # Will create .nii and raw .npz files
    # in $savedir
    if [ $mask_file == 0 ]; then
        interpolate_command="python scripts/rbfinterp_mp_large.py \
        --nifti $niftis \
        --timeint $timeints \
        --savedir $savedir"
    else
        interpolate_command="python scripts/rbfinterp_mp_large.py \
        --nifti $niftis \
        --timeint $timeints \
        --mask $mask_file \
        --savedir $savedir"
    fi
    echo $interpolate_command
    # This command runs the interpolation program:
    #
    if [ $run_evals == 1 ] ; then
        eval $interpolate_command
    fi
    #
    
    # Rename (if necessary) .nii files so that
    # .nii files and consequently .png files
    # from all partitions correspoind to each other
    # by having globally increasing numbers as file names
    # (indicating a time point and affected by time_step.
    # if time_step = 1 , then the time between each .nii file will 
    # be approximately 1 day, 
    # assuming scripts/get-timeints-days.py was used to calculate time
    # intervals from examination date formats in a .txt file.)
    if [ $i -gt 0 ] ; then
        first_volume_number=$(($num_processed_volumes-$i+1))
        rename_command="bash scripts/rename-files.sh $savedir/nii $first_volume_number"
        echo $rename_command
        
        # There are now two copies of a .nii with
        # number equal to first_volume_number
        # -> take the mean of the two niftis and
        # overwrite this nifti
        # (the one in this partition with number first_volume_number)
        # with the mean version
        printf -v number_padded "%03d" $first_volume_number
        prev_savedir=$patient_sequence_output_dir/$prev_exam_num_begin-$exam_num_begin
        prev_nii=$prev_savedir/nii/$number_padded.nii
        mean_niftis_command="python scripts/mean-nifti.py $prev_nii $savedir/nii/$number_padded.nii $savedir/nii/$number_padded.nii"
        echo $mean_niftis_command
        
        #
        if [ $run_evals == 1 ] ; then
            eval $rename_command
            eval $mean_niftis_command
        fi
    fi
        
    # Render .png snapshot of .nii files 
    # using fsleyes
    # Will be saved in $savedir/png
    render_command="bash scripts/render-frames-tp.sh $savedir/nii $savedir/png"
    echo $render_command
    #
    if [ $run_evals == 1 ] ; then
        eval $render_command
    fi
    #

    # Move rendered frames to a common folder for all partitions
    # The move will not overwrite (-n)
    # This means that the png for
    # each real (not interpolated) 
    # examination for i > 0
    # will always come from the rendring
    # of the last real volume
    # of the previous interpolation, if 
    # an interpolation on a previous partition 
    # was done 
    #move_pngs_command="mv -vn $savedir/png/*.png $png_folder"
    # This version overwrites target files if existing in src
    move_pngs_command="mv -v $savedir/png/*.png $png_folder"
    echo $move_pngs_command
    #
    if [ $run_evals == 1 ] ; then
        eval $move_pngs_command
    fi
    #
    remove_old_png_folder_command="rmdir $savedir/png"
    # Will also remove the first png for i > 0
    # that was not moved since it was already rendered
    # by previous interpolation run partition
    #remove_old_png_folder_command="rm -rd $savedir/png"
    echo $remove_old_png_folder_command
    #
    if [ $run_evals == 1 ] ; then
        eval $remove_old_png_folder_command
    fi
    #
    # Last, update num_processed_intervals to keep track
    # of where we are in the in the time axis in the
    # interpolation process
    num_processed_intervals=$(($num_processed_intervals+$num_timeints))
    # Also, save the number of processed (real and interpolated)
    # volumes, for use in renaming files in next loop
    for k in ${timeints_arr[@]}; do
        let num_processed_volumes+=$k
    done
    #num_completed_volumes=$(($num_processed_volumes-(num_processed_intervals-1)))
done

# Lastly, create a single .mp4 video and .gif based on all interpolated
# png files in 
animate_command="bash scripts/make-animation.sh $png_folder $patient_sequence_output_dir"
echo $animate_command
#
if [ $run_evals == 1 ] ; then
    eval $animate_command
fi
#
