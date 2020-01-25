# Run
# bash rename-files.sh <nifti path> <starting number>
# ex:
# bash scripts/rename-files.sh data/output/elies_7_14_each_fourth/niicopy 53
readarray niftis < <(ls $1/*.nii)
starting_number=$2

# Reverse iterate over files
# since file names are going
# to be renamed by adding an
# integer to the file number.
for (( idx=${#niftis[@]}-1 ; idx>=0 ; idx-- )) ; do
    # Get file
    nifti=${niftis[idx]}
    # Get file name
    png_file_name=$(basename $nifti)
    # Remove file suffix
    png_file_name=${png_file_name%.nii}
    # Remove 0 prefix two times if possible
    png_file_name=${png_file_name#0}
    png_file_name=${png_file_name#0}
    # Create new file name
    png_new_file_name=$(($png_file_name + $starting_number - 1))
    # zero pad new integer file name to have fixed length of 3 digits
    printf -v png_new_file_name "%03d" $png_new_file_name
    # Create new file
    png_new_file=$(dirname ${nifti%.nii})/$png_new_file_name.nii
    # Rename old file into new file
    mv -v $nifti $png_new_file
done
