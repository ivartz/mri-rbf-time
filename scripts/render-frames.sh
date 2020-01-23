# Run
# bash render-frames.sh <nifti path> <png save path>

#readarray niftis < <(ls ../data/output/elies_1_7_each_fourth/nii/*.nii)
readarray niftis < <(ls $1/*.nii)
#save_path="../data/output/elies_1_7_each_fourth/png"
save_path=$2

# Make directory if not exists
[ -d $save_path ] || mkdir $save_path

make_png() {
    nifti=$1
    save_path=$2
    png_file_name=$(basename -- ${nifti%.nii})
    png_file=$save_path/$png_file_name
    fsleyes render --outfile $png_file --scene ortho --worldLoc 20.61959330240886 36.27085673014321 11.2251215764921 --displaySpace $nifti --xcentre  0.01661 -0.01234 --ycentre -0.07621 -0.02504 --zcentre -0.03193  0.07084 --xzoom 900.0 --yzoom 1200.0 --zzoom 850.0 --hideLabels --labelSize 14 --layout horizontal --hideCursor --bgColour 0.0 0.0 0.0 --fgColour 1.0 1.0 1.0 --cursorColour 0.0 1.0 0.0 --showColourBar --colourBarLocation top --colourBarLabelSide top-left --performance 3 $nifti --name $png_file_name --overlayType volume --alpha 100.0 --brightness 49.999999999993584 --contrast 50.0 --cmap greyscale --negativeCmap greyscale --displayRange 0.0 255.0 --clippingRange 0.0 265.0 --gamma 0.0 --cmapResolution 256 --interpolation none --numSteps 100 --blendFactor 0.1 --smoothing 0 --resolution 100 --numInnerSteps 10 --clipMode intersection --volume 0
}

export -f make_png

printf "%s\n" "${niftis[@]}" | xargs -I nifti -n 1 -P $(nproc) bash -c 'make_png "$@"' _ nifti $save_path

