# Run
# bash make-animation.sh <png path> <animation save path>

#png_path="../data/output/elies_1_7_each_fourth/png"
png_path=$1
save_path=$2

# Make directory if not exists
[ -d $save_path ] || mkdir $save_path

ffmpeg -framerate "8" -pattern_type glob -i "$png_path/*.png" -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -preset slow -profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 1 $png_path/video.mp4

ffmpeg -i $png_path/video.mp4 -vf palettegen $png_path/palette.png

ffmpeg -i $png_path/video.mp4 -i $png_path/palette.png -filter_complex "scale=-1:-1:flags=lanczos[x];[x][1:v]paletteuse" $png_path/animation.gif

rm $png_path/palette.png

# Move results to save_path
mv $png_path/video.mp4 $save_path
mv $png_path/animation.gif $save_path
