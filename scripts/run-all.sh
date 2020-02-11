
patient_sequence_output_dir=\
/media/ivar/SSD700GB/gitprojects/Longitudinal_Study/data/output/sailor-patient/FLAIR-gauss-2

[ -d $patient_sequence_output_dir ] || mkdir -p $patient_sequence_output_dir

rm $patient_sequence_output_dir/runlog.txt

command="bash scripts/run-tp.sh \
    1 \
    ../Elies-longitudinal-data-test \
    ../Elies-longitudinal-data-test/timeints_mod.txt \
    $patient_sequence_output_dir \
    0 \
    4 \
    1 \
    1 \
    2>&1 | tee $patient_sequence_output_dir/runlog.txt"
    
eval $command

#../Elies-longitudinal-data-test/flairbinmask.nii
