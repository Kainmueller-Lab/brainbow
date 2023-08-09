#!/bin/bash

root=/nrs/saalfeld/maisl/flylight_benchmark/brainbow
in_folder=/nrs/saalfeld/maisl/data/flylight/flylight_complete/fold1
out_folder=$root/denoised/psd_0_05
sigma_psd=0.05
clip_max=1500

log_dir=$root/log
mkdir $log_dir
mkdir $out_folder

samples=($(ls $in_folder ))

for sample in "${samples[@]}";
do
    sn="$(basename $sample)"
    sn="${sn%.*}"
    in_file=$in_folder/$sample
    echo $in_file
    log_file=$log_dir/${sn}_denoise.out
    echo $log_file
    bsub -n 16 -W 12:00 -o $log_file python denoise.py --in_file $in_file --out_folder $out_folder --sigma_psd $sigma_psd --clip_max 1500
done
