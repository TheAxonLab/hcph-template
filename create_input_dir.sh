#!/bin/bash

bids_dir="/data/datasets/hcph-dataset"
sub="sub-001"
sub_dir="${bids_dir}/${sub}"

files_list=$( ls $sub_dir/ses-0*/anat/*undistorted*.nii.gz )

save_dir="/home/acionca/Documents/data/hcph-template"
template_name="multivar-v00"
save_loc="${save_dir}/${template_name}"

mkdir -p $save_loc
echo "Emptying existing file"
printf "" > "${save_loc}/templateInput.csv"

prefix_char=""

for file in $files_list
do
    fname=$( basename $file )
    modality=$( echo $fname | grep -o -E 'T[0-9]w' )
    ses_id=$( echo $fname | grep -o -E 'ses-[0-9]+' | grep -o -E '[0-9]+' )
    run_id=$( echo $fname | grep -o -E 'run-[0-9]' | grep -o -E '[0-9]' )
    file_rename="sub-${ses_id}_${modality}.nii.gz"

    if [ "$run_id" != "" ] && [ "$run_id" != "1" ]
    then
        echo "Not converting multiple runs for ${fname} !"
    else
        echo "${file} --> ${save_loc}/${file_rename}"
        ln -s $file ${save_loc}/${file_rename}
        if [ "$modality" == "T1w" ]
        then
            printf "${prefix_char}%s," $file_rename >> "${save_loc}/templateInput.csv"
        elif [ "$modality" == "T2w" ]
        then
            printf "$file_rename" >> "${save_loc}/templateInput.csv"
        fi
    fi
    prefix_char="\n"
done