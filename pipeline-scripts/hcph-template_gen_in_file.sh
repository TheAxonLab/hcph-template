#!/usr/bin/env bash -l

#########################################################################################################################################
########################################## Main program #################################################################################
#########################################################################################################################################
# How to run:
# sh pipeline-scripts/hcph-template_gen_in_file.sh ~/data/hcph-template/multivar-v00/derivatives/allInRef-noskull-i4

bids_dir=$1
filename=$2

if [ "$filename" == "" ]
then
    files_list=$( ls $bids_dir/*N4corrdenhist*.nii.gz )
else
    files_list=$( cat $bids_dir/$filename )
fi

savename="templateInput-HPC.csv"

save_loc=$bids_dir

echo "Creating file ${savename}"
printf "" > "${save_loc}/${savename}"

prefix_char=""

for file in $files_list
do
    fname=$( basename $file )
    modality=$( echo $fname | grep -o -E 'T[0-9]w' )

    if [ "$modality" == "T1w" ]
    then
        printf "${prefix_char}%s," $fname >> "${save_loc}/templateInput-HPC.csv"
    elif [ "$modality" == "T2w" ]
    then
        printf "$fname" >> "${save_loc}/templateInput-HPC.csv"
    else
        echo "Modality not recognized for ${fname} !"
    fi
    prefix_char="\n"
done

printf "\n" >> "${save_loc}/${savename}"
