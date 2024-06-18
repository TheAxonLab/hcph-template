#!/usr/bin/env bash -l

#########################################################################################################################################
########################################## Main program #################################################################################
#########################################################################################################################################
# How to run:
# sh interpolation/pipeline-scripts/align_with_t1w-local.sh ~/data/hcph-template/multivar-v00/derivatives/histomatch-noskull ~/data/hcph-template/multivar-v00/derivatives/histomatch-noskull/allImages.txt

bids_dir=$1 

# Specify output dir below !
out_dir=$bids_dir/../allInRef-noskull
mkdir -p $out_dir

IdFile=$2

Ia=$(head $IdFile | tail -1);

while read Ia
do
    fullname=$(cut -d "." -f 1 <<< $Ia);

    subjId=$(cut -d "_" -f 1 <<< $fullname);
    sesID=$(cut -d "_" -f 2 <<< $fullname);
    # imaMod=$(cut -d "_" -f 2 <<< $fullname);
    imaMod=$( echo $fullname | rev | cut -d "_" -f 1 | rev )
    Id=$subjId'_'$sesID'_'$imaMod

    echo "Processing "$Id
    imFile=$bids_dir/$Ia

    ####### Defining the output files
    imgInRef=$out_dir/$subjId'_'$sesID'_space-inRef_desc-N4corrdenhist_'$imaMod'.nii.gz'
    transform=$out_dir/$subjId'_'$sesID'_'$imaMod'-to-T1w_'
    #target_img="$bids_dir/${Ia/$imaMod/"T1w"}"
    target_img="$bids_dir/${Ia/$imaMod/T1w}"
    target_rename="${imgInRef/$imaMod/T1w}"
    
    echo Creating symlink at: $target_rename
    ln -s $target_img $target_rename

    if [ ! -f $imgInRef ]; then
        antsRegistration -d 3 -v -m MI[$target_img, $imFile, 1,32, Regular, 0.1] -t Affine[0.25] \
            -c [10000x10000x10000, 1e-4, 10] -f 4x2x1 -s 0.6x0.2x0mm --float 1 -o [$transform, $imgInRef]
        
        if [ ! -f $imgInRef ]; then
            echo "Registration has failed for session " $Id
        fi
    else
        echo "Target file is already there. Skipping registration step"
    fi
done < $IdFile
