#!/usr/bin/env bash -l



#########################################################################################################################################
########################################## Main program #################################################################################
#########################################################################################################################################

bids_dir=$1 
out_dir=$bids_dir/../allInRef
IdFile=$2 

Ia=$(head -"$SLURM_ARRAY_TASK_ID" $IdFile | tail -1);

# echo $bVal
fullname=$(cut -d "." -f 1 <<< $Ia);

subjId=$(cut -d "_" -f 1 <<< $fullname);
sesID=$(cut -d "_" -f 2 <<< $fullname);
# imaMod=$(cut -d "_" -f 2 <<< $fullname);
imaMod=$( echo $fullname | rev | cut -d "_" -f 1 | rev )
Id=$subjId'_'$sesID'_'$imaMod

#lock_dir=$out_dir/'processing-lockfiles'
#mkdir -p $lock_dir

#lockfile=$lock_dir/$Ia".lock"
#if [ ! -f $lockfile ]; then
        #touch $lockfile
if [ "" == "" ]; then
        echo "Processing "$Id
        imFile=$bids_dir/$Ia

        ####### Creating the ouput logs
        # 0.Creating the output folder
        #logsdir=$out_dir/'pipeline-errors'
        #mkdir -p $logsdir

        ####### Running N4 and denoising using ANTs
        imgInRef=$out_dir/$subjId'_'$sesID'_space-inRef_desc-N4corrdenhist_'$imaMod'.nii.gz'
	transform=$out_dir/$subjId'_'$sesID'_'$imaMod'-to-T1w_'
        #target_img="$bids_dir/${Ia/$imaMod/"T1w"}"
	target_img="$bids_dir/${Ia/$imaMod/T1w}"
	target_rename="${imgInRef/$imaMod/T1w}"

	echo Creating symlink at: $target_rename
	ln -s $target_img $target_rename
	

        if [ ! -f $imgInRef ]; then
            # mkdir -p $antsN4bfDir
	    mkdir -p $out_dir
            antsRegistration -d 3 -v -m MI[$target_img, $imFile, 1,32, Regular, 0.1] -t Affine[0.25] \
                -c [10000x10000x10000, 1e-4, 10] -f 4x2x1 -s 0.6x0.2x0mm --float 1 -o [$transform, $imgInRef]
            
            if [ ! -f $imgInRef ]; then
                echo "Stage 1: N4 Bias field correction has failed for subject " $Id >> $logsdir/$Id'.pipelineerrors.log'
            fi
        fi
fi
