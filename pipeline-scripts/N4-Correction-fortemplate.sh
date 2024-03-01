#!/usr/bin/env bash -l



#########################################################################################################################################
########################################## Main program #################################################################################
#########################################################################################################################################

bids_dir=$1 
out_dir=$bids_dir'/derivatives'
IdFile=$2 

Ia=$(head -"$SLURM_ARRAY_TASK_ID" $IdFile | tail -1);

# echo $bVal  
fullname=$(cut -d "." -f 1 <<< $Ia);

subjId=$(cut -d "_" -f 1 <<< $fullname);
imaMod=$(cut -d "_" -f 2 <<< $fullname);
Id=$subjId'_'$imaMod

lock_dir=$out_dir/'processing-lockfiles'
mkdir -p $lock_dir

#lockfile=$lock_dir/$Ia".lock"
#if [ ! -f $lockfile ]; then
        #touch $lockfile
if [ "" == "" ]; then
        echo "Processing "$Id
        imFile=$bids_dir/$Ia

        ####### Creating the ouput logs
        # 0.Creating the output folder
        logsdir=$out_dir/'pipeline-errors'
        mkdir -p $logsdir

        ####### Running N4 and denoising using ANTs
        antsN4bfDir=$out_dir/'ants-t1N4bfcorr-b80-norm'/
        n4corrt1=$antsN4bfDir/$subjId'_desc-N4corr_'$imaMod'.nii.gz'
	n4corrbf=$antsN4bfDir/$subjId'_desc-N4corr_biasf'$imaMod'.nii.gz'
        n4corrt1den=$antsN4bfDir/$subjId'_desc-N4corrden_'$imaMod'.nii.gz'
	n4corrt1dennorm=$antsN4bfDir/$subjId'_desc-N4corrdennorm_'$imaMod'.nii.gz'

        if [ ! -f $n4corrt1den ]; then
          mkdir -p $antsN4bfDir
          # N4BiasFieldCorrection -d 3 -i $imFile -o $n4corrt1
	  N4BiasFieldCorrection -d 3 -b [80] -i $imFile -o [$n4corrt1,$n4corrbf]
          DenoiseImage -d 3 -i $n4corrt1 -o $n4corrt1den --verbose 1
	  mri_normalize $n4corrt1den $n4corrt1dennorm
          if [ ! -f $n4corrt1den ]; then
            echo "Stage 1: N4 Bias field correction has failed for subject " $Id >> $logsdir/$Id'.pipelineerrors.log'
          fi
        fi
        t1File=$n4corrt1den
fi
