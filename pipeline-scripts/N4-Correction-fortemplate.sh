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
sesID=$(cut -d "_" -f 2 <<< $fullname);
# imaMod=$(cut -d "_" -f 2 <<< $fullname);
imaMod=$( echo $fullname | rev | cut -d "_" -f 1 | rev )
Id=$subjId'_'$sesID'_'$imaMod

lock_dir=$out_dir/'processing-lockfiles'
#mkdir -p $lock_dir

echo "Running"

#lockfile=$lock_dir/$Ia".lock"
#if [ ! -f $lockfile ]; then
#        touch $lockfile
if [ 1 ]; then
	echo "Processing "$Id
        imFile=$bids_dir/$Ia

        ####### Creating the ouput logs
        # 0.Creating the output folder
        logsdir=$out_dir/'pipeline-errors'
        mkdir -p $logsdir

        ####### Running N4 and denoising using ANTs
        antsN4bfDir=$out_dir/'ants-t1N4bfcorr-ss60'/
	#tmpDir=$antsN4bfDir/'N4pipeline'

	padded=$antsN4bfDir/$subjId'_'$sesID'_desc-padded_'$imaMod'.nii.gz'
	skullStripped=$antsN4bfDir/$subjId'_'$sesID'_desc-noSkull_'$imaMod'.nii.gz'
	brainMask=$antsN4bfDir/$subjId'_'$sesID'_desc-brainmask_'$imaMod'.nii.gz'
        n4corrt1=$antsN4bfDir/$subjId'_'$sesID'_desc-N4corr_'$imaMod'.nii.gz'
	n4corrbf=$antsN4bfDir/$subjId'_'$sesID'_desc-N4corr_biasf'$imaMod'.nii.gz'
	n4corrt1den=$antsN4bfDir/$subjId'_'$sesID'_desc-N4corrden_'$imaMod'.nii.gz'
	#n4corrt1dennorm=$antsN4bfDir/$subjId'_'$sesID'_desc-N4corrdennorm_'$imaMod'.nii.gz'

	echo $skullStripped

        if [ ! -f $n4corrt1den ]; then
	  mkdir -p $antsN4bfDir
          # N4BiasFieldCorrection -d 3 -i $imFile -o $n4corrt1
	  # N4BiasFieldCorrection -d 3 -b [80] -i $skullStripped -o [$n4corrt1,$n4corrbf]
	  # mri_watershed $imFile $skullStripped
	  ImageMath 3 $padded PadImage $imFile 25
	  mri_synthstrip -i $padded -o $skullStripped -m $brainMask
	  N4BiasFieldCorrection -d 3 -b [60] -i $padded -o [$n4corrt1,$n4corrbf] -w $brainMask --verbose 1
	  DenoiseImage -d 3 -i $n4corrt1 -o $n4corrt1den --verbose 1
	  # mri_normalize -gentle $n4corrt1den $n4corrt1dennorm
          
	  if [ ! -f $n4corrt1den ]; then
            echo "Stage 1: N4 Bias field correction has failed for subject " $Id >> $logsdir/$Id'.pipelineerrors.log'
          fi
        fi
fi
