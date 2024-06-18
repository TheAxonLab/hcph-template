#!/usr/bin/env bash -l

#########################################################################################################################################
########################################## Main program #################################################################################
#########################################################################################################################################
# How to run:
# sh pipeline-scripts/N4-correction-local.sh ~/data/hcph-template/multivar-v00 ~/data/hcph-template/multivar-v00/allImages.txt

bids_dir=$1 
out_dir=$bids_dir'/derivatives'
IdFile=$2 

Ia=$(head $IdFile | tail -1);

echo "Running"
while read Ia
do
	fullname=$(cut -d "." -f 1 <<< $Ia);

	subjId=$(cut -d "_" -f 1 <<< $fullname);
	sesID=$(cut -d "_" -f 2 <<< $fullname);

	imaMod=$( echo $fullname | rev | cut -d "_" -f 1 | rev )
	Id=$subjId'_'$sesID'_'$imaMod

	echo "Processing "$Id
	imFile=$bids_dir/$Ia

	####### Running N4 and denoising using ANTs
	antsN4bfDir=$out_dir/'ants-t1N4bfcorr-ss60'/

	padded=$antsN4bfDir/$subjId'_'$sesID'_desc-padded_'$imaMod'.nii.gz'
	skullStripped=$antsN4bfDir/$subjId'_'$sesID'_desc-noSkull_'$imaMod'.nii.gz'
	brainMask=$antsN4bfDir/$subjId'_'$sesID'_desc-brainmask_'$imaMod'.nii.gz'
	n4corrt1=$antsN4bfDir/$subjId'_'$sesID'_desc-N4corr_'$imaMod'.nii.gz'
	n4corrbf=$antsN4bfDir/$subjId'_'$sesID'_desc-N4corr_biasf'$imaMod'.nii.gz'
	n4corrt1den=$antsN4bfDir/$subjId'_'$sesID'_desc-N4corrden_'$imaMod'.nii.gz'

	if [ ! -f $n4corrt1den ]; then
		mkdir -p $antsN4bfDir
		
		ImageMath 3 $padded PadImage $imFile 25
		mri_synthstrip -i $padded -o $skullStripped -m $brainMask
		N4BiasFieldCorrection -d 3 -b [60] -i $padded -o [$n4corrt1,$n4corrbf] -w $brainMask --verbose 1
		DenoiseImage -d 3 -i $n4corrt1 -o $n4corrt1den --verbose 1

		if [ ! -f $n4corrt1den ]; then
			echo "Stage 1: N4 Bias field correction has failed for subject " $Id
		fi
	fi
done < $IdFile
