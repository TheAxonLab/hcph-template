#!/usr/bin/env bash -l



#########################################################################################################################################
########################################## Main program #################################################################################
#########################################################################################################################################
# How to run:
# sh interpolation/pipeline-scripts/template_to_reference.sh ~/data/hcph-template/multivar-v00/derivatives/diswe-interp/my_template.nii.gz ~/path_to_mni/mni_template.nii.gz MNI152NLin2009cAsym Affine
# 
# Will create:
# my_template_space-MNI152NLin2009cAsymAffine.nii.gz
# and
# transform_space-MNI152NLin2009cAsymAffine.mat


moving_img=$1
fixed_img=$2
space_name=$3

if [ $4 == "" ]
then
    transform_type="Affine"
else
    transform_type=$4
fi

savedir=$( dirname $moving_img )

####### Defining the output files

moving_no_ext=$( basename $moving_img)
savename="${moving_no_ext/.nii.gz/}"

transform=$savedir/${savename}_to_${space_name}${transform_type}
imgInRef=$savedir/${savename}_MovedTo${space_name}${transform_type}.nii.gz

antsRegistration -d 3 -v -m MI[$fixed_img, $moving_img, 1,32, Regular, 0.1] -t $transform_type[0.25] \
    -c [10000x10000x10000, 1e-4, 10] -f 4x2x1 -s 0.6x0.2x0mm --float 1 -o [$transform, $imgInRef]

#antsRegistration -d 3 -v -m MI[$fixed_img, $moving_img, 1,32, Regular, 0.1] -t $transform_type[0.25] \
#    -c [10000x10000x10000, 1e-4, 10] -f 4x2x1 -s 0.6x0.2x0mm --float 1 -o [$transform, $imgInRef]

#antsRegistration -d 3 -v -m MI[$fixed_img, $moving_img, 1,32, Regular, 0.1] -t $transform_type[0.25] \
#    -c [10000x10000x10000, 1e-4, 10] -f 4x2x1 -s 0.6x0.2x0mm --float 1 -o [$transform, $imgInRef]