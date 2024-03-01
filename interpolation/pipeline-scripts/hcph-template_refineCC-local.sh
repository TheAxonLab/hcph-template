#!/bin/bash

#path_to_data="/data/hagmann_group/acionca/multivar-v00/derivatives/allInRef"
path_to_data="/Users/acionca/data/hcph-template/multivar-v00/derivatives/refine01"
cd $path_to_data

first_line=$( head -n 1 templateInput-HPC.csv )
first_t1="A_tpl_template0.nii.gz"
first_t2="A_tpl_template1.nii.gz"

# Setup for building in MNI
# path_to_mni="/data/hagmann_group/acionca/mni_template"
# mni_template=$path_to_mni"/mni_template-res0.8mm.nii.gz"

#path_to_script="/home/yalemang/matlab/scripts/ClusterScripts"
#path_to_script="/home/al1150/scripts"
path_to_script="/Users/acionca/code/hcph-template/interpolation/pipeline-scripts"


#bash $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 3 -k 2 -f 6x4x2x1 -s 4x2x1x0vox -q 100x100x70x20 -t Affine -m MI -c 5 -n 0 -o A_tpl_ -z $mni_template -z $mni_template -y 0 -r 1 templateInput-HPC.csv

#bash $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 3 -k 2 -f 6x4x2x1 -s 4x2x1x0vox -q 100x100x70x20 -t Affine -m MI -c 5 -n 0 \
#	-o A_tpl_ -z $first_t1 -z $first_t2 -y 0 -r 1 templateInput-HPC.csv

bash $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 1 -k 2 -f 6x4x2x1 -s 4x2x1x0vox -q 50x100x70x20 -t Affine -m CC -c 0 -n 0 \
	-o Refined_ -z $first_t1 -z $first_t2 -y 0 -r 0 templateInput-HPC.csv
