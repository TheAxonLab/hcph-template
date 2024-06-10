#!/bin/bash

#SBATCH --partition cluster2
#SBATCH --job-name=antsMultiMNI          # Job name
#SBATCH --mail-type=END                          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexandre.cionca@gmail.com       # Where to send mail
#SBATCH --ntasks=1                               # Run on a single core
#SBATCH --cpus-per-task=1
#SBATCH --time=0-23:59:00                        # Time limit hrs:min:sec
#SBATCH --output=/data/hagmann_group/acionca/logs/out/%x-%j-%A_%a.out     # Standard output log
#SBATCH --error=/data/hagmann_group/acionca/logs/err/%x-%j-%A_%a.err      # Standard error log

#path_to_data="/data/hagmann_group/yaleman/Alex/multivar-v00/derivatives/ants-t1N4bfcorr-b80-noSkull"
#path_to_data="/data/hagmann_group/acionca/multivar-v00/derivatives/histomatch"
path_to_data="/data/hagmann_group/acionca/multivar-v00/derivatives/allInRef"
cd $path_to_data

first_line=$( head -n 1 templateInput-HPC.csv )
first_t1=$( cut -d "," -f 1 <<< $first_line)
first_t2=$( cut -d "," -f 2 <<< $first_line)

# Setup for building in MNI
path_to_mni="/data/hagmann_group/acionca/mni_template"
mni_template=$path_to_mni"/mni_template-res0.8mm.nii.gz"

#path_to_script="/home/yalemang/matlab/scripts/ClusterScripts"
path_to_script="/home/al1150/scripts"


#bash $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 3 -k 2 -f 6x4x2x1 -s 4x2x1x0vox -q 100x100x70x20 -t Affine -m MI -c 5 -n 0 -o A_tpl_ -z $mni_template -z $mni_template -y 0 -r 1 templateInput-HPC.csv
bash $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 3 -k 2 -f 6x4x2x1 -s 4x2x1x0vox -q 100x100x70x20 -t Affine -m MI -c 5 -n 0 \
	-o A_tpl_ -z $first_t1 -z $first_t2 -y 0 -r 1 templateInput-HPC.csv
