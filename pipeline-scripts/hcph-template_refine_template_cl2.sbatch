#!/bin/bash

#SBATCH --partition cluster2
#SBATCH --job-name=antsRefine          # Job name
#SBATCH --mail-type=END                          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexandre.cionca@gmail.com       # Where to send mail
#SBATCH --ntasks=1                               # Run on a single core
#SBATCH --cpus-per-task=1
#SBATCH --time=0-23:59:00                        # Time limit hrs:min:sec
#SBATCH --output=/data/hagmann_group/acionca/logs/out/%x-%j-%A_%a.out     # Standard output log
#SBATCH --error=/data/hagmann_group/acionca/logs/err/%x-%j-%A_%a.err      # Standard error log

#path_to_data="/data/hagmann_group/yaleman/Alex/multivar-v00/derivatives/histomatch/"
path_to_data="/data/hagmann_group/acionca/multivar-v00/derivatives/histomatch/"
cd $path_to_data

#existing_template="A_tpl_template0.nii.gz"
#initial_template="Intial_affine_template.nii.gz"

path_to_reference="/data/hagmann_group/acionca/multivar-v00/derivatives/histomatch"
t1w_ref=$path_to_reference"/A_tpl_template0.nii.gz"
t2w_ref=$path_to_reference"/A_tpl_template1.nii.gz"

#cp $existing_template $initial_template

#path_to_script="/home/yalemang/matlab/scripts/ClusterScripts"
path_to_script="/home/al1150/scripts"

bash $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 2 -k 2 -f 6x4x2x1 -s 3x2x1x0vox -q 100x100x70x20 -t Affine -m MI -c 5 -r 0 -n 0 -o A_tpl_ -z $t1w_ref -z $t2w_ref templateInput-HPC.csv
