#!/bin/bash

#SBATCH --partition cluster2
#SBATCH --job-name=antsMultiTpl          # Job name
#SBATCH --mail-type=END                          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexandre.cionca@gmail.com       # Where to send mail
#SBATCH --ntasks=1                               # Run on a single core
#SBATCH --cpus-per-task=1
#SBATCH --time=0-23:59:00                        # Time limit hrs:min:sec
#SBATCH --output=/data/hagmann_group/yaleman/Alex/derivatives/out/%x-%j-%A_%a.out     # Standard output log
#SBATCH --error=/data/hagmann_group/yaleman/Alex/derivatives/err/%x-%j-%A_%a.err      # Standard error log

path_to_data="/data/hagmann_group/yaleman/Alex/multivar-v00/derivatives/ants-t1N4bfcorr-b80-noSkull"
cd $path_to_data

path_to_script="/home/yalemang/matlab/scripts/ClusterScripts"

bash $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 4 -k 2 -f 6x4x2x1 -s 4x2x1x0vox -q 100x100x70x20 -t Affine -m MI -c 5 -r 1 -n 0 -o A_tpl_ templateInput-HPC.csv
