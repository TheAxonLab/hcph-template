#!/bin/bash

#SBATCH --partition cluster2
#SBATCH --job-name=antsMultivarTemplate          # Job name
#SBATCH --mail-type=END                          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=name.surname@whatever.com       # Where to send mail
#SBATCH --ntasks=1                               # Run on a single core
#SBATCH --cpus-per-task=1
#SBATCH --time=0-23:59:00                        # Time limit hrs:min:sec
#SBATCH --output=/path/to/data/derivatives/out/%x-%j-%A_%a.out     # Standard output log
#SBATCH --error=/path/to/data/derivatives/err/%x-%j-%A_%a.err      # Standard error log

path_to_data="path/to/data/derivatives/ants-t1N4bfcorr-b80-norm/"
cd $path_to_data

path_to_script="/path/to/script"

sh $path_to_script/antsMultivariateTemplateConstruction2-mod.sh -d 3 -b 1 -i 1 -k 2 -f 6x4x2x1 -s 4x2x1x0vox -q 200x100x50x0 -t Affine -m CC -c 5 -r 1 -n 0 -o A_tpl_ templateInput-HPC.csv
