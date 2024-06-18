#!/bin/bash

#SBATCH --partition cluster2
#SBATCH --job-name=antsReg-T1w                             # Job name
#SBATCH --mail-type=END                                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexandre.cionca@gmail.com                        # Where to send mail
#SBATCH --ntasks=1                                              # Run on a single core
#SBATCH --cpus-per-task=1
#SBATCH --array=1-38%38
#SBATCH --time=0-00:10:00                                         # Time limit hrs:min:sec
#SBATCH --output=/data/hagmann_group/acionca/logs/out/%x-%j-%A_%a.out      # Standard output log
#SBATCH --error=/data/hagmann_group/acionca/logs/err/%x-%j-%A_%a.err      # Standard error log
sh align_with_t1w-SLURM.sh /data/hagmann_group/acionca/multivar-v00/derivatives/ants-t1N4bfcorr-ss-histomatch /data/hagmann_group/acionca/multivar-v00/derivatives/ants-t1N4bfcorr-ss-histomatch/allImages.txt
