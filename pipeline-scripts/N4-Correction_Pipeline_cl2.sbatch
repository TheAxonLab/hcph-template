#!/bin/bash

#SBATCH --partition cluster
#SBATCH --job-name=n4ss-4i                             # Job name
#SBATCH --mail-type=END                                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexandre.cionca@gmail.com                        # Where to send mail
#SBATCH --ntasks=1                                              # Run on a single core
#SBATCH --cpus-per-task=1
#SBATCH --array=1-76%76
#SBATCH --time=0-01:00:00                                         # Time limit hrs:min:sec
#SBATCH --output=/data/hagmann_group/acionca/logs/out/%x-%j-%A_%a.out      # Standard output log
#SBATCH --error=/data/hagmann_group/acionca/logs/err/%x-%j-%A_%a.err      # Standard error log
sh N4-Correction-fortemplate.sh /data/hagmann_group/acionca/multivar-v00 /data/hagmann_group/acionca/multivar-v00/allImages.txt
