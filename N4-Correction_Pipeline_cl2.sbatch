#!/usr/bin/env bash -l

#SBATCH --partition cluster2
#SBATCH --job-name=n4correct                             # Job name
#SBATCH --mail-type=END                                         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=name.surname@whatever.com                        # Where to send mail
#SBATCH --ntasks=1                                              # Run on a single core
#SBATCH --cpus-per-task=1
#SBATCH --array=1-70%70
#SBATCH --time=0-02:00:00                                         # Time limit hrs:min:sec
#SBATCH --output=/path/to/data/derivatives/out/%x-%j-%A_%a.out     # Standard output log
#SBATCH --error=/path/to/data/derivatives/err/%x-%j-%A_%a.err      # Standard error log
sh N4-Correction-fortemplate.sh /path/to/data /path/to/data/allImages.txt
