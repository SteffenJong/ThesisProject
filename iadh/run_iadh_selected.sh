#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=15G
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_iadh_selected.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_iadh_selected.err

cd /home/sdejong/thesis/iadh
i-adhore ath_bol_selected.ini