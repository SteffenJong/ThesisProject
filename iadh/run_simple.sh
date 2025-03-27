#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/simple.out
#SBATCH --error=/home/sdejong/thesis/iadh/simple.err

cd /home/sdejong/thesis/iadh
python anno_to_genelist.py --input annotation.all_transcripts.ath.tsv --output_dir ath_selected