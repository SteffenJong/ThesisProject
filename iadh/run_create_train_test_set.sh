#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=15G
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.err

cd /home/jong505/thesis/iadh
python create_training_tsv.py --merged_iadh_tsv iadh_out/ath_bol_aar/merged_results.tsv \
--refseqs data/ath.fasta.gz data/aar.fasta.gz data/bol.fasta.gz \
--segment_length 7 \
--output iadh_out/ath_bol_aar/train_test2.tsv
