#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=15G
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.err

cd /home/jong505/thesis/model
python create_embeddings.py \
    --dataframes "data/aar_ath_bol_chi_cpa_tha/sm7_50000_test_seq.tsv" "data/aar_ath_bol_chi_cpa_tha/sm7_50000_train_seq.tsv" "data/aar_ath_bol_chi_cpa_tha/sm7_50000_val_seq.tsv" \
    --output_prefix data/aar_ath_bol_chi_cpa_tha/sm7_50000_div3_embeddings.tsv \
    --output_prefix_raw data/aar_ath_bol_chi_cpa_tha/sm7_50000_div3