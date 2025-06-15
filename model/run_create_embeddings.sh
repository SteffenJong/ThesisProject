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
    --dataframes "data/aar_ath_bol_chi_cpa_tha/medium_2g_500_test.tsv" "data/aar_ath_bol_chi_cpa_tha/medium_2g_500_train.tsv" "data/aar_ath_bol_chi_cpa_tha/medium_2g_500_val.tsv" \
    --output data/aar_ath_bol_chi_cpa_tha/medium_2g_500_pca_embeddings.tsv 