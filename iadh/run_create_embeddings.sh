#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=15G
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.err

cd /home/jong505/thesis/iadh
python create_embeddings.py --train_test iadh_out/ath_bol_aar/simple_train_test_randomandsim.tsv --output iadh_out/ath_bol_aar/emebddings_blocks_24_mlp_l3_sttras_30_A0.tsv