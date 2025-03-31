#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=15G
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.err

source /home/sdejong/.bashrc
source activate jp
cd /home/sdejong/thesis/iadh
python collect_results_iadh.py --iadhdir ath_bol \
--annofiles annotation.all_transcripts.ath.tsv annotation.all_transcripts.bol.tsv \
--output IADH_results.tsv