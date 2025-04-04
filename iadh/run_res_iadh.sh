#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=15G
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_res_iadh-%j.err

cd /home/jong505/thesis/iadh
python collect_results_iadh.py --iadhdir iadh_out/ath_bol_aar \
--annofiles data/annotation.all_transcripts.ath.csv.gz data/annotation.all_transcripts.bol.csv.gz data/annotation.all_transcripts.aar.csv.gz \
--output iadh_out/ath_bol_aar/merged_results.tsv