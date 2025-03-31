#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_preprocess-%j.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_preprocess-%j.err

cd /home/sdejong/thesis/iadh
python parse_gene_family.py --input genefamily_data.HOMFAM.tsv --output gene_fam_parsed.tsv
python anno_to_genelist.py --input annotation.all_transcripts.ath.tsv --output_dir ath --remove_more True
python anno_to_genelist.py --input annotation.all_transcripts.bol.tsv --output_dir bol 
python add_singelton_gene_fam.py --gene_list_dir ath
python add_singelton_gene_fam.py --gene_list_dir bol