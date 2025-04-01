#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --account=cseduproject
#SBATCH --output=/home/sdejong/thesis/iadh/logs/run_preprocess-%j.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/run_preprocess-%j.err

cd /home/jong505/thesis/iadh
python parse_gene_family.py --input data/genefamily_data.HOMFAM.csv.gz --output data/gene_fam_parsed.tsv
python anno_to_genelist.py --input data/annotation.all_transcripts.ath.csv.gz --output_dir data/ath --remove_more True
python anno_to_genelist.py --input data/annotation.all_transcripts.bol.csv.gz --output_dir data/bol --remove_more True
python add_singelton_gene_fam.py --gene_list_dir ath --gene_fam data/gene_fam_parsed.tsv
python add_singelton_gene_fam.py --gene_list_dir bol --gene_fam data/gene_fam_parsed.tsv