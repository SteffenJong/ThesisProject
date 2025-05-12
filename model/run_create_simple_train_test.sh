trans_p="/home/jong505/thesis/iadh/data"

cd /home/jong505/thesis/model
python Create_simple_train_tsv.py \
    --gene_fam "${trans_p}/gene_fam_parsed.tsv" \
    --refseqs "${trans_p}/annotation.all_transcripts.ath.csv.gz" "${trans_p}/annotation.all_transcripts.bol.csv.gz" "${trans_p}/annotation.all_transcripts.aar.csv.gz" \
    --output_prefix data/ath_bol_aar/simple_sf_nsnf_500 \
    --output_prefix_raw data/ath_bol_aar/simple_sf_nsnf_500 \
    --samplesize_per_organism 500
    