trans_p="/home/jong505/thesis/iadh/data"

cd /home/jong505/thesis/model
python create_medium_train_tsv.py \
    --gene_fam "${trans_p}/gene_fam_parsed.tsv" \
    --refseqs "${trans_p}/annotation.all_transcripts.aar.csv.gz" "${trans_p}/annotation.all_transcripts.ath.csv.gz" "${trans_p}/annotation.all_transcripts.bol.csv.gz" \
            "${trans_p}/annotation.all_transcripts.chi.csv.gz" "${trans_p}/annotation.all_transcripts.cpa.csv.gz" "${trans_p}/annotation.all_transcripts.tha.csv.gz" \
    --output_prefix data/aar_ath_bol_chi_cpa_tha/medium_2g_500_noflip \
    --samplesize_per_organism 500
    