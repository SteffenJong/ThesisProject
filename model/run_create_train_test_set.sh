data_p="/home/jong505/thesis/iadh/data"

cd /home/jong505/thesis/model
python create_training_tsv.py \
    --merged_iadh_tsv /home/jong505/thesis/iadh/iadh_out/ath_bol_aar/merged_results1.tsv \
    --refseqs "${data_p}/ath.fasta.gz" "${data_p}/aar.fasta.gz" "${data_p}/bol.fasta.gz" \
    --segment_length 4 \
    --output_prefix data/ath_bol_aar/sm4 \
    --output_prefix_raw data/ath_bol_aar/sm4
    
