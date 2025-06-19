data_p="/home/jong505/thesis/iadh/data"

cd /home/jong505/thesis/model
python create_training_tsv.py \
    --merged_iadh_tsv /home/jong505/thesis/iadh/iadh_out/aar_ath_bol_chi_cpa_tha/merged_results.tsv \
    --gene_fam /home/jong505/thesis/iadh/data/gene_fam_parsed.tsv \
    --list_elements /home/jong505/thesis/iadh/iadh_out/aar_ath_bol_chi_cpa_tha/list_elements.txt \
    --refseqs "${data_p}/ath.fasta.gz" "${data_p}/aar.fasta.gz" "${data_p}/bol.fasta.gz" "${data_p}/tha.fasta.gz" "${data_p}/chi.fasta.gz" "${data_p}/cpa.fasta.gz" \
    --segment_length 7 \
    --output_prefix data/aar_ath_bol_chi_cpa_tha/sm7_50000_new_neg \
    --output_prefix_seq data/aar_ath_bol_chi_cpa_tha/sm7_50000_new_neg \
    --max_len_nuc 50000
    
