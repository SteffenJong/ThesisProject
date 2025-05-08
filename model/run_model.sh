cd /home/jong505/thesis/model
python run_model.py \
    --train data/ath_bol_aar/simple_blocks_24_mlp_l3_rsrns_same_length_axis1_test.tsv  \
    --test data/ath_bol_aar/simple_blocks_24_mlp_l3_rsrns_same_length_axis1_train.tsv  \
    --val data/ath_bol_aar/simple_blocks_24_mlp_l3_rsrns_same_length_axis1_val.tsv  \
    --model simple \
    --epochs 2 \
    --batch_size 10 \
    --output_prefix results/testing/simple30_ \
    --validation True