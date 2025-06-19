cd /home/jong505/thesis/model
CUDA_VISIBLE_DEVICES=0

# python run_grid_search.py \
#     --prefix data/ath_bol_aar/simple_sf_nsnf_500 \
#     --batches 32 64 128 256 \
#     --layers 1 2 3 \
#     --dropout 0.1 0.3 0.5 \
#     --max_epochs 1000 \
#     --datal_type old \
#     --device cuda

python run_grid_search.py \
    --prefix data/ath_bol_aar/simple_sf_nsnf_500 \
    --batches 256 512 \
    --layers 1 \
    --dropout 0.3 \
    --max_epochs 1000 \
    --datal_type old \
    --hidden_size 128 256 512 1024 \
    --result_file gridsearch2_results \
    --device cuda

# python run_grid_search.py \
#     --prefix data/ath_bol_aar/simple_sf_nsnf_500 \
#     --batches 256 \
#     --layers 1 \
#     --dropout 0.3 \
#     --max_epochs 1000 \
#     --datal_type old \
#     --hidden_size 128 \
#     --result_file gridsearch2_results \
#     --device cuda

# python run_grid_search.py \
#     --prefix data/aar_ath_bol_chi_cpa_tha/medium_2g_500 \
#     --batches 32 64 128 256 \
#     --layers 1 2 3 \
#     --dropout 0.1 0.3 0.5 \
#     --max_epochs 1000 \
#     --input_siz 3840 \
#     --em_type avg \
#     --device cuda

# python run_grid_search.py \
#     --prefix data/aar_ath_bol_chi_cpa_tha/medium_2g_500_noflip \
#     --batches 32 64 128 256 \
#     --layers 1 2 3 \
#     --dropout 0.1 0.3 0.5 \
#     --max_epochs 1000 \
#     --input_siz 3840 \
#     --em_type avg \
#     --device cuda

# python run_grid_search.py \
#     --prefix data/aar_ath_bol_chi_cpa_tha/medium_2g_500 \
#     --batches 32 64 128 256 \
#     --layers 1 2 3 \
#     --dropout 0.1 0.3 0.5 \
#     --max_epochs 1000 \
#     --input_siz 11520 \
#     --em_type div3 \
#     --device cuda

python run_grid_search.py \
    --prefix data/aar_ath_bol_chi_cpa_tha/sm7_50000_new_neg \
    --batches 32 64 128 256 \
    --layers 1 2 3 \
    --dropout 0.1 0.3 0.5 \
    --max_epochs 1000 \
    --input_siz 3840 \
    --em_type avg \
    --device cuda    