cd /home/jong505/thesis/model
CUDA_VISIBLE_DEVICES=0

# python run_grid_search.py \
#     --prefix data/aar_ath_bol_chi_cpa_tha/medium_2g_500_noflip \
#     --batches 32 64 128 256 \
#     --layers 1 2 3 \
#     --dropout 0.1 0.3 0.5 \
#     --max_epochs 1000 \
#     --device cuda

# python run_grid_search.py \
#     --prefix data/ath_bol_aar/simple_sf_nsnf_500 \
#     --batches 32 64 128 256 \
#     --layers 1 2 3 \
#     --dropout 0.1 0.3 0.5 \
#     --max_epochs 1000 \
#     --device cuda

python run_grid_search.py \
    --prefix data/aar_ath_bol_chi_cpa_tha/sm7_50000 \
    --batches 32 64 128 256 \
    --layers 1 2 3 \
    --dropout 0.1 0.3 0.5 \
    --max_epochs 1000 \
    --device cuda