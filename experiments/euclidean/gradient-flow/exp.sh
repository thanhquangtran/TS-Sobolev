#!/bin/bash

# Change to workspace root directory
cd "$(dirname "$0")/../.." || exit

# Ablation study: TSW (p=1) and Sobolev (p=1.2, 1.5, 2)
# For each: chain vs concurrent, uniform vs distance-based
# Total: 4 TSW settings + 12 Sobolev settings = 16 settings
# All settings run automatically in one command

# Run both experiments in parallel on different GPUs
# For 8gaussians dataset on GPU 4
CUDA_VISIBLE_DEVICES=4 python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_tsw_sl 0.005 \
    --delta 1 \
    --dataset_name "8gaussians" --std 0.001 --num_seeds 5 &

# For higher dimensional dataset on GPU 5
CUDA_VISIBLE_DEVICES=5 python experiments/gradient-flow/GradientFlow.py \
    --num_iter 2500 --L 100 --n_lines 4 --lr_tsw_sl 0.05 \
    --delta 1 \
    --dataset_name "gaussian_30d_small_v" --std 0.001 --num_seeds 5 &

# Wait for both processes to complete
wait
