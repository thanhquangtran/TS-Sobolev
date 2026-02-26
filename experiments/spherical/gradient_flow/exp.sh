#!/bin/bash

# Run lssot experiment (s3w already completed)

# Previous experiments (commented out)
# CUDA_VISIBLE_DEVICES=0 python3 main.py -d ssw &
# CUDA_VISIBLE_DEVICES=1 python3 main.py -d s3w &
# CUDA_VISIBLE_DEVICES=2 python3 main.py -d ri_s3w_1 &
# CUDA_VISIBLE_DEVICES=3 python3 main.py -d ri_s3w_5 &
# CUDA_VISIBLE_DEVICES=4 python3 main.py -d ari_s3w &
# CUDA_VISIBLE_DEVICES=5 python3 main.py -d stsw --p 1 --delta 50 &
# CUDA_VISIBLE_DEVICES=6 python3 main.py -d sbsts --p 1.5 --delta 1 --lr 0.05 &
# CUDA_VISIBLE_DEVICES=7 python3 main.py -d sbsts --p 2 --delta 1 --lr 0.05 &
# wait

# Run lssot ablation with different learning rates
# Learning rate ablation (fair comparison: num_projections=1000 matches other baselines)
CUDA_VISIBLE_DEVICES=0 python3 main.py -d lssot --seed 0 --lr 0.005 &
CUDA_VISIBLE_DEVICES=1 python3 main.py -d lssot --seed 0 --lr 0.01 &
CUDA_VISIBLE_DEVICES=2 python3 main.py -d lssot --seed 0 --lr 0.02 &
CUDA_VISIBLE_DEVICES=3 python3 main.py -d lssot --seed 0 --lr 0.05 &

# Wait for all processes to complete
wait

# Figure
# python3 plot_loss.py