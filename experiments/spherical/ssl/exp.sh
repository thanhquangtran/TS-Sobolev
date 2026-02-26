# export CUDA_VISIBLE_DEVICES=0  # Commented out, using CUDA_VISIBLE_DEVICES per command instead

# Previous experiments (commented out)
# ## Ours
# python3 train_eval.py --method sbsts \
#     --ntrees 200 --nlines 20 --delta 2 \
#     --unif_w 10 --feat_dim 10 --epochs 200 \
#     --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
#     --lr 0.05 --seed 0 --p 2
#
# python3 train_eval.py --method sbsts \
#     --ntrees 200 --nlines 20 --delta 2 \
#     --unif_w 10 --feat_dim 10 --epochs 200 \
#     --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
#     --lr 0.05 --seed 0 --p 1.5

# Run lssot only (s3w already completed)
CUDA_VISIBLE_DEVICES=5 python3 train_eval.py --method lssot \
    --unif_w 10 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --lr 0.05 --seed 0 --epochs 200