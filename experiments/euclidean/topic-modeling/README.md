# Code submission for Tree-Sliced Sobolev IPM
## Topic modeling task

Requirements:
- python 3.10
- pytorch 2.8.0 with cuda 12.6
- other requirements are in requirements.txt and requirements_dev.txt

To train on spherical setting with M10 dataset
```
  python train.py --datasets M10 --epochs 100 --beta 1 --batch_size 64 --dist unif_sphere --dropout 0.5 --loss_type sbstsw --delta 2 --num_projections 2000 --n_trees 100 --p 1.5 --seeds 6
```
