## Tree-Sliced Sobolev IPM

[![Conference](https://img.shields.io/badge/ICLR_2026-accepted-blue)](https://openreview.net/forum?id=HHNQSXaLkF)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📄 **Paper**: [Tree-Sliced Sobolev IPM](https://openreview.net/pdf?id=HHNQSXaLkF) (ICLR 2026)

**Tree-Sliced Sobolev IPM (TS-Sobolev)** extends Tree-Sliced Wasserstein (TSW) to higher order by using regularized Sobolev IPMs over probability measures on tree metrics. It:

- **Remains tractable for all orders** $p \ge 1$; for $p > 1$ it has the *same computational complexity* as TSW at $p = 1$.
- **Recovers TSW exactly** at $p = 1$, serving as a drop-in replacement with the added flexibility of tuning $p$.
- **Provides more favorable optimization landscapes** for $p > 1$, with stricter convexity and smoother gradients that help capture fine-grained structure TSW misses.
- **Extends to the spherical setting**, yielding the Spherical TS-Sobolev (STS-Sobolev) for probability measures on hyperspheres.
- **Outperforms existing methods** across gradient flows, generative modeling, self-supervised learning, and topic modeling in both Euclidean and spherical settings.

---

### Requirements

To create the conda environment used in our experiments:

```bash
conda env create --file=environment.yaml
conda activate ts-sobolev
pip install -e .
```

This installs PyTorch with CUDA 11.8 and all Python dependencies needed to run the Euclidean and spherical experiments.

---

### Quick Start

The core Euclidean TS-Sobolev implementation lives in `src/tree_sliced/ts_sobolev.py` (class `TSSobolev`), with tree generation utilities in `src/tree_sliced/utils.py`. After installing the package (`pip install -e .`), both are importable from the `tree_sliced` namespace.

Below is a minimal example that computes a TS-Sobolev distance between two point clouds in $\mathbb{R}^d$:

```python
import torch
from tree_sliced.ts_sobolev import TSSobolev
from tree_sliced.utils import generate_trees_frames

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tree system parameters
ntrees = 250   # number of trees (L in the paper)
nlines = 4     # lines per tree (k in the paper)
d = 64         # data dimension (must satisfy d >= nlines for gaussian_orthogonal)

# Sample two empirical measures (same number of points)
N = 500
X = torch.randn(N, d, device=device)
Y = torch.randn(N, d, device=device)

# Center tree roots near the data mean
mean_X = X.mean(dim=0, keepdim=True)   # shape (1, d)

# Sample concurrent-line tree systems
# gaussian_orthogonal produces orthonormal line directions (requires d >= nlines)
theta, intercept = generate_trees_frames(
    ntrees=ntrees,
    nlines=nlines,
    d=d,
    mean=mean_X,
    std=0.1,
    gen_mode="gaussian_orthogonal",
    device=device,
)

# Instantiate TS-Sobolev (p=2, distance-based mass splitting)
ts_sobolev = TSSobolev(p=2, delta=2, device=device)

distance = ts_sobolev(X, Y, theta, intercept)
print(f"TS-Sobolev distance: {distance.item():.6f}")
```

---

### Experiments

The `experiments/` folder is split into **Euclidean** and **spherical** settings.

#### Euclidean (`experiments/euclidean/`, Section 4.1)

- **Gradient flow on $\mathbb{R}^d$** (`gradient-flow/`)
  - `GradientFlow.py` — particle transport toward Gaussian targets; benchmarks SW, TSW, and TS-Sobolev.
  - `spectral_bias_exp.py` — **spectral bias analysis**: shows that TS-Sobolev\(_2\) recovers both low- and high-frequency components of a 1-D target density where TSW (p=1) is bad at high-frequency components. See `gradient-flow/README.md` for full usage.
- **Denoising Diffusion GAN** (`denoising-diffusion-gan/`) — TS-Sobolev as a discriminator loss for CIFAR-10 generation.
- **Topic modeling** (`topic-modeling/`) — Euclidean WAE-based topic models comparing TSW and TS-Sobolev as latent-space regularizers.

#### Spherical (`experiments/spherical/`,  Section 4.2)

- **Gradient flow on $\mathbb{S}^d$** (`gradient_flow/`) — particle transport on the hypersphere using the Spherical Tree-Sliced Sobolev IPM (STS-Sobolev).
- **Self-supervised learning** (`ssl/`) — TS-Sobolev / STS-Sobolev as a uniformity loss on the unit hypersphere for representation learning.

Each subdirectory has its own README or inline comments describing configuration and how to run the corresponding experiments.

---

### Acknowledgments

This code builds on prior work in Tree-Sliced and Sliced Optimal Transport, including:

- [Db-TSW](https://github.com/Fsoft-AIC/DbTSW)
- [NonlinearTSW](https://github.com/thanhquangtran/NonlinearTSW)
- [PartialTSW](https://github.com/thanhquangtran/PartialTSW)
- [FW-TSW](https://github.com/thanhquangtran/FW-TSW)

---

### Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{tran2026treesliced,
    title={Tree-sliced Sobolev {IPM}},
    author={Viet-Hoang Tran and Thanh Tran and Thanh Chu and Duy-Tung Pham and Trung-Khang Tran and Tam Le and Tan Minh Nguyen},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=HHNQSXaLkF}
}
```

---

### License

This project is licensed under the MIT License; see the `LICENSE` file for details.