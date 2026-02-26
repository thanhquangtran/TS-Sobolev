## Tree-Sliced Sobolev IPM

[![Conference](https://img.shields.io/badge/ICLR-2026-in%20review-blue)](https://openreview.net/forum?id=HHNQSXaLkF)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

📄 **Paper**: [Tree-Sliced Sobolev IPM](https://openreview.net/pdf?id=HHNQSXaLkF) (ICLR 2026)

**Tree-Sliced Sobolev IPM (TS-Sobolev)** is a tree-sliced metric that aggregates regularized Sobolev Integral Probability Metrics (IPMs) over random tree systems. It:

- **Remains tractable for all orders** \(p \ge 1\); for \(p > 1\) it has the *same computational complexity* as Tree-Sliced Wasserstein (TSW) at \(p = 1\).
- **Recovers TSW exactly** at \(p = 1\), so it serves as a drop-in replacement for TSW in practical applications, while allowing the additional flexibility of changing \(p\).
- **Leveraging higher-order metrics** \((p > 1)\,\,\)provides more favorable optimization landscapes, with stricter convexity and smoother gradients than the \(p = 1\) case.
- **Extends to the spherical setting**, where it yields the Spherical Tree-Sliced Sobolev IPM (STS-Sobolev) for probability measures on hyperspheres.

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

The core Euclidean TS-Sobolev implementation lives in `src/tree_sliced/ts_sobolev.py` (class `SbTS`), with tree generation utilities in `src/tree_sliced/utils.py`. After installing the package (`pip install -e .`), both are importable from the `tree_sliced` namespace.

Below is a minimal example that computes a TS-Sobolev distance between two point clouds in \(\mathbb{R}^d\):

```python
import torch
from tree_sliced.ts_sobolev import SbTS
from tree_sliced.utils import generate_trees_frames

device = "cuda" if torch.cuda.is_available() else "cpu"

# Tree system parameters
ntrees = 250   # number of trees (L in the paper)
nlines = 4     # lines per tree (k in the paper)
d = 3          # data dimension

# Sample two empirical measures (same number of points)
N = 100
X = torch.randn(N, d, device=device)
Y = torch.randn(N, d, device=device)

# Sample concurrent-line tree systems
theta, intercept = generate_trees_frames(
    ntrees=ntrees,
    nlines=nlines,
    d=d,
    gen_mode="gaussian_orthogonal",
    device=device,
)

# Instantiate TS-Sobolev with order p > 1
ts_sobolev = SbTS(
    p=1.5,        # order of the Sobolev IPM
    delta=2.0,    # temperature for distance-based mass splitting
    device=device
)

distance = ts_sobolev(X, Y, theta, intercept)
print(f"TS-Sobolev_p distance: {distance.item():.6f}")
```

---

### Experiments

The `experiments/` folder is split into **Euclidean** and **spherical** settings.

#### Euclidean (`experiments/euclidean/`)

- **Gradient flow on \(\mathbb{R}^d\)** (`gradient-flow/`, Section 4.1)
  - `GradientFlow.py` — particle transport toward Gaussian targets; benchmarks SW, TSW, and TS-Sobolev.
  - `spectral_bias_exp.py` — **spectral bias analysis**: shows that TS-Sobolev\(_2\) recovers both low- and high-frequency components of a 1-D target density where TSW (p=1) is bad at high-frequency components. See `gradient-flow/README.md` for full usage.
- **Denoising Diffusion GAN** (`denoising-diffusion-gan/`) — TS-Sobolev as a discriminator loss for CIFAR-10 generation.
- **Topic modeling** (`topic-modeling/`) — Euclidean WAE-based topic models comparing TSW and TS-Sobolev as latent-space regularizers.

#### Spherical (`experiments/spherical/`)

- **Gradient flow on \(\mathbb{S}^d\)** (`gradient_flow/`, Section 4.2) — particle transport on the hypersphere using the Spherical Tree-Sliced Sobolev IPM (STS-Sobolev).
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