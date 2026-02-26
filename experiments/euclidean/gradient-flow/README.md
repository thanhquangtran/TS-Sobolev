# Euclidean Gradient Flow

Source code for the Euclidean gradient flow experiments in the paper (Section 4.1). Contains two complementary experiments:

1. **Standard gradient flow** (`GradientFlow.py`) — transports particles toward a 2-D or high-dimensional Gaussian target, benchmarking convergence speed across methods.
2. **Spectral bias experiment** (`spectral_bias_exp.py`) — demonstrates that TS-Sobolev captures fine-grained, high-frequency details of a target density that TSW misses, by measuring low- and high-frequency spectral errors during a 1-D gradient flow.

## Requirements

Install the top-level conda environment (see the repository root README), which satisfies all dependencies including:

```
torch>=1.13.0
matplotlib
scikit-image
scikit-learn
tqdm
numpy
wandb
```

---

## Experiment 1: Standard Gradient Flow

Run `GradientFlow.py` to compare SW, TSW, and TS-Sobolev on particle transport tasks.

**Arguments:**

| Argument | Description | Default |
|---|---|---|
| `--num_iter` | Number of gradient steps | — |
| `--L` | Number of trees | — |
| `--n_lines` | Lines per tree | — |
| `--lr_sw` | Learning rate for SW | — |
| `--lr_tsw_sl` | Learning rate for TSW / TS-Sobolev | — |
| `--delta` | Distance-based mass-splitting temperature | — |
| `--p` | Sobolev order \(p\) | — |
| `--dataset_name` | Dataset / target distribution name | — |
| `--std` | Std for tree root generation | — |
| `--num_seeds` | Number of random seeds | — |

**Example:**

```bash
python GradientFlow.py \
  --num_iter 2500 --L 100 --n_lines 4 \
  --lr_sw 0.005 --lr_tsw_sl 50.0 \
  --delta 1.0 --p 2 \
  --dataset_name "gaussian_20d_small_v" \
  --std 0.001 --num_seeds 10
```

---

## Experiment 2: Spectral Bias

`spectral_bias_exp.py` shows that TS-Sobolev (p=2) captures **both low- and high-frequency components** of a target density, whereas TSW (p=1) exhibits a spectral bias that causes it to under-fit fine-grained structure.

**Setup:** particles are initialized on \([0,1]\) and transported toward a 1-D target density

\[
p_{\text{target}}(x) \propto 1 + 0.5\sin(2\pi \cdot 2 \cdot x) + 0.3\sin(2\pi \cdot 20 \cdot x),
\]

which contains a low-frequency mode (k=2) and a high-frequency mode (k=20). After each gradient step the spectral error at both frequencies is measured via DFT.

**Run:**

```bash
python spectral_bias_exp.py
```

This uses the default settings (8 seeds, 1000 iterations, lr=3e-5, 10 000 particles) and saves:

- `spectral_bias_results_std.pdf` — side-by-side plots of low- and high-frequency error for TSW vs TS-Sobolev\(_2\), with ±1 std shading across seeds.
- `spectral_results_std.txt` — raw numerical results (mean ± std per iteration).

The experiment runs seeds in parallel, automatically distributing work across available GPUs.
