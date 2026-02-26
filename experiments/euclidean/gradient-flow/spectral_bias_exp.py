import os
import sys
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to sys.path to allow imports from core and db_tsw
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.gradient_flow import TWD, SbTS
from tree_sliced.utils import generate_trees_frames

# --- 1. Setup Target Density ---

def target_density(x):
    """
    p_target(x) = 1 + 0.5*sin(2*pi*k_low*x) + 0.3*sin(2*pi*k_high*x)
    Normalized on [0, 1].
    """
    k_low = 2
    k_high = 20
    return 1 + 0.5 * np.sin(2 * np.pi * k_low * x) + 0.3 * np.sin(2 * np.pi * k_high * x)

def sample_target(n_samples):
    """
    Rejection sampling for the target density.
    """
    samples = []
    max_val = 1.8 # Upper bound for p_target
    while len(samples) < n_samples:
        x = np.random.uniform(0, 1, n_samples * 2)
        u = np.random.uniform(0, max_val, n_samples * 2)
        p_x = target_density(x)
        accepted = x[u <= p_x]
        samples.extend(accepted)
    return np.array(samples[:n_samples])

# --- 3. Metric: Spectral Error Analysis ---

def compute_spectral_error(particles, k_low=2, k_high=20, n_bins=200):
    """
    Computes the spectral error at specific frequencies.
    """
    # Empirical density
    hist, bin_edges = np.histogram(particles, bins=n_bins, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Target density at bin centers
    target_vals = target_density(bin_centers)
    
    # DFT
    fft_current = np.fft.fft(hist)
    fft_target = np.fft.fft(target_vals)
    
    # Frequencies corresponding to FFT indices
    err_low = np.abs(fft_target[k_low] - fft_current[k_low])
    err_high = np.abs(fft_target[k_high] - fft_current[k_high])
    
    return err_low, err_high

# --- 2. Task: Gradient Flow ---

def run_experiment(method='TSW', p=1, n_particles=10000, n_iter=100, log_interval=10, lr=0.01, seed=0, disable_tqdm=False, device_id=None, X_target_np=None):
    # Set random seeds - ensure each thread has its own random state
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Set device - use specific GPU if device_id is provided, otherwise auto-detect
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Initialize particles from Uniform[0, 1] - this varies by seed
    Y_init = np.random.uniform(0, 1, (n_particles, 1))
    Y = torch.tensor(Y_init, dtype=torch.float, device=device, requires_grad=True)
    
    # Use provided target samples (same for all seeds) or generate if not provided
    if X_target_np is None:
        X_target_np = sample_target(n_particles)
    X_target = torch.tensor(X_target_np.reshape(-1, 1), dtype=torch.float, device=device)
    
    optimizer = optim.Adam([Y], lr=lr)
    
    # Setup for Tree Sliced metrics
    n_projs = 100
    n_lines = 4 
    
    low_freq_errors = []
    high_freq_errors = []
    iterations = []
    
    # Parameters for loss
    delta = 2.0 
    mass_division = 'distance_based' 
    std = 0.01
    mean_X = torch.mean(X_target, dim=0, keepdim=True)
    
    for t in tqdm(range(n_iter), desc=f"{method} p={p} seed={seed}", disable=disable_tqdm):
        # Generate trees/slices
        theta, intercept = generate_trees_frames(
            ntrees=n_projs,
            nlines=n_lines,
            d=1,
            mean=mean_X,
            std=std,
            gen_mode='gaussian_raw',
            device=device
        )
        
        loss = 0
        if method == 'TSW':
            loss = TWD(X_target, Y, theta, intercept, mass_division=mass_division, p=p, delta=delta, device=device)
        elif method == 'Sobolev':
            loss = SbTS(X_target, Y, theta, intercept, mass_division=mass_division, p=p, delta=delta, device=device)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clamp particles
        with torch.no_grad():
            Y.clamp_(0, 1)
            
        if t % log_interval == 0 or t == n_iter - 1:
            Y_np = Y.detach().cpu().numpy().flatten()
            l_err, h_err = compute_spectral_error(Y_np)
            low_freq_errors.append(l_err)
            high_freq_errors.append(h_err)
            iterations.append(t)
            
    return iterations, low_freq_errors, high_freq_errors

def run_single_seed_wrapper(args):
    """Wrapper function for parallel execution"""
    method, p, n_iter, log_interval, lr, seed, device_id, X_target_np = args
    return run_experiment(method, p, n_particles=10000, n_iter=n_iter, log_interval=log_interval, lr=lr, seed=seed, disable_tqdm=True, device_id=device_id, X_target_np=X_target_np)

def run_multi_seed(method, p, n_iter, log_interval, lr, seeds, n_jobs=None, X_target_np=None, processes_per_gpu=1):
    all_low = []
    all_high = []
    iters = None
    
    # Use provided target samples or generate if not provided
    if X_target_np is None:
        np.random.seed(42)  # Default seed if not provided
        X_target_np = sample_target(10000)
        print(f"Generated target samples with seed 42 (same for all runs)")
    else:
        print(f"Using provided target samples (same for all runs)")
    
    # Detect available GPUs
    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s), allowing {processes_per_gpu} processes per GPU")
        # Use GPUs if available, otherwise fall back to CPU
        if num_gpus > 0:
            # Assign each seed to a GPU (round-robin, one process per GPU)
            gpu_ids = [seed % num_gpus for seed in seeds]
            if n_jobs is None:
                n_jobs = min(len(seeds), num_gpus)
            print(f"Setting n_jobs={n_jobs} (one process per GPU)")
        else:
            gpu_ids = [None] * len(seeds)
            if n_jobs is None:
                n_jobs = min(len(seeds), os.cpu_count() or 1)
    else:
        gpu_ids = [None] * len(seeds)
        if n_jobs is None:
            n_jobs = min(len(seeds), os.cpu_count() or 1)
    
    # Prepare arguments for parallel execution with GPU assignment
    # Pass the same X_target_np to all experiments
    args_list = [(method, p, n_iter, log_interval, lr, seed, gpu_id, X_target_np) 
                  for seed, gpu_id in zip(seeds, gpu_ids)]
    
    # Print GPU assignment summary
    if torch.cuda.is_available() and num_gpus > 0:
        gpu_assignments = {}
        for seed, gpu_id in zip(seeds, gpu_ids):
            if gpu_id not in gpu_assignments:
                gpu_assignments[gpu_id] = []
            gpu_assignments[gpu_id].append(seed)
        for gpu, gpu_seeds in sorted(gpu_assignments.items()):
            print(f"GPU {gpu}: {len(gpu_seeds)} processes (seeds {sorted(gpu_seeds)})")
    
    # Run in parallel using ThreadPoolExecutor (better for GPU workloads)
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(run_single_seed_wrapper, args): args[-3] for args in args_list}  # -3 is seed index (method, p, n_iter, log_interval, lr, seed, gpu_id, X_target_np)
        
        results = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{method} p={p} (parallel)"):
            seed = futures[future]
            try:
                iters, low, high = future.result()
                results[seed] = (low, high)
            except Exception as e:
                print(f"Error in seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    # Sort by seed to maintain order
    sorted_seeds = sorted(results.keys())
    for seed in sorted_seeds:
        low, high = results[seed]
        all_low.append(low)
        all_high.append(high)
        
    all_low = np.array(all_low)
    all_high = np.array(all_high)
    
    mean_low = np.mean(all_low, axis=0)
    std_low = np.std(all_low, axis=0, ddof=1)  # Use sample std (ddof=1) for unbiased estimate
    mean_high = np.mean(all_high, axis=0)
    std_high = np.std(all_high, axis=0, ddof=1)
    
    # Print some diagnostics
    print(f"{method} p={p}: Low freq - Mean range: [{mean_low.min():.4f}, {mean_low.max():.4f}], Std range: [{std_low.min():.4f}, {std_low.max():.4f}]")
    print(f"{method} p={p}: High freq - Mean range: [{mean_high.min():.4f}, {mean_high.max():.4f}], Std range: [{std_high.min():.4f}, {std_high.max():.4f}]")
    
    return iters, mean_low, std_low, mean_high, std_high

# --- Main Execution ---

if __name__ == "__main__":
    n_seeds = 8
    seeds = range(n_seeds)
    n_iter = 1000
    log_interval = 50
    lr = 3e-5
    
    # Generate target samples once - same for both TSW and Sobolev
    np.random.seed(42)
    X_target_np = sample_target(10000)
    print(f"Generated target samples with seed 42 (same for TSW and Sobolev)")
    
    # Run TSW (p=1) with the same target data
    iters, mean_low_tsw, std_low_tsw, mean_high_tsw, std_high_tsw = run_multi_seed('TSW', 1, n_iter, log_interval, lr, seeds, X_target_np=X_target_np)
    
    # Run Sobolev (p=2) with the same target data
    _, mean_low_sob, std_low_sob, mean_high_sob, std_high_sob = run_multi_seed('Sobolev', 2, n_iter, log_interval, lr, seeds, X_target_np=X_target_np)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Low Frequency
    plt.subplot(1, 2, 1)
    plt.plot(iters, mean_low_tsw, label='TSW', marker='o')
    plt.fill_between(iters, mean_low_tsw - std_low_tsw, mean_low_tsw + std_low_tsw, alpha=0.3)
    
    plt.plot(iters, mean_low_sob, label=r'TS-Sobolev$_2$', marker='x')
    plt.fill_between(iters, mean_low_sob - std_low_sob, mean_low_sob + std_low_sob, alpha=0.3)
    
    plt.title('Low Frequency Error')
    plt.xlabel('Iteration')
    plt.ylabel('Spectral Error')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    # High Frequency
    plt.subplot(1, 2, 2)
    plt.plot(iters, mean_high_tsw, label='TSW', marker='o')
    plt.fill_between(iters, mean_high_tsw - std_high_tsw, mean_high_tsw + std_high_tsw, alpha=0.3)
    
    plt.plot(iters, mean_high_sob, label=r'TS-Sobolev$_2$', marker='x')
    plt.fill_between(iters, mean_high_sob - std_high_sob, mean_high_sob + std_high_sob, alpha=0.3)
    
    plt.title('High Frequency Error')
    plt.xlabel('Iteration')
    plt.ylabel('Spectral Error')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig('spectral_bias_results_std.pdf', dpi=100)
    print("Results saved to spectral_bias_results_std.pdf")
    
    # Save results to text file
    with open('spectral_results_std.txt', 'w') as f:
        f.write("Iter, TSW_Low_Mean, TSW_Low_Std, TSW_High_Mean, TSW_High_Std, Sob_Low_Mean, Sob_Low_Std, Sob_High_Mean, Sob_High_Std\n")
        for i in range(len(iters)):
            f.write(f"{iters[i]}, {mean_low_tsw[i]}, {std_low_tsw[i]}, {mean_high_tsw[i]}, {std_high_tsw[i]}, "
                    f"{mean_low_sob[i]}, {std_low_sob[i]}, {mean_high_sob[i]}, {std_high_sob[i]}\n")
    print("Results saved to spectral_results_std.txt")
