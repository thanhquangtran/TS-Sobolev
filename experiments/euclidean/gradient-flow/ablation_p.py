import os
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as pl
import time

from core.utils_GF import load_data, w2
import core.gradient_flow as gradient_flow
from tree_sliced.utils import generate_trees_frames

from tqdm import tqdm
# Configuration
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()
dataset_name = args.dataset

nofiterations = 1000
# Arrays to store results
results = {}
seeds = [1]
Ps = [1.2, 1.5, 1.7, 2, 5, 10]
for p in Ps:
    results[p] = {'raw_w2': np.zeros((nofiterations, len(seeds)))}

Xs = []
for i, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    N = 100  # Number of samples from p_X
    Xs.append(load_data(name=dataset_name, n_samples=N, dim=2))
    Xs[i] -= Xs[i].mean(dim=0)[np.newaxis, :]  # Normalization

for p in Ps:
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        X = Xs[i].detach().clone()
        meanX = 0
        _, d = X.shape

        # Use GPU if available, CPU otherwise
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the initial distribution
        temp = np.random.normal(loc=meanX, scale=.25, size=(N, d))

        # Define the variables to store the loss (2-Wasserstein distance)
        dist = 'w2'
        w2_dist = np.nan * np.zeros((nofiterations))

        # Define the optimizers and gradient flow objects
        Y = torch.tensor(temp, dtype=torch.float, device=device, requires_grad=True)
        optimizer = optim.Adam([Y], lr=args.lr)

        mean_X = torch.mean(X, dim=0, keepdim=True).to(device)
        std_X = torch.std(X, dim=0, keepdim=True).to(device)

        for t in tqdm(range(nofiterations)):
            loss = 0
            start_time = time.time()  # Start timing
            theta_twd, intercept_twd = generate_trees_frames(
                ntrees=25,
                nlines=4,
                d=X.shape[1],
                mean=mean_X,
                std=0.001,
                gen_mode='gaussian_raw',
                device='cuda'
            )  # orthogonal
            loss += gradient_flow.SbTS(X=X.to(device), Y=Y, theta=theta_twd, intercept=intercept_twd, 
                                       mass_division='distance_based', p=p, delta=1.0)
            end_time = time.time()  # End timing
            # print(f"Time taken for TWD orthogonal: {end_time - start_time:.4f} seconds")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if dist == 'w2':
                w2_dist[t] = w2(X.detach().cpu().numpy(), Y.detach().cpu().numpy())

        results[p]['raw_w2'][:, i] = w2_dist

#Calculate the mean and standard deviation for each iteration
for title in Ps:
    results[title]['mean'] = np.mean(results[title]['raw_w2'], axis=1)
    results[title]['std'] =  np.std(results[title]['raw_w2'], axis=1)
    results[title]['log10_w2'] = np.log10(results[title]['raw_w2'])
    results[title]['mean_log10'] =  np.mean(results[title]['log10_w2'], axis=1)
    results[title]['std_log10'] =  np.std(results[title]['log10_w2'], axis=1)

for t in range(nofiterations):
    log_dict = {}
    for title in Ps:
        log_dict[f'{title}/mean'] = results[title]['mean'][t]
        log_dict[f'{title}/mean_log10'] = results[title]['mean_log10'][t]
        log_dict[f'{title}/std'] = results[title]['std'][t]
        log_dict[f'{title}/std_log10'] = results[title]['std_log10'][t]

# Plot the results
pl.figure(figsize=(8, 6))

# Plot SW with mean and shaded standard deviation (log scale)
for i, title in enumerate(Ps):
    pl.plot(results[title]['mean_log10'], label=f"p={title}")
    pl.fill_between(range(nofiterations), 
                    results[title]['mean_log10'] - results[title]['std_log10'], 
                    results[title]['mean_log10'] + results[title]['std_log10'], alpha=0.2)

# Finalize the plot with dataset name in the title
os.makedirs("figure", exist_ok=True)
pl.title(f'Log Wasserstein Distance ({dataset_name})', fontsize=13)
pl.xlabel('Iterations', fontsize=13)
pl.ylabel(r'$W_2$ Distance (log scale)', fontsize=13)
pl.legend()
pl.grid(True)
pl.savefig(f"figure/ablation_p_{dataset_name}.png")
pl.clf()
