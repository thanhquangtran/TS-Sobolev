import os
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as pl
import time
import wandb
import pickle

from core.utils_GF import load_data, w2
import core.gradient_flow as gradient_flow
from tree_sliced.utils import generate_trees_frames
import cfg
args = cfg.parse_args()
from tqdm import tqdm

# Configuration
dataset_name = args.dataset_name
nofiterations = args.num_iter
seeds = range(1, args.num_seeds+1)

# Define ablation study settings
# TSW (p=1): chain-uniform, chain-distance_based, concurrent-uniform, concurrent-distance_based
# Sobolev (p=1.2, 1.5, 2): chain-uniform, chain-distance_based, concurrent-uniform, concurrent-distance_based for each p

ablation_settings = []

# TSW (p=1) settings
ablation_settings.append({'method': 'TSW', 'p': 1, 'structure': 'chain', 'mass_division': 'uniform', 'title': 'TSW-p1-chain-uniform'})
ablation_settings.append({'method': 'TSW', 'p': 1, 'structure': 'chain', 'mass_division': 'distance_based', 'title': 'TSW-p1-chain-distance'})
ablation_settings.append({'method': 'TSW', 'p': 1, 'structure': 'concurrent', 'mass_division': 'uniform', 'title': 'TSW-p1-concurrent-uniform'})
ablation_settings.append({'method': 'TSW', 'p': 1, 'structure': 'concurrent', 'mass_division': 'distance_based', 'title': 'TSW-p1-concurrent-distance'})

# Sobolev settings for p=1.2, 1.5, 2
for p in [1.2, 1.5, 2]:
    ablation_settings.append({'method': 'Sobolev', 'p': p, 'structure': 'chain', 'mass_division': 'uniform', 'title': f'SbTS-p{p}-chain-uniform'})
    ablation_settings.append({'method': 'Sobolev', 'p': p, 'structure': 'chain', 'mass_division': 'distance_based', 'title': f'SbTS-p{p}-chain-distance'})
    ablation_settings.append({'method': 'Sobolev', 'p': p, 'structure': 'concurrent', 'mass_division': 'uniform', 'title': f'SbTS-p{p}-concurrent-uniform'})
    ablation_settings.append({'method': 'Sobolev', 'p': p, 'structure': 'concurrent', 'mass_division': 'distance_based', 'title': f'SbTS-p{p}-concurrent-distance'})

# Arrays to store results
results = {}
for setting in ablation_settings:
    results[setting['title']] = {'raw_w2': np.zeros((nofiterations, len(seeds)))}

Xs = []
for i, seed in enumerate(seeds):
    np.random.seed(seed)
    torch.manual_seed(seed)
    N = 500  # Number of samples from p_X
    Xs.append(load_data(name=dataset_name, n_samples=N, dim=2))
    Xs[i] -= Xs[i].mean(dim=0)[np.newaxis, :]  # Normalization

# Learning rate for all methods
lr = args.lr_tsw_sl
n_projs = int(args.L / args.n_lines)

for setting_idx, setting in enumerate(ablation_settings):
    title = setting['title']
    method = setting['method']
    p = setting['p']
    structure = setting['structure']
    mass_division = setting['mass_division']
    
    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        X = Xs[i].detach().clone()
        meanX = 0
        _, d = X.shape

        # Construct folder name based on hyperparameters
        args_dict = vars(args)
        folder_info = '-'.join([f"{key.replace('_', '')}{value}" for key, value in args_dict.items()])
        results_folder = f"./Results_reduced/Gradient_Flow_{folder_info}/seed{seed}"
        os.makedirs(results_folder, exist_ok=True)

        foldername = os.path.join(results_folder, 'Gifs', dataset_name + '_Comparison')
        os.makedirs(foldername, exist_ok=True)

        # Use GPU if available, CPU otherwise
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the initial distribution
        temp = np.random.normal(loc=meanX, scale=.25, size=(N, d))

        # Define the variables to store the loss (2-Wasserstein distance)
        dist = 'w2'
        w2_dist = np.nan * np.zeros((nofiterations))

        # Define the optimizers and gradient flow objects
        Y = torch.tensor(temp, dtype=torch.float, device=device, requires_grad=True)
        optimizer = optim.Adam([Y], lr=lr)

        mean_X = torch.mean(X, dim=0, keepdim=True).to(device)
        std_X = torch.std(X, dim=0, keepdim=True).to(device)

        for t in tqdm(range(nofiterations), desc=f"{title} seed{seed}"):
            loss = 0

            if structure == 'chain':
                # Generate trees for chain structure
                # Generate root with proper mean and std
                root = torch.randn(n_projs, 1, d, device=device) * args.std + mean_X
                # Generate theta
                theta = torch.randn(n_projs, args.n_lines, d, device=device)
                theta = theta / torch.norm(theta, dim=-1, keepdim=True)
                # Generate source offsets
                source = (0.1 - (-0.1)) * torch.rand(n_projs, args.n_lines - 1, device=device) + (-0.1)
                # Compute bias and intercept
                theta_mul_source = torch.einsum('tld,tl->tld', theta[:, :args.n_lines - 1, :], source)
                theta_mul_source_cummulative = torch.cumsum(theta_mul_source, dim=1)
                bias = theta_mul_source_cummulative + root
                intercept = torch.cat((root, bias), dim=1)
                subsequent_sources = torch.cat((bias, torch.zeros(n_projs, 1, d, device=device)), dim=1)
                
                if method == 'TSW':
                    loss += gradient_flow.TWD_Chain(
                        X=X.to(device), 
                        Y=Y, 
                        theta=theta, 
                        intercept=intercept, 
                        subsequent_sources=subsequent_sources,
                        mass_division=mass_division, 
                        p=p, 
                        delta=args.delta
                    )
                elif method == 'Sobolev':
                    loss += gradient_flow.SbTS_Chain(
                        X=X.to(device), 
                        Y=Y, 
                        theta=theta, 
                        intercept=intercept, 
                        subsequent_sources=subsequent_sources,
                        mass_division=mass_division, 
                        p=p, 
                        delta=args.delta
                    )
            else:  # concurrent
                # Generate trees for concurrent structure
                theta, intercept = generate_trees_frames(
                    ntrees=n_projs,
                    nlines=args.n_lines,
                    d=d,
                    mean=mean_X,
                    std=args.std,
                    gen_mode='gaussian_raw',
                    device=device
                )
                
                if method == 'TSW':
                    loss += gradient_flow.TWD(
                        X=X.to(device), 
                        Y=Y, 
                        theta=theta, 
                        intercept=intercept, 
                        mass_division=mass_division, 
                        p=p, 
                        delta=args.delta
                    )
                elif method == 'Sobolev':
                    loss += gradient_flow.SbTS(
                        X=X.to(device), 
                        Y=Y, 
                        theta=theta, 
                        intercept=intercept, 
                        mass_division=mass_division, 
                        p=p, 
                        delta=args.delta
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dist == 'w2' and (t + 1) % 500 == 0:
                w2_dist[t] = w2(X.detach().cpu().numpy(), Y.detach().cpu().numpy())

        results[title]['raw_w2'][:, i] = w2_dist
        
        # Save results to text file
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{title}_results.txt", "a") as f:
            a = ""
            a += f"{folder_info}_seed{seed}\n"
            stp = [499, 999, 1499, 1999, 2499]
            for step in stp:
                if step < nofiterations:
                    data = results[title]['raw_w2'][step, :i+1]  # Only use completed seeds up to current seed
                    if len(data) > 0 and not np.isnan(data).all():
                        a += f"{data.mean():.2e} & "
            a += "\n"
            f.write(a)
