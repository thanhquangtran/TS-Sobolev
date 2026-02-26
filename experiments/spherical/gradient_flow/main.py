import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from itertools import cycle
from scipy.stats import gaussian_kde
from tqdm.auto import trange
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from train import run_exp

import sys
sys.path.append('../')
import utils.vmf as vmf_utils
import utils.plot as plot_utils
from utils.func import set_seed
from methods import s3wd, sswd, stswd, sbstsd, lssotd

def plot_result(X, out_path):
    k = gaussian_kde(X.T)
    fig, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': "mollweide"})
    plot_utils.projection_mollweide(lambda x: k.pdf(x.T), ax)
    plt.savefig(out_path)
    plt.close(fig)

def get_run_name(args):
    if "sts" in args.d_func:
        return f"{args.d_func}-delta_{args.delta}-p_{args.p}"
    elif args.d_func == "lssot":
        # Include lr and num_projections in name for ablation tracking
        name = f"{args.d_func}-lr_{args.lr}"
        if hasattr(args, 'num_projections') and args.num_projections != 1000:
            name += f"-nproj_{args.num_projections}"
        if hasattr(args, 'ref_size') and args.ref_size is not None:
            name += f"-ref_{args.ref_size}"
        return name
    return f"{args.d_func}"
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('gradient flow parameters')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--d_func', '-d', type=str, default="stsw")
    parser.add_argument('--ntry', type=int, default=10)
    parser.add_argument('--ntrees', '-nt', type=int, default=200)
    parser.add_argument('--nlines', '-nl', type=int, default=5)
    parser.add_argument('--delta', type=float, default=2)
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--epochs', '-ep', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch_size', type=int, default=2400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_projections', type=int, default=1000, help='Number of projections for lssot (default: 1000, matches other baselines)')
    parser.add_argument('--ref_size', type=int, default=None, help='Reference size for lssot LCOT (default: None, uses batch size)')
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = args.device
    os.makedirs('figures', exist_ok=True)

    phi = (1 + np.sqrt(5)) / 2
    vs = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ])
    mus = F.normalize(torch.tensor(vs, dtype=torch.float), p=2, dim=-1)
    X = []
    kappa = 50 
    N = 200   
    for mu in mus:
        vmf = vmf_utils.rand_vmf(mu, kappa=kappa, N=N)
        X += list(vmf)
    X = np.array(X)
    X = torch.tensor(X, dtype=torch.float)
    Xt = X.clone().detach()
    trainloader = DataLoader(Xt, batch_size=args.batch_size, shuffle=True)
    dataiter = iter(cycle(trainloader))

    if args.d_func == "stsw":
        d_func = stswd.stswd
        d_args = {'p': args.p, 'ntrees': args.ntrees, 'nlines': args.nlines, 'delta': args.delta, 'device': device}
    elif args.d_func == "sbsts":
        d_func = sbstsd.sbsts
        d_args = {'p': args.p, 'ntrees': args.ntrees, 'nlines': args.nlines, 'delta': args.delta, 'device': device}
    elif args.d_func == "ari_s3w":
        d_func = s3wd.ari_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 30, 'pool_size': 1000}
    elif args.d_func == "s3w":
        d_func = s3wd.s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None}
    elif args.d_func == "ri_s3w_1":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 1}
    elif args.d_func == "ri_s3w_5":
        d_func = s3wd.ri_s3wd
        d_args = {'p': 2, 'n_projs': 1000, 'device': device, 'h': None, 'n_rotations': 5}
    elif args.d_func == "ssw":
        d_func = sswd.sswd
        d_args = {'p': 2, 'num_projections': 1000, 'device': device}
    elif args.d_func == "lssot":
        d_func = lssotd.lssotd
        # ref_size=None means use batch size dynamically
        # num_projections matches other baselines (s3w, ssw use 1000)
        num_projections = getattr(args, 'num_projections', 1000)
        ref_size = getattr(args, 'ref_size', None)
        d_args = {'num_projections': num_projections, 'ref_size': ref_size, 'device': device, 'seed': args.seed}
    else:
        raise Exception(f"Loss function {args.d_func} is not supported")
    
    results = []
    for s in range(args.ntry):
        results.append(run_exp(dataiter, d_func, d_args, mus, batch_size=args.batch_size, n_steps=args.epochs, 
                                           lr=args.lr, kappa=kappa, device=device, random_seed=s))

    runtimes = [r[0] for r in results]
    nll= np.array([r[1] for r in results])
    w = np.array([r[2] for r in results])
    log_wd = np.log10(w)
    iter_nll = np.mean(nll, axis=0)
    iter_log_wd = np.mean(log_wd, axis=0)
    iter_std = np.std(log_wd, axis=0)

    res_df = pd.read_csv("results.csv") if os.path.exists("results.csv") else pd.DataFrame()
    to_add = pd.DataFrame({
        "run_name": [get_run_name(args)],
        "iter_50": [iter_log_wd[0]],
        "iter_100": [iter_log_wd[1]],
        "iter_150": [iter_log_wd[2]],
        "iter_200": [iter_log_wd[3]],
        "iter_250": [iter_log_wd[4]],
        "iter_300": [iter_log_wd[5]],
        "iter_350": [iter_log_wd[6]],
        "iter_400": [iter_log_wd[7]],
        "iter_450": [iter_log_wd[8]],
        "iter_500": [iter_log_wd[9]],
        "mean_runtime": [np.mean(runtimes)],
        "lr": [args.lr],
        "seed": [args.seed],
    })
    res_df = pd.concat([res_df, to_add], ignore_index=True)
    res_df.to_csv("results.csv", index=False)
    
    # best = np.argmin(w)
    # X0_best = results[best][0]
    # plot_result(X0_best, f"figures/{get_run_name(args)}.png")

    os.makedirs("logs", exist_ok=True)
    with open(f"all_resutls.txt", "a") as f:
        f.write(f"{get_run_name(args)}\n") 
        f.write(f"\t{iter_log_wd[0]:.3f} $\pm$ {iter_std[0]:.3f} & {iter_log_wd[1]:.3f} $\pm$ {iter_std[1]:.3f} & \
{iter_log_wd[2]:.3f} $\pm$ {iter_std[2]:.3f} & {iter_log_wd[3]:.3f} $\pm$ {iter_std[3]:.3f} & {iter_log_wd[4]:.3f} $\pm$ {iter_std[4]:.3f}\n\n")
