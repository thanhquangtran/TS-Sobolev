#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wae.py
@Time    :   2020/10/04 19:53:52
@Author  :   Leilan Zhang
@Version :   1.0
@Contact :   zhangleilan@gmail.com
@Desc    :   None
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from octis.models.sb_tsw import TSSobolev, sb_generate_trees_frames
from octis.models.tsw.tsw import TSW, generate_trees_frames

# WAE model
class WAE(nn.Module):
    def __init__(self, encode_dims=[2000, 1024, 512, 20], decode_dims=[20, 1024, 2000], dropout=0.0, nonlin='relu'):
        super(WAE, self).__init__()
        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-1)
        })

        self.decoder = nn.ModuleDict({
            f'dec_{i}': nn.Linear(decode_dims[i], decode_dims[i+1])
            for i in range(len(decode_dims)-1)
        })
        self.latent_dim = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
        self.nonlin = {'relu': F.relu, 'sigmoid': torch.sigmoid}[nonlin]
        self.z_dim = encode_dims[-1]

    def encode(self, x):
        hid = x
        n_layers = len(self.encoder)
        for i, (_, layer) in enumerate(self.encoder.items()):
            if i < n_layers - 1:
                hid = self.dropout(layer(hid))
                hid = self.nonlin(hid)
            else:
                hid = layer(hid)
        return hid

    def decode(self, z):
        hid = z
        for i, (_, layer) in enumerate(self.decoder.items()):
            hid = layer(hid)
            if i < len(self.decoder)-1:
                hid = self.nonlin(self.dropout(hid))
        return hid

    def forward(self, x):
        z = self.encode(x)
        theta = F.softmax(z, dim=1)
        x_reconst = self.decode(theta)
        return x_reconst, z

    def sample(self, dist='dirichlet', batch_size=256, dirichlet_alpha=0.1, ori_data=None):
        if dist == 'dirichlet':
            z_true = np.random.dirichlet(
                np.ones(self.z_dim)*dirichlet_alpha, size=batch_size)
            z_true = torch.from_numpy(z_true).float()
            return z_true
        elif dist == 'gaussian':
            z_true = torch.randn(batch_size, self.z_dim).float()
            # z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
            return z_true
        elif dist == 'gmm_std':
            odes = np.eye(self.z_dim)*20
            ides = np.random.randint(low=0, high=self.z_dim, size=batch_size)
            mus = odes[ides]
            sigmas = np.ones((batch_size, self.z_dim))*0.2*20
            z_true = np.random.normal(mus, sigmas)
            # z_true = F.softmax(torch.from_numpy(z_true).float(), dim=1)
            z_true = torch.from_numpy(z_true).float()
            return z_true
        elif dist=='gmm_ctm' and ori_data!=None:
            with torch.no_grad():
                hid_vecs = self.encode(ori_data).cpu().numpy()
                gmm = GaussianMixture(n_components=self.z_dim,covariance_type='full',max_iter=200)
                gmm.fit(hid_vecs)
                #hid_vecs = torch.from_numpy(hid_vecs).to(self.device)
                gmm_spls, _spl_lbls = gmm.sample(n_samples=len(ori_data))
                z = torch.from_numpy(gmm_spls).float()
                # theta_prior = F.softmax(z,dim=1)
                return z
        else:
            raise
            return self.sample(dist='dirichlet',batch_size=batch_size)

    def sbtsw_loss(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        sbtsw = TSSobolev(p=p, delta=delta)
        theta, intercept = sb_generate_trees_frames(ntrees, nlines, dn)
        return sbtsw(z, prior, theta, intercept)

    def c0tsw_loss(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        TW_obj = TSW(ntrees=ntrees, nlines=nlines, p=p, delta=delta, mass_division='distance_based', ftype='circular_r0')
        theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
        return TW_obj(z, prior, theta, intercept)

    def c1tsw_loss(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        TW_obj = TSW(ntrees=ntrees, nlines=nlines, p=p, delta=delta, mass_division='distance_based', ftype='circular', radius=0.01)
        theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
        return TW_obj(z, prior, theta, intercept)

    def spatialtsw_loss(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        TW_obj = TSW(ntrees=ntrees, nlines=nlines, p=p, delta=delta, mass_division='distance_based', ftype='pow')
        theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
        return TW_obj(z, prior, theta, intercept)

    def dbtsw_loss(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        TW_obj = TSW(ntrees=ntrees, nlines=nlines, p=p, delta=delta, mass_division='distance_based')
        theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
        return TW_obj(z, prior, theta, intercept)

    def rptsw_loss(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        TW_obj = TSW(ntrees=ntrees, mass_division='distance_based', delta=delta, p=p)
        theta, intercept = generate_trees_frames(ntrees, nlines, dn, intercept_mode="geometric_median", gen_mode="random_path", X=z, Y=prior)
        return TW_obj(z, prior, theta, intercept)

    def rptsw_loss_root(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        TW_obj = TSW(ntrees=ntrees, mass_division='distance_based', delta=delta, p=p)
        theta, intercept = generate_trees_frames(ntrees, nlines, dn, intercept_mode="geometric_median", gen_mode="gaussian_raw", X=z, Y=prior)
        return TW_obj(z, prior, theta, intercept)
    
    def tswsl_loss(self, z, prior, p=2, delta=1, ntrees=100, nlines=50):
        dn = z.shape[-1]
        TW_obj = TSW(ntrees=ntrees, nlines=nlines, p=p, delta=delta, mass_division='uniform')
        theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_raw")
        return TW_obj(z, prior, theta, intercept)

    def sw_loss(self, z, prior, p=2, nlines=50):
        return SW(z, prior, L=nlines, p=p, device=z.device)

    def rpsw_loss(self, z, prior, p=2, nlines=50, kappa=50):
        return RPSW(z, prior, L=nlines, p=p, device=z.device, kappa=kappa)

    def ebrpsw_loss(self, z, prior, p=2, nlines=50, kappa=50):
        return EBRPSW(z, prior, L=nlines, p=p, device=z.device, kappa=kappa)

    def dsw_loss(self, z, prior, p=2, nlines=50, kappa=50, s_lr=0.1, n_lr=100):
        return DSW(z, prior, L=nlines, p=p, kappa=kappa, s_lr=s_lr, n_lr=n_lr, device=z.device)

    def isebsw_loss(self, z, prior, p=2, nlines=50):
        return ISEBSW(z, prior, L=nlines, p=p, device=z.device)

    def mmd_loss(self, x, y, device, t=0.1, kernel='diffusion'):
        '''
        computes the mmd loss with information diffusion kernel
        :param x: batch_size * latent dimension
        :param y:
        :param t:
        :return:
        '''
        eps = 1e-6
        n, d = x.shape
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(i+1, n):
                    sum_xx = sum_xx + torch.norm(x[i]-x[j], p=1).to(device)
            sum_xx = sum_xx / (n * (n-1))

            sum_yy = torch.zeros(1).to(device)
            for i in range(y.shape[0]):
                for j in range(i+1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

            sum_xy = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())

            def diffusion_kernel(a, tmpt, dim):
                # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
                return torch.exp(-torch.acos(a).pow(2)) / tmpt

            off_diag = 1 - torch.eye(n).to(device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t, d-1)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t, d-1)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t, d-1)
            sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
            sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy
