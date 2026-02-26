import torch
import torch.nn.functional as F
import sys
import os

# Add path to import LSSOT from topic-modeling
sys.path.append(os.path.join(os.path.dirname(__file__), '../../experiments/topic-modeling/octis/models'))
from lssot import LSSOT

def lssotd(X, Y, num_projections=1000, ref_size=None, device='cpu', seed=0):
    """
    Compute the LSSOT (Linear Sliced Spherical Optimal Transport) distance.
    
    Parameters:
    X: torch.Tensor, shape (n_samples_x, dim)
        Samples in the source domain
    Y: torch.Tensor, shape (n_samples_y, dim)
        Samples in the target domain
    num_projections: int
        Number of projections
    ref_size: int, optional
        Reference size for LCOT. If None, uses batch size.
    device: str
        Device to run on
    seed: int
        Random seed for projections
    """
    X = X.to(device)
    Y = Y.to(device)
    
    # Ensure inputs are normalized and have valid values
    X = F.normalize(X, p=2, dim=-1)
    Y = F.normalize(Y, p=2, dim=-1)
    if torch.isnan(X).any() or torch.isinf(X).any() or torch.isnan(Y).any() or torch.isinf(Y).any():
        # Return a small non-zero value if input is invalid
        return torch.tensor(1e-6, device=device, requires_grad=True)
    
    # Create uniform weights (1/(2n) as in wae_sp.py)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    x_weights = torch.ones(n_x, device=device) / n_x / 2
    y_weights = torch.ones(n_y, device=device) / n_y / 2
    
    # Use batch size as ref_size if not specified, but ensure it's reasonable
    if ref_size is None:
        ref_size = max(n_x, n_y)
    
    # Ensure ref_size is at least 10 to avoid numerical issues
    ref_size = max(ref_size, 10)
    
    # Initialize LSSOT (create fresh each time)
    # Note: seed=0 is used for deterministic behavior
    lssot = LSSOT(num_projections=num_projections, ref_size=ref_size, device=device, seed=seed)
    
    # Compute distance with gradient clipping protection
    try:
        dist = lssot(X, x_weights, Y, y_weights)
        # Clip to prevent extreme values
        dist = torch.clamp(dist, min=1e-8, max=1e6)
        if torch.isnan(dist) or torch.isinf(dist):
            return torch.tensor(1e-6, device=device, requires_grad=True)
        return dist
    except:
        # Fallback to small value if computation fails
        return torch.tensor(1e-6, device=device, requires_grad=True)

def lssotd_unif(X, num_projections=1000, ref_size=None, device='cpu', seed=0):
    """
    Compute the LSSOT distance to uniform distribution.
    
    Parameters:
    X: torch.Tensor, shape (n_samples_x, dim)
        Samples in the source domain
    num_projections: int
        Number of projections
    ref_size: int, optional
        Reference size for LCOT. If None, uses batch size.
    device: str
        Device to run on
    seed: int
        Random seed for projections
    """
    X = X.to(device)
    
    # Ensure X is normalized and has valid values
    X = F.normalize(X, p=2, dim=-1)
    if torch.isnan(X).any() or torch.isinf(X).any():
        # Return a small non-zero value if input is invalid
        return torch.tensor(1e-6, device=device, requires_grad=True)
    
    # Create uniform weights (1/n for uniform case)
    n_x = X.shape[0]
    x_weights = torch.ones(n_x, device=device) / n_x
    
    # Use batch size as ref_size if not specified, but ensure it's reasonable
    if ref_size is None:
        ref_size = n_x
    
    # Ensure ref_size is at least 10 to avoid numerical issues
    ref_size = max(ref_size, 10)
    
    # Initialize LSSOT (create fresh each time)
    # Note: seed=0 is used for deterministic behavior
    lssot = LSSOT(num_projections=num_projections, ref_size=ref_size, device=device, seed=seed)
    
    # Compute distance to uniform with gradient clipping protection
    try:
        dist = lssot(X, x_weights)
        # Clip to prevent extreme values
        dist = torch.clamp(dist, min=1e-8, max=1e6)
        if torch.isnan(dist) or torch.isinf(dist):
            return torch.tensor(1e-6, device=device, requires_grad=True)
        return dist
    except:
        # Fallback to small value if computation fails
        return torch.tensor(1e-6, device=device, requires_grad=True)

