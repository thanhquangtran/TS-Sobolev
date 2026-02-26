import torch
import torch.nn.functional as F
import sys
import os

# Add path to import LSSOT from topic-modeling
sys.path.append(os.path.join(os.path.dirname(__file__), '../../experiments/topic-modeling/octis/models'))
from lssot import LSSOT

def lssotd_unif(X, num_projections=1000, ref_size=None, device='cpu', seed=0):
    """
    Compute the LSSOT distance to uniform distribution with numerical stability fixes.
    """
    X = X.to(device)
    
    # Ensure X is normalized and has valid values
    X = F.normalize(X, p=2, dim=-1)
    
    # Check for NaN/Inf in input
    if torch.isnan(X).any() or torch.isinf(X).any():
        return torch.tensor(0.0, device=device, requires_grad=False)
    
    # Add small noise (jitter) to prevent mode collapse and division by zero in interpolation
    # This is critical for stability when distributions are degenerate
    X = X + torch.rand_like(X) * 1e-5
    X = F.normalize(X, p=2, dim=-1)
    
    n_x = X.shape[0]
    x_weights = torch.ones(n_x, device=device) / n_x
    
    # Use batch size as ref_size if not specified
    if ref_size is None:
        ref_size = n_x
    
    # Ensure ref_size is at least 10 to avoid numerical issues
    ref_size = max(ref_size, 10)
    
    # Initialize LSSOT
    lssot = LSSOT(num_projections=num_projections, ref_size=ref_size, device=device, seed=seed)
    
    # Compute distance directly without try/except to allow gradients to flow
    # The jitter above prevents the numerical instability
    dist = lssot(X, x_weights)
    
    # Basic check for NaN output
    if torch.isnan(dist) or torch.isinf(dist):
        return torch.tensor(0.0, device=device, requires_grad=False)
        
    return dist
