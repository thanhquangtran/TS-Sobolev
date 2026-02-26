import torch
import os
import random
import torch
import numpy as np

def strfdelta(tdelta, fmt):
    d = {}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

def h(y):
    epsilon = 1e-6
    # y -> [-1, 1]
    y = y.mean(dim=-1, keepdim=True)
    # [-1, 1] in (-1 - epsilon, 1 + epsilon) -> (0, pi)
    shifted = (y + 1 + epsilon) * torch.pi / (2 + 2 * epsilon)
    assert torch.all((0 < shifted) & (shifted < torch.pi)), "h(y) should be in (0, pi)"
    return shifted

def transform(X):
    """
    X: (N, d)
    
    Map x_i to (cos(h(x_i), sin(h(x_i)) * x_i))
    """
    return torch.cat([torch.cos(h(X)), torch.sin(h(X)) * X], dim=-1)

def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(manualSeed)