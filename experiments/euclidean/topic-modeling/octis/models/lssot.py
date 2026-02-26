import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import contextlib

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


def sort_measure(mu_values,mu_weights):
    mu_sorter = torch.argsort(mu_values, -1)
    mu_values = torch.take_along_dim(mu_values, mu_sorter, -1)
    mu_weights = torch.take_along_dim(mu_weights, mu_sorter, -1)
    return mu_values,mu_weights


class LCOT_torch(nn.Module):
    def __init__(self,device, refsize=None, *args, **kwargs):
        super(LCOT_torch, self).__init__(*args, **kwargs)
        self.device = device
        self.ref = torch.linspace(0,1,refsize+1)[:-1].to(device)
        self.N = len(self.ref)
        self.dx = 1./self.N

    def empirical_cdf(self, samples, weights):
        # Returns samples in order and cumulutative probs at those points, to plot the cdf do plt.plot(sorted_samples,cumulative_probs)
        sorted_samples, sorted_weights = sort_measure(samples, weights)
        cumulative_probs = torch.cumsum(sorted_weights, -1).to(self.device)
        return sorted_samples, cumulative_probs

    def ecdf(self, samples, weights, xnew):
        int_x = torch.floor(xnew).to(self.device)
        rest_x = xnew-int_x
        xs, ys = self.empirical_cdf(samples, weights)
        return int_x + Interp1d()(xs, ys,rest_x).to(self.device)


    def emb(self, samples, weights):
        l, n = samples.shape
        alpha=(torch.sum(samples*weights, dim=-1)/torch.sum(weights, dim=-1)-1/2)[:,None]
        xnew = torch.linspace(-1,2,3*self.N).repeat(l, 1).to(self.device)
        x = self.ref.repeat(l, 1)
        embedd = Interp1d()(self.ecdf(samples, weights, xnew), xnew, x-alpha).to(self.device)-x
        return embedd

    def cost(self,x1, x1_weights, x2=None, x2_weigths=None):
        x1_hat = self.emb(x1, x1_weights)
        if x2 == None: #when x2 is the uniform distribution
            # Add epsilon for numerical stability of sqrt derivative at 0
            return torch.sqrt(((torch.minimum(abs(x1_hat),1-abs(x1_hat))**2).sum(-1)).mean() + 1e-8)
        x2_hat = self.emb(x2, x2_weigths)
        # Add epsilon for numerical stability of sqrt derivative at 0
        return torch.sqrt(((torch.minimum(abs(x2_hat-x1_hat),1-abs(x2_hat-x1_hat))**2).sum(-1)).mean() + 1e-8)
  
  


class LSSOT(nn.Module):
    def __init__(self, num_projections, ref_size, device, seed=0):
        super(LSSOT, self).__init__()
        self.device = device
        self.num_projections = num_projections
        self.ref_size = ref_size
        self.lcot = LCOT_torch(device=device, refsize=self.ref_size)
        self.seed = seed

    def slice(self, x, x_weights, cap=1e-6):
        x = F.normalize(x, p=2, dim=-1)
        slice_weights = x_weights.repeat(self.num_projections, 1)
        modified_weights = slice_weights.clone()
        n, d = x.shape
        # Uniform and independent samples on the Stiefel manifold V_{d,2}
        torch.manual_seed(self.seed)
        Z = torch.randn((self.num_projections,d,2), device=self.device)
        U, _ = torch.linalg.qr(Z)
        x = x[None, :, :]@U
        # Apply \epsilon cap
        ignore_ind = torch.norm(x, dim=-1) <= cap
        modified_weights[ignore_ind] = 0
        x = F.normalize(x, p=2, dim=-1)
        x = (torch.atan2(-x[:,:,1], -x[:,:,0])+torch.pi)/(2*torch.pi)
        # slice_weights = slice_weights / slice_weights.sum(-1).unsqueeze(-1)
        modified_weights = modified_weights + ((slice_weights-modified_weights).sum(-1) / ignore_ind.logical_not().sum(-1)).unsqueeze(-1)
        modified_weights[ignore_ind] = 0
        return x, modified_weights
    
    def embed(self, x, x_weights):
        x, w = self.slice(x, x_weights)
        return self.lcot.emb(x, w)

    def forward(self, x1, x1_weights, x2=None, x2_weights=None):
        x1, x1_w = self.slice(x1, x1_weights)
        if  x2 is not None:
            x2, x2_w = self.slice(x2, x2_weights)
            return self.lcot.cost(x1, x1_w, x2, x2_w)
        return self.lcot.cost(x1, x1_w)
    