import torch
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

def transform(X):
    """
    X: (N, d)
    
    Map x_i to (cos(h(x_i), sin(h(x_i)) * x_i))
    """
    return torch.cat([torch.cos(h(X)), torch.sin(h(X)) * X], dim=-1)

class TSSobolev():
    def __init__(self, p=2, delta=2, mass_division='distance_based', device="cuda"):
        """
        Class for computing the Tree Sliced Sobolev distance between two distributions.
        Args:
            p: level of the norm
            delta: negative inverse of softmax temperature for distance based mass division
            mass_division: how to divide the mass, one of 'uniform', 'distance_based'
            device: device to run the code, follow torch convention
        """
        self.device = device
        self.p = p
        self.delta = delta
        self.mass_division = mass_division

        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"

    def __call__(self, X, Y, theta, intercept):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        
        combined_axis_coordinate, mass_XY = self.get_mass_and_coordinate(X, Y, theta, intercept)
        tw = self.tw_concurrent_lines(mass_XY, combined_axis_coordinate)[0]

        return tw

    def tw_concurrent_lines(self, mass_XY, combined_axis_coordinate):
        """
        Args:
            mass_XY: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, num_lines, 2 * num_points)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_XY.shape[0], mass_XY.shape[1]

        # generate the cumulative sum of mass
        sub_mass = torch.gather(mass_XY, 2, indices)
        sub_mass_target_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_right_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_target_cumsum
        mask_right = torch.nonzero(coord_sorted > 0, as_tuple=True)
        sub_mass_target_cumsum[mask_right] = sub_mass_right_cumsum[mask_right]
        # generate the cumulative sum of length
        sub_length_target_cumsum = coord_sorted - coord_sorted[..., 0:1]
        sub_length_right_cumsum = coord_sorted[..., -1:] - coord_sorted
        sub_length_target_cumsum[mask_right] = sub_length_right_cumsum[mask_right]

        ### compute edge length
        # add root to the sorted coordinate by insert 0 to the first position <= 0
        root = torch.zeros(num_trees, num_lines, 1, device=self.device) 
        root_indices = torch.searchsorted(coord_sorted, root)
        coord_sorted_with_root = torch.zeros(num_trees, num_lines, mass_XY.shape[2] + 1, device=self.device)
        # distribute other points to the correct position
        edge_mask = torch.ones_like(coord_sorted_with_root, dtype=torch.bool)
        edge_mask.scatter_(2, root_indices, False)
        coord_sorted_with_root[edge_mask] = coord_sorted.flatten()
        # compute edge length
        edge_length = coord_sorted_with_root[:, :, 1:] - coord_sorted_with_root[:, :, :-1]

        ### compute beta edge 
        if self.p == 2:
            beta_edge_weight = torch.log(1 + edge_length/(1 + sub_length_target_cumsum))
        else:
            beta_edge_weight = ((1 + sub_length_target_cumsum + edge_length) ** (2 - self.p)
                          - (1 + sub_length_target_cumsum) ** (2 - self.p)) / (2 - self.p)
        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * beta_edge_weight
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)

        return tw, sub_mass_target_cumsum, beta_edge_weight

    def get_mass_and_coordinate(self, X, Y, theta, intercept):
        # for the last dimension
        # 0, 1, 2, ...., N -1 is of distribution 1
        # N, N + 1, ...., 2N -1 is of distribution 2
        N, dn = X.shape
        mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)
        mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)
        massXY = torch.cat((mass_X, -mass_Y), dim=2)

        return combined_axis_coordinate, massXY

    def project(self, input, theta, intercept):
        N, d = input.shape
        num_trees = theta.shape[0]
        num_lines = theta.shape[1]
        
        # all lines has the same point which is root
        input_translated = (input - intercept) #[T,B,D]
        # projected cordinate
        # 'tld,tdb->tlb'
        axis_coordinate = torch.matmul(theta, input_translated.transpose(1, 2))
        input_projected_translated = torch.einsum('tlb,tld->tlbd', axis_coordinate, theta)
        
        if self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)
        elif self.mass_division =='distance_based':
            dist = (torch.norm(input_projected_translated - input_translated.unsqueeze(1), dim = -1))
            weight = -self.delta*dist
            mass_input = torch.softmax(weight, dim=-2)/N
        
        return mass_input, axis_coordinate


class SbTS(TSSobolev):
    def __init__(self, p=2, delta=2, device="cuda"):
        super().__init__(p=p, delta=delta, device=device, mass_division='distance_based')


def sb_generate_trees_frames(ntrees, nlines, d, mean=128, std=0.1, device='cuda', gen_mode='gaussian_raw'):    
    # random root as gaussian distribution with given mean and std
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], "Invalid gen_mode"
    root = torch.randn(ntrees, 1, d, device=device) * std + mean
    intercept = root
    if gen_mode == 'gaussian_raw':
        theta = torch.randn(ntrees, nlines, d, device=device)
        theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "Support dim should be greater than or equal to number of lines to generate orthogonal lines"
        theta = torch.randn(ntrees, d, nlines, device=device)
        theta = svd_orthogonalize(theta)
        theta = theta.transpose(-2, -1)
    return theta, intercept


def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U


class SbSTSD():
    def __init__(self, ntrees=200, nlines=5, p=2, delta=2, device="cuda", type="normal"):
        """
        Class for computing the TW distance between two point clouds
        Args:
            mlp: The model that is used to divide the mass
            ntrees: Number of trees
            nlines: Number of lines per tree
            p: level of the norm
            delta: negative inverse of softmax temperature for distance based mass division
            device: device to run the code, follow torch convention
        """
        self.ntrees = ntrees
        self.nlines = nlines
        self.p = p
        self.delta = delta
        self.device = device
        self.eps = 1e-6
        if type not in ["normal", "generalized"]:
            raise ValueError("type should be either normal or generalized")
        self.type = type

    def __call__(self, X, Y):
        if self.type == "generalized":
            X = transform(X)
            Y = transform(Y)
            
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Get mass
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        root, intercept = self.generate_spherical_trees_frames(d=dn)
        
        combined_axis_coordinate, mass_X, mass_Y = self.get_mass_and_coordinate(X, Y, root, intercept)
        stw = self.stw_concurrent_lines(mass_X, mass_Y, combined_axis_coordinate)[0]

        return stw

    def stw_concurrent_lines(self, mass_X, mass_Y, combined_axis_coordinate):
        """
        Args:
            mass_X: (num_trees, num_lines, 2 * num_points)
            mass_Y: (num_trees, num_lines, 2 * num_points)
            combined_axis_coordinate: (num_trees, 2 * num_points)
        """
        coord_sorted, indices = torch.sort(combined_axis_coordinate, dim=-1)
        num_trees, num_lines = mass_X.shape[0], mass_X.shape[1]
        indices = indices.unsqueeze(1).repeat(1, num_lines, 1)

        # generate the cumulative sum of mass
        mass_X_sorted = torch.gather(mass_X, 2, indices)
        mass_Y_sorted = torch.gather(mass_Y, 2, indices)
        sub_mass = mass_X_sorted - mass_Y_sorted
        sub_mass_cumsum = torch.cumsum(sub_mass, dim=-1)
        sub_mass_target_cumsum = sub_mass + torch.sum(sub_mass, dim=-1, keepdim=True) - sub_mass_cumsum #(ntrees, nlines, 2*npoints)
        # generate the cumulative sum of length
        sub_length_target_cumsum = coord_sorted[..., -1:] - coord_sorted
        sub_length_target_cumsum = sub_length_target_cumsum.unsqueeze(1)

        ### compute edge length
        edge_length = torch.diff(coord_sorted, prepend=torch.zeros((num_trees, 1), device=coord_sorted.device), dim=-1)
        edge_length = edge_length.unsqueeze(1) #(ntrees, 1, 2*npoints)

        ### compute beta edge 
        if self.p == 2:
            beta_edge_weight = torch.log(1 + edge_length/(1 + sub_length_target_cumsum))
        else:
            beta_edge_weight = ((1 + sub_length_target_cumsum + edge_length) ** (2 - self.p)
                          - (1 + sub_length_target_cumsum) ** (2 - self.p)) / (2 - self.p)

        # compute TW distance
        subtract_mass = (torch.abs(sub_mass_target_cumsum) ** self.p) * beta_edge_weight
        subtract_mass_sum = torch.sum(subtract_mass, dim=[-1,-2])
        tw = torch.mean(subtract_mass_sum) ** (1/self.p)

        return tw, sub_mass_target_cumsum, beta_edge_weight


    def get_mass_and_coordinate(self, X, Y, root, intercept):
        # for the last dimension
        # 0, 1, 2, ...., N -1 is of distribution 1
        # N, N + 1, ...., 2N -1 is of distribution 2
        N, dn = X.shape
        mass_X, axis_coordinate_X = self.project(X, root=root, intercept=intercept)
        mass_Y, axis_coordinate_Y = self.project(Y, root=root, intercept=intercept)
        mass_X = torch.cat((mass_X, torch.zeros((mass_X.shape[0], mass_X.shape[1], N), device=self.device)), dim=2)
        mass_Y = torch.cat((torch.zeros((mass_Y.shape[0], mass_Y.shape[1], N), device=self.device), mass_Y), dim=2)

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=-1)

        return combined_axis_coordinate, mass_X, mass_Y

    def project(self, input, root, intercept):
        """
        Args:
            input: (N, d)
            root: (ntrees, 1, d)
            intercept: (ntrees, nlines, d)
        
        Returns:
            mass_input: (ntrees, nlines, N)
            axis_coordinate: (ntrees, N)
        """
        N = input.shape[0]
        ntrees, nlines, d = intercept.shape
        # project input on great circle.
        input_alpha = root @ input.T #(ntrees, 1, N)
        input_pc = input - input_alpha.transpose(1, 2) @ root #(ntrees, N, d)
        input_pc = F.normalize(input_pc, p=2, dim=-1)
         
        ## get axis_coordinate 
        # coord based on distance from root to projections
        root_input_cosine = (root @ input.T).squeeze(1) #(ntrees, N) coordinate in vector root.
        axis_coordinate = torch.acos(torch.clamp(root_input_cosine, -1 + self.eps, 1 - self.eps)) #(ntrees, N)
        
        ## divide mass
        dist_cosine = intercept @ input_pc.transpose(1, 2) #(ntrees, nlines, N)
        dist = torch.acos(torch.clamp(dist_cosine, -1 + self.eps, 1 - self.eps)) 
        scale = torch.sin(axis_coordinate).unsqueeze(1) # (ntrees, 1, N)
        dist = dist * scale
        weight = -self.delta*dist #(ntrees, nlines, N)
        mass_input = torch.softmax(weight, dim=-2)/N

        return mass_input, axis_coordinate

    def generate_spherical_trees_frames(self, d):
        root = torch.randn(self.ntrees, 1, d, device=self.device)
        root = root / torch.norm(root, dim=-1, keepdim=True)
        # root = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample((self.ntrees, 1)).to(self.device)
        # root = root / torch.norm(root, dim=-1, keepdim=True)
        # intercept = MultivariateNormal(torch.zeros(d), torch.eye(d)).sample((self.ntrees, self.nlines)).to(self.device)
        intercept = torch.randn(self.ntrees, self.nlines, d, device=self.device)
        intercept_proj = intercept @ root.transpose(1, 2) #(ntrees, nlines, 1)
        intercept = intercept - intercept_proj @ root #(ntrees, nlines, d)
        intercept = F.normalize(intercept, p=2, dim=-1)

        return root, intercept

def unif_hypersphere(shape, device):
    samples = torch.randn(shape, device=device)
    samples = F.normalize(samples, p=2, dim=-1)
    return samples

def sbsts(X, Y, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal'):
    TW_obj = SbSTSD(ntrees=ntrees, nlines=nlines, p=p, delta=delta, device=device, type=type)
    stswd = TW_obj(X, Y)
    return stswd

def sbsts_unif(X, ntrees=250, nlines=4, p=2, delta=2, device='cuda', type='normal'):
    Y_unif = unif_hypersphere(X.shape, device=X.device) 
    stswd_unif = sbsts(X, Y_unif, ntrees, nlines, p, delta, device, type)
    return stswd_unif


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## Test case 1    
    TW_obj = SbTS(device=device, delta=-1, p=2)
      
    theta = torch.tensor([[[1., 1.],[2., -1.]]])
    theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    intercept = torch.tensor([[[0., 0.]]])
    X = torch.tensor([[2., 0.], [4., 0.]]).to(device)
    Y = torch.tensor([[2., 1.], [6., 0.]]).to(device)
    sbts = TW_obj(X, Y, theta, intercept)
    assert torch.allclose(sbts, torch.tensor([0.426344]), rtol=0, atol=1e-5)

    ## Test case 2   
    TW_obj = SbTS(device=device, delta=-1, p=3)
      
    theta = torch.tensor([[[1., 1.],[2., -1.]]])
    theta = theta / torch.norm(theta, dim=-1, keepdim=True)
    intercept = torch.tensor([[[0., 0.]]])
    X = torch.tensor([[2., 0.], [4., 0.]]).to(device)
    Y = torch.tensor([[2., 1.], [6., 0.]]).to(device)
    sbts = TW_obj(X, Y, theta, intercept)
    assert torch.allclose(sbts, torch.tensor([0.349254]), rtol=0, atol=1e-5)

    ## Memory profiling
    from torch.profiler import profile, record_function, ProfilerActivity
    N = 50000
    M = 50000
    dn = dm = 1000
    ntrees = 100
    nlines = 10
    
    TW_obj = torch.compile(SbTS())
    
    
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    TW_obj(X, Y, theta, intercept)
    
    theta, intercept = generate_trees_frames(ntrees, nlines, dn, gen_mode="gaussian_orthogonal")
    X = torch.rand(N, dn).to("cuda")
    Y = torch.rand(M, dm).to("cuda")
    torch.cuda.reset_peak_memory_stats(device=None)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        tw = TW_obj(X, Y, theta, intercept)

    prof.export_chrome_trace("trace_concurrent.json")
    with open("profile_result_concurrent.txt", "w") as f:
        table_str = prof.key_averages().table(sort_by="cpu_time_total", top_level_events_only=True)
        f.write(table_str)
        print(table_str)
    print(torch.cuda.max_memory_allocated(device=None) / 1024 / 1024)

