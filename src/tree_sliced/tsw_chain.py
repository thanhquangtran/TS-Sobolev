import torch

class TSWChain():

    def __init__(self, nofprojections=1000, nlines=5, p=2, delta=2, mass_division='uniform', device="cuda"):

        self.nofprojections = nofprojections

        self.device = device

        self.nlines = nlines

        self.p = p

        self.delta = delta

        self.mass_division = mass_division

        assert self.mass_division in ['uniform', 'distance_based'], \
            "Invalid mass division. Must be one of 'uniform', 'distance_based'"

    def __call__(self, X, Y, theta, intercept, subsequent_sources):

        X = X.to(self.device)

        Y = Y.to(self.device)

        # Get mass

        N, dn = X.shape

        M, dm = Y.shape

        assert dn == dm and M == N

        combined_axis_coordinate_with_intercept, mass_X, mass_Y = self.get_mass_and_coordinate(X, Y, theta, intercept, subsequent_sources)

        combined_axis_coordinate_with_intercept[:, -1, 1] = 1e3

        point_to_source, source_to_source = self.get_H_seq_of_line(combined_axis_coordinate_with_intercept)

        tree_mass = self.compute_tree_mass(source_to_source, point_to_source, mass_X, mass_Y)

        

        dt_combined_axis_coordinate_with_intercept = torch.sort(combined_axis_coordinate_with_intercept, dim=-1)

        combined_axis_coordinate_with_intercept_sorted = dt_combined_axis_coordinate_with_intercept.values

        edge_length = combined_axis_coordinate_with_intercept_sorted[:, :, 1:] - combined_axis_coordinate_with_intercept_sorted[:, :, :-1]

        edge_length = edge_length.view(edge_length.size(0), -1).unsqueeze(1).clone()

        
        # Use simple edge_length for TWD (no Sobolev weights)
        substract_mass = ((torch.abs(tree_mass)) ** self.p) * edge_length
        substract_mass = substract_mass.view(substract_mass.size(0), -1)

        substract_mass_sum = torch.sum(substract_mass, dim = 1)

        tw = torch.mean(substract_mass_sum) ** (1/self.p)

        return tw

    def get_mass_and_coordinate(self, X, Y, theta, intercept, subsequent_sources):

        # for the last dimension

        # 0, 1, 2, ...., N -1 is of distribution 1

        # N, N + 1, ...., 2N -1 is of distribution 2

        N, dn = X.shape

        M, dm = Y.shape

        assert dn == dm and M == N

        subsequent_sources_translated = subsequent_sources - intercept

        

        subsequent_sources_coordinate = torch.einsum('abc,abc->ab', subsequent_sources_translated, theta).unsqueeze(-1)

        theta_norm = torch.norm(theta, dim=-1, keepdim=True)

        theta = theta / theta_norm

        mass_X, axis_coordinate_X = self.project(X, theta=theta, intercept=intercept)

        mass_Y, axis_coordinate_Y = self.project(Y, theta=theta, intercept=intercept)

        mass_X = torch.cat((mass_X, torch.zeros((mass_X.shape[0], mass_X.shape[1], N), device=self.device)), dim=2)

        mass_Y = torch.cat((torch.zeros((mass_Y.shape[0], mass_Y.shape[1], M), device=self.device), mass_Y), dim=2)

        mass_X = torch.transpose(mass_X, -2, -1)

        mass_Y = torch.transpose(mass_Y, -2, -1)

        mass_X = mass_X.flatten(-2, -1).unsqueeze(-2)

        mass_Y = mass_Y.flatten(-2, -1).unsqueeze(-2)

        combined_axis_coordinate = torch.cat((axis_coordinate_X, axis_coordinate_Y), dim=2)

        intercept_coordinate = torch.zeros((combined_axis_coordinate.shape[0], combined_axis_coordinate.shape[1], 1), device=self.device)

        combined_axis_coordinate_with_intercept = torch.cat((intercept_coordinate, subsequent_sources_coordinate, combined_axis_coordinate), dim=2)

        return combined_axis_coordinate_with_intercept, mass_X, mass_Y      

    def project(self, input, theta, intercept):

        N, d = input.shape

        num_trees = theta.shape[0]

        num_lines = theta.shape[1]

        input = input.unsqueeze(0).unsqueeze(0).repeat(theta.shape[0], theta.shape[1], 1, 1)

        intercept = intercept.unsqueeze(2).repeat(1, 1, N, 1)

        input_translated = input - intercept

        axis_coordinate = torch.einsum('teld,ted->tel', input_translated, theta)

        # Compute projected points for distance-based mass division
        if self.mass_division == 'distance_based':
            input_projected_translated = torch.einsum('teld,ted->teld', axis_coordinate.unsqueeze(-1), theta)
            dist = torch.norm(input_projected_translated - input_translated, dim=-1)
            weight = -self.delta * dist
            mass_input = torch.softmax(weight, dim=-2) / N
        elif self.mass_division == 'uniform':
            mass_input = torch.ones((num_trees, num_lines, N), device=self.device) / (N * num_lines)

        return mass_input, axis_coordinate

    def find_indices(self, tensor, values):

        bsz, num_row, num_col = tensor.shape

        temp =  torch.nonzero(tensor[..., None] == values)

        indices = temp[:, :-1]

        index_type = temp[:, -1]

        output = torch.full([values.shape[0], bsz, num_row],

                        float(-1e-9), device=tensor.device, dtype=torch.float)

        output[index_type, indices[:, 0], indices[:, 1]] = indices[:, 2].float()

        return output

    def get_H_seq_of_line(self, coord_matrix):

        num_tree, num_line, num_point_per_line = coord_matrix.shape

        num_projection_point = num_point_per_line - 2

        num_segment = num_point_per_line - 1

        coord_matrix_sorted, indices = torch.sort(coord_matrix, dim=2)

        del coord_matrix_sorted

        mask = indices - 2

        point_to_find = torch.tensor([-2, -1, *list(range(0, num_projection_point))], dtype=torch.int64, device=mask.device)

        indices_source_branch_proj_point = self.find_indices(mask, point_to_find)

        indices_source_point = indices_source_branch_proj_point[0].unsqueeze(0).repeat(num_projection_point+1, 1, 1).unsqueeze(0)

        indices_source_branch_proj_point = indices_source_branch_proj_point[1:].unsqueeze(0)

        indices_source_branch_proj_point, _ = torch.cat([indices_source_point, indices_source_branch_proj_point], dim=0).sort(dim=0)

        source_to_branch_proj_point_left = torch.zeros([num_projection_point+1, num_tree, num_line, num_segment], device=mask.device, dtype=torch.float)

        source_to_branch_proj_point_right = torch.zeros_like(source_to_branch_proj_point_left, device=mask.device, dtype=torch.float)

        ones = torch.ones_like(source_to_branch_proj_point_left, device=mask.device, dtype=torch.float)

        source_to_branch_proj_point_left.scatter_(dim=-1, index=indices_source_branch_proj_point[0].unsqueeze(-1).long(), src=ones)

        source_to_branch_proj_point_right.scatter_(dim=-1, index=(indices_source_branch_proj_point[1].unsqueeze(-1) - 1).long(), src=ones)

        source_to_branch_proj_point_left = torch.cumsum(source_to_branch_proj_point_left, dim=-1)

        source_to_branch_proj_point_right = torch.cumsum(source_to_branch_proj_point_right.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))

        source_to_branch_proj_point = source_to_branch_proj_point_left * source_to_branch_proj_point_right

        source_to_proj_point = source_to_branch_proj_point[1:].transpose(0, 1)

        return source_to_proj_point, source_to_branch_proj_point[0]

    def compute_tree_mass(self, source_to_source, point_to_source, mass_X, mass_Y):

        """

        Args:

            source_to_source: (num_trees, num_lines, num_segments)

            point_to_source: (num_trees, num_projection_points, num_lines, num_segments)

            mass_X: (num_trees, 1, num_line * num_projection_points)

            mass_Y: (num_trees, 1, num_line * num_projection_points)

        """

        num_trees, num_lines, num_segments = source_to_source.shape

        num_projection_points = num_segments - 1

        sub_mass = mass_X - mass_Y

        sub_mass = sub_mass.reshape(num_trees, num_projection_points, num_lines)

        """

        mass = [

                [[ p_1l_1, ... , p_1l_n, ...., p_ml_1, ... , p_ml_n ]]

                ....

                [[........]]

            ]

        mass_cumsum is expected to be

            [

                [[ p_1l_1 + ... + p_1l_n, ... , p_1l_(n-1) + p_1l_n, p_1l_n, ....]]

                ....

                [[........]]

            ]

        where p_il_j means ith projection of i-th point on j-th line.

        """

        mass_cumsum = sub_mass.sum(dim=1)

        mass_cumsum = mass_cumsum + torch.sum(mass_cumsum, dim=1, keepdims=True) - torch.cumsum(mass_cumsum, dim=1)

        mass_cumsum = torch.cat((mass_cumsum[:, 1:], torch.zeros((num_trees, 1), device=mass_X.device)), dim=1)

        first_mass = source_to_source * mass_cumsum.unsqueeze(2)

        second_mass = point_to_source * sub_mass.unsqueeze(3)

        second_mass = second_mass.sum(1)

        

        return (first_mass + second_mass).reshape(num_trees, 1, -1)

