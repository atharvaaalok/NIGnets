import torch
import torch.nn as nn
import torch.nn.functional as F


class MinMaxNet(nn.Module):

    def __init__(self, input_dim, n_groups, nodes_per_group, monotonicity = None):
        super().__init__()

        self.input_dim = input_dim
        self.n_groups = n_groups
        self.nodes_per_group = nodes_per_group

        # If no monotonicity info is provided, assume increasing monotonicity for all inputs
        if monotonicity is None:
            monotonicity = [1] * input_dim
        
        if len(monotonicity) != input_dim:
            raise ValueError(
                f'Expected monotonicity to have length {input_dim}, got {len(monotonicity)}.'
            )
        
        # Convert monotonicity from list of -1, 0, 1 to actual signs
        # We store them in a buffer so that pytorch does not treat them as parameters
        self.register_buffer(
            "mono_signs",
            torch.tensor(monotonicity, dtype = torch.float32).view(1, 1, -1)
        )
        
        # raw_weights will be exponentiated in forward() for monotonicity constraints
        self.raw_weights = nn.Parameter(
            torch.empty(n_groups, nodes_per_group, input_dim)
        )
        nn.init.trunc_normal_(self.raw_weights)

        self.biases = nn.Parameter(
            torch.zeros(n_groups, nodes_per_group)
        )
    

    def forward(self, x):
        # Constrain weights according to monotonicity
        # - If monotonic sign is +1, we use exp(raw_weights) to force positivity
        # - If monotonic sign is -1, we use -exp(raw_weights) to force negativity
        # - If monotonic sign is 0, we use raw_weights and leave them unconstrained
        w_exp = torch.exp(self.raw_weights)
        sign_matrix = torch.sign(self.mono_signs).expand_as(self.raw_weights)
        w_actual = torch.where(
            sign_matrix == 0,
            self.raw_weights,
            sign_matrix * w_exp
        )
        
        x_expanded = x.unsqueeze(1).unsqueeze(1)    # (batch_size, 1, 1, input_dim)
        w_expanded = w_actual.unsqueeze(0)          # (1, n_groups, nodes_per_group, input_dim)
        bias_expanded = self.biases.unsqueeze(0)    # (1, n_groups, nodes_per_group)

        lin = (x_expanded * w_expanded).sum(dim = -1)
        lin_out = lin + bias_expanded               # (batch_size, n_groups, nodes_per_group)

        # Within each group, take the max over nodes
        group_out = lin_out.max(dim = 2).values     # (batch_size, n_groups)
        # Take the min across groups
        y = group_out.min(dim = 1).values           # (batch_size,)

        return y