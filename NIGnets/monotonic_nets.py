import torch
import torch.nn as nn
import torch.nn.functional as F


class MinMaxNet(nn.Module):
    """
    MinMax Monotonic Network architecture as described in [1]_ that computes a "min of max"
    operation across groups of linear units, with constraints on monotonicity for each input
    dimension.

    Attributes
    ----------
    input_dim : int
        Number of input features.
    n_groups : int
        Number of groups.
    nodes_per_group : int
        Number of nodes within each group.
    mono_signs : torch.Tensor
        A buffer indicating the monotonic signs for each input dimension of shape (1, 1, input_dim).
        Derived from the 'monotonicity' list.
    raw_weights : torch.nn.Parameter
        The raw weights for the linear units of shape (n_groups, nodes_per_group, input_dim).
        These are exponentiated or left unconstrained in `forward()` based on `mono_signs`.
    biases : torch.nn.Parameter
        Bias terms for each node of shape (n_groups, nodes_per_group).
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    n_groups : int
        Number of groups.
    nodes_per_group : int
        Number of nodes within each group.
    monotonicity: list[int], optional
        List specifying monotonicity constraints for each input dimension.
        Possible values for each element:
          - 1  -> strictly increasing
          - -1 -> strictly decreasing
          - 0  -> no monotonic constraint
        If None, a list of all +1 is used (strictly increasing). The default is None.
    
    References
    ----------
    .. [1] Sill, J. (1997). Monotonic networks. Advances in neural information processing systems,
        10.
    """

    def __init__(
        self,
        input_dim: int,
        n_groups: int,
        nodes_per_group: int,
        monotonicity: list = None
    ) -> None:
        """
        Constructs all the necessary attributes for the MinMaxNet.

        Raises
        ------
        ValueError
            If the length of `monotonicity` is not equal to `input_dim`.
        """

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
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MinMaxNet.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size,).
        """

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