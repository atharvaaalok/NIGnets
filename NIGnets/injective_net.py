import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Callable


class NIGnet(nn.Module):
    """
    Neural Injective Geometry network (NIGnet) for mapping a parameter `t` in [0, 1] onto a simple
    closed curve.

    Attributes
    ----------
    available_intersection_modes : list[str]
        List of valid intersection modes. Valid values are ['possible', 'impossible']
    intersection : str
        The chosen intersection mode that determines the linear layer to be `nn.Linear` or
        `ExpLinear`.
    closed_transform : Callable[[torch.Tensor], torch.Tensor]
        A function that maps `t` (in [0, 1]) onto the unit circle.
    linear_act_stack : nn.Sequential
        A sequence of alternating linear layers (or ExpLinear layers) and activation functions.
    
    Parameters
    ----------
    layer_count : int
        The number of (linear + activation) layers in the network. There is one extra linear layer
        at the end.
    act_fn : Callable[[], torch.nn.Module]
        Activation function constructor (e.g. `nn.ReLU`)
    intersection : str, optional
        Intersection mode determining layer type, by default 'possible'.
        Must be one of ['possible', 'impossible'].
        If set to 'possible', standard nn.Linear is used;
        If set to 'impossible', ExpLinear is used.
    """

    available_intersection_modes = ['possible', 'impossible']

    def __init__(
        self,
        layer_count: int,
        act_fn: Callable[[], nn.Module],
        intersection: str = 'possible'
    ) -> None:
        """
        Initialize NIGnet with specified architecture and intersection mode.
        
        Raises
        ------
        ValueError
            If `intersection` is not one of the supported modes.
        """

        super(NIGnet, self).__init__()

        if intersection not in self.available_intersection_modes:
            raise ValueError(f'Invalid intersection mode. ' \
                             f'Choose from {self.available_intersection_modes}')
        self.intersection = intersection

        # Define the transformation from t on the [0, 1] interval to unit circle for closed shapes
        self.closed_transform = lambda t: torch.hstack([
            torch.cos(2 * torch.pi * t),
            torch.sin(2 * torch.pi * t)
        ])

        Linear_class = nn.Linear if intersection == 'possible' else ExpLinear
        
        layers = []
        for i in range(layer_count):
            layers.append(Linear_class(2, 2))
            layers.append(act_fn())
        layers.append(Linear_class(2, 2))
        
        self.linear_act_stack = nn.Sequential(*layers)
    

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the NIGnet.

        Parameters
        ----------
        t : torch.Tensor
            A tensor of shape (N, 1) with values in [0, 1] representing the parameter for the simple
            closed curve.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 2) containing 2D coordinates of points on the simple closed
            curve represented by the network.
        """

        X = self.closed_transform(t)
        X = self.linear_act_stack(X)
        return X


class ExpLinear(nn.Module):
    """Linear layer with matrix-exponentiated weight transformation.

    Performs linear transformation using W = exp(weight_matrix) instead of direct weight matrix
    multiplication. This ensures injectivity of the transformation.

    Attributes
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    W : torch.nn.Parameter
        The learnable weight matrix, which is exponentiated in the forward pass.
    bias : torch.nn.Parameter or None
        Optional learnable bias term for each output feature.

    Parameters
    ----------
    in_features : int
        Number of features in the input.
    out_features : int
        Number of features in the output.
    bias : bool, optional
        If True, a learnable bias is included, by default True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Initialize the ExpLinear module.
        
        Raises
        ------
        AssertionError
            If in_features != out_features.
        """
        
        super().__init__()

        assert in_features == out_features, 'ExpLinear requires in_features == out_features'

        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the layer using Kaiming uniform initialization for the weight
        matrix, and a uniform distribution for the bias.
        """

        nn.init.kaiming_uniform_(self.W, a = math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of ExpLinear.

        The weight matrix `self.W` is exponentiated using `torch.matrix_exp` before being applied to
        the input `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_features).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_features)
        """
        
        exp_weight = torch.matrix_exp(self.W)
        return F.linear(x, exp_weight.t(), self.bias)